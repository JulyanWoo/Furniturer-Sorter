//! Model-free local image correspondences with geometric verification.
//! Appearance similarity retrieves candidates; only registered details establish identity.
use crate::{cache::CacheStore, fingerprint::load_oriented_thumbnail, scanner::ScannedImage};
use bincode::Options;
use image::{GrayImage, imageops};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::{
    fs,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
};

const VERSION: u32 = 6;
const RASTER: usize = 96;
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LocalPoint {
    pub x: f32,
    pub y: f32,
    pub descriptor: [u64; 4],
    pub plain_descriptor: [u64; 4],
    pub gradients: Vec<u8>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LocalSignature {
    pub version: u32,
    pub points: Vec<LocalPoint>,
    pub raster: Vec<u8>,
}
fn random(state: &mut u64) -> u64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    *state
}
fn sample(image: &GrayImage, x: i32, y: i32) -> f32 {
    image.get_pixel(
        x.clamp(0, image.width() as i32 - 1) as u32,
        y.clamp(0, image.height() as i32 - 1) as u32,
    )[0] as f32
}

fn gradient_descriptor(image: &GrayImage, x: i32, y: i32) -> Vec<u8> {
    let mut direction = [0.0f32; 36];
    for dy in (-8..=8).step_by(2) {
        for dx in (-8..=8).step_by(2) {
            let gx = sample(image, x + dx + 1, y + dy) - sample(image, x + dx - 1, y + dy);
            let gy = sample(image, x + dx, y + dy + 1) - sample(image, x + dx, y + dy - 1);
            let angle = gy.atan2(gx).rem_euclid(std::f32::consts::TAU);
            direction[(angle * 36.0 / std::f32::consts::TAU) as usize % 36] +=
                (gx * gx + gy * gy).sqrt();
        }
    }
    let peak = (0..36)
        .max_by(|&a, &b| direction[a].total_cmp(&direction[b]))
        .unwrap();
    let angle = (peak as f32 + 0.5) * std::f32::consts::TAU / 36.0;
    let (sin, cos) = angle.sin_cos();
    let mut bins = [0.0f32; 128];
    for v in 0..12 {
        for u in 0..12 {
            let dx = u as f32 * 2.0 - 11.0;
            let dy = v as f32 * 2.0 - 11.0;
            let px = x + (cos * dx - sin * dy).round() as i32;
            let py = y + (sin * dx + cos * dy).round() as i32;
            let gx = sample(image, px + 1, py) - sample(image, px - 1, py);
            let gy = sample(image, px, py + 1) - sample(image, px, py - 1);
            let dir = (gy.atan2(gx) - angle).rem_euclid(std::f32::consts::TAU) * 8.0
                / std::f32::consts::TAU;
            let base = (v / 3 * 4 + u / 3) * 8;
            let weight = (gx * gx + gy * gy).sqrt();
            let fraction = dir - dir.floor();
            bins[base + (dir as usize % 8)] += weight * (1.0 - fraction);
            bins[base + ((dir as usize + 1) % 8)] += weight * fraction;
        }
    }
    let sum = bins.iter().sum::<f32>().max(0.01);
    bins.iter()
        .map(|v| ((v / sum).sqrt() * 255.0).round().clamp(0.0, 255.0) as u8)
        .collect()
}

/// Ignore letterboxed screen controls only when a substantial central photo is
/// surrounded by mostly black bands on both sides. Ordinary photos stay intact.
fn photo_region(image: &GrayImage) -> GrayImage {
    let (w, h) = image.dimensions();
    if w < 40 || h < 80 {
        return image.clone();
    }
    let photo_row = |y| (0..w).filter(|&x| image.get_pixel(x, y)[0] > 24).count() * 2 > w as usize;
    let mut best = (0u32, 0u32);
    let mut start = 0;
    for y in 0..=h {
        if y < h && photo_row(y) {
            continue;
        }
        if y - start > best.1 - best.0 {
            best = (start, y);
        }
        start = y + 1;
    }
    let (top, bottom) = best;
    if top > h / 12 && h - bottom > h / 12 && bottom - top > h / 4 {
        imageops::crop_imm(image, 0, top, w, bottom - top).to_image()
    } else {
        image.clone()
    }
}

pub fn extract(source: &GrayImage) -> LocalSignature {
    let region = photo_region(source);
    let image = &region;
    let mut points = Vec::new();
    let mut offsets = Vec::new();
    let mut rng = 0x76ac_42f1_8dce_9103;
    for _ in 0..256 {
        offsets.push((
            random(&mut rng) as i32 % 13,
            random(&mut rng) as i32 % 13,
            random(&mut rng) as i32 % 13,
            random(&mut rng) as i32 % 13,
        ));
    }
    for scale in [1.0, 0.75] {
        let w = (image.width() as f32 * scale) as u32;
        let h = (image.height() as f32 * scale) as u32;
        if w < 40 || h < 40 {
            continue;
        }
        let level = imageops::blur(
            &imageops::resize(image, w, h, imageops::FilterType::Triangle),
            0.8,
        );
        let mut corners = Vec::new();
        for y in (15..h as i32 - 15).step_by(2) {
            for x in (15..w as i32 - 15).step_by(2) {
                let (mut xx, mut yy, mut xy) = (0.0, 0.0, 0.0);
                for dy in -2..=2 {
                    for dx in -2..=2 {
                        let gx =
                            sample(&level, x + dx + 1, y + dy) - sample(&level, x + dx - 1, y + dy);
                        let gy =
                            sample(&level, x + dx, y + dy + 1) - sample(&level, x + dx, y + dy - 1);
                        xx += gx * gx;
                        yy += gy * gy;
                        xy += gx * gy;
                    }
                }
                let score = xx * yy - xy * xy - 0.04 * (xx + yy).powi(2);
                if score > 1000.0 {
                    corners.push((x, y, score));
                }
            }
        }
        corners.sort_by(|a, b| b.2.total_cmp(&a.2));
        let response_floor = corners.first().map_or(1e5, |c| (c.2 * 1e-5).max(1e5));
        let mut chosen: Vec<(i32, i32)> = Vec::new();
        let mut grid = [0u8; 64];
        for (x, y, score) in corners {
            if score < response_floor {
                break;
            }
            let cell =
                ((y as usize * 8 / h as usize).min(7)) * 8 + (x as usize * 8 / w as usize).min(7);
            if grid[cell] >= 3
                || chosen
                    .iter()
                    .any(|&(a, b)| (x - a).pow(2) + (y - b).pow(2) < 49)
            {
                continue;
            }
            chosen.push((x, y));
            grid[cell] += 1;
            // Intensity-centroid orientation makes descriptors tolerant of small camera rolls.
            let (mut mx, mut my) = (0.0, 0.0);
            for dy in -8..=8 {
                for dx in -8..=8 {
                    let value = sample(&level, x + dx, y + dy);
                    mx += dx as f32 * value;
                    my += dy as f32 * value;
                }
            }
            let angle = my.atan2(mx);
            let (sin, cos) = angle.sin_cos();
            let mut descriptor = [0u64; 4];
            let mut plain_descriptor = [0u64; 4];
            for (bit, &(ax, ay, bx, by)) in offsets.iter().enumerate() {
                let at = |dx: i32, dy: i32| {
                    sample(
                        &level,
                        x + (cos * dx as f32 - sin * dy as f32).round() as i32,
                        y + (sin * dx as f32 + cos * dy as f32).round() as i32,
                    )
                };
                if sample(&level, x + ax, y + ay) < sample(&level, x + bx, y + by) {
                    plain_descriptor[bit / 64] |= 1 << (bit % 64);
                }
                if at(ax, ay) < at(bx, by) {
                    descriptor[bit / 64] |= 1 << (bit % 64);
                }
            }
            points.push(LocalPoint {
                x: x as f32 / w as f32,
                y: y as f32 / h as f32,
                descriptor,
                plain_descriptor,
                gradients: gradient_descriptor(&level, x, y),
            });
            if chosen.len() >= 160 {
                break;
            }
        }
    }
    LocalSignature {
        version: VERSION,
        points,
        raster: imageops::resize(
            image,
            RASTER as u32,
            RASTER as u32,
            imageops::FilterType::Triangle,
        )
        .into_raw(),
    }
}

pub fn load_signatures(
    images: &[ScannedImage],
    cache: &CacheStore,
    cancel: &Arc<AtomicBool>,
) -> Vec<Option<LocalSignature>> {
    let dir = cache.cache_dir().join(format!("local_identity_v{VERSION}"));
    let _ = fs::create_dir_all(&dir);
    images
        .par_iter()
        .map(|image| {
            if cancel.load(Ordering::Relaxed) {
                return None;
            }
            let sha = image.content_sha256.as_deref()?;
            if sha.len() != 64 || !sha.bytes().all(|b| b.is_ascii_hexdigit()) {
                return None;
            }
            let file = dir.join(format!("{sha}.bin"));
            if fs::metadata(&file).is_ok_and(|m| m.len() <= 128 * 1024) {
                if let Ok(bytes) = fs::read(&file) {
                    if let Ok(sig) = bincode::DefaultOptions::new()
                        .with_fixint_encoding()
                        .with_limit(128 * 1024)
                        .deserialize::<LocalSignature>(&bytes)
                    {
                        if sig.version == VERSION
                            && sig.points.len() <= 320
                            && sig.raster.len() == RASTER * RASTER
                            && sig.points.iter().all(|p| {
                                p.gradients.len() == 128
                                    && (0.0..=1.0).contains(&p.x)
                                    && (0.0..=1.0).contains(&p.y)
                            })
                        {
                            return Some(sig);
                        }
                    }
                }
            }
            let image = load_oriented_thumbnail(&image.path, 480).ok()?.to_luma8();
            let sig = extract(&image);
            if let Ok(bytes) = bincode::serialize(&sig) {
                let _ = fs::write(file, bytes);
            }
            Some(sig)
        })
        .collect()
}
