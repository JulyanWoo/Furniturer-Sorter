//! Model-free local image correspondences with geometric verification.
//! Appearance similarity retrieves candidates; only registered details establish identity.
use crate::{cache::CacheStore, fingerprint::load_oriented_thumbnail, scanner::ScannedImage};
use bincode::Options;
use image::{GrayImage, imageops};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::{
    collections::HashMap,
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
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct LocalEvidence {
    pub tentative_matches: usize,
    pub inliers: usize,
    pub inlier_fraction: f32,
    pub coverage_a: f32,
    pub coverage_b: f32,
    pub occupied_cells: usize,
    pub median_error: f32,
    pub verified: bool,
    pub registered_overlap: f32,
    pub overlap_cells: usize,
    pub gradient_matching: bool,
}
pub type LocalEvidenceMap = HashMap<(usize, usize), LocalEvidence>;

/// Bound verification work to broad nearest candidates, deduplicated by content SHA.
/// No image path, timestamp or human label contributes to identity decisions.
pub fn candidate_evidence(
    images: &[ScannedImage],
    signatures: &[Option<LocalSignature>],
    pairs: &[crate::similarity::PairComparison],
    cache: &CacheStore,
    cancel: &Arc<AtomicBool>,
) -> LocalEvidenceMap {
    let mut first = HashMap::new();
    let representatives: Vec<_> = images
        .iter()
        .enumerate()
        .map(|(i, im)| {
            *first
                .entry(
                    im.content_sha256
                        .as_ref()
                        .map(|h| format!("sha256:{h}"))
                        .unwrap_or_else(|| im.stable_id()),
                )
                .or_insert(i)
        })
        .collect();
    let mut scores: HashMap<(usize, usize), f32> = HashMap::new();
    for p in pairs {
        let a = representatives[p.i];
        let b = representatives[p.j];
        if a == b {
            continue;
        }
        let d = &p.score.deterministic;
        let score = p
            .score
            .combined
            .max(0.5 * d.structure + 0.5 * d.partial_view);
        scores
            .entry((a.min(b), a.max(b)))
            .and_modify(|s| *s = s.max(score))
            .or_insert(score);
    }
    let mut neighbors = vec![Vec::new(); images.len()];
    for (&(a, b), &s) in &scores {
        neighbors[a].push((b, s));
        neighbors[b].push((a, s));
    }
    let mut candidates = std::collections::HashSet::new();
    for (a, list) in neighbors.iter_mut().enumerate() {
        list.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        for &(b, _) in list.iter().take(64) {
            candidates.insert((a.min(b), a.max(b)));
        }
    }
    let mut candidates: Vec<_> = candidates.into_iter().collect();
    candidates.sort_unstable();
    let mut cache_key = VERSION.to_le_bytes().to_vec();
    cache_key.extend_from_slice(&1u32.to_le_bytes()); // Verification algorithm version.
    for &(a, b) in &candidates {
        cache_key.extend_from_slice(images[a].content_sha256.as_deref().unwrap_or("").as_bytes());
        cache_key.extend_from_slice(images[b].content_sha256.as_deref().unwrap_or("").as_bytes());
    }
    let cache_file = cache.cache_dir().join(format!(
        "registered_pairs_{}.bin",
        crate::fingerprint::sha256_hex(&cache_key)
    ));
    if !cancel.load(Ordering::Relaxed)
        && fs::metadata(&cache_file).is_ok_and(|m| m.len() <= 64 * 1024 * 1024)
    {
        if let Ok(bytes) = fs::read(&cache_file) {
            if bytes.len() >= 8
                && u64::from_le_bytes(bytes[..8].try_into().unwrap()) == candidates.len() as u64
            {
                if let Ok(entries) = bincode::DefaultOptions::new()
                    .with_fixint_encoding()
                    .with_limit(64 * 1024 * 1024)
                    .deserialize::<Vec<Option<LocalEvidence>>>(&bytes)
                {
                    if entries.len() == candidates.len() {
                        return candidates
                            .into_iter()
                            .zip(entries)
                            .filter_map(|(k, v)| v.map(|v| (k, v)))
                            .collect();
                    }
                }
            }
        }
    }
    let result: LocalEvidenceMap = candidates
        .par_iter()
        .filter_map(|&(a, b)| {
            if cancel.load(Ordering::Relaxed) {
                return None;
            }
            let (Some(sa), Some(sb)) = (&signatures[a], &signatures[b]) else {
                return None;
            };
            Some(((a, b), compare(sa, sb)))
        })
        .collect();
    if !cancel.load(Ordering::Relaxed) {
        let entries: Vec<_> = candidates.iter().map(|k| result.get(k).cloned()).collect();
        if let Ok(bytes) = bincode::serialize(&entries) {
            let _ = fs::write(cache_file, bytes);
        }
    }
    result
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

fn gradient_matches(a: &LocalSignature, b: &LocalSignature) -> Vec<(usize, usize)> {
    let mut aa = vec![(usize::MAX, u32::MAX, u32::MAX); a.points.len()];
    let mut bb = vec![(usize::MAX, u32::MAX); b.points.len()];
    for (i, x) in a
        .points
        .iter()
        .enumerate()
        .filter(|(_, p)| p.gradients.len() == 128)
    {
        for (j, y) in b
            .points
            .iter()
            .enumerate()
            .filter(|(_, p)| p.gradients.len() == 128)
        {
            let d: u32 = x
                .gradients
                .iter()
                .zip(&y.gradients)
                .map(|(&a, &b)| (a as i32 - b as i32).pow(2) as u32)
                .sum();
            if d < aa[i].1 {
                aa[i].2 = aa[i].1;
                aa[i].1 = d;
                aa[i].0 = j;
            } else if d < aa[i].2 {
                aa[i].2 = d;
            }
            if d < bb[j].1 {
                bb[j] = (i, d);
            }
        }
    }
    aa.iter()
        .enumerate()
        .filter_map(|(i, &(j, d, next))| {
            (j != usize::MAX && d < 24000 && (d as f32) < 0.64 * next as f32 && bb[j].0 == i)
                .then_some((i, j))
        })
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

fn distance(a: &LocalPoint, b: &LocalPoint) -> u32 {
    let rotated: u32 = a
        .descriptor
        .iter()
        .zip(&b.descriptor)
        .map(|(x, y)| (x ^ y).count_ones())
        .sum();
    let plain: u32 = a
        .plain_descriptor
        .iter()
        .zip(&b.plain_descriptor)
        .map(|(x, y)| (x ^ y).count_ones())
        .sum();
    rotated.min(plain)
}

fn affine(
    a: &LocalPoint,
    b: &LocalPoint,
    c: &LocalPoint,
    u: &LocalPoint,
    v: &LocalPoint,
    w: &LocalPoint,
) -> Option<[f32; 6]> {
    let det = (b.x - a.x) * (c.y - a.y) - (c.x - a.x) * (b.y - a.y);
    if det.abs() < 0.003 {
        return None;
    }
    let fit = |p: f32, q: f32, r: f32| {
        let x = ((q - p) * (c.y - a.y) - (r - p) * (b.y - a.y)) / det;
        let y = ((b.x - a.x) * (r - p) - (c.x - a.x) * (q - p)) / det;
        (x, y, p - x * a.x - y * a.y)
    };
    let (xx, xy, xt) = fit(u.x, v.x, w.x);
    let (yx, yy, yt) = fit(u.y, v.y, w.y);
    let area = xx * yy - xy * yx;
    if !(0.10..=6.0).contains(&area) {
        return None;
    }
    Some([xx, xy, xt, yx, yy, yt])
}

/// Validate an alignment with many textured patches, not blank walls or global color.
fn registered_overlap(a: &LocalSignature, b: &LocalSignature, t: [f32; 9]) -> (f32, usize) {
    if a.raster.len() != RASTER * RASTER || b.raster.len() != RASTER * RASTER {
        return (0.0, 0);
    }
    let pixel = |r: &[u8], x: f32, y: f32| {
        let x = x * (RASTER - 1) as f32;
        let y = y * (RASTER - 1) as f32;
        let ix = (x.floor() as usize).min(RASTER - 2);
        let iy = (y.floor() as usize).min(RASTER - 2);
        let dx = x - ix as f32;
        let dy = y - iy as f32;
        (r[iy * RASTER + ix] as f32 * (1.0 - dx) + r[iy * RASTER + ix + 1] as f32 * dx) * (1.0 - dy)
            + (r[(iy + 1) * RASTER + ix] as f32 * (1.0 - dx)
                + r[(iy + 1) * RASTER + ix + 1] as f32 * dx)
                * dy
    };
    let (mut considered, mut good, mut valid) = (0, 0, 0);
    let mut cells = 0u16;
    for gy in 0..20 {
        for gx in 0..20 {
            let x = 0.05 + gx as f32 * 0.90 / 19.0;
            let y = 0.05 + gy as f32 * 0.90 / 19.0;
            let mut av = [0.0; 9];
            let mut bv = [0.0; 9];
            let mut valid_patch = true;
            for (idx, (dx, dy)) in (-1..=1)
                .flat_map(|dy| (-1..=1).map(move |dx| (dx, dy)))
                .enumerate()
            {
                let ax = x + dx as f32 * 0.023;
                let ay = y + dy as f32 * 0.023;
                let divisor = t[6] * ax + t[7] * ay + t[8];
                if divisor.abs() < 0.1 {
                    valid_patch = false;
                    break;
                }
                let bx = (t[0] * ax + t[1] * ay + t[2]) / divisor;
                let by = (t[3] * ax + t[4] * ay + t[5]) / divisor;
                if !(0.01..0.99).contains(&bx) || !(0.01..0.99).contains(&by) {
                    valid_patch = false;
                    break;
                }
                av[idx] = pixel(&a.raster, ax, ay);
                bv[idx] = pixel(&b.raster, bx, by);
            }
            if !valid_patch {
                continue;
            }
            valid += 1;
            let am = av.iter().sum::<f32>() / 9.0;
            let bm = bv.iter().sum::<f32>() / 9.0;
            let aa = av.iter().map(|v| (v - am).powi(2)).sum::<f32>();
            if aa < 9.0 * 64.0 {
                continue;
            }
            considered += 1;
            let bb = bv.iter().map(|v| (v - bm).powi(2)).sum::<f32>();
            if bb < 9.0 * 36.0 {
                continue;
            }
            let corr = av
                .iter()
                .zip(&bv)
                .map(|(a, b)| (a - am) * (b - bm))
                .sum::<f32>()
                / (aa * bb).sqrt();
            if corr >= 0.80 {
                good += 1;
                cells |= 1 << ((gy / 5) * 4 + gx / 5);
            }
        }
    }
    if considered < 48 || valid < 160 {
        return (0.0, 0);
    }
    (good as f32 / considered as f32, cells.count_ones() as usize)
}

fn projective(
    a: &LocalSignature,
    b: &LocalSignature,
    matches: &[(usize, usize)],
    ids: [usize; 4],
) -> Option<[f32; 9]> {
    let mut rows = [[0.0f64; 9]; 8];
    for (k, idx) in ids.into_iter().enumerate() {
        let (i, j) = matches[idx];
        let p = &a.points[i];
        let q = &b.points[j];
        let (x, y, u, v) = (p.x as f64, p.y as f64, q.x as f64, q.y as f64);
        rows[2 * k] = [x, y, 1.0, 0.0, 0.0, 0.0, -u * x, -u * y, u];
        rows[2 * k + 1] = [0.0, 0.0, 0.0, x, y, 1.0, -v * x, -v * y, v];
    }
    for col in 0..8 {
        let pivot = (col..8).max_by(|&i, &j| rows[i][col].abs().total_cmp(&rows[j][col].abs()))?;
        if rows[pivot][col].abs() < 1e-7 {
            return None;
        }
        rows.swap(col, pivot);
        let divisor = rows[col][col];
        for x in col..9 {
            rows[col][x] /= divisor;
        }
        for row in 0..8 {
            if row != col {
                let scale = rows[row][col];
                for x in col..9 {
                    rows[row][x] -= scale * rows[col][x];
                }
            }
        }
    }
    let mut t = [0.0; 9];
    for i in 0..8 {
        t[i] = rows[i][8] as f32;
    }
    t[8] = 1.0;
    let mut corners = Vec::new();
    for (x, y) in [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)] {
        let z = t[6] * x + t[7] * y + 1.0;
        if z < 0.15 || !z.is_finite() {
            return None;
        }
        let u = (t[0] * x + t[1] * y + t[2]) / z;
        let v = (t[3] * x + t[4] * y + t[5]) / z;
        if !u.is_finite() || !v.is_finite() || u.abs() > 4.0 || v.abs() > 4.0 {
            return None;
        }
        corners.push((u, v));
    }
    let area = (0..4)
        .map(|i| corners[i].0 * corners[(i + 1) % 4].1 - corners[(i + 1) % 4].0 * corners[i].1)
        .sum::<f32>()
        * 0.5;
    if !(0.10..6.0).contains(&area) {
        return None;
    }
    Some(t)
}

fn projective_fallback(
    a: &LocalSignature,
    b: &LocalSignature,
    matches: &[(usize, usize)],
) -> (f32, usize, usize) {
    let mut rng = 0x831c_694f_d028_a62b;
    let mut proposals = Vec::new();
    for _ in 0..96 {
        let ids = std::array::from_fn(|_| random(&mut rng) as usize % matches.len());
        if (0..4).any(|i| (i + 1..4).any(|j| ids[i] == ids[j])) {
            continue;
        }
        let Some(t) = projective(a, b, matches, ids) else {
            continue;
        };
        let support = matches
            .iter()
            .filter(|&&(i, j)| {
                let p = &a.points[i];
                let q = &b.points[j];
                let z = t[6] * p.x + t[7] * p.y + 1.0;
                ((t[0] * p.x + t[1] * p.y + t[2]) / z - q.x).powi(2)
                    + ((t[3] * p.x + t[4] * p.y + t[5]) / z - q.y).powi(2)
                    < 0.024f32.powi(2)
            })
            .count();
        if support < 4 || support as f32 / (matches.len() as f32) < 0.60 {
            continue;
        }
        if proposals.iter().any(|(_, previous): &(usize, [f32; 9])| {
            previous
                .iter()
                .zip(t)
                .map(|(a, b)| (a - b).abs())
                .sum::<f32>()
                < 0.01
        }) {
            continue;
        }
        proposals.push((support, t));
    }
    proposals.sort_by(|a, b| b.0.cmp(&a.0));
    let mut best = (0.0, 0, 0);
    for (support, t) in proposals.into_iter().take(12) {
        let (score, cells) = registered_overlap(a, b, t);
        if score > best.0 {
            best = (score, cells, support);
        }
    }
    best
}

pub fn compare(a: &LocalSignature, b: &LocalSignature) -> LocalEvidence {
    if a.points.len() < 12 || b.points.len() < 12 {
        return LocalEvidence::default();
    }
    let mut best_a = vec![(usize::MAX, 257u32, 257u32); a.points.len()];
    let mut best_b = vec![(usize::MAX, 257u32); b.points.len()];
    for (i, x) in a.points.iter().enumerate() {
        for (j, y) in b.points.iter().enumerate() {
            let d = distance(x, y);
            if d < best_a[i].1 {
                best_a[i].2 = best_a[i].1;
                best_a[i].1 = d;
                best_a[i].0 = j;
            } else if d < best_a[i].2 {
                best_a[i].2 = d;
            }
            if d < best_b[j].1 {
                best_b[j] = (i, d);
            }
        }
    }
    let matches: Vec<_> = best_a
        .iter()
        .enumerate()
        .filter_map(|(i, &(j, d, next))| {
            (d <= 65 && (d as f32) < 0.78 * next as f32 && best_b[j].0 == i).then_some((i, j))
        })
        .collect();
    let binary = verify_correspondences(a, b, &matches, false);
    if binary.verified {
        return binary;
    }
    let gradient = gradient_matches(a, b);
    if gradient.len() >= 4 {
        let alternate = verify_correspondences(a, b, &gradient, true);
        if alternate.verified || alternate.inliers > binary.inliers {
            return alternate;
        }
    }
    binary
}

fn verify_correspondences(
    a: &LocalSignature,
    b: &LocalSignature,
    matches: &[(usize, usize)],
    gradient_matching: bool,
) -> LocalEvidence {
    let mut result = LocalEvidence {
        tentative_matches: matches.len(),
        gradient_matching,
        ..Default::default()
    };
    if matches.len() < 4 {
        return result;
    }
    let mut rng = 0x1cef_785a_f123_469d;
    let mut best = Vec::new();
    let mut best_transform = None;
    for _ in 0..256 {
        let ids = [
            random(&mut rng) as usize % matches.len(),
            random(&mut rng) as usize % matches.len(),
            random(&mut rng) as usize % matches.len(),
        ];
        if ids[0] == ids[1] || ids[0] == ids[2] || ids[1] == ids[2] {
            continue;
        }
        let [(i, j), (k, l), (m, n)] = ids.map(|v| matches[v]);
        let Some(t) = affine(
            &a.points[i],
            &a.points[k],
            &a.points[m],
            &b.points[j],
            &b.points[l],
            &b.points[n],
        ) else {
            continue;
        };
        let inliers: Vec<_> = matches
            .iter()
            .enumerate()
            .filter_map(|(idx, &(i, j))| {
                let x = &a.points[i];
                let y = &b.points[j];
                let e = ((t[0] * x.x + t[1] * x.y + t[2] - y.x).powi(2)
                    + (t[3] * x.x + t[4] * x.y + t[5] - y.y).powi(2))
                .sqrt();
                (e < 0.024).then_some((idx, e))
            })
            .collect();
        if inliers.len() > best.len() {
            best = inliers;
            best_transform = Some([t[0], t[1], t[2], t[3], t[4], t[5], 0.0, 0.0, 1.0]);
        }
    }
    result.inliers = best.len();
    result.inlier_fraction = best.len() as f32 / matches.len() as f32;
    let extent = |source: bool| {
        let (mut minx, mut miny, mut maxx, mut maxy) = (1f32, 1f32, 0f32, 0f32);
        let mut cells = 0u16;
        for &(idx, _) in &best {
            let (i, j) = matches[idx];
            let p = if source { &a.points[i] } else { &b.points[j] };
            minx = minx.min(p.x);
            miny = miny.min(p.y);
            maxx = maxx.max(p.x);
            maxy = maxy.max(p.y);
            cells |= 1 << ((p.y * 4.0).min(3.0) as usize * 4 + (p.x * 4.0).min(3.0) as usize);
        }
        (
            if best.is_empty() {
                0.0
            } else {
                ((maxx - minx) * (maxy - miny)).max(0.0)
            },
            cells.count_ones() as usize,
        )
    };
    let (ca, ga) = extent(true);
    let (cb, gb) = extent(false);
    result.coverage_a = ca;
    result.coverage_b = cb;
    result.occupied_cells = ga.min(gb);
    let mut errors: Vec<_> = best.iter().map(|(_, e)| *e).collect();
    errors.sort_by(f32::total_cmp);
    result.median_error = errors.get(errors.len() / 2).copied().unwrap_or(1.0);
    result.verified = result.inliers >= 10
        && result.inlier_fraction >= 0.40
        && ca >= 0.07
        && cb >= 0.07
        && result.occupied_cells >= 3;
    if let Some(t) = best_transform {
        if result.inliers >= 4 && result.inlier_fraction >= 0.60 && ca >= 0.07 && cb >= 0.07 {
            let (agreement, cells) = registered_overlap(a, b, t);
            result.registered_overlap = agreement;
            result.overlap_cells = cells;
            if agreement >= 0.75 && cells >= 8 {
                result.verified = true;
            }
        }
    }
    if !result.verified {
        let (agreement, cells, support) = projective_fallback(a, b, &matches);
        if agreement > result.registered_overlap {
            result.registered_overlap = agreement;
            result.overlap_cells = cells;
        }
        if agreement >= 0.75 && cells >= 8 {
            result.verified = true;
            result.inliers = support;
        }
    }
    if gradient_matching && result.verified {
        result.verified = result.registered_overlap >= 0.60 && result.overlap_cells >= 6;
    }
    result
}
