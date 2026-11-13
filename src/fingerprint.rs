use image::imageops::FilterType;
use image::{DynamicImage, ImageReader};
use image_hasher::{HashAlg, HasherConfig};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::io::Cursor;
use std::path::Path;

pub const FINGERPRINT_VERSION: u32 = 5;
const MAX_IMAGE_FILE_BYTES: u64 = 250 * 1024 * 1024;
const MAX_IMAGE_PIXELS: u64 = 150_000_000;
const WORK_DIM: u32 = 128;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[repr(u8)]
pub enum MaterialFamily {
    WhiteLaminate = 0,
    LightWood = 1,
    WarmWood = 2,
    DarkWood = 3,
    GrayLaminate = 4,
    Black = 5,
    GraniteStone = 6,
    Other = 7,
}

impl MaterialFamily {
    pub fn name(self) -> &'static str {
        match self {
            Self::WhiteLaminate => "White Finish",
            Self::LightWood => "Light Wood",
            Self::WarmWood => "Warm Wood",
            Self::DarkWood => "Dark Wood",
            Self::GrayLaminate => "Gray Finish",
            Self::Black => "Black Finish",
            Self::GraniteStone => "Stone / Granite",
            Self::Other => "Other",
        }
    }

    pub fn from_repr(val: u8) -> Option<Self> {
        match val {
            0 => Some(Self::WhiteLaminate),
            1 => Some(Self::LightWood),
            2 => Some(Self::WarmWood),
            3 => Some(Self::DarkWood),
            4 => Some(Self::GrayLaminate),
            5 => Some(Self::Black),
            6 => Some(Self::GraniteStone),
            _ => Some(Self::Other),
        }
    }

    pub fn from_lab(l_mean: f32, a_mean: f32, b_mean: f32, chroma_mean: f32, l_std: f32) -> Self {
        if l_mean < 28.0 && chroma_mean < 14.0 {
            Self::Black
        } else if l_mean > 76.0 && chroma_mean < 13.0 {
            Self::WhiteLaminate
        } else if chroma_mean < 13.0 && l_mean >= 28.0 && l_mean <= 76.0 {
            Self::GrayLaminate
        } else if b_mean > 7.0 && a_mean > 1.5 {
            if l_mean < 50.0 {
                Self::DarkWood
            } else if l_mean <= 70.0 {
                Self::WarmWood
            } else {
                Self::LightWood
            }
        } else if l_std > 20.0 && chroma_mean < 16.0 {
            Self::GraniteStone
        } else {
            Self::Other
        }
    }

    pub fn compatibility(self, other: Self) -> (f32, bool) {
        if self == other {
            return (1.0, false);
        }
        use MaterialFamily::*;
        match (self, other) {
            // Highly compatible wood neighbors
            (LightWood, WarmWood) | (WarmWood, LightWood) => (0.78, false),
            (WarmWood, DarkWood) | (DarkWood, WarmWood) => (0.75, false),
            // Moderate compatible neutrals
            (WhiteLaminate, GrayLaminate) | (GrayLaminate, WhiteLaminate) => (0.70, false),
            (GrayLaminate, Black) | (Black, GrayLaminate) => (0.65, false),
            (GraniteStone, GrayLaminate) | (GrayLaminate, GraniteStone) => (0.60, false),
            // HARD contradictions
            (WhiteLaminate, DarkWood) | (DarkWood, WhiteLaminate) => (0.0, true),
            (WhiteLaminate, Black) | (Black, WhiteLaminate) => (0.0, true),
            (Black, LightWood) | (LightWood, Black) => (0.0, true),
            (Black, WarmWood) | (WarmWood, Black) => (0.10, true),
            _ => (0.40, false),
        }
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct FurnitureTopologySignature {
    /// Normalized vertical partition coordinates [0..1]
    pub vertical_partitions: Vec<f32>,
    /// Normalized horizontal partition coordinates [0..1]
    pub horizontal_partitions: Vec<f32>,
    /// 3x3 structural cell density grid
    pub cell_occupancy: [f32; 9],
    /// Dominant material family
    pub dominant_material: u8,
    /// Countertop / upper surface signature
    pub surface_roughness: f32,
    pub surface_color: [f32; 3],
    pub surface_y: f32,
    pub has_prominent_surface: bool,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct RegionSignature {
    /// Region identifier (0: Center, 1: Top-Left, 2: Top-Right, 3: Bottom-Left, 4: Bottom-Right).
    pub id: u8,
    /// Coarse normalized Lab color: [L/100, (a+80)/160, (b+80)/160].
    pub coarse_color: [f32; 3],
    /// Normalized mean edge energy in this region [0..1].
    pub edge_energy: f32,
    /// Ratio of vertical edge energy to total edge energy: gx / (gx + gy + eps).
    pub vertical_ratio: f32,
    /// 8-bin horizontal edge energy profile across rows.
    pub horiz_profile: Vec<f32>,
    /// 8-bin vertical edge energy profile across columns.
    pub vert_profile: Vec<f32>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct FurnitureGeometrySignature {
    /// 16-bin vertical edge energy distribution across columns (detects vertical door seams, stiles, divisions).
    pub vertical_profile: Vec<f32>,
    /// 16-bin horizontal edge energy distribution across rows (detects drawers, shelves, horizontals).
    pub horizontal_profile: Vec<f32>,
    /// Ratio of vertical edge energy to total edge energy: Ev / (Ev + Eh).
    pub vertical_edge_ratio: f32,
    /// Bilateral symmetry score [0..1] of vertical edges across vertical center line.
    pub symmetry_score: f32,
    /// Center-to-border edge ratio: inner 60% vs outer 20% border.
    pub center_to_border_ratio: f32,
    /// Estimated horizontal division count (drawers/shelves periodicity).
    pub horizontal_division_count: f32,
    /// Estimated vertical division count (doors/panels periodicity).
    pub vertical_division_count: f32,
    /// 3x3 central structural edge density grid (9 normalized values).
    pub structural_grid: Vec<f32>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ImageFingerprint {
    pub version: u32,
    pub width: u32,
    pub height: u32,
    pub aspect_ratio: f32,
    /// 64-bit DCT perceptual hash.
    pub phash: [u8; 8],
    /// 64-bit double-gradient perceptual hash.
    pub dhash: [u8; 8],
    /// Coarse normalized global Lab summary: [L_mean, a_mean, b_mean, chroma_mean, L_std].
    pub coarse_color: [f32; 5],
    /// 76-bin normalized Lab histogram: 64 bins (8x8 a*b* chromaticity) + 12 bins (L* lightness).
    pub color_histogram: Vec<f32>,
    /// Top 3 quantized Lab colors: 3 * [L_norm, a_norm, b_norm, weight].
    pub dominant_colors: Vec<f32>,
    /// 4x4 spatial layout grid: 16 * [L, a, b, L_std, edge_density, vertical_ratio].
    pub spatial_grid: Vec<f32>,
    /// Sobel edge orientation signature: 8 global + 4*8 quadrant bins = 40 floats.
    pub edge_signature: Vec<f32>,
    /// 16-bin Center-Symmetric LBP texture histogram.
    pub texture_signature: Vec<f32>,
    /// Laplacian sharpness estimate for representative selection.
    pub sharpness: f32,
    /// Central edge ratio (framing score) for representative selection.
    pub center_focus: f32,
    /// Structural furniture geometry signature (vertical/horizontal profiles, symmetry, divisions).
    #[serde(default)]
    pub geometry: FurnitureGeometrySignature,
    /// Multi-scale region signatures for partial-view matching (Center + 4 quadrants).
    #[serde(default)]
    pub regions: Vec<RegionSignature>,
    /// Structural furniture topology signature (normalized partition coordinates, materials, countertop surface).
    #[serde(default)]
    pub topology: FurnitureTopologySignature,
}

/// Decodes an image safely, applies EXIF orientation, extracts compact features on a
/// 128x128 working buffer, and immediately drops full-resolution pixel data.
pub fn extract_fingerprint(path: &Path) -> Result<ImageFingerprint, String> {
    let (fp, _) = extract_fingerprint_and_hash(path)?;
    Ok(fp)
}

/// Reads the image once, computes a stable SHA-256 content digest for caching/evaluation
/// identity, extracts the compact fingerprint, and drops raw pixel buffers immediately.
pub fn extract_fingerprint_and_hash(path: &Path) -> Result<(ImageFingerprint, String), String> {
    let meta = std::fs::metadata(path).map_err(|e| format!("Stat failed: {e}"))?;
    if meta.len() > MAX_IMAGE_FILE_BYTES {
        return Err(format!(
            "Image exceeds {} MB safety limit",
            MAX_IMAGE_FILE_BYTES / (1024 * 1024)
        ));
    }

    let bytes = std::fs::read(path).map_err(|e| format!("Read failed: {e}"))?;
    let content_sha256 = sha256_hex(&bytes);
    let fp = extract_fingerprint_from_bytes(&bytes)?;
    Ok((fp, content_sha256))
}

pub fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    let mut out = String::with_capacity(64);
    for b in digest {
        use std::fmt::Write;
        let _ = write!(&mut out, "{b:02x}");
    }
    out
}

pub fn extract_fingerprint_from_bytes(bytes: &[u8]) -> Result<ImageFingerprint, String> {
    let exif_orientation = read_exif_orientation(bytes);

    let reader = ImageReader::new(Cursor::new(bytes))
        .with_guessed_format()
        .map_err(|e| format!("Format detection failed: {e}"))?;
    let (raw_w, raw_h) = reader
        .into_dimensions()
        .map_err(|e| format!("Header decode failed: {e}"))?;
    if raw_w == 0 || raw_h == 0 {
        return Err("Image has zero dimensions".to_string());
    }
    if (raw_w as u64) * (raw_h as u64) > MAX_IMAGE_PIXELS {
        return Err("Image exceeds maximum pixel count limit".to_string());
    }

    let decoded = ImageReader::new(Cursor::new(bytes))
        .with_guessed_format()
        .map_err(|e| format!("Format detection failed: {e}"))?
        .decode()
        .map_err(|e| format!("Decode failed: {e}"))?;

    let oriented = apply_exif_orientation(decoded, exif_orientation);
    let width = oriented.width();
    let height = oriented.height();
    let aspect_ratio = width as f32 / height.max(1) as f32;

    // Immediately resize to a compact working representation and drop the full image.
    let work_img = oriented.resize_exact(WORK_DIM, WORK_DIM, FilterType::Triangle);
    drop(oriented);

    let phash = compute_hash(&work_img, true);
    let dhash = compute_hash(&work_img, false);
    let rgb = work_img.to_rgb8();
    drop(work_img);

    Ok(compute_features_from_128_rgb(
        rgb.as_raw(),
        width,
        height,
        aspect_ratio,
        phash,
        dhash,
    ))
}

/// Decodes and applies EXIF orientation, returning a bounded thumbnail image (max side `max_side`).
pub fn load_oriented_thumbnail(path: &Path, max_side: u32) -> Result<DynamicImage, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("Read failed: {e}"))?;
    let exif_orientation = read_exif_orientation(&bytes);
    let decoded = ImageReader::new(Cursor::new(&bytes))
        .with_guessed_format()
        .map_err(|e| format!("Format detection failed: {e}"))?
        .decode()
        .map_err(|e| format!("Decode failed: {e}"))?;
    let oriented = apply_exif_orientation(decoded, exif_orientation);
    let (w, h) = (oriented.width(), oriented.height());
    if w <= max_side && h <= max_side {
        Ok(oriented)
    } else {
        Ok(oriented.resize(max_side, max_side, FilterType::Triangle))
    }
}

/// Decodes and applies EXIF orientation, returning an RGB image resized for ONNX inference.
pub fn load_oriented_for_onnx(
    path: &Path,
    target_w: u32,
    target_h: u32,
) -> Result<DynamicImage, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("Read failed: {e}"))?;
    let exif_orientation = read_exif_orientation(&bytes);
    let decoded = ImageReader::new(Cursor::new(&bytes))
        .with_guessed_format()
        .map_err(|e| format!("Format detection failed: {e}"))?
        .decode()
        .map_err(|e| format!("Decode failed: {e}"))?;
    let oriented = apply_exif_orientation(decoded, exif_orientation);
    Ok(oriented.resize_exact(target_w, target_h, FilterType::Triangle))
}

fn read_exif_orientation(bytes: &[u8]) -> Option<u32> {
    let mut cursor = Cursor::new(bytes);
    let exif = exif::Reader::new().read_from_container(&mut cursor).ok()?;
    let field = exif.get_field(exif::Tag::Orientation, exif::In::PRIMARY)?;
    field.value.get_uint(0)
}

fn apply_exif_orientation(img: DynamicImage, orientation: Option<u32>) -> DynamicImage {
    match orientation {
        Some(2) => img.fliph(),
        Some(3) => img.rotate180(),
        Some(4) => img.flipv(),
        Some(5) => img.rotate90().fliph(),
        Some(6) => img.rotate90(),
        Some(7) => img.rotate270().fliph(),
        Some(8) => img.rotate270(),
        _ => img,
    }
}

fn compute_hash(img: &DynamicImage, use_dct_phash: bool) -> [u8; 8] {
    let cfg = if use_dct_phash {
        HasherConfig::new()
            .hash_size(8, 8)
            .preproc_dct()
            .hash_alg(HashAlg::Median)
    } else {
        HasherConfig::new()
            .hash_size(8, 8)
            .hash_alg(HashAlg::DoubleGradient)
    };
    let hasher = cfg.to_hasher();
    let hash = hasher.hash_image(img);
    let mut out = [0u8; 8];
    let bytes = hash.as_bytes();
    let len = bytes.len().min(8);
    out[..len].copy_from_slice(&bytes[..len]);
    out
}

fn compute_features_from_128_rgb(
    rgb: &[u8],
    width: u32,
    height: u32,
    aspect_ratio: f32,
    phash: [u8; 8],
    dhash: [u8; 8],
) -> ImageFingerprint {
    let n = (WORK_DIM * WORK_DIM) as usize;
    let mut l_plane = vec![0.0f32; n];
    let mut a_plane = vec![0.0f32; n];
    let mut b_plane = vec![0.0f32; n];

    let mut ab_hist = vec![0.0f32; 64];
    let mut l_hist = vec![0.0f32; 12];

    // 4x4x4 quantized Lab bins for dominant colors
    let mut dom_counts = [0u32; 64];
    let mut dom_sums = [[0.0f32; 3]; 64];

    let mut sum_l = 0.0f32;
    let mut sum_a = 0.0f32;
    let mut sum_b = 0.0f32;
    let mut sum_chroma = 0.0f32;

    for i in 0..n {
        let idx = i * 3;
        let (l, a, b) = srgb_to_lab(rgb[idx], rgb[idx + 1], rgb[idx + 2]);
        l_plane[i] = l;
        a_plane[i] = a;
        b_plane[i] = b;

        sum_l += l;
        sum_a += a;
        sum_b += b;
        sum_chroma += (a * a + b * b).sqrt();

        let a_bin = (((a + 50.0) / 100.0).clamp(0.0, 0.9999) * 8.0) as usize;
        let b_bin = (((b + 50.0) / 100.0).clamp(0.0, 0.9999) * 8.0) as usize;
        ab_hist[a_bin * 8 + b_bin] += 1.0;

        let l_bin = ((l / 100.0).clamp(0.0, 0.9999) * 12.0) as usize;
        l_hist[l_bin] += 1.0;

        let ql = ((l / 100.0).clamp(0.0, 0.9999) * 4.0) as usize;
        let qa = (((a + 60.0) / 120.0).clamp(0.0, 0.9999) * 4.0) as usize;
        let qb = (((b + 60.0) / 120.0).clamp(0.0, 0.9999) * 4.0) as usize;
        let qidx = (ql * 16) + (qa * 4) + qb;
        dom_counts[qidx] += 1;
        dom_sums[qidx][0] += l;
        dom_sums[qidx][1] += a;
        dom_sums[qidx][2] += b;
    }

    let nf = n as f32;
    let mean_l = sum_l / nf;
    let mean_a = sum_a / nf;
    let mean_b = sum_b / nf;
    let mean_chroma = sum_chroma / nf;

    let var_l = l_plane
        .iter()
        .map(|&l| {
            let d = l - mean_l;
            d * d
        })
        .sum::<f32>()
        / nf;
    let std_l = var_l.sqrt();

    let coarse_color = [
        (mean_l / 100.0).clamp(0.0, 1.0),
        ((mean_a + 80.0) / 160.0).clamp(0.0, 1.0),
        ((mean_b + 80.0) / 160.0).clamp(0.0, 1.0),
        (mean_chroma / 100.0).clamp(0.0, 1.0),
        (std_l / 50.0).clamp(0.0, 1.0),
    ];

    l1_normalize(&mut ab_hist);
    l1_normalize(&mut l_hist);
    let mut color_histogram = Vec::with_capacity(76);
    color_histogram.extend_from_slice(&ab_hist);
    color_histogram.extend_from_slice(&l_hist);

    // Top 3 dominant colors
    let mut dom_indices: Vec<(usize, u32)> = dom_counts
        .iter()
        .copied()
        .enumerate()
        .filter(|&(_, c)| c > 0)
        .collect();
    dom_indices.sort_unstable_by(|a, b| b.1.cmp(&a.1));
    let mut dominant_colors = vec![0.0f32; 12];
    for (slot, (qidx, count)) in dom_indices.into_iter().take(3).enumerate() {
        let cf = count as f32;
        let base = slot * 4;
        dominant_colors[base] = (dom_sums[qidx][0] / cf / 100.0).clamp(0.0, 1.0);
        dominant_colors[base + 1] = ((dom_sums[qidx][1] / cf + 80.0) / 160.0).clamp(0.0, 1.0);
        dominant_colors[base + 2] = ((dom_sums[qidx][2] / cf + 80.0) / 160.0).clamp(0.0, 1.0);
        dominant_colors[base + 3] = cf / nf;
    }

    // Sobel edges + orientation histograms + CS-LBP texture + sharpness
    let w = WORK_DIM as usize;
    let mut gx_abs_plane = vec![0.0f32; n];
    let mut gy_abs_plane = vec![0.0f32; n];
    let mut mag_plane = vec![0.0f32; n];

    let mut global_orient = [0.0f32; 8];
    let mut quad_orient = [[0.0f32; 8]; 4];
    let mut texture_hist = vec![0.0f32; 16];

    let mut lap_sum = 0.0f32;
    let mut lap_count = 0usize;
    let mut total_edge_mag = 0.0f32;
    let mut center_edge_mag = 0.0f32;

    for y in 1..(w - 1) {
        for x in 1..(w - 1) {
            let idx = y * w + x;
            let tl = l_plane[(y - 1) * w + (x - 1)];
            let t = l_plane[(y - 1) * w + x];
            let tr = l_plane[(y - 1) * w + (x + 1)];
            let l = l_plane[y * w + (x - 1)];
            let c = l_plane[idx];
            let r = l_plane[y * w + (x + 1)];
            let bl = l_plane[(y + 1) * w + (x - 1)];
            let b = l_plane[(y + 1) * w + x];
            let br = l_plane[(y + 1) * w + (x + 1)];

            let gx = (-tl - 2.0 * l - bl) + (tr + 2.0 * r + br);
            let gy = (-tl - 2.0 * t - tr) + (bl + 2.0 * b + br);
            let mag = (gx * gx + gy * gy).sqrt();

            gx_abs_plane[idx] = gx.abs();
            gy_abs_plane[idx] = gy.abs();
            mag_plane[idx] = mag;

            total_edge_mag += mag;
            if (32..96).contains(&x) && (32..96).contains(&y) {
                center_edge_mag += mag;
                let lap = (4.0 * c - t - b - l - r).abs();
                lap_sum += lap;
                lap_count += 1;
            }

            if mag > 4.0 {
                let mut angle = gy.atan2(gx);
                if angle < 0.0 {
                    angle += std::f32::consts::PI;
                }
                let bin = ((angle / std::f32::consts::PI).clamp(0.0, 0.9999) * 8.0) as usize;
                global_orient[bin] += mag;
                let q = (usize::from(y >= 64) * 2) + usize::from(x >= 64);
                quad_orient[q][bin] += mag;
            }

            // CS-LBP (4 opposite pairs around pixel with small noise deadband)
            const EPS: f32 = 2.0;
            let mut code = 0usize;
            if r - l > EPS {
                code |= 1;
            }
            if br - tl > EPS {
                code |= 2;
            }
            if b - t > EPS {
                code |= 4;
            }
            if bl - tr > EPS {
                code |= 8;
            }
            texture_hist[code] += 1.0;
        }
    }

    let mut global_orient_vec = global_orient.to_vec();
    l1_normalize(&mut global_orient_vec);
    let mut edge_signature = Vec::with_capacity(40);
    edge_signature.extend_from_slice(&global_orient_vec);
    for q in &mut quad_orient {
        let mut qv = q.to_vec();
        l1_normalize(&mut qv);
        edge_signature.extend_from_slice(&qv);
    }

    l1_normalize(&mut texture_hist);

    // 4x4 Spatial Grid (32x32 pixels per cell)
    let mut spatial_grid = Vec::with_capacity(16 * 6);
    for gy_cell in 0..4 {
        for gx_cell in 0..4 {
            let x0 = gx_cell * 32;
            let y0 = gy_cell * 32;
            let mut cl = 0.0f32;
            let mut ca = 0.0f32;
            let mut cb = 0.0f32;
            let mut cmag = 0.0f32;
            let mut cgx = 0.0f32;
            let mut cgy = 0.0f32;
            let cell_n = (32 * 32) as f32;

            for yy in y0..(y0 + 32) {
                let row = yy * w;
                for xx in x0..(x0 + 32) {
                    let i = row + xx;
                    cl += l_plane[i];
                    ca += a_plane[i];
                    cb += b_plane[i];
                    cmag += mag_plane[i];
                    cgx += gx_abs_plane[i];
                    cgy += gy_abs_plane[i];
                }
            }

            let m_l = cl / cell_n;
            let mut cell_var_l = 0.0f32;
            for yy in y0..(y0 + 32) {
                let row = yy * w;
                for xx in x0..(x0 + 32) {
                    let d = l_plane[row + xx] - m_l;
                    cell_var_l += d * d;
                }
            }
            let c_std_l = (cell_var_l / cell_n).sqrt();

            spatial_grid.push((m_l / 100.0).clamp(0.0, 1.0));
            spatial_grid.push(((ca / cell_n + 80.0) / 160.0).clamp(0.0, 1.0));
            spatial_grid.push(((cb / cell_n + 80.0) / 160.0).clamp(0.0, 1.0));
            spatial_grid.push((c_std_l / 40.0).clamp(0.0, 1.0));
            spatial_grid.push((cmag / cell_n / 180.0).clamp(0.0, 1.0));
            spatial_grid.push((cgx / (cgx + cgy + 1e-4)).clamp(0.0, 1.0));
        }
    }

    let sharpness = if lap_count > 0 {
        (lap_sum / lap_count as f32 / 25.0).clamp(0.0, 1.0)
    } else {
        0.0
    };
    let center_focus = (center_edge_mag / (total_edge_mag + 1e-5)).clamp(0.0, 1.0);

    // Furniture Geometry Signature computation
    let mut vert_profile = vec![0.0f32; 16];
    let mut horiz_profile = vec![0.0f32; 16];
    let mut sum_gx_all = 0.0f32;
    let mut sum_gy_all = 0.0f32;
    let mut center_edge_sum = 0.0f32;
    let mut border_edge_sum = 0.0f32;
    let mut grid3x3_sums = [0.0f32; 9];

    for y in 0..w {
        let r_bin = (y / 8).min(15);
        let g3_y = if y < 42 {
            0
        } else if y < 85 {
            1
        } else {
            2
        };
        let is_border_y = y < 24 || y >= 104;

        for x in 0..w {
            let c_bin = (x / 8).min(15);
            let g3_x = if x < 42 {
                0
            } else if x < 85 {
                1
            } else {
                2
            };
            let is_border_x = x < 24 || x >= 104;

            let idx = y * w + x;
            let gx = gx_abs_plane[idx];
            let gy = gy_abs_plane[idx];
            let mag = mag_plane[idx];

            vert_profile[c_bin] += gx;
            horiz_profile[r_bin] += gy;
            sum_gx_all += gx;
            sum_gy_all += gy;

            grid3x3_sums[g3_y * 3 + g3_x] += mag;

            if is_border_x || is_border_y {
                border_edge_sum += mag;
            } else {
                center_edge_sum += mag;
            }
        }
    }

    let mut norm_vp = vert_profile.clone();
    l1_normalize(&mut norm_vp);
    let mut norm_hp = horiz_profile.clone();
    l1_normalize(&mut norm_hp);

    let vertical_edge_ratio = sum_gx_all / (sum_gx_all + sum_gy_all + 1e-4);
    let center_to_border_ratio = center_edge_sum / (center_edge_sum + border_edge_sum + 1e-4);

    let mut sym_diff = 0.0f32;
    for c in 0..8 {
        sym_diff += (norm_vp[c] - norm_vp[15 - c]).abs();
    }
    let symmetry_score = (1.0 - sym_diff).clamp(0.0, 1.0);

    let h_divs = count_prominent_peaks(&norm_hp, 0.015);
    let v_divs = count_prominent_peaks(&norm_vp, 0.015);

    let mut structural_grid = grid3x3_sums.to_vec();
    l1_normalize(&mut structural_grid);

    let mut cell_occupancy = [0.0f32; 9];
    if structural_grid.len() >= 9 {
        cell_occupancy.copy_from_slice(&structural_grid[..9]);
    }

    let vp_peaks = extract_peak_positions(&norm_vp, 0.015);
    let hp_peaks = extract_peak_positions(&norm_hp, 0.015);

    let geometry = FurnitureGeometrySignature {
        vertical_profile: norm_vp,
        horizontal_profile: norm_hp,
        vertical_edge_ratio,
        symmetry_score,
        center_to_border_ratio,
        horizontal_division_count: h_divs,
        vertical_division_count: v_divs,
        structural_grid,
    };

    // Multi-Scale Region Signatures:
    // 0: Center (x: 32..96, y: 32..96)
    // 1: Top-Left (x: 0..64, y: 0..64)
    // 2: Top-Right (x: 64..128, y: 0..64)
    // 3: Bottom-Left (x: 0..64, y: 64..128)
    // 4: Bottom-Right (x: 64..128, y: 64..128)
    let regions = vec![
        extract_region_signature(
            0,
            32,
            96,
            32,
            96,
            w,
            &l_plane,
            &a_plane,
            &b_plane,
            &gx_abs_plane,
            &gy_abs_plane,
            &mag_plane,
        ),
        extract_region_signature(
            1,
            0,
            64,
            0,
            64,
            w,
            &l_plane,
            &a_plane,
            &b_plane,
            &gx_abs_plane,
            &gy_abs_plane,
            &mag_plane,
        ),
        extract_region_signature(
            2,
            64,
            128,
            0,
            64,
            w,
            &l_plane,
            &a_plane,
            &b_plane,
            &gx_abs_plane,
            &gy_abs_plane,
            &mag_plane,
        ),
        extract_region_signature(
            3,
            0,
            64,
            64,
            128,
            w,
            &l_plane,
            &a_plane,
            &b_plane,
            &gx_abs_plane,
            &gy_abs_plane,
            &mag_plane,
        ),
        extract_region_signature(
            4,
            64,
            128,
            64,
            128,
            w,
            &l_plane,
            &a_plane,
            &b_plane,
            &gx_abs_plane,
            &gy_abs_plane,
            &mag_plane,
        ),
    ];

    let dominant_material = MaterialFamily::from_lab(
        coarse_color[0],
        coarse_color[1],
        coarse_color[2],
        coarse_color[3],
        coarse_color[4],
    ) as u8;

    let avg_hp: f32 = horiz_profile.iter().sum::<f32>() / 16.0;
    let mut max_hp_r = 0usize;
    let mut max_hp_val = 0.0f32;
    for r in 2..11 {
        if horiz_profile[r] > max_hp_val {
            max_hp_val = horiz_profile[r];
            max_hp_r = r;
        }
    }

    let has_prominent_surface = max_hp_val > 1.35 * avg_hp.max(1e-4);
    let surface_y = (max_hp_r as f32 + 0.5) / 16.0;

    let mut s_l_sum = 0.0f32;
    let mut s_a_sum = 0.0f32;
    let mut s_b_sum = 0.0f32;
    let mut s_l_sq = 0.0f32;
    let s_count = (8 * w) as f32;
    let y_start = max_hp_r * 8;
    for y in y_start..(y_start + 8).min(w) {
        for x in 0..w {
            let idx = y * w + x;
            let l = l_plane[idx];
            s_l_sum += l;
            s_a_sum += a_plane[idx];
            s_b_sum += b_plane[idx];
            s_l_sq += l * l;
        }
    }
    let s_l_mean = s_l_sum / s_count;
    let surface_color = [s_l_mean, s_a_sum / s_count, s_b_sum / s_count];
    let surface_roughness = ((s_l_sq / s_count) - (s_l_mean * s_l_mean)).max(0.0).sqrt();

    let topology = FurnitureTopologySignature {
        vertical_partitions: vp_peaks,
        horizontal_partitions: hp_peaks,
        cell_occupancy,
        dominant_material,
        surface_y,
        surface_roughness,
        surface_color,
        has_prominent_surface,
    };

    ImageFingerprint {
        version: FINGERPRINT_VERSION,
        width,
        height,
        aspect_ratio,
        phash,
        dhash,
        coarse_color,
        color_histogram,
        dominant_colors,
        spatial_grid,
        edge_signature,
        texture_signature: texture_hist,
        sharpness,
        center_focus,
        geometry,
        regions,
        topology,
    }
}

fn extract_region_signature(
    id: u8,
    x0: usize,
    x1: usize,
    y0: usize,
    y1: usize,
    w: usize,
    l_plane: &[f32],
    a_plane: &[f32],
    b_plane: &[f32],
    gx_abs: &[f32],
    gy_abs: &[f32],
    mag: &[f32],
) -> RegionSignature {
    let rw = x1 - x0;
    let rh = y1 - y0;
    let total_pixels = (rw * rh) as f32;

    let mut sum_l = 0.0f32;
    let mut sum_a = 0.0f32;
    let mut sum_b = 0.0f32;
    let mut sum_gx = 0.0f32;
    let mut sum_gy = 0.0f32;
    let mut sum_mag = 0.0f32;

    let mut hp = vec![0.0f32; 8];
    let mut vp = vec![0.0f32; 8];

    for y in y0..y1 {
        let r_bin = (((y - y0) * 8) / rh).min(7);
        let row_idx = y * w;
        for x in x0..x1 {
            let c_bin = (((x - x0) * 8) / rw).min(7);
            let idx = row_idx + x;
            sum_l += l_plane[idx];
            sum_a += a_plane[idx];
            sum_b += b_plane[idx];

            let egx = gx_abs[idx];
            let egy = gy_abs[idx];
            let emag = mag[idx];

            sum_gx += egx;
            sum_gy += egy;
            sum_mag += emag;

            hp[r_bin] += emag;
            vp[c_bin] += emag;
        }
    }

    l1_normalize(&mut hp);
    l1_normalize(&mut vp);

    let edge_energy = (sum_mag / total_pixels / 180.0).clamp(0.0, 1.0);
    let vertical_ratio = (sum_gx / (sum_gx + sum_gy + 1e-4)).clamp(0.0, 1.0);

    RegionSignature {
        id,
        coarse_color: [
            (sum_l / total_pixels / 100.0).clamp(0.0, 1.0),
            ((sum_a / total_pixels + 80.0) / 160.0).clamp(0.0, 1.0),
            ((sum_b / total_pixels + 80.0) / 160.0).clamp(0.0, 1.0),
        ],
        edge_energy,
        vertical_ratio,
        horiz_profile: hp,
        vert_profile: vp,
    }
}

pub fn region_similarity(ra: &RegionSignature, rb: &RegionSignature) -> f32 {
    let dl = (ra.coarse_color[0] - rb.coarse_color[0]).abs();
    let da = (ra.coarse_color[1] - rb.coarse_color[1]).abs();
    let db = (ra.coarse_color[2] - rb.coarse_color[2]).abs();
    let color_sim = (1.0 - (0.4 * dl + 0.3 * da + 0.3 * db) * 3.0).clamp(0.0, 1.0);

    let de = (ra.edge_energy - rb.edge_energy).abs();
    let dvr = (ra.vertical_ratio - rb.vertical_ratio).abs();
    let edge_sim = (1.0 - (0.5 * de + 0.5 * dvr) * 2.5).clamp(0.0, 1.0);

    let hp_sim = ruzicka_slice(&ra.horiz_profile, &rb.horiz_profile);
    let vp_sim = ruzicka_slice(&ra.vert_profile, &rb.vert_profile);
    let profile_sim = 0.5 * hp_sim + 0.5 * vp_sim;

    (0.40 * profile_sim + 0.35 * color_sim + 0.25 * edge_sim).clamp(0.0, 1.0)
}

#[inline]
pub fn ruzicka_slice(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len().min(b.len());
    let mut num = 0.0f32;
    let mut den = 0.0f32;
    for i in 0..len {
        num += a[i].min(b[i]);
        den += a[i].max(b[i]);
    }
    if den <= 1e-6 {
        0.0
    } else {
        (num / den).clamp(0.0, 1.0)
    }
}

/// Evaluates partial-view match between two images.
/// Finds maximum similarity across significant regions (ignoring blank background regions).
pub fn partial_view_similarity(a: &ImageFingerprint, b: &ImageFingerprint) -> f32 {
    if a.regions.is_empty() || b.regions.is_empty() {
        return 0.0;
    }
    let mut max_sim = 0.0f32;
    const MIN_EDGE_ENERGY: f32 = 0.035;

    for ra in &a.regions {
        if ra.edge_energy < MIN_EDGE_ENERGY {
            continue;
        }
        for rb in &b.regions {
            if rb.edge_energy < MIN_EDGE_ENERGY {
                continue;
            }
            let s = region_similarity(ra, rb);
            if s > max_sim {
                max_sim = s;
            }
        }
    }
    max_sim
}

fn count_prominent_peaks(profile: &[f32], min_prominence: f32) -> f32 {
    if profile.len() < 3 {
        return 0.0;
    }
    let mut peaks = 0.0f32;
    for i in 1..(profile.len() - 1) {
        let v = profile[i];
        if v > profile[i - 1] && v > profile[i + 1] {
            let left_diff = v - profile[i - 1];
            let right_diff = v - profile[i + 1];
            if left_diff >= min_prominence || right_diff >= min_prominence {
                peaks += 1.0;
            }
        }
    }
    peaks
}

pub fn extract_peak_positions(profile: &[f32], min_prominence: f32) -> Vec<f32> {
    if profile.len() < 3 {
        return Vec::new();
    }
    let mut peaks = Vec::new();
    let n = profile.len() as f32;
    for i in 1..(profile.len() - 1) {
        let v = profile[i];
        if v > profile[i - 1] && v > profile[i + 1] {
            let left_diff = v - profile[i - 1];
            let right_diff = v - profile[i + 1];
            if left_diff >= min_prominence || right_diff >= min_prominence {
                peaks.push((i as f32 + 0.5) / n);
            }
        }
    }
    peaks
}

pub fn compare_topology(
    a: &FurnitureTopologySignature,
    b: &FurnitureTopologySignature,
    partial_ok: bool,
) -> (f32, f32, bool) {
    let mat_a = MaterialFamily::from_repr(a.dominant_material).unwrap_or(MaterialFamily::Other);
    let mat_b = MaterialFamily::from_repr(b.dominant_material).unwrap_or(MaterialFamily::Other);

    let mut is_hard = false;
    let mut contradiction = 0.0f32;

    // 1. Material compatibility
    let (mat_sim, mat_hard) = mat_a.compatibility(mat_b);
    if mat_hard {
        is_hard = true;
        contradiction += 0.55;
    } else if mat_sim < 0.60 {
        contradiction += (0.60 - mat_sim) * 0.40;
    }

    // 2. Cell occupancy match (3x3 grid)
    let grid_sim = ruzicka_slice(&a.cell_occupancy, &b.cell_occupancy);

    // 3. Partition topology
    let (vp_match, vp_contra, vp_hard) =
        match_partitions(&a.vertical_partitions, &b.vertical_partitions, partial_ok);
    let (hp_match, hp_contra, hp_hard) = match_partitions(
        &a.horizontal_partitions,
        &b.horizontal_partitions,
        partial_ok,
    );

    if vp_hard || hp_hard {
        is_hard = true;
    }
    contradiction += 0.35 * vp_contra + 0.25 * hp_contra;

    // 4. Surface / Countertop comparison
    let mut surf_match = 0.5f32;
    if a.has_prominent_surface && b.has_prominent_surface {
        let y_diff = (a.surface_y - b.surface_y).abs();
        let col_diff = (a.surface_color[0] - b.surface_color[0]).abs() * 0.5
            + (a.surface_color[1] - b.surface_color[1]).abs()
            + (a.surface_color[2] - b.surface_color[2]).abs();

        if col_diff > 38.0 || (y_diff > 0.32 && !partial_ok) {
            is_hard = true;
            contradiction += 0.45;
        } else {
            surf_match = (1.0 - col_diff / 40.0).clamp(0.0, 1.0) * 0.6
                + (1.0 - y_diff * 2.5).clamp(0.0, 1.0) * 0.4;
            if col_diff > 22.0 {
                contradiction += 0.15;
            }
        }
    } else if a.has_prominent_surface != b.has_prominent_surface && !partial_ok {
        contradiction += 0.10;
    }

    let identity =
        (0.35 * vp_match + 0.25 * hp_match + 0.20 * grid_sim + 0.10 * mat_sim + 0.10 * surf_match)
            .clamp(0.0, 1.0);

    if is_hard {
        contradiction = contradiction.max(0.65);
    }
    let final_contradiction = contradiction.clamp(0.0, 1.0);

    (identity, final_contradiction, is_hard)
}

fn match_partitions(p_a: &[f32], p_b: &[f32], partial_ok: bool) -> (f32, f32, bool) {
    if p_a.is_empty() && p_b.is_empty() {
        return (0.80, 0.0, false);
    }
    if p_a.is_empty() || p_b.is_empty() {
        let count = p_a.len().max(p_b.len());
        if count >= 3 && !partial_ok {
            return (0.20, 0.60, true);
        } else if count >= 2 && !partial_ok {
            return (0.35, 0.35, false);
        } else {
            return (0.50, 0.15, false);
        }
    }

    let tol = if partial_ok { 0.14 } else { 0.09 };
    let mut matches = 0usize;
    for &va in p_a {
        if p_b.iter().any(|&vb| (va - vb).abs() <= tol) {
            matches += 1;
        }
    }

    let frac_a = matches as f32 / p_a.len() as f32;
    let mut matches_b = 0usize;
    for &vb in p_b {
        if p_a.iter().any(|&va| (va - vb).abs() <= tol) {
            matches_b += 1;
        }
    }
    let frac_b = matches_b as f32 / p_b.len() as f32;

    let match_score = (frac_a + frac_b) * 0.5;

    let diff_count = (p_a.len() as isize - p_b.len() as isize).unsigned_abs();
    let mut contra = (1.0 - match_score) * 0.60;
    let mut is_hard = false;

    if !partial_ok && diff_count >= 3 {
        is_hard = true;
        contra += 0.40;
    } else if !partial_ok && diff_count >= 2 {
        contra += 0.20;
    }

    (match_score, contra.clamp(0.0, 1.0), is_hard)
}

fn l1_normalize(v: &mut [f32]) {
    let sum: f32 = v.iter().sum();
    if sum > 1e-6 {
        for x in v.iter_mut() {
            *x /= sum;
        }
    }
}

/// Converts sRGB (0..=255) to CIE L*a*b* (D65).
pub fn srgb_to_lab(r: u8, g: u8, b: u8) -> (f32, f32, f32) {
    let rl = srgb_to_linear(r as f32 / 255.0);
    let gl = srgb_to_linear(g as f32 / 255.0);
    let bl = srgb_to_linear(b as f32 / 255.0);

    let x = (0.4124564 * rl + 0.3575761 * gl + 0.1804375 * bl) / 0.95047;
    let y = 0.2126729 * rl + 0.7151522 * gl + 0.0721750 * bl;
    let z = (0.0193339 * rl + 0.1191920 * gl + 0.9503041 * bl) / 1.08883;

    let fx = lab_f(x);
    let fy = lab_f(y);
    let fz = lab_f(z);

    let l = 116.0 * fy - 16.0;
    let a = 500.0 * (fx - fy);
    let b_val = 200.0 * (fy - fz);
    (l, a, b_val)
}

#[inline]
fn srgb_to_linear(c: f32) -> f32 {
    if c <= 0.04045 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

#[inline]
fn lab_f(t: f32) -> f32 {
    const EPS: f32 = 216.0 / 24389.0;
    const KAPPA: f32 = 24389.0 / 27.0;
    if t > EPS {
        t.cbrt()
    } else {
        (KAPPA * t + 16.0) / 116.0
    }
}
