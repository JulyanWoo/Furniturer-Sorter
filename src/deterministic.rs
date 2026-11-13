use crate::fingerprint::ImageFingerprint;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DeterministicWeights {
    /// Structural furniture geometry weight (~35-45%, default 0.38).
    pub structure: f32,
    /// Material & wood tone / color weight (~15-25%, default 0.22).
    pub material_color: f32,
    /// Partial-view multi-scale region matching weight (default 0.15).
    #[serde(default = "default_partial_view_weight")]
    pub partial_view: f32,
    /// Spatial scene / room layout context weight (default 0.08).
    pub scene_context: f32,
    /// Center-symmetric texture / surface pattern weight (default 0.09).
    pub texture: f32,
    /// Perceptual hash supporting weight (default 0.08).
    pub phash: f32,
    // Legacy fields for backward compatibility with existing configs:
    #[serde(default)]
    pub color: f32,
    #[serde(default)]
    pub layout: f32,
    #[serde(default)]
    pub edges: f32,
}

fn default_partial_view_weight() -> f32 {
    0.15
}

impl Default for DeterministicWeights {
    fn default() -> Self {
        Self {
            structure: 0.38,
            material_color: 0.22,
            partial_view: 0.15,
            scene_context: 0.08,
            texture: 0.09,
            phash: 0.08,
            color: 0.22,
            layout: 0.08,
            edges: 0.38,
        }
    }
}

impl DeterministicWeights {
    pub fn normalized(&self) -> Self {
        let sum_v3 = self.structure
            + self.material_color
            + self.partial_view
            + self.scene_context
            + self.texture
            + self.phash;
        if sum_v3 > 1e-4 {
            return Self {
                structure: self.structure / sum_v3,
                material_color: self.material_color / sum_v3,
                partial_view: self.partial_view / sum_v3,
                scene_context: self.scene_context / sum_v3,
                texture: self.texture / sum_v3,
                phash: self.phash / sum_v3,
                color: self.material_color / sum_v3,
                layout: self.scene_context / sum_v3,
                edges: self.structure / sum_v3,
            };
        }
        let sum_legacy = self.color + self.layout + self.edges + self.phash + self.texture;
        if sum_legacy > 1e-4 {
            let s = self.edges / sum_legacy;
            let mc = self.color / sum_legacy;
            let sc = self.layout / sum_legacy;
            let tx = self.texture / sum_legacy;
            let ph = self.phash / sum_legacy;
            return Self {
                structure: s,
                material_color: mc,
                partial_view: 0.0,
                scene_context: sc,
                texture: tx,
                phash: ph,
                color: mc,
                layout: sc,
                edges: s,
            };
        }
        Self::default()
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct DeterministicComponents {
    /// Structure & geometry similarity (profiles, divisions, symmetry, edges).
    pub structure: f32,
    /// Material & color tone similarity (wood tone, robust chromaticity).
    pub material_color: f32,
    /// Partial-view region similarity across multi-scale crops.
    #[serde(default)]
    pub partial_view: f32,
    /// Scene & spatial layout similarity (downweighted background/borders).
    pub scene_context: f32,
    /// Texture signature similarity.
    pub texture: f32,
    /// Perceptual hash similarity (pHash + dHash).
    pub phash: f32,
    /// Family similarity: "How visually similar are these furniture scenes?"
    #[serde(default)]
    pub family_similarity: f32,
    /// Project identity score: "How much positive evidence suggests this is the same physical project?"
    #[serde(default)]
    pub project_identity: f32,
    /// Contradiction score: "How much evidence suggests these cannot be the same project?"
    #[serde(default)]
    pub contradiction: f32,
    /// Flag indicating whether a hard contradiction was triggered.
    #[serde(default)]
    pub hard_contradiction: bool,
    /// Topology match score.
    #[serde(default)]
    pub topology_match: f32,
    /// Legacy fields for backward compatibility:
    pub color: f32,
    pub layout: f32,
    pub edges: f32,
    /// Combined deterministic similarity score.
    pub combined: f32,
    pub phash_hamming: u32,
    pub dhash_hamming: u32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PrefilterConfig {
    /// Skip pairs whose aspect ratios diverge beyond this factor unless pHash is very close.
    pub max_aspect_ratio_factor: f32,
    /// Maximum coarse chromaticity L1 difference before dropping candidate.
    pub max_coarse_chroma_diff: f32,
    /// Maximum coarse luminance difference before dropping candidate.
    pub max_coarse_luma_diff: f32,
    /// Always keep pairs with pHash Hamming distance <= this threshold.
    pub bypass_phash_hamming: u32,
}

impl Default for PrefilterConfig {
    fn default() -> Self {
        Self {
            max_aspect_ratio_factor: 2.5,
            max_coarse_chroma_diff: 0.28,
            max_coarse_luma_diff: 0.38,
            bypass_phash_hamming: 14,
        }
    }
}

#[inline]
pub fn hamming_u64(a: &[u8; 8], b: &[u8; 8]) -> u32 {
    let va = u64::from_le_bytes(*a);
    let vb = u64::from_le_bytes(*b);
    (va ^ vb).count_ones()
}

/// Fast O(1) candidate prefilter so expensive histogram/grid/edge comparisons
/// do not run across every N^2 pair.
pub fn passes_candidate_prefilter(
    a: &ImageFingerprint,
    b: &ImageFingerprint,
    cfg: &PrefilterConfig,
) -> bool {
    let ph_dist = hamming_u64(&a.phash, &b.phash);
    if ph_dist <= cfg.bypass_phash_hamming {
        return true;
    }

    let ar_factor = if a.aspect_ratio > b.aspect_ratio {
        a.aspect_ratio / b.aspect_ratio.max(0.05)
    } else {
        b.aspect_ratio / a.aspect_ratio.max(0.05)
    };
    if ar_factor > cfg.max_aspect_ratio_factor {
        return false;
    }

    let l_diff = (a.coarse_color[0] - b.coarse_color[0]).abs();
    if l_diff > cfg.max_coarse_luma_diff {
        return false;
    }

    let chroma_diff = (a.coarse_color[1] - b.coarse_color[1]).abs()
        + (a.coarse_color[2] - b.coarse_color[2]).abs()
        + 0.5 * (a.coarse_color[3] - b.coarse_color[3]).abs();
    if chroma_diff > cfg.max_coarse_chroma_diff {
        return false;
    }

    // Coarse layout mean luminance/color check across 4 corner cells (O(1) 12-float L1)
    let mut grid_l1 = 0.0f32;
    for c in [0usize, 3, 12, 15] {
        let base = c * 6;
        grid_l1 += (a.spatial_grid[base] - b.spatial_grid[base]).abs()
            + (a.spatial_grid[base + 1] - b.spatial_grid[base + 1]).abs()
            + (a.spatial_grid[base + 2] - b.spatial_grid[base + 2]).abs();
    }
    grid_l1 <= 1.5
}

pub fn compare_deterministic(
    a: &ImageFingerprint,
    b: &ImageFingerprint,
    weights: &DeterministicWeights,
) -> DeterministicComponents {
    let w = weights.normalized();

    let structure = furniture_structure_similarity(a, b);
    let material_color = material_color_similarity(a, b, structure);
    let partial_view = crate::fingerprint::partial_view_similarity(a, b);
    let scene_context = scene_context_similarity(&a.spatial_grid, &b.spatial_grid);
    let texture = ruzicka_similarity(&a.texture_signature, &b.texture_signature);

    let phash_hamming = hamming_u64(&a.phash, &b.phash);
    let dhash_hamming = hamming_u64(&a.dhash, &b.dhash);
    // Map Hamming [0..32] -> [1.0..0.0] so random/unrelated pairs contribute 0.0 rather than 0.50.
    let p_sim = (1.0 - (phash_hamming as f32 / 32.0)).max(0.0);
    let d_sim = (1.0 - (dhash_hamming as f32 / 32.0)).max(0.0);
    let phash = (0.65 * p_sim + 0.35 * d_sim).clamp(0.0, 1.0);

    // Legacy fields for backward compatibility:
    let color = color_similarity(a, b);
    let layout = layout_similarity(&a.spatial_grid, &b.spatial_grid);
    let edges = edge_similarity(&a.edge_signature, &b.edge_signature);

    let combined = (w.structure * structure
        + w.material_color * material_color
        + w.partial_view * partial_view
        + w.scene_context * scene_context
        + w.texture * texture
        + w.phash * phash)
        .clamp(0.0, 1.0);

    // Family Similarity: "How visually similar are these furniture scenes?"
    let family_similarity = (0.28 * material_color
        + 0.22 * texture
        + 0.20 * scene_context
        + 0.18 * structure
        + 0.12 * phash)
        .clamp(0.0, 1.0);

    // Topology match & contradiction:
    let opposing_structure = (a.topology.horizontal_partitions.len()
        > a.topology.vertical_partitions.len() + 1
        && b.topology.vertical_partitions.len() > b.topology.horizontal_partitions.len() + 1)
        || (b.topology.horizontal_partitions.len() > b.topology.vertical_partitions.len() + 1
            && a.topology.vertical_partitions.len() > a.topology.horizontal_partitions.len() + 1);
    let is_subset_structure = (a.topology.vertical_partitions.len()
        < b.topology.vertical_partitions.len())
        || (b.topology.vertical_partitions.len() < a.topology.vertical_partitions.len())
        || (a.topology.horizontal_partitions.len() < b.topology.horizontal_partitions.len())
        || (b.topology.horizontal_partitions.len() < a.topology.horizontal_partitions.len());
    let partial_ok = !opposing_structure && is_subset_structure && partial_view >= 0.78;
    let (topology_match, topology_contra, hard_contra) =
        crate::fingerprint::compare_topology(&a.topology, &b.topology, partial_ok);

    // Project Identity Score: "How much positive evidence suggests this is the same physical project?"
    let project_identity = (0.30 * structure
        + 0.26 * topology_match
        + 0.20 * partial_view
        + 0.14 * material_color
        + 0.10 * texture)
        .clamp(0.0, 1.0);

    // Contradiction Score: "How much evidence suggests these cannot be the same project?"
    let mut contradiction = topology_contra;
    if hard_contra {
        contradiction = contradiction.max(0.65);
    }
    if !partial_ok {
        let div_diff = (a.geometry.vertical_division_count - b.geometry.vertical_division_count)
            .abs()
            + (a.geometry.horizontal_division_count - b.geometry.horizontal_division_count).abs();
        if div_diff >= 3.0 {
            contradiction = (contradiction + 0.25).clamp(0.0, 1.0);
        }
    }
    let contradiction = contradiction.clamp(0.0, 1.0);

    DeterministicComponents {
        structure,
        material_color,
        partial_view,
        scene_context,
        color,
        layout,
        edges,
        phash,
        texture,
        combined,
        phash_hamming,
        dhash_hamming,
        family_similarity,
        project_identity,
        contradiction,
        hard_contradiction: hard_contra,
        topology_match,
    }
}

/// Evaluates furniture identity structure: vertical door/stile profiles, horizontal shelf/drawer profiles,
/// Sobel edge orientation distributions, symmetry, and division periodicity.
pub fn furniture_structure_similarity(a: &ImageFingerprint, b: &ImageFingerprint) -> f32 {
    let vp_len = a.geometry.vertical_profile.len();
    if vp_len >= 16 && b.geometry.vertical_profile.len() >= 16 {
        let vert_sim =
            ruzicka_similarity(&a.geometry.vertical_profile, &b.geometry.vertical_profile);
        let horiz_sim = ruzicka_similarity(
            &a.geometry.horizontal_profile,
            &b.geometry.horizontal_profile,
        );
        let orient_sim = edge_similarity(&a.edge_signature, &b.edge_signature);

        let grid_sim =
            if a.geometry.structural_grid.len() >= 9 && b.geometry.structural_grid.len() >= 9 {
                ruzicka_similarity(&a.geometry.structural_grid, &b.geometry.structural_grid)
            } else {
                0.5
            };

        let sym_sim = (1.0 - (a.geometry.symmetry_score - b.geometry.symmetry_score).abs() * 2.0)
            .clamp(0.0, 1.0);
        let vr_sim = (1.0
            - (a.geometry.vertical_edge_ratio - b.geometry.vertical_edge_ratio).abs() * 2.5)
            .clamp(0.0, 1.0);
        let div_sim = (1.0
            - (a.geometry.horizontal_division_count - b.geometry.horizontal_division_count).abs()
                * 0.25)
            .clamp(0.0, 1.0);
        let scalar_match = 0.40 * sym_sim + 0.30 * vr_sim + 0.30 * div_sim;

        (0.28 * vert_sim
            + 0.28 * horiz_sim
            + 0.20 * orient_sim
            + 0.12 * grid_sim
            + 0.12 * scalar_match)
            .clamp(0.0, 1.0)
    } else {
        // Fallback for fingerprints lacking geometry signature
        edge_similarity(&a.edge_signature, &b.edge_signature)
    }
}

/// Evaluates material tone and chromaticity (wood species, painted finishes), ensuring
/// color strengthens a structural match but exposure/lighting shifts do not veto a strong structural match.
pub fn material_color_similarity(
    a: &ImageFingerprint,
    b: &ImageFingerprint,
    struct_sim: f32,
) -> f32 {
    if a.color_histogram.len() < 76 || b.color_histogram.len() < 76 {
        return 0.0;
    }
    let ab_sim = ruzicka_similarity(&a.color_histogram[..64], &b.color_histogram[..64]);
    let l_sim = ruzicka_similarity(&a.color_histogram[64..76], &b.color_histogram[64..76]);
    let dom_sim = dominant_color_similarity(&a.dominant_colors, &b.dominant_colors);

    // When structural match is strong (>= 0.70), exposure differences (lightness L*) receive lower penalty
    let (w_ab, w_l, w_dom) = if struct_sim >= 0.70 {
        (0.65, 0.10, 0.25)
    } else {
        (0.55, 0.20, 0.25)
    };

    let base = (w_ab * ab_sim + w_l * l_sim + w_dom * dom_sim).clamp(0.0, 1.0);
    if struct_sim >= 0.80 && ab_sim >= 0.65 {
        base.max(0.72)
    } else {
        base
    }
}

/// Evaluates spatial scene context, downweighting outer room borders (floors, ceilings, walls)
/// to prioritize the central furniture region.
pub fn scene_context_similarity(grid_a: &[f32], grid_b: &[f32]) -> f32 {
    if grid_a.len() != 96 || grid_b.len() != 96 {
        return 0.0;
    }
    let mut weighted_sim = 0.0f32;
    let mut total_w = 0.0f32;
    for cell in 0..16 {
        let base = cell * 6;
        let dl = (grid_a[base] - grid_b[base]).abs();
        let da = (grid_a[base + 1] - grid_b[base + 1]).abs();
        let db = (grid_a[base + 2] - grid_b[base + 2]).abs();
        let d_std = (grid_a[base + 3] - grid_b[base + 3]).abs();
        let d_edge = (grid_a[base + 4] - grid_b[base + 4]).abs();
        let d_vert = (grid_a[base + 5] - grid_b[base + 5]).abs();

        let cell_dist =
            0.18 * dl + 0.16 * da + 0.16 * db + 0.14 * d_std + 0.22 * d_edge + 0.14 * d_vert;
        let cell_sim = (1.0 - cell_dist * 3.6).clamp(0.0, 1.0);

        // Center 4 cells (5, 6, 9, 10): weight 2.5
        // Border 8 cells (1, 2, 4, 7, 8, 11, 13, 14): weight 1.0
        // Corner 4 cells (0, 3, 12, 15): weight 0.5
        let w = match cell {
            5 | 6 | 9 | 10 => 2.5f32,
            0 | 3 | 12 | 15 => 0.5f32,
            _ => 1.0f32,
        };
        weighted_sim += w * cell_sim;
        total_w += w;
    }
    (weighted_sim / total_w.max(1.0)).clamp(0.0, 1.0)
}

pub fn color_similarity(a: &ImageFingerprint, b: &ImageFingerprint) -> f32 {
    if a.color_histogram.len() < 76 || b.color_histogram.len() < 76 {
        return 0.0;
    }
    // 64-bin a*b* chromaticity histogram + 12-bin L* histogram + dominant color alignment
    let ab_sim = ruzicka_similarity(&a.color_histogram[..64], &b.color_histogram[..64]);
    let l_sim = ruzicka_similarity(&a.color_histogram[64..76], &b.color_histogram[64..76]);
    let dom_sim = dominant_color_similarity(&a.dominant_colors, &b.dominant_colors);
    (0.60 * ab_sim + 0.20 * l_sim + 0.20 * dom_sim).clamp(0.0, 1.0)
}

fn dominant_color_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() < 12 || b.len() < 12 {
        return 0.0;
    }
    let mut score = 0.0f32;
    let mut total_w = 0.0f32;
    for slot_a in 0..3 {
        let ba = slot_a * 4;
        let wa = a[ba + 3];
        if wa <= 1e-4 {
            continue;
        }
        let mut best = 0.0f32;
        for slot_b in 0..3 {
            let bb = slot_b * 4;
            let wb = b[bb + 3];
            if wb <= 1e-4 {
                continue;
            }
            let dl = (a[ba] - b[bb]).abs();
            let da = (a[ba + 1] - b[bb + 1]).abs();
            let db = (a[ba + 2] - b[bb + 2]).abs();
            let dist = (0.4 * dl * dl + 1.8 * da * da + 1.8 * db * db).sqrt();
            let sim = (1.0 - dist * 2.2).clamp(0.0, 1.0);
            if sim > best {
                best = sim;
            }
        }
        score += wa * best;
        total_w += wa;
    }
    if total_w > 1e-5 {
        (score / total_w).clamp(0.0, 1.0)
    } else {
        0.0
    }
}

pub fn layout_similarity(grid_a: &[f32], grid_b: &[f32]) -> f32 {
    if grid_a.len() != 96 || grid_b.len() != 96 {
        return 0.0;
    }
    let mut total_sim = 0.0f32;
    for cell in 0..16 {
        let base = cell * 6;
        let dl = (grid_a[base] - grid_b[base]).abs();
        let da = (grid_a[base + 1] - grid_b[base + 1]).abs();
        let db = (grid_a[base + 2] - grid_b[base + 2]).abs();
        let d_std = (grid_a[base + 3] - grid_b[base + 3]).abs();
        let d_edge = (grid_a[base + 4] - grid_b[base + 4]).abs();
        let d_vert = (grid_a[base + 5] - grid_b[base + 5]).abs();

        let cell_dist =
            0.18 * dl + 0.16 * da + 0.16 * db + 0.14 * d_std + 0.22 * d_edge + 0.14 * d_vert;
        total_sim += (1.0 - cell_dist * 3.6).clamp(0.0, 1.0);
    }
    (total_sim / 16.0).clamp(0.0, 1.0)
}

pub fn edge_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != 40 || b.len() != 40 {
        return 0.0;
    }
    let global_sim = ruzicka_similarity(&a[..8], &b[..8]);
    let mut quad_sim = 0.0f32;
    for q in 0..4 {
        let start = 8 + q * 8;
        quad_sim += ruzicka_similarity(&a[start..start + 8], &b[start..start + 8]);
    }
    quad_sim /= 4.0;
    (0.45 * global_sim + 0.55 * quad_sim).clamp(0.0, 1.0)
}

/// Ruzicka (weighted Jaccard) histogram similarity: sum(min(a,b)) / sum(max(a,b)).
#[inline]
pub fn ruzicka_similarity(a: &[f32], b: &[f32]) -> f32 {
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

#[inline]
pub fn histogram_intersection(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len().min(b.len());
    let mut sum = 0.0f32;
    for i in 0..len {
        sum += a[i].min(b[i]);
    }
    sum.clamp(0.0, 1.0)
}
