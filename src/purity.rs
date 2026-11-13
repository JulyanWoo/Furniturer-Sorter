//! Conservative v3 local-core formation with explicit consistency admission.
//! Broad family and v4 project-identity scores have no membership authority.
use crate::clustering::{GroupKind, ImageGroup, PossibleSplit};
use crate::embeddings::ImageFeatures;
use crate::fingerprint::{ImageFingerprint, MaterialFamily};
use crate::similarity::PairComparison;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct AdmissionConfig {
    pub majority_fraction: f32,
    pub minimum_consensus: f32,
    pub minimum_geometry: f32,
    pub minimum_local_layout: f32,
    pub minimum_material: f32,
    pub maximum_contradiction: f32,
}

impl Default for AdmissionConfig {
    fn default() -> Self {
        Self {
            majority_fraction: 2.0 / 3.0,
            minimum_consensus: 0.72,
            minimum_geometry: 0.68,
            minimum_local_layout: 0.68,
            minimum_material: 0.60,
            maximum_contradiction: 0.28,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CoarseFurnitureFamily {
    KitchenInstallation,
    TallStorage,
    DrawerCabinet,
    Unknown,
}

/// Only recognize strong structural extremes. A surface alone is not a kitchen.
/// Desk/TV/bedroom semantics cannot safely be inferred from these cached features.
pub fn coarse_family(f: &ImageFingerprint) -> CoarseFurnitureFamily {
    let g = &f.geometry;
    if g.structural_grid.len() != 9 || g.vertical_profile.len() < 16 {
        return CoarseFurnitureFamily::Unknown;
    }
    if f.aspect_ratio > 1.6
        && f.topology.has_prominent_surface
        && g.vertical_division_count >= 3.0
        && g.horizontal_division_count >= 2.0
    {
        CoarseFurnitureFamily::KitchenInstallation
    } else if f.aspect_ratio < 0.65
        && !f.topology.has_prominent_surface
        && g.vertical_edge_ratio > 0.72
        && g.vertical_division_count >= 2.0
        && g.horizontal_division_count <= 1.0
    {
        CoarseFurnitureFamily::TallStorage
    } else if !f.topology.has_prominent_surface
        && g.horizontal_division_count >= 5.0
        && g.vertical_division_count <= 1.0
        && g.vertical_edge_ratio < 0.28
    {
        CoarseFurnitureFamily::DrawerCabinet
    } else {
        CoarseFurnitureFamily::Unknown
    }
}

pub fn family_compatible(a: CoarseFurnitureFamily, b: CoarseFurnitureFamily) -> bool {
    a == b || a == CoarseFurnitureFamily::Unknown || b == CoarseFurnitureFamily::Unknown
}

pub fn membership_strength(p: &PairComparison) -> f32 {
    let d = &p.score.deterministic;
    (0.50 * d.structure + 0.25 * d.partial_view + 0.25 * d.material_color
        - 0.20 * p.score.contradiction)
        .clamp(0.0, 1.0)
}

pub fn pair_compatible(
    p: &PairComparison,
    features: &[ImageFeatures],
    cfg: &AdmissionConfig,
) -> bool {
    if p.score.hard_contradiction || p.score.contradiction > cfg.maximum_contradiction {
        return false;
    }
    if let (Some(a), Some(b)) = (features.get(p.i), features.get(p.j)) {
        if !family_compatible(coarse_family(&a.fingerprint), coarse_family(&b.fingerprint)) {
            return false;
        }
        let ma = MaterialFamily::from_repr(a.fingerprint.topology.dominant_material).unwrap();
        let mb = MaterialFamily::from_repr(b.fingerprint.topology.dominant_material).unwrap();
        if ma.compatibility(mb).1 {
            return false;
        }
    }
    if p.score.relation.is_duplicate() {
        return true;
    }
    let d = &p.score.deterministic;
    d.structure >= cfg.minimum_geometry
        && d.partial_view >= cfg.minimum_local_layout
        && d.material_color >= cfg.minimum_material
        && membership_strength(p) >= cfg.minimum_consensus
}

#[derive(Debug, Default)]
pub struct GroupConsensusScore {
    pub median: f32,
    pub compatible_fraction: f32,
    pub major_contradiction: bool,
    pub medoid_supported: bool,
}

impl GroupConsensusScore {
    pub fn allows(&self, cfg: &AdmissionConfig) -> bool {
        !self.major_contradiction
            && self.medoid_supported
            && self.median >= cfg.minimum_consensus
            && self.compatible_fraction >= cfg.majority_fraction.clamp(0.60, 1.0)
    }
}

/// All members act as representatives (groups are bounded to 48). No viewpoint
/// or inconvenient low-scoring member is discarded by a top-match selection.
pub fn group_consensus(
    candidate: usize,
    members: &[usize],
    medoid: usize,
    pairs: &HashMap<(usize, usize), &PairComparison>,
    features: &[ImageFeatures],
    cfg: &AdmissionConfig,
    min_pair: f32,
) -> GroupConsensusScore {
    let mut result = GroupConsensusScore::default();
    if members.is_empty() {
        return result;
    }
    let mut values = Vec::new();
    let mut supported = 0;
    for &m in members {
        let key = (candidate.min(m), candidate.max(m));
        let Some(p) = pairs.get(&key) else {
            result.major_contradiction = true;
            continue;
        };
        let s = membership_strength(p);
        values.push(if p.score.relation.is_duplicate() {
            1.0
        } else {
            s
        });
        let compatible = pair_compatible(p, features, cfg);
        supported += usize::from(compatible);
        if m == medoid {
            result.medoid_supported = compatible;
        }
        let family_or_material_mismatch =
            features
                .get(candidate)
                .zip(features.get(m))
                .is_some_and(|(a, b)| {
                    !family_compatible(coarse_family(&a.fingerprint), coarse_family(&b.fingerprint))
                        || MaterialFamily::from_repr(a.fingerprint.topology.dominant_material)
                            .unwrap()
                            .compatibility(
                                MaterialFamily::from_repr(b.fingerprint.topology.dominant_material)
                                    .unwrap(),
                            )
                            .1
                });
        if p.score.hard_contradiction
            || p.score.contradiction > cfg.maximum_contradiction
            || p.score.combined < min_pair
            || family_or_material_mismatch
        {
            result.major_contradiction = true;
        }
    }
    values.sort_by(f32::total_cmp);
    if values.len() == members.len() {
        let mid = values.len() / 2;
        result.median = if values.len() % 2 == 0 {
            (values[mid - 1] + values[mid]) * 0.5
        } else {
            values[mid]
        };
    }
    result.compatible_fraction = supported as f32 / members.len() as f32;
    result
}

/// Review-only audit, including low support even without a topology contradiction.
pub fn audit_groups(
    groups: &[ImageGroup],
    pairs: &HashMap<(usize, usize), &PairComparison>,
    features: &[ImageFeatures],
    cfg: &AdmissionConfig,
) -> Vec<PossibleSplit> {
    let mut findings = Vec::new();
    for g in groups
        .iter()
        .filter(|g| g.kind != GroupKind::Ungrouped && g.member_indices.len() >= 3)
    {
        let ms = &g.member_indices;
        let strength = |a: usize, b: usize| {
            if a == b {
                1.0
            } else {
                pairs
                    .get(&(a.min(b), a.max(b)))
                    .map_or(0.0, |p| membership_strength(p))
            }
        };
        for &x in ms {
            let others: Vec<_> = ms.iter().copied().filter(|&y| y != x).collect();
            let support = others
                .iter()
                .filter(|&&y| {
                    pairs
                        .get(&(x.min(y), x.max(y)))
                        .is_some_and(|p| pair_compatible(p, features, cfg))
                })
                .count();
            if support as f32 / (others.len() as f32) < 0.34 {
                findings.push(PossibleSplit {
                    group_id: g.id.clone(),
                    group_name: g.name.clone(),
                    outlier_idx: Some(x),
                    subcluster_a: others,
                    subcluster_b: vec![x],
                    cross_contradiction: 1.0 - support as f32 / (ms.len() - 1) as f32,
                    reason:
                        "Potential Outlier: weak agreement with the group; human review required"
                            .into(),
                });
            }
        }
        if ms.len() < 4 {
            continue;
        }
        let mut worst = (ms[0], ms[1], 1.0);
        for (i, &a) in ms.iter().enumerate() {
            for &b in &ms[i + 1..] {
                let s = strength(a, b);
                if s < worst.2 {
                    worst = (a, b, s);
                }
            }
        }
        let (a, b): (Vec<_>, Vec<_>) = ms
            .iter()
            .copied()
            .partition(|&m| strength(m, worst.0) >= strength(m, worst.1));
        if a.len() < 2 || b.len() < 2 {
            continue;
        }
        let mean = |left: &[usize], right: &[usize], within: bool| {
            let mut sum = 0.0;
            let mut count = 0;
            for &x in left {
                for &y in right {
                    if x != y && (!within || x < y) {
                        sum += strength(x, y);
                        count += 1;
                    }
                }
            }
            sum / (count.max(1) as f32)
        };
        let cross = mean(&a, &b, false);
        if mean(&a, &a, true) >= cfg.minimum_consensus
            && mean(&b, &b, true) >= cfg.minimum_consensus
            && cross < 0.60
        {
            findings.push(PossibleSplit {
                group_id: g.id.clone(),
                group_name: g.name.clone(),
                outlier_idx: None,
                subcluster_a: a,
                subcluster_b: b,
                cross_contradiction: 1.0 - cross,
                reason:
                    "Possible Split: coherent subgroups with weak cross-member support; review only"
                        .into(),
            });
        }
    }
    findings
}
