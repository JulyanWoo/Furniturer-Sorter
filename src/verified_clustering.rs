//! Initial cluster construction from SHA atoms and verified local correspondences.
//! This is not a post-clustering automatic merge refinement pass.
use crate::{
    clustering::{self, ClusteringConfig, ClusteringOutput, GroupKind, ImageGroup, MergeRank},
    embeddings::ImageFeatures,
    local_identity::LocalEvidenceMap,
    scanner::ScannedImage,
    similarity::PairComparison,
};
use std::collections::HashMap;

#[derive(Default)]
pub struct HumanConstraints {
    pub same: Vec<(String, String)>,
    pub different: Vec<(String, String)>,
}
impl HumanConstraints {
    pub fn from_labels(store: &crate::evaluation::EvaluationLabelStore) -> Self {
        use crate::evaluation::{GroundTruthLabel, LabelSource};
        let mut constraints = Self::default();
        // External evaluation fixtures never become inference rules.
        for p in store
            .pairs
            .iter()
            .filter(|p| p.source == LabelSource::ManualCorrection)
        {
            let stable = |id: &str| id.starts_with("sha256:") || id.starts_with("openimg:");
            if !stable(&p.image_a) || !stable(&p.image_b) {
                continue;
            }
            match p.label {
                GroundTruthLabel::SameProject => constraints
                    .same
                    .push((p.image_a.clone(), p.image_b.clone())),
                GroundTruthLabel::DifferentProject => constraints
                    .different
                    .push((p.image_a.clone(), p.image_b.clone())),
            }
        }
        constraints.same.sort();
        constraints.different.sort();
        constraints
    }
}

pub fn cluster(
    images: &[ScannedImage],
    features: &[ImageFeatures],
    pairs: &[PairComparison],
    evidence: &LocalEvidenceMap,
    cfg: &ClusteringConfig,
) -> ClusteringOutput {
    cluster_with_constraints(
        images,
        features,
        pairs,
        evidence,
        cfg,
        &HumanConstraints::default(),
    )
}

pub fn cluster_with_constraints(
    images: &[ScannedImage],
    features: &[ImageFeatures],
    pairs: &[PairComparison],
    evidence: &LocalEvidenceMap,
    cfg: &ClusteringConfig,
    constraints: &HumanConstraints,
) -> ClusteringOutput {
    let n = images.len();
    if n == 0 {
        return ClusteringOutput::default();
    }
    let ids: Vec<_> = images
        .iter()
        .map(|im| {
            im.content_sha256
                .as_ref()
                .map(|h| format!("sha256:{h}"))
                .unwrap_or_else(|| im.stable_id())
        })
        .collect();
    let stable_ids: Vec<_> = images.iter().map(ScannedImage::stable_id).collect();
    let index: HashMap<_, _> = stable_ids
        .iter()
        .enumerate()
        .map(|(i, id)| (id.as_str(), i))
        .collect();
    let cannot: Vec<_> = constraints
        .different
        .iter()
        .filter_map(|(a, b)| Some((*index.get(a.as_str())?, *index.get(b.as_str())?)))
        .collect();
    let conflicts = |left: &[usize], right: &[usize]| {
        cannot.iter().any(|&(a, b)| {
            (left.iter().any(|&i| stable_ids[i] == stable_ids[a])
                && right.iter().any(|&i| stable_ids[i] == stable_ids[b]))
                || (left.iter().any(|&i| stable_ids[i] == stable_ids[b])
                    && right.iter().any(|&i| stable_ids[i] == stable_ids[a]))
        })
    };
    let mut first = HashMap::new();
    let canonical: Vec<_> = ids
        .iter()
        .enumerate()
        .map(|(i, id)| *first.entry(id.as_str()).or_insert(i))
        .collect();
    let mut atoms: HashMap<&str, Vec<usize>> = HashMap::new();
    for (i, id) in ids.iter().enumerate() {
        atoms.entry(id).or_default().push(i);
    }
    let mut cores: Vec<Vec<usize>> = atoms.into_values().collect();
    cores.sort_by_key(|c| c[0]);
    let mut owner = vec![0; n];
    for (i, c) in cores.iter().enumerate() {
        for &m in c {
            owner[m] = i;
        }
    }
    let near: std::collections::HashSet<_> = pairs
        .iter()
        .filter(|p| p.score.relation.is_duplicate())
        .filter_map(|p| {
            let a = canonical[p.i];
            let b = canonical[p.j];
            if a == b {
                return None;
            }
            let key = (a.min(b), a.max(b));
            let e = evidence.get(&key)?;
            (e.verified
                && e.inliers >= 20
                && e.inlier_fraction >= 0.75
                && e.coverage_a.min(e.coverage_b) >= 0.25)
                .then_some(key)
        })
        .collect();
    let mut near_edges: Vec<_> = near.iter().copied().collect();
    near_edges.sort_unstable();
    for (a, b) in near_edges {
        let ca = owner[a];
        let cb = owner[b];
        if ca == cb || conflicts(&cores[ca], &cores[cb]) {
            continue;
        }
        // Complete-link equivalence prevents a transitive pan across different views.
        let equivalent = cores[ca].iter().all(|&x| {
            cores[cb].iter().all(|&y| {
                let x = canonical[x];
                let y = canonical[y];
                x == y || near.contains(&(x.min(y), x.max(y)))
            })
        });
        if !equivalent {
            continue;
        }
        let moved = std::mem::take(&mut cores[cb]);
        for m in moved {
            owner[m] = ca;
            cores[ca].push(m);
        }
    }
    let view_owner = owner.clone();
    // Apply explicitly confirmed membership before constructing automatic groups.
    // DifferentProject wins over inconsistent transitive SameProject links.
    for (a, b) in &constraints.same {
        let (Some(&a), Some(&b)) = (index.get(a.as_str()), index.get(b.as_str())) else {
            continue;
        };
        let ca = owner[a];
        let cb = owner[b];
        if ca == cb || conflicts(&cores[ca], &cores[cb]) {
            continue;
        }
        let moved = std::mem::take(&mut cores[cb]);
        for m in moved {
            owner[m] = ca;
            cores[ca].push(m);
        }
    }
    let manual_owner = owner.clone();
    let confirmed_roots: std::collections::HashSet<_> = constraints
        .same
        .iter()
        .filter_map(|(a, b)| {
            let a = *index.get(a.as_str())?;
            let b = *index.get(b.as_str())?;
            (manual_owner[a] == manual_owner[b]).then_some(manual_owner[a])
        })
        .collect();
    let verified_views: std::collections::HashSet<_> = evidence
        .iter()
        .filter(|(_, e)| e.verified)
        .map(|(&(a, b), _)| {
            let a = view_owner[a];
            let b = view_owner[b];
            (a.min(b), a.max(b))
        })
        .collect();
    let mut edges: Vec<_> = evidence
        .iter()
        .filter(|(_, e)| e.verified)
        .map(|(&(a, b), e)| {
            (
                a,
                b,
                e.inliers as f32 * e.inlier_fraction * (e.coverage_a.min(e.coverage_b)).sqrt(),
            )
        })
        .collect();
    edges.sort_by(|a, b| b.2.total_cmp(&a.2).then(a.0.cmp(&b.0)).then(a.1.cmp(&b.1)));
    let linked = |a: usize, b: usize| {
        let va = view_owner[a];
        let vb = view_owner[b];
        va == vb
            || manual_owner[a] == manual_owner[b]
            || verified_views.contains(&(va.min(vb), va.max(vb)))
    };
    let representatives = |members: &[usize]| {
        let mut seen = std::collections::HashSet::new();
        members
            .iter()
            .copied()
            .filter(|&m| seen.insert(view_owner[m]))
            .collect::<Vec<_>>()
    };
    let mut blocked = 0;
    for (a, b, _) in edges {
        let ca = owner[a];
        let cb = owner[b];
        if ca == cb {
            continue;
        }
        if conflicts(&cores[ca], &cores[cb]) {
            blocked += 1;
            continue;
        }
        if cores[ca].len() + cores[cb].len() > cfg.max_group_size {
            continue;
        }
        let left = representatives(&cores[ca]);
        let right = representatives(&cores[cb]);
        let agreement = |from: &[usize], to: &[usize]| {
            from.iter()
                .filter(|&&x| {
                    let support =
                        to.iter().filter(|&&y| linked(x, y)).count() as f32 / to.len() as f32;
                    support >= cfg.admission.majority_fraction
                })
                .count() as f32
                / from.len() as f32
        };
        // Each supported view must match a majority across the other core, and a
        // majority of views on both sides must qualify. Sparse cross-links cannot
        // join two coherent cores. Copies contribute exactly one vote.
        if agreement(&left, &right) < cfg.admission.majority_fraction
            || agreement(&right, &left) < cfg.admission.majority_fraction
        {
            blocked += 1;
            continue;
        }
        let moved = std::mem::take(&mut cores[cb]);
        for m in moved {
            owner[m] = ca;
            cores[ca].push(m);
        }
    }
    let score_map: HashMap<_, _> = pairs
        .iter()
        .map(|p| ((p.i.min(p.j), p.i.max(p.j)), p.score.combined))
        .collect();
    let relations: HashMap<_, _> = pairs
        .iter()
        .map(|p| ((p.i.min(p.j), p.i.max(p.j)), p.score.relation))
        .collect();
    let mut groups = Vec::new();
    let mut ungrouped = Vec::new();
    for mut members in cores.into_iter().filter(|c| !c.is_empty()) {
        members.sort_unstable();
        let (avg, duplicates) = clustering::group_stats(&members, &score_map, &relations);
        let kind = if members.len() > 1 {
            if duplicates || representatives(&members).len() == 1 {
                GroupKind::Duplicates
            } else {
                GroupKind::FurnitureProject
            }
        } else if pairs.iter().any(|p| {
            (p.i == members[0] || p.j == members[0]) && p.score.combined >= cfg.review_threshold
        }) {
            GroupKind::CandidateGroup
        } else {
            ungrouped.extend(members);
            continue;
        };
        let signature = clustering::build_group_signature(&members, features, &score_map);
        let confirmed = confirmed_roots.contains(&manual_owner[members[0]])
            && members
                .iter()
                .all(|&m| manual_owner[m] == manual_owner[members[0]]);
        groups.push(ImageGroup {
            id: String::new(),
            name: String::new(),
            kind,
            representative_idx: Some(signature.medoid_idx),
            representative_members: signature.representative_members.clone(),
            member_indices: members,
            avg_similarity: avg,
            confirmed,
            nearest_related: None,
            signature: Some(signature),
        });
    }
    groups.sort_by(|a, b| {
        let order = |k| match k {
            GroupKind::FurnitureProject => 0,
            GroupKind::Duplicates => 1,
            _ => 2,
        };
        order(a.kind)
            .cmp(&order(b.kind))
            .then(b.member_indices.len().cmp(&a.member_indices.len()))
            .then(a.member_indices[0].cmp(&b.member_indices[0]))
    });
    let mut counts = [0; 3];
    for g in &mut groups {
        let (k, prefix) = match g.kind {
            GroupKind::FurnitureProject => (0, "Furniture"),
            GroupKind::Duplicates => (1, "Duplicates"),
            _ => (2, "Candidate"),
        };
        counts[k] += 1;
        g.id = format!("{prefix}_{:03}", counts[k]);
        g.name = format!("{prefix} {:03}", counts[k]);
    }
    let (mut suggestions, related) = clustering::compute_group_to_group_similarity(
        &mut groups,
        features,
        &score_map,
        cfg.possible_merge_threshold,
    );
    for (g, related) in groups.iter_mut().zip(related) {
        g.nearest_related = related;
    }
    // Global diagnostics can suggest, but cannot rank an unregistered family as a very strong identity.
    for suggestion in &mut suggestions {
        let a = groups
            .iter()
            .find(|g| g.id == suggestion.group_a_id)
            .unwrap();
        let b = groups
            .iter()
            .find(|g| g.id == suggestion.group_b_id)
            .unwrap();
        let left = representatives(&a.member_indices);
        let right = representatives(&b.member_indices);
        let support = left
            .iter()
            .filter(|&&x| right.iter().any(|&y| linked(x, y)))
            .count() as f32
            / left.len() as f32;
        if support < cfg.admission.majority_fraction {
            suggestion.rank = MergeRank::Possible;
        }
    }
    let count = groups.len() + usize::from(!ungrouped.is_empty());
    // Audit the graph of verified details. Absence is weak support for review,
    // never proof of a different project and never an automatic removal.
    let mut audit_pairs = Vec::new();
    let mut audit_groups = groups.clone();
    for g in &mut audit_groups {
        g.member_indices = representatives(&g.member_indices);
        for (i, &a) in g.member_indices.iter().enumerate() {
            for &b in &g.member_indices[i + 1..] {
                let strength = if linked(a, b) { 1.0 } else { 0.3 };
                audit_pairs.push(PairComparison {
                    i: a.min(b),
                    j: a.max(b),
                    score: crate::embeddings::SimilarityScore {
                        deterministic: crate::deterministic::DeterministicComponents {
                            structure: strength,
                            partial_view: strength,
                            material_color: strength,
                            ..Default::default()
                        },
                        embedding: None,
                        combined: strength,
                        used_onnx: false,
                        relation: crate::embeddings::PairRelation::RelatedUncertain,
                        family_similarity: 0.0,
                        project_identity: 0.0,
                        contradiction: 0.0,
                        hard_contradiction: false,
                    },
                });
            }
        }
    }
    let lookup = audit_pairs.iter().map(|p| ((p.i, p.j), p)).collect();
    let mut possible_splits =
        crate::purity::audit_groups(&audit_groups, &lookup, &[], &cfg.admission);
    for split in &mut possible_splits {
        split.reason = if split.outlier_idx.is_some() {
            "Potential Outlier: few verified shared details; review this view before making changes"
                .into()
        } else {
            "Possible Split: local correspondence subgroups have weak cross-view support; human review required".into()
        };
    }
    let detected_outliers = possible_splits
        .iter()
        .filter(|p| p.outlier_idx.is_some())
        .count();
    groups.push(ImageGroup {
        id: "Ungrouped".into(),
        name: "Ungrouped".into(),
        kind: GroupKind::Ungrouped,
        representative_idx: ungrouped.first().copied(),
        member_indices: ungrouped,
        avg_similarity: 0.0,
        confirmed: false,
        nearest_related: None,
        representative_members: vec![],
        signature: None,
    });
    ClusteringOutput {
        groups,
        possible_merges: suggestions,
        blocked_chaining_edges: blocked,
        initial_group_count: count,
        auto_merges_performed: 0,
        possible_splits,
        detected_outliers,
        ..Default::default()
    }
}
