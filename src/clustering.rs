use crate::embeddings::{ImageFeatures, PairRelation};
use crate::purity::{AdmissionConfig, group_consensus, pair_compatible};
use crate::similarity::PairComparison;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ClusteringConfig {
    /// Admission policy is independent from broad retrieval and v4 identity scores.
    #[serde(default)]
    pub admission: AdmissionConfig,
    pub same_project_threshold: f32,
    pub review_threshold: f32,
    /// Minimum pairwise similarity required between cross-pairs
    /// to prevent transitive A-B-C chaining false merges.
    pub cluster_min_pairwise: f32,
    /// Minimum average cross-cluster similarity required to merge two clusters.
    pub cluster_avg_threshold: f32,
    pub max_group_size: usize,
    /// Neighborhood size K for Top-K and Mutual Nearest Neighbor graph (default 15).
    pub top_k: usize,
    /// Threshold for suggesting possible merges between separate groups (default 0.74).
    pub possible_merge_threshold: f32,
    /// Automatic group merge confidence threshold (default 0.85).
    #[serde(default = "default_auto_merge_threshold")]
    pub auto_merge_threshold: f32,
    /// Maximum iterative group merge refinement passes (default 2).
    #[serde(default = "default_max_merge_passes")]
    pub max_merge_passes: usize,
}

fn default_auto_merge_threshold() -> f32 {
    0.85
}

fn default_max_merge_passes() -> usize {
    0
}

impl Default for ClusteringConfig {
    fn default() -> Self {
        Self {
            admission: AdmissionConfig::default(),
            same_project_threshold: 0.72,
            review_threshold: 0.56,
            cluster_min_pairwise: 0.50,
            cluster_avg_threshold: 0.68,
            max_group_size: 48,
            top_k: 15,
            possible_merge_threshold: 0.74,
            auto_merge_threshold: 0.85,
            max_merge_passes: 0,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum GroupKind {
    FurnitureProject,
    CandidateGroup,
    Duplicates,
    Review,
    Ungrouped,
}

impl GroupKind {
    pub fn badge(self) -> &'static str {
        match self {
            Self::FurnitureProject => "Same project",
            Self::CandidateGroup => "Candidate match",
            Self::Duplicates => "Duplicate",
            Self::Review => "Possible match",
            Self::Ungrouped => "Different / Isolated",
        }
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct GroupSignature {
    pub medoid_idx: usize,
    /// 2 to 4 diverse representative members covering multiple views/angles.
    pub representative_members: Vec<usize>,
    /// Normalized mean embedding vector (if ONNX embeddings are present).
    pub centroid_embedding: Option<Vec<f32>>,
    pub avg_vertical_profile: Vec<f32>,
    pub avg_horizontal_profile: Vec<f32>,
    pub avg_structural_grid: Vec<f32>,
    pub avg_vertical_edge_ratio: f32,
    pub avg_symmetry_score: f32,
    pub avg_center_to_border: f32,
    pub avg_horizontal_division_count: f32,
    pub avg_vertical_division_count: f32,
    pub avg_coarse_color: [f32; 5],
    pub avg_color_histogram: Vec<f32>,
    pub avg_internal_similarity: f32,
    pub min_internal_similarity: f32,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct PossibleMergeBreakdown {
    #[serde(default)]
    pub family_similarity: f32,
    #[serde(default)]
    pub cross_member_support: f32,
    pub structure: f32,
    pub embedding: Option<f32>,
    pub cross_view: f32,
    pub color: f32,
    pub partial_match: f32,
    pub mutual_neighbor: bool,
    #[serde(default)]
    pub project_identity: f32,
    #[serde(default)]
    pub contradiction: f32,
    #[serde(default)]
    pub hard_contradiction: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ImageGroup {
    pub id: String,
    pub name: String,
    pub kind: GroupKind,
    pub representative_idx: Option<usize>,
    pub member_indices: Vec<usize>,
    pub avg_similarity: f32,
    pub confirmed: bool,
    #[serde(default)]
    pub nearest_related: Option<(String, f32)>,
    #[serde(default)]
    pub representative_members: Vec<usize>,
    #[serde(default)]
    pub signature: Option<GroupSignature>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum MergeRank {
    VeryStrong,
    Strong,
    Possible,
}

impl MergeRank {
    pub fn badge(self) -> &'static str {
        match self {
            Self::VeryStrong => "Very Strong",
            Self::Strong => "Strong",
            Self::Possible => "Possible",
        }
    }
}

fn default_merge_rank() -> MergeRank {
    MergeRank::Possible
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PossibleSplit {
    pub group_id: String,
    pub group_name: String,
    pub outlier_idx: Option<usize>,
    pub subcluster_a: Vec<usize>,
    pub subcluster_b: Vec<usize>,
    pub cross_contradiction: f32,
    pub reason: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PossibleMergeSuggestion {
    pub group_a_id: String,
    pub group_a_name: String,
    pub group_b_id: String,
    pub group_b_name: String,
    pub similarity: f32,
    pub rep_similarity: f32,
    #[serde(default)]
    pub breakdown: PossibleMergeBreakdown,
    #[serde(default = "default_merge_rank")]
    pub rank: MergeRank,
    #[serde(default)]
    pub project_identity: f32,
    #[serde(default)]
    pub contradiction: f32,
    pub dismissed: bool,
}

#[derive(Clone, Copy, Debug)]
pub struct IdentityCompatibilityGate {
    pub min_identity: f32,
    pub max_contradiction: f32,
}

impl Default for IdentityCompatibilityGate {
    fn default() -> Self {
        Self {
            min_identity: 0.65,
            max_contradiction: 0.28,
        }
    }
}

impl IdentityCompatibilityGate {
    pub fn allows_edge(
        &self,
        family_sim: f32,
        identity: f32,
        contradiction: f32,
        hard_contradiction: bool,
    ) -> bool {
        if hard_contradiction || contradiction > 0.35 {
            return false;
        }
        family_sim >= 0.70 || (identity >= 0.65 && contradiction <= 0.25)
    }

    pub fn allows_group_merge(
        &self,
        family_sim: f32,
        identity: f32,
        contradiction: f32,
        hard_contradiction: bool,
    ) -> bool {
        if hard_contradiction || contradiction > self.max_contradiction {
            return false;
        }
        identity >= self.min_identity && family_sim >= 0.70
    }
}

#[derive(Clone, Debug)]
pub struct RefinementEntry {
    pub kind: GroupKind,
    pub members: Vec<usize>,
    pub sig: GroupSignature,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ClusteringOutput {
    pub groups: Vec<ImageGroup>,
    pub blocked_chaining_edges: usize,
    pub possible_merges: Vec<PossibleMergeSuggestion>,
    #[serde(default)]
    pub possible_splits: Vec<PossibleSplit>,
    #[serde(default)]
    pub detected_outliers: usize,
    #[serde(default)]
    pub initial_group_count: usize,
    #[serde(default)]
    pub auto_merges_performed: usize,
}

pub fn compute_top_k_neighbors(
    num_images: usize,
    pairs: &[PairComparison],
    k: usize,
) -> Vec<Vec<(usize, f32)>> {
    let mut neighbors: Vec<Vec<(usize, f32)>> = vec![Vec::new(); num_images];
    for p in pairs {
        neighbors[p.i].push((p.j, p.score.combined));
        neighbors[p.j].push((p.i, p.score.combined));
    }
    for list in &mut neighbors {
        list.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
        if list.len() > k {
            list.truncate(k);
        }
    }
    neighbors
}

pub fn compute_mutual_nearest_neighbors(top_k: &[Vec<(usize, f32)>]) -> HashSet<(usize, usize)> {
    let mut mnn = HashSet::new();
    for (i, list) in top_k.iter().enumerate() {
        for &(j, _) in list {
            if j > i
                && top_k
                    .get(j)
                    .map_or(false, |jl| jl.iter().any(|&(nbr, _)| nbr == i))
            {
                mnn.insert((i, j));
            }
        }
    }
    mnn
}

pub fn jaccard_neighborhood(top_k: &[Vec<(usize, f32)>], a: usize, b: usize) -> f32 {
    let set_a: HashSet<usize> = top_k
        .get(a)
        .map(|l| l.iter().map(|&(x, _)| x).collect())
        .unwrap_or_default();
    let set_b: HashSet<usize> = top_k
        .get(b)
        .map(|l| l.iter().map(|&(x, _)| x).collect())
        .unwrap_or_default();
    if set_a.is_empty() && set_b.is_empty() {
        return 0.0;
    }
    let intersection = set_a.intersection(&set_b).count();
    let union = set_a.union(&set_b).count();
    if union == 0 {
        0.0
    } else {
        intersection as f32 / union as f32
    }
}

pub fn cluster_images(
    num_images: usize,
    features: &[ImageFeatures],
    pairs: &[PairComparison],
    cfg: &ClusteringConfig,
) -> ClusteringOutput {
    if num_images == 0 {
        return ClusteringOutput::default();
    }

    // Lookup map for O(1) pairwise similarity & relation queries
    let mut score_map: HashMap<(usize, usize), f32> = HashMap::with_capacity(pairs.len());
    let mut rel_map: HashMap<(usize, usize), PairRelation> = HashMap::with_capacity(pairs.len());
    let mut contra_map: HashMap<(usize, usize), (f32, bool)> = HashMap::with_capacity(pairs.len());
    let mut best_neighbor: Vec<(f32, usize)> = vec![(0.0, usize::MAX); num_images];

    for p in pairs {
        let key = (p.i.min(p.j), p.i.max(p.j));
        score_map.insert(key, p.score.combined);
        rel_map.insert(key, p.score.relation);
        contra_map.insert(key, (p.score.contradiction, p.score.hard_contradiction));
        if p.score.combined > best_neighbor[p.i].0 {
            best_neighbor[p.i] = (p.score.combined, p.j);
        }
        if p.score.combined > best_neighbor[p.j].0 {
            best_neighbor[p.j] = (p.score.combined, p.i);
        }
    }

    let top_k = compute_top_k_neighbors(num_images, pairs, cfg.top_k);
    let mnn = compute_mutual_nearest_neighbors(&top_k);

    let pair_lookup: HashMap<_, _> = pairs
        .iter()
        .map(|p| ((p.i.min(p.j), p.i.max(p.j)), p))
        .collect();
    // v3 deterministic local ordering; v4 identity and family scores are diagnostics only.
    let mut scored_edges: Vec<(usize, usize, f32, bool)> = pairs
        .iter()
        .filter(|p| {
            (p.score.relation.is_duplicate() || p.score.combined >= cfg.same_project_threshold)
                && pair_compatible(p, features, &cfg.admission)
        })
        .map(|p| (p.i, p.j, p.score.combined, false))
        .collect();

    // Prioritize edges with MNN support and higher score
    scored_edges.sort_unstable_by(|a, b| b.2.total_cmp(&a.2));

    let mut clusters: Vec<Vec<usize>> = (0..num_images).map(|i| vec![i]).collect();
    let mut cluster_of: Vec<usize> = (0..num_images).collect();
    let mut blocked_chaining_edges = 0usize;
    let mut chaining_flagged_nodes: HashSet<usize> = HashSet::new();

    for (a, b, _edge_score, _is_mnn) in scored_edges {
        let ca = cluster_of[a];
        let cb = cluster_of[b];
        if ca == cb {
            continue;
        }
        if clusters[ca].len() + clusters[cb].len() > cfg.max_group_size {
            continue;
        }

        // Existing local cores never merge automatically. Only singleton admission.
        if clusters[ca].len() > 1 && clusters[cb].len() > 1 {
            continue;
        }
        let (candidate, core) = if clusters[ca].len() == 1 {
            (clusters[ca][0], &clusters[cb])
        } else {
            (clusters[cb][0], &clusters[ca])
        };
        let medoid = select_representative(core, features, &score_map);
        if !group_consensus(
            candidate,
            core,
            medoid,
            &pair_lookup,
            features,
            &cfg.admission,
            cfg.cluster_min_pairwise,
        )
        .allows(&cfg.admission)
        {
            blocked_chaining_edges += 1;
            chaining_flagged_nodes.insert(a);
            chaining_flagged_nodes.insert(b);
            continue;
        }

        // Anti-chaining cluster consistency & contradiction gate check:
        // Every cross-pair must clear cluster_min_pairwise AND have low contradiction without hard contradiction.
        let mut consistent = true;
        let mut sum_cross = 0.0f32;
        let mut count_cross = 0usize;

        'outer: for &x in &clusters[ca] {
            for &y in &clusters[cb] {
                let key = (x.min(y), x.max(y));
                let s = score_map.get(&key).copied().unwrap_or(0.0);
                if s < cfg.cluster_min_pairwise {
                    consistent = false;
                    break 'outer;
                }
                if let Some(&(contra, is_hard)) = contra_map.get(&key) {
                    if is_hard || contra > 0.35 {
                        consistent = false;
                        break 'outer;
                    }
                }
                sum_cross += s;
                count_cross += 1;
            }
        }

        if !consistent || count_cross == 0 {
            blocked_chaining_edges += 1;
            chaining_flagged_nodes.insert(a);
            chaining_flagged_nodes.insert(b);
            continue;
        }

        let avg_cross = sum_cross / count_cross as f32;
        if avg_cross < cfg.cluster_avg_threshold {
            blocked_chaining_edges += 1;
            chaining_flagged_nodes.insert(a);
            chaining_flagged_nodes.insert(b);
            continue;
        }

        // Merge smaller cluster into larger cluster
        let (target, source) = if clusters[ca].len() >= clusters[cb].len() {
            (ca, cb)
        } else {
            (cb, ca)
        };
        let moved = std::mem::take(&mut clusters[source]);
        for m in moved {
            cluster_of[m] = target;
            clusters[target].push(m);
        }
    }

    // Separate High Confidence project groups, Probable Candidate subgroups, Duplicates, and Ungrouped
    let mut project_groups: Vec<ImageGroup> = Vec::new();
    let mut duplicate_groups: Vec<ImageGroup> = Vec::new();
    let mut unassigned_indices: Vec<usize> = Vec::new();

    for members in clusters {
        if members.is_empty() {
            continue;
        }
        if members.len() == 1 {
            unassigned_indices.push(members[0]);
            continue;
        }

        let (avg_similarity, all_duplicates) = group_stats(&members, &score_map, &rel_map);
        let rep = select_representative(&members, features, &score_map);

        if all_duplicates {
            duplicate_groups.push(ImageGroup {
                id: String::new(),
                name: String::new(),
                kind: GroupKind::Duplicates,
                representative_idx: Some(rep),
                member_indices: members,
                avg_similarity,
                confirmed: false,
                nearest_related: None,
                representative_members: vec![rep],
                signature: None,
            });
        } else {
            project_groups.push(ImageGroup {
                id: String::new(),
                name: String::new(),
                kind: GroupKind::FurnitureProject,
                representative_idx: Some(rep),
                member_indices: members,
                avg_similarity,
                confirmed: false,
                nearest_related: None,
                representative_members: vec![rep],
                signature: None,
            });
        }
    }

    // Partition unassigned images into coherent Probable Candidate Subgroups vs Truly Ungrouped
    // (Abolishing the monolithic flat Review bucket)
    let mut candidate_edges: Vec<(usize, usize, f32)> = Vec::new();
    for (i_idx, &u1) in unassigned_indices.iter().enumerate() {
        for &u2 in &unassigned_indices[i_idx + 1..] {
            let key = (u1.min(u2), u1.max(u2));
            if let Some(&s) = score_map.get(&key) {
                if s >= cfg.review_threshold || mnn.contains(&key) {
                    candidate_edges.push((u1, u2, s));
                }
            }
        }
    }
    candidate_edges.sort_unstable_by(|a, b| b.2.total_cmp(&a.2));

    let mut cand_cluster_of: HashMap<usize, usize> = HashMap::new();
    let mut cand_clusters: Vec<Vec<usize>> = Vec::new();

    for &u in &unassigned_indices {
        let cid = cand_clusters.len();
        cand_cluster_of.insert(u, cid);
        cand_clusters.push(vec![u]);
    }

    for (a, b, _s) in candidate_edges {
        let ca = cand_cluster_of[&a];
        let cb = cand_cluster_of[&b];
        if ca == cb {
            continue;
        }
        if cand_clusters[ca].len() + cand_clusters[cb].len() > 16 {
            continue;
        }

        let ok = if cand_clusters[ca].len() > 1 && cand_clusters[cb].len() > 1 {
            false
        } else {
            let (candidate, core) = if cand_clusters[ca].len() == 1 {
                (cand_clusters[ca][0], &cand_clusters[cb])
            } else {
                (cand_clusters[cb][0], &cand_clusters[ca])
            };
            let medoid = select_representative(core, features, &score_map);
            group_consensus(
                candidate,
                core,
                medoid,
                &pair_lookup,
                features,
                &cfg.admission,
                cfg.cluster_min_pairwise,
            )
            .allows(&cfg.admission)
        };

        if ok {
            let moved = std::mem::take(&mut cand_clusters[cb]);
            for m in moved {
                cand_cluster_of.insert(m, ca);
                cand_clusters[ca].push(m);
            }
        }
    }

    let mut probable_candidate_groups: Vec<ImageGroup> = Vec::new();
    let mut ungrouped_members: Vec<usize> = Vec::new();

    for members in cand_clusters {
        if members.is_empty() {
            continue;
        }
        if members.len() >= 2 {
            let (avg_sim, _) = group_stats(&members, &score_map, &rel_map);
            let rep = select_representative(&members, features, &score_map);
            probable_candidate_groups.push(ImageGroup {
                id: String::new(),
                name: String::new(),
                kind: GroupKind::CandidateGroup,
                representative_idx: Some(rep),
                member_indices: members,
                avg_similarity: avg_sim,
                confirmed: false,
                nearest_related: None,
                representative_members: vec![rep],
                signature: None,
            });
        } else {
            let single = members[0];
            let best_s = best_neighbor[single].0;
            // Singletons with a strong neighborhood link (e.g. blocked by anti-chaining from a project)
            // become a 1-image candidate subgroup instead of being dumped as isolated.
            if best_s >= cfg.review_threshold || chaining_flagged_nodes.contains(&single) {
                probable_candidate_groups.push(ImageGroup {
                    id: String::new(),
                    name: String::new(),
                    kind: GroupKind::CandidateGroup,
                    representative_idx: Some(single),
                    member_indices: vec![single],
                    avg_similarity: best_s,
                    confirmed: false,
                    nearest_related: None,
                    representative_members: vec![single],
                    signature: None,
                });
            } else {
                ungrouped_members.push(single);
            }
        }
    }

    // =========================================================================
    // PASS 2: Group Merge Refinement (Cross-View Furniture Identity)
    // =========================================================================
    let initial_group_count = project_groups.len()
        + probable_candidate_groups.len()
        + duplicate_groups.len()
        + (if ungrouped_members.is_empty() { 0 } else { 1 });

    let mut refinement_pool: Vec<RefinementEntry> = Vec::new();
    for g in project_groups
        .into_iter()
        .chain(probable_candidate_groups.into_iter())
    {
        let sig = build_group_signature(&g.member_indices, features, &score_map);
        refinement_pool.push(RefinementEntry {
            kind: g.kind,
            members: g.member_indices,
            sig,
        });
    }

    // CrossGroupAutomaticMerge = OFF, regardless of legacy persisted merge settings.
    // Group signatures and ranking remain available for human-confirmed suggestions.
    let total_auto_merges = 0;

    // Partition refinement pool into Project groups vs Candidate groups
    let mut final_project_groups: Vec<ImageGroup> = Vec::new();
    let mut final_candidate_groups: Vec<ImageGroup> = Vec::new();

    for entry in refinement_pool {
        let (avg_sim, _) = group_stats(&entry.members, &score_map, &rel_map);
        let rep = entry.sig.medoid_idx;
        let rep_members = entry.sig.representative_members.clone();
        let sig = Some(entry.sig);

        let is_project = entry.kind == GroupKind::FurnitureProject;

        if is_project {
            final_project_groups.push(ImageGroup {
                id: String::new(),
                name: String::new(),
                kind: GroupKind::FurnitureProject,
                representative_idx: Some(rep),
                member_indices: entry.members,
                avg_similarity: avg_sim,
                confirmed: false,
                nearest_related: None,
                representative_members: rep_members,
                signature: sig,
            });
        } else {
            final_candidate_groups.push(ImageGroup {
                id: String::new(),
                name: String::new(),
                kind: GroupKind::CandidateGroup,
                representative_idx: Some(rep),
                member_indices: entry.members,
                avg_similarity: avg_sim,
                confirmed: false,
                nearest_related: None,
                representative_members: rep_members,
                signature: sig,
            });
        }
    }

    // Sort Project groups descending by size, then similarity
    final_project_groups.sort_by(|a, b| {
        b.member_indices
            .len()
            .cmp(&a.member_indices.len())
            .then_with(|| b.avg_similarity.total_cmp(&a.avg_similarity))
    });

    let mut furniture_num = 1usize;
    for g in &mut final_project_groups {
        g.id = format!("Furniture_{furniture_num:03}");
        g.name = format!("Furniture {furniture_num:03}");
        furniture_num += 1;
    }

    // Duplicates groups
    duplicate_groups.sort_by(|a, b| {
        b.member_indices
            .len()
            .cmp(&a.member_indices.len())
            .then_with(|| b.avg_similarity.total_cmp(&a.avg_similarity))
    });
    let mut dup_num = 1usize;
    for g in &mut duplicate_groups {
        g.id = format!("Duplicates_{dup_num:03}");
        g.name = format!("Duplicates {dup_num:03}");
        dup_num += 1;
    }

    // Candidate groups
    final_candidate_groups.sort_by(|a, b| {
        b.member_indices
            .len()
            .cmp(&a.member_indices.len())
            .then_with(|| b.avg_similarity.total_cmp(&a.avg_similarity))
    });
    let mut cand_num = 1usize;
    for g in &mut final_candidate_groups {
        g.id = format!("Candidate_{cand_num:03}");
        g.name = format!("Candidate {cand_num:03}");
        cand_num += 1;
    }

    let mut all_groups = final_project_groups;
    all_groups.extend(duplicate_groups);
    all_groups.extend(final_candidate_groups);

    // Compute final group-to-group merge suggestions and nearest related indicators
    let (possible_merges, related_pairs) = compute_group_to_group_similarity(
        &mut all_groups,
        features,
        &score_map,
        cfg.possible_merge_threshold,
    );

    for (g_idx, related) in related_pairs.into_iter().enumerate() {
        if let Some(g) = all_groups.get_mut(g_idx) {
            g.nearest_related = related;
        }
    }

    // Truly unmatched images
    all_groups.push(ImageGroup {
        id: "Ungrouped".to_string(),
        name: "Ungrouped".to_string(),
        kind: GroupKind::Ungrouped,
        representative_idx: ungrouped_members.first().copied(),
        member_indices: ungrouped_members,
        avg_similarity: 0.0,
        confirmed: false,
        nearest_related: None,
        representative_members: Vec::new(),
        signature: None,
    });

    let possible_splits =
        crate::purity::audit_groups(&all_groups, &pair_lookup, features, &cfg.admission);
    let detected_outliers = possible_splits
        .iter()
        .filter(|s| s.outlier_idx.is_some())
        .count();

    ClusteringOutput {
        groups: all_groups,
        blocked_chaining_edges,
        possible_merges,
        possible_splits,
        detected_outliers,
        initial_group_count,
        auto_merges_performed: total_auto_merges,
    }
}

pub fn build_group_signature(
    members: &[usize],
    features: &[ImageFeatures],
    score_map: &HashMap<(usize, usize), f32>,
) -> GroupSignature {
    if members.is_empty() {
        return GroupSignature::default();
    }
    let medoid = select_representative(members, features, score_map);

    // Select 2 to 4 diverse representative members covering distinct views
    let mut rep_members = vec![medoid];
    if members.len() > 1 {
        let mut min_sim = f32::INFINITY;
        let mut second = medoid;
        for &m in members {
            if m == medoid {
                continue;
            }
            let k = (m.min(medoid), m.max(medoid));
            let s = score_map.get(&k).copied().unwrap_or(0.0);
            if s < min_sim {
                min_sim = s;
                second = m;
            }
        }
        if second != medoid {
            rep_members.push(second);
        }

        if members.len() > 2 {
            let mut best_cand = None;
            let mut best_min_dist = f32::NEG_INFINITY;
            for &m in members {
                if rep_members.contains(&m) {
                    continue;
                }
                let min_s_to_reps = rep_members
                    .iter()
                    .map(|&r| {
                        let k = (m.min(r), m.max(r));
                        score_map.get(&k).copied().unwrap_or(0.0)
                    })
                    .fold(f32::INFINITY, f32::min);
                let dist = 1.0 - min_s_to_reps;
                if dist > best_min_dist {
                    best_min_dist = dist;
                    best_cand = Some(m);
                }
            }
            if let Some(third) = best_cand {
                rep_members.push(third);
            }
        }

        if members.len() > 3 {
            let mut best_cand = None;
            let mut best_min_dist = f32::NEG_INFINITY;
            for &m in members {
                if rep_members.contains(&m) {
                    continue;
                }
                let min_s_to_reps = rep_members
                    .iter()
                    .map(|&r| {
                        let k = (m.min(r), m.max(r));
                        score_map.get(&k).copied().unwrap_or(0.0)
                    })
                    .fold(f32::INFINITY, f32::min);
                let dist = 1.0 - min_s_to_reps;
                if dist > best_min_dist {
                    best_min_dist = dist;
                    best_cand = Some(m);
                }
            }
            if let Some(fourth) = best_cand {
                rep_members.push(fourth);
            }
        }
    }

    // Geometry aggregation
    let mut avg_vp = vec![0.0f32; 16];
    let mut avg_hp = vec![0.0f32; 16];
    let mut avg_grid = vec![0.0f32; 9];
    let mut sum_vr = 0.0f32;
    let mut sum_sym = 0.0f32;
    let mut sum_c2b = 0.0f32;
    let mut sum_h_div = 0.0f32;
    let mut sum_v_div = 0.0f32;

    // Color aggregation
    let mut sum_coarse = [0.0f32; 5];
    let mut sum_hist = vec![0.0f32; 76];

    // Embedding aggregation
    let mut emb_sum = Vec::new();
    let mut emb_count = 0usize;

    for &m in members {
        let fp = &features[m].fingerprint;
        if fp.geometry.vertical_profile.len() >= 16 {
            for (idx, &v) in fp.geometry.vertical_profile.iter().take(16).enumerate() {
                avg_vp[idx] += v;
            }
        }
        if fp.geometry.horizontal_profile.len() >= 16 {
            for (idx, &v) in fp.geometry.horizontal_profile.iter().take(16).enumerate() {
                avg_hp[idx] += v;
            }
        }
        if fp.geometry.structural_grid.len() >= 9 {
            for (idx, &v) in fp.geometry.structural_grid.iter().take(9).enumerate() {
                avg_grid[idx] += v;
            }
        }
        sum_vr += fp.geometry.vertical_edge_ratio;
        sum_sym += fp.geometry.symmetry_score;
        sum_c2b += fp.geometry.center_to_border_ratio;
        sum_h_div += fp.geometry.horizontal_division_count;
        sum_v_div += fp.geometry.vertical_division_count;

        for (idx, &c) in fp.coarse_color.iter().enumerate() {
            sum_coarse[idx] += c;
        }
        if fp.color_histogram.len() >= 76 {
            for (idx, &h) in fp.color_histogram.iter().take(76).enumerate() {
                sum_hist[idx] += h;
            }
        }

        if let Some(emb) = &features[m].embedding {
            if emb_sum.is_empty() {
                emb_sum = vec![0.0f32; emb.len()];
            }
            if emb_sum.len() == emb.len() {
                for (i, &x) in emb.iter().enumerate() {
                    emb_sum[i] += x;
                }
                emb_count += 1;
            }
        }
    }

    let n = members.len() as f32;
    l1_norm(&mut avg_vp);
    l1_norm(&mut avg_hp);
    l1_norm(&mut avg_grid);
    l1_norm(&mut sum_hist);
    for c in &mut sum_coarse {
        *c /= n;
    }

    let centroid_embedding = if emb_count > 0 {
        let mut norm_sq = 0.0f32;
        for x in &emb_sum {
            norm_sq += x * x;
        }
        let norm = norm_sq.sqrt().max(1e-6);
        for x in &mut emb_sum {
            *x /= norm;
        }
        Some(emb_sum)
    } else {
        None
    };

    // Internal pairwise stats
    let mut min_sim = 1.0f32;
    let mut sum_sim = 0.0f32;
    let mut count_sim = 0usize;
    for (i_idx, &m1) in members.iter().enumerate() {
        for &m2 in &members[i_idx + 1..] {
            let k = (m1.min(m2), m1.max(m2));
            let s = score_map.get(&k).copied().unwrap_or(0.0);
            sum_sim += s;
            count_sim += 1;
            if s < min_sim {
                min_sim = s;
            }
        }
    }
    let avg_sim = if count_sim > 0 {
        sum_sim / count_sim as f32
    } else {
        1.0
    };
    if count_sim == 0 {
        min_sim = 1.0;
    }

    GroupSignature {
        medoid_idx: medoid,
        representative_members: rep_members,
        centroid_embedding,
        avg_vertical_profile: avg_vp,
        avg_horizontal_profile: avg_hp,
        avg_structural_grid: avg_grid,
        avg_vertical_edge_ratio: sum_vr / n,
        avg_symmetry_score: sum_sym / n,
        avg_center_to_border: sum_c2b / n,
        avg_horizontal_division_count: sum_h_div / n,
        avg_vertical_division_count: sum_v_div / n,
        avg_coarse_color: sum_coarse,
        avg_color_histogram: sum_hist,
        avg_internal_similarity: avg_sim,
        min_internal_similarity: min_sim,
    }
}

pub fn compute_group_merge_score(
    sig_a: &GroupSignature,
    sig_b: &GroupSignature,
    members_a: &[usize],
    members_b: &[usize],
    features: &[ImageFeatures],
    score_map: &HashMap<(usize, usize), f32>,
    is_mutual: bool,
    is_triad: bool,
) -> (f32, PossibleMergeBreakdown) {
    // 1. Representative-to-representative similarity
    let rep_k = (
        sig_a.medoid_idx.min(sig_b.medoid_idx),
        sig_a.medoid_idx.max(sig_b.medoid_idx),
    );
    let s_rep = score_map.get(&rep_k).copied().unwrap_or(0.0);

    // 2. Cross-member matches: take representative members (2-4 each) plus up to 8 members
    let mut cross_scores = Vec::new();
    let candidates_a: Vec<usize> = sig_a
        .representative_members
        .iter()
        .copied()
        .chain(members_a.iter().copied().take(8))
        .collect::<HashSet<_>>()
        .into_iter()
        .collect();
    let candidates_b: Vec<usize> = sig_b
        .representative_members
        .iter()
        .copied()
        .chain(members_b.iter().copied().take(8))
        .collect::<HashSet<_>>()
        .into_iter()
        .collect();

    for &ma in &candidates_a {
        for &mb in &candidates_b {
            let k = (ma.min(mb), ma.max(mb));
            if let Some(&s) = score_map.get(&k) {
                cross_scores.push(s);
            }
        }
    }
    cross_scores.sort_unstable_by(|a, b| b.total_cmp(a));
    let s_max = cross_scores.first().copied().unwrap_or(s_rep);
    let s_top3_avg = if !cross_scores.is_empty() {
        let n = cross_scores.len().min(3);
        cross_scores.iter().take(n).sum::<f32>() / n as f32
    } else {
        s_rep
    };
    let s_cross = 0.40 * s_max + 0.35 * s_top3_avg + 0.25 * s_rep;

    // 3. Geometry agreement
    let s_vp =
        crate::fingerprint::ruzicka_slice(&sig_a.avg_vertical_profile, &sig_b.avg_vertical_profile);
    let s_hp = crate::fingerprint::ruzicka_slice(
        &sig_a.avg_horizontal_profile,
        &sig_b.avg_horizontal_profile,
    );
    let s_grid =
        crate::fingerprint::ruzicka_slice(&sig_a.avg_structural_grid, &sig_b.avg_structural_grid);
    let s_sym =
        (1.0 - (sig_a.avg_symmetry_score - sig_b.avg_symmetry_score).abs() * 2.0).clamp(0.0, 1.0);
    let s_div = (1.0
        - (sig_a.avg_horizontal_division_count - sig_b.avg_horizontal_division_count).abs() * 0.25)
        .clamp(0.0, 1.0);
    let s_geom =
        (0.30 * s_vp + 0.30 * s_hp + 0.18 * s_grid + 0.12 * s_sym + 0.10 * s_div).clamp(0.0, 1.0);

    // 4. Color / material agreement
    let s_ab = if sig_a.avg_color_histogram.len() >= 64 && sig_b.avg_color_histogram.len() >= 64 {
        crate::fingerprint::ruzicka_slice(
            &sig_a.avg_color_histogram[..64],
            &sig_b.avg_color_histogram[..64],
        )
    } else {
        0.5
    };
    let s_l = if sig_a.avg_color_histogram.len() >= 76 && sig_b.avg_color_histogram.len() >= 76 {
        crate::fingerprint::ruzicka_slice(
            &sig_a.avg_color_histogram[64..76],
            &sig_b.avg_color_histogram[64..76],
        )
    } else {
        0.5
    };
    let s_color = (0.75 * s_ab + 0.25 * s_l).clamp(0.0, 1.0);

    // 5. Centroid embedding
    let s_emb = match (&sig_a.centroid_embedding, &sig_b.centroid_embedding) {
        (Some(ea), Some(eb)) if ea.len() == eb.len() => {
            let dot: f32 = ea.iter().zip(eb.iter()).map(|(&x, &y)| x * y).sum();
            Some(dot.clamp(0.0, 1.0))
        }
        _ => None,
    };

    // 6. Partial view match: best partial match between representative members
    let mut s_partial = 0.0f32;
    if !features.is_empty() {
        for &ra in &sig_a.representative_members {
            for &rb in &sig_b.representative_members {
                if let (Some(fa), Some(fb)) = (features.get(ra), features.get(rb)) {
                    let sp = crate::fingerprint::partial_view_similarity(
                        &fa.fingerprint,
                        &fb.fingerprint,
                    );
                    if sp > s_partial {
                        s_partial = sp;
                    }
                }
            }
        }
    }

    let mutual_bonus = if is_mutual { 0.04 } else { 0.0 } + if is_triad { 0.03 } else { 0.0 };

    let base_score = match s_emb {
        Some(e) => 0.32 * s_geom + 0.23 * s_cross + 0.20 * e + 0.13 * s_color + 0.12 * s_partial,
        None => 0.38 * s_geom + 0.28 * s_cross + 0.18 * s_color + 0.16 * s_partial,
    };
    let merge_score = (base_score + mutual_bonus).clamp(0.0, 1.0);

    // 7. Identity & Contradiction across representative members
    let mut id_sum = 0.0f32;
    let mut contra_sum = 0.0f32;
    let mut hard_contra = false;
    let mut rep_pair_count = 0usize;

    if !features.is_empty() {
        for &ra in &sig_a.representative_members {
            for &rb in &sig_b.representative_members {
                if let (Some(fa), Some(fb)) = (features.get(ra), features.get(rb)) {
                    let (id, contra, hard) = crate::fingerprint::compare_topology(
                        &fa.fingerprint.topology,
                        &fb.fingerprint.topology,
                        s_partial >= 0.65,
                    );
                    id_sum += id;
                    contra_sum += contra;
                    if hard {
                        hard_contra = true;
                    }
                    rep_pair_count += 1;
                }
            }
        }
    }

    let avg_identity = if rep_pair_count > 0 {
        (0.50 * s_geom + 0.30 * (id_sum / rep_pair_count as f32) + 0.20 * s_cross).clamp(0.0, 1.0)
    } else {
        s_geom
    };
    let mut avg_contra = if rep_pair_count > 0 {
        contra_sum / rep_pair_count as f32
    } else {
        0.0
    };
    if hard_contra {
        avg_contra = avg_contra.max(0.65);
    }

    let mut family_sum = 0.0;
    let mut identity_sum = 0.0;
    let mut contradiction_sum = 0.0;
    let mut cross_supported = 0;
    let mut cross_count = 0;
    let mut any_hard = false;
    for &a in members_a {
        for &b in members_b {
            if let (Some(fa), Some(fb)) = (features.get(a), features.get(b)) {
                let score = crate::similarity::compare_pair_direct(
                    fa,
                    fb,
                    &crate::similarity::SimilarityConfig::default(),
                );
                let pair = PairComparison { i: a, j: b, score };
                family_sum += pair.score.family_similarity;
                identity_sum += pair.score.project_identity;
                contradiction_sum += pair.score.contradiction;
                any_hard |= pair.score.hard_contradiction;
                cross_supported += usize::from(pair_compatible(
                    &pair,
                    features,
                    &AdmissionConfig::default(),
                ));
                cross_count += 1;
            }
        }
    }
    let denominator = cross_count.max(1) as f32;

    let breakdown = PossibleMergeBreakdown {
        family_similarity: family_sum / denominator,
        cross_member_support: cross_supported as f32 / denominator,
        structure: s_geom,
        embedding: s_emb,
        cross_view: s_cross,
        color: s_color,
        partial_match: s_partial,
        mutual_neighbor: is_mutual,
        project_identity: if cross_count > 0 {
            identity_sum / denominator
        } else {
            avg_identity
        },
        contradiction: if cross_count > 0 {
            contradiction_sum / denominator
        } else {
            avg_contra
        },
        hard_contradiction: any_hard || hard_contra,
    };

    (merge_score, breakdown)
}

pub fn check_merge_consistency(
    sig_a: &GroupSignature,
    sig_b: &GroupSignature,
    members_a: &[usize],
    members_b: &[usize],
    features: &[ImageFeatures],
    score_map: &HashMap<(usize, usize), f32>,
    breakdown: &PossibleMergeBreakdown,
    max_group_size: usize,
) -> bool {
    // 1. Size constraint
    if members_a.len() + members_b.len() > max_group_size {
        return false;
    }

    // 2. Strong structural consistency
    if breakdown.structure < 0.68 {
        return false;
    }

    // 3. Minimum material/color agreement
    if breakdown.color < 0.52 {
        return false;
    }

    // 4. Cross-view cohesion
    if breakdown.cross_view < 0.58 {
        return false;
    }

    // 5. Contradictory cross-members check:
    // If any cross pair between members has pairwise similarity < 0.50, reject auto-merge (anti-chaining guard).
    for &ma in members_a {
        for &mb in members_b {
            let k = (ma.min(mb), ma.max(mb));
            if let Some(&s) = score_map.get(&k) {
                if s < 0.50 {
                    return false;
                }
            }
        }
    }

    // 6. No member pair in rep_members(A) x rep_members(B) has contradictory structural fingerprint
    if !features.is_empty() {
        for &ra in &sig_a.representative_members {
            for &rb in &sig_b.representative_members {
                if let (Some(fa), Some(fb)) = (features.get(ra), features.get(rb)) {
                    let st = crate::deterministic::furniture_structure_similarity(
                        &fa.fingerprint,
                        &fb.fingerprint,
                    );
                    if st < 0.38 {
                        return false;
                    }
                }
            }
        }
    }

    // 7. Identity Compatibility Gate
    if breakdown.hard_contradiction {
        return false;
    }
    if breakdown.contradiction > 0.28 {
        return false;
    }
    if breakdown.project_identity < 0.65 {
        return false;
    }

    true
}

/// Evaluates group-to-group similarity combining representative similarity,
/// best cross-member match, average top-3 cross-member matches, and mutual neighbor links.
pub fn compute_group_to_group_similarity(
    groups: &mut [ImageGroup],
    features: &[ImageFeatures],
    score_map: &HashMap<(usize, usize), f32>,
    threshold: f32,
) -> (Vec<PossibleMergeSuggestion>, Vec<Option<(String, f32)>>) {
    let mut suggestions = Vec::new();
    let mut nearest_related = vec![None; groups.len()];

    for i in 0..groups.len() {
        if matches!(groups[i].kind, GroupKind::Ungrouped) || groups[i].member_indices.is_empty() {
            continue;
        }

        // Ensure signature is built
        if groups[i].signature.is_none() {
            let sig = build_group_signature(&groups[i].member_indices, features, score_map);
            groups[i].representative_members = sig.representative_members.clone();
            groups[i].signature = Some(sig);
        }

        let mut best_for_i: Option<(String, f32)> = None;

        for j in (i + 1)..groups.len() {
            if matches!(groups[j].kind, GroupKind::Ungrouped) || groups[j].member_indices.is_empty()
            {
                continue;
            }

            if groups[j].signature.is_none() {
                let sig = build_group_signature(&groups[j].member_indices, features, score_map);
                groups[j].representative_members = sig.representative_members.clone();
                groups[j].signature = Some(sig);
            }

            let sig_i = groups[i].signature.as_ref().unwrap();
            let sig_j = groups[j].signature.as_ref().unwrap();

            let rep_k = (
                sig_i.medoid_idx.min(sig_j.medoid_idx),
                sig_i.medoid_idx.max(sig_j.medoid_idx),
            );
            let s_rep = score_map.get(&rep_k).copied().unwrap_or(0.0);

            let (merge_score, breakdown) = compute_group_merge_score(
                sig_i,
                sig_j,
                &groups[i].member_indices,
                &groups[j].member_indices,
                features,
                score_map,
                false,
                false,
            );

            let rank = if breakdown.project_identity >= 0.80
                && breakdown.contradiction <= 0.18
                && breakdown.cross_member_support >= 0.67
                && breakdown.structure >= 0.70
                && breakdown.color >= 0.60
                && !breakdown.hard_contradiction
            {
                MergeRank::VeryStrong
            } else if breakdown.project_identity >= 0.70
                && breakdown.contradiction <= 0.28
                && breakdown.cross_member_support >= 0.60
                && !breakdown.hard_contradiction
            {
                MergeRank::Strong
            } else {
                MergeRank::Possible
            };

            if merge_score >= threshold {
                suggestions.push(PossibleMergeSuggestion {
                    group_a_id: groups[i].id.clone(),
                    group_a_name: groups[i].name.clone(),
                    group_b_id: groups[j].id.clone(),
                    group_b_name: groups[j].name.clone(),
                    similarity: merge_score,
                    rep_similarity: s_rep,
                    rank,
                    project_identity: breakdown.project_identity,
                    contradiction: breakdown.contradiction,
                    breakdown,
                    dismissed: false,
                });
            }

            if merge_score >= 0.65 {
                if best_for_i.as_ref().map_or(true, |(_, s)| merge_score > *s) {
                    best_for_i = Some((groups[j].name.clone(), merge_score));
                }
                if nearest_related[j]
                    .as_ref()
                    .map_or(true, |(_, s)| merge_score > *s)
                {
                    nearest_related[j] = Some((groups[i].name.clone(), merge_score));
                }
            }
        }

        if let Some(pair) = best_for_i {
            if nearest_related[i]
                .as_ref()
                .map_or(true, |(_, s)| pair.1 > *s)
            {
                nearest_related[i] = Some(pair);
            }
        }
    }

    suggestions.sort_unstable_by(|a, b| b.similarity.total_cmp(&a.similarity));
    (suggestions, nearest_related)
}

fn l1_norm(v: &mut [f32]) {
    let sum: f32 = v.iter().sum();
    if sum > 1e-6 {
        for x in v.iter_mut() {
            *x /= sum;
        }
    }
}

pub fn group_stats(
    members: &[usize],
    score_map: &HashMap<(usize, usize), f32>,
    rel_map: &HashMap<(usize, usize), PairRelation>,
) -> (f32, bool) {
    if members.len() < 2 {
        return (1.0, false);
    }
    let mut sum = 0.0f32;
    let mut count = 0usize;
    let mut all_dups = true;
    for (ai, &i) in members.iter().enumerate() {
        for &j in &members[ai + 1..] {
            let key = (i.min(j), i.max(j));
            let s = score_map.get(&key).copied().unwrap_or(0.0);
            sum += s;
            count += 1;
            if let Some(rel) = rel_map.get(&key) {
                if !rel.is_duplicate() {
                    all_dups = false;
                }
            } else {
                all_dups = false;
            }
        }
    }
    let avg = if count > 0 { sum / count as f32 } else { 0.0 };
    (avg, all_dups)
}

/// Medoid-like group representative selection considering:
/// - average intra-cluster similarity (60%)
/// - relative resolution (15%)
/// - sharpness (15%)
/// - central framing focus (10%)
pub fn select_representative(
    members: &[usize],
    features: &[ImageFeatures],
    score_map: &HashMap<(usize, usize), f32>,
) -> usize {
    if members.len() <= 1 {
        return members[0];
    }

    let max_log_res = members
        .iter()
        .map(|&m| {
            let fp = &features[m].fingerprint;
            ((fp.width as f32) * (fp.height as f32)).max(1.0).log2()
        })
        .fold(1.0f32, f32::max);

    let max_sharp = members
        .iter()
        .map(|&m| features[m].fingerprint.sharpness)
        .fold(0.01f32, f32::max);

    let mut best_member = members[0];
    let mut best_score = f32::NEG_INFINITY;

    for &m in members {
        let mut sim_sum = 0.0f32;
        for &other in members {
            if other == m {
                continue;
            }
            let key = (m.min(other), m.max(other));
            sim_sum += score_map.get(&key).copied().unwrap_or(0.0);
        }
        let avg_sim = sim_sum / (members.len() - 1) as f32;
        let fp = &features[m].fingerprint;
        let res_norm = ((fp.width as f32) * (fp.height as f32)).max(1.0).log2() / max_log_res;
        let sharp_norm = (fp.sharpness / max_sharp).clamp(0.0, 1.0);
        let center_norm = fp.center_focus.clamp(0.0, 1.0);

        let total = 0.60 * avg_sim + 0.15 * res_norm + 0.15 * sharp_norm + 0.10 * center_norm;
        if total > best_score {
            best_score = total;
            best_member = m;
        }
    }

    best_member
}
