use crate::cache::CacheStore;
use crate::clustering::{ClusteringOutput, GroupKind, ImageGroup};
use crate::embeddings::ImageFeatures;
use crate::scanner::ScannedImage;
use crate::similarity::{
    PairComparison, SimilarityConfig, ambiguous_image_indices, compare_embedding_only,
    compare_pair_direct,
};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::fmt::Write as _;
use std::fs;
use std::path::PathBuf;

pub const EVALUATION_STORE_VERSION: u32 = 1;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum GroundTruthLabel {
    SameProject,
    DifferentProject,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum LabelSource {
    ManualComparison,
    ManualCorrection,
    ExternalImport,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LabeledPair {
    /// Stable image identity (`openimg:<sha>`, `sha256:<sha>`, or legacy relative key).
    pub image_a: String,
    pub image_b: String,
    /// Short non-personal display label (basename / relative path), never a full local absolute path.
    #[serde(default)]
    pub display_a: String,
    #[serde(default)]
    pub display_b: String,
    pub label: GroundTruthLabel,
    #[serde(default = "default_label_source")]
    pub source: LabelSource,
}

fn default_label_source() -> LabelSource {
    LabelSource::ManualComparison
}

/// Optional compatibility schema for headless `--labels` imports using `rel_path_a`/`rel_path_b`.
#[derive(Clone, Debug, Deserialize)]
struct LegacyLabeledPair {
    rel_path_a: String,
    rel_path_b: String,
    label: GroundTruthLabel,
}

#[derive(Clone, Debug, Deserialize)]
struct LegacyEvaluationStore {
    pairs: Vec<LegacyLabeledPair>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HumanProject {
    pub id: String,
    pub name: String,
    pub member_hashes: HashSet<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EvaluationLabelStore {
    pub version: u32,
    pub pairs: Vec<LabeledPair>,
    #[serde(default)]
    pub projects: Vec<HumanProject>,
}

impl Default for EvaluationLabelStore {
    fn default() -> Self {
        Self {
            version: EVALUATION_STORE_VERSION,
            pairs: Vec::new(),
            projects: Vec::new(),
        }
    }
}

impl EvaluationLabelStore {
    /// Explicit coherent membership, keyed only by stable SHA identities.
    /// Merge overlapping human projects while preserving an existing project ID.
    pub fn record_coherent_members(&mut self, mut hashes: HashSet<String>) {
        hashes.retain(|h| h.starts_with("sha256:") || h.starts_with("openimg:"));
        if hashes.is_empty() {
            return;
        }
        let mut overlaps: Vec<_> = self
            .projects
            .iter()
            .filter(|p| !p.member_hashes.is_disjoint(&hashes))
            .map(|p| p.id.clone())
            .collect();
        overlaps.sort();
        let id = overlaps.first().cloned().unwrap_or_else(|| {
            let first = hashes.iter().min().unwrap();
            format!("human_{}", first.replace(':', "_"))
        });
        let name = self
            .projects
            .iter()
            .find(|p| p.id == id)
            .map(|p| p.name.clone())
            .unwrap_or_else(|| "Human-confirmed project".into());
        for project in &self.projects {
            if overlaps.contains(&project.id) {
                hashes.extend(project.member_hashes.iter().cloned());
            }
        }
        self.projects.retain(|p| !overlaps.contains(&p.id));
        self.upsert_human_project(&id, &name, hashes);
    }
    pub fn canonical_pair_key<'a>(a: &'a str, b: &'a str) -> (&'a str, &'a str) {
        if a <= b { (a, b) } else { (b, a) }
    }

    pub fn upsert_human_project(&mut self, id: &str, name: &str, hashes: HashSet<String>) {
        if let Some(existing) = self.projects.iter_mut().find(|p| p.id == id) {
            existing.name = name.to_string();
            existing.member_hashes.extend(hashes);
        } else {
            self.projects.push(HumanProject {
                id: id.to_string(),
                name: name.to_string(),
                member_hashes: hashes,
            });
        }
    }

    pub fn upsert(
        &mut self,
        id_a: &str,
        id_b: &str,
        display_a: &str,
        display_b: &str,
        label: GroundTruthLabel,
        source: LabelSource,
    ) {
        if id_a == id_b {
            return;
        }
        let (ka, kb, da, db) = if id_a <= id_b {
            (id_a, id_b, display_a, display_b)
        } else {
            (id_b, id_a, display_b, display_a)
        };
        if let Some(existing) = self
            .pairs
            .iter_mut()
            .find(|p| p.image_a == ka && p.image_b == kb)
        {
            existing.label = label;
            existing.source = source;
            if !da.is_empty() {
                existing.display_a = da.to_string();
            }
            if !db.is_empty() {
                existing.display_b = db.to_string();
            }
        } else {
            self.pairs.push(LabeledPair {
                image_a: ka.to_string(),
                image_b: kb.to_string(),
                display_a: da.to_string(),
                display_b: db.to_string(),
                label,
                source,
            });
        }
    }

    pub fn count_by_label(&self) -> (usize, usize) {
        let same = self
            .pairs
            .iter()
            .filter(|p| p.label == GroundTruthLabel::SameProject)
            .count();
        let diff = self
            .pairs
            .iter()
            .filter(|p| p.label == GroundTruthLabel::DifferentProject)
            .count();
        (same, diff)
    }

    pub fn lookup_label(
        &self,
        img_a: &ScannedImage,
        img_b: &ScannedImage,
    ) -> Option<GroundTruthLabel> {
        let id_a = img_a.stable_id();
        let id_b = img_b.stable_id();
        let (ka, kb) = Self::canonical_pair_key(&id_a, &id_b);
        if let Some(p) = self
            .pairs
            .iter()
            .find(|p| p.image_a == ka && p.image_b == kb)
        {
            return Some(p.label);
        }
        let (ra, rb) = Self::canonical_pair_key(&img_a.relative_path, &img_b.relative_path);
        self.pairs
            .iter()
            .find(|p| p.image_a == ra && p.image_b == rb)
            .map(|p| p.label)
    }
}

pub fn labels_file_path(cache_store: &CacheStore) -> PathBuf {
    cache_store.eval_dir().join("labels.json")
}

pub fn load_evaluation_labels(cache_store: &CacheStore) -> EvaluationLabelStore {
    let path = labels_file_path(cache_store);
    let Ok(raw) = fs::read(&path) else {
        return EvaluationLabelStore::default();
    };
    parse_labels_bytes(&raw).unwrap_or_default()
}

pub fn parse_labels_bytes(raw: &[u8]) -> Result<EvaluationLabelStore, String> {
    let clean = raw.strip_prefix(&[0xEF, 0xBB, 0xBF]).unwrap_or(raw);
    if let Ok(store) = serde_json::from_slice::<EvaluationLabelStore>(clean) {
        return Ok(store);
    }
    if let Ok(legacy) = serde_json::from_slice::<LegacyEvaluationStore>(clean) {
        let mut store = EvaluationLabelStore::default();
        for p in legacy.pairs {
            store.upsert(
                &p.rel_path_a,
                &p.rel_path_b,
                &p.rel_path_a,
                &p.rel_path_b,
                p.label,
                LabelSource::ExternalImport,
            );
        }
        return Ok(store);
    }
    Err("Failed to parse evaluation labels JSON".to_string())
}

pub fn save_evaluation_labels(
    cache_store: &CacheStore,
    store: &EvaluationLabelStore,
) -> Result<PathBuf, String> {
    fs::create_dir_all(cache_store.eval_dir())
        .map_err(|e| format!("Failed to create evaluation dir: {e}"))?;
    let path = labels_file_path(cache_store);
    let json = serde_json::to_string_pretty(store)
        .map_err(|e| format!("Failed to serialize evaluation labels: {e}"))?;
    fs::write(&path, json).map_err(|e| format!("Failed to write {}: {e}", path.display()))?;
    Ok(path)
}

/// Captures `SameProject` labels between a confirmed group's representative and its members.
pub fn capture_confirm_group(
    store: &mut EvaluationLabelStore,
    group: &ImageGroup,
    images: &[ScannedImage],
) {
    if !matches!(
        group.kind,
        GroupKind::FurnitureProject | GroupKind::Duplicates
    ) || group.member_indices.len() < 2
    {
        return;
    }
    let anchor = group
        .representative_idx
        .filter(|r| group.member_indices.contains(r))
        .unwrap_or(group.member_indices[0]);
    let Some(img_anchor) = images.get(anchor) else {
        return;
    };
    let anchor_id = img_anchor.stable_id();
    store.record_coherent_members(
        group
            .member_indices
            .iter()
            .filter_map(|&m| images.get(m))
            .map(ScannedImage::stable_id)
            .collect(),
    );
    for &m in &group.member_indices {
        if m == anchor {
            continue;
        }
        if let Some(img_m) = images.get(m) {
            store.upsert(
                &anchor_id,
                &img_m.stable_id(),
                &img_anchor.file_name,
                &img_m.file_name,
                GroundTruthLabel::SameProject,
                LabelSource::ManualCorrection,
            );
        }
    }
}

/// Captures `DifferentProject` labels when an image is explicitly removed from a project group to Ungrouped.
pub fn capture_remove_from_group(
    store: &mut EvaluationLabelStore,
    source_group: &ImageGroup,
    removed_indices: &[usize],
    images: &[ScannedImage],
) {
    if !matches!(
        source_group.kind,
        GroupKind::FurnitureProject | GroupKind::Duplicates
    ) {
        return;
    }
    let removed_hashes: HashSet<_> = removed_indices
        .iter()
        .filter_map(|&m| images.get(m))
        .map(ScannedImage::stable_id)
        .collect();
    let remaining_hashes: HashSet<_> = source_group
        .member_indices
        .iter()
        .filter(|m| !removed_indices.contains(m))
        .filter_map(|&m| images.get(m))
        .map(ScannedImage::stable_id)
        .collect();
    for project in &mut store.projects {
        if !project.member_hashes.is_disjoint(&remaining_hashes) {
            project
                .member_hashes
                .retain(|h| !removed_hashes.contains(h));
        }
    }
    store.projects.retain(|p| !p.member_hashes.is_empty());
    store.pairs.retain(|p| {
        p.label != GroundTruthLabel::SameProject
            || !((removed_hashes.contains(&p.image_a) && remaining_hashes.contains(&p.image_b))
                || (removed_hashes.contains(&p.image_b) && remaining_hashes.contains(&p.image_a)))
    });
    let remaining_anchor = source_group
        .representative_idx
        .filter(|r| !removed_indices.contains(r) && source_group.member_indices.contains(r))
        .or_else(|| {
            source_group
                .member_indices
                .iter()
                .copied()
                .find(|m| !removed_indices.contains(m))
        });
    let Some(anchor_idx) = remaining_anchor else {
        return;
    };
    let Some(img_anchor) = images.get(anchor_idx) else {
        return;
    };
    let anchor_id = img_anchor.stable_id();
    for &r in removed_indices {
        if let Some(img_r) = images.get(r) {
            store.upsert(
                &anchor_id,
                &img_r.stable_id(),
                &img_anchor.file_name,
                &img_r.file_name,
                GroundTruthLabel::DifferentProject,
                LabelSource::ManualCorrection,
            );
        }
    }
}

/// Captures `DifferentProject` (against source project) and `SameProject` (against destination project)
/// when the user moves selected images between groups.
pub fn capture_move_members(
    store: &mut EvaluationLabelStore,
    source_group: &ImageGroup,
    target_group: &ImageGroup,
    moved_indices: &[usize],
    images: &[ScannedImage],
) {
    if matches!(
        source_group.kind,
        GroupKind::FurnitureProject | GroupKind::Duplicates
    ) {
        capture_remove_from_group(store, source_group, moved_indices, images);
    }

    if matches!(
        target_group.kind,
        GroupKind::FurnitureProject | GroupKind::Duplicates
    ) {
        let target_anchor = target_group
            .representative_idx
            .filter(|r| target_group.member_indices.contains(r))
            .or_else(|| target_group.member_indices.first().copied());
        if let Some(anchor_idx) = target_anchor
            && let Some(img_anchor) = images.get(anchor_idx)
        {
            let anchor_id = img_anchor.stable_id();
            for &m in moved_indices {
                if let Some(img_m) = images.get(m) {
                    store.upsert(
                        &anchor_id,
                        &img_m.stable_id(),
                        &img_anchor.file_name,
                        &img_m.file_name,
                        GroundTruthLabel::SameProject,
                        LabelSource::ManualCorrection,
                    );
                }
            }
        }
    }
}

/// Captures `SameProject` when two project groups are merged.
pub fn capture_merge_groups(
    store: &mut EvaluationLabelStore,
    source_group: &ImageGroup,
    target_group: &ImageGroup,
    images: &[ScannedImage],
) {
    let target_anchor = target_group
        .representative_idx
        .filter(|r| target_group.member_indices.contains(r))
        .or_else(|| target_group.member_indices.first().copied());
    let Some(t_idx) = target_anchor else {
        return;
    };
    let Some(img_t) = images.get(t_idx) else {
        return;
    };
    let t_id = img_t.stable_id();
    store.record_coherent_members(
        source_group
            .member_indices
            .iter()
            .chain(&target_group.member_indices)
            .filter_map(|&m| images.get(m))
            .map(ScannedImage::stable_id)
            .collect(),
    );
    for &m in &source_group.member_indices {
        if let Some(img_m) = images.get(m) {
            store.upsert(
                &t_id,
                &img_m.stable_id(),
                &img_t.file_name,
                &img_m.file_name,
                GroundTruthLabel::SameProject,
                LabelSource::ManualCorrection,
            );
        }
    }
}

/// Captures `DifferentProject` when two groups are kept separate / dismissed.
pub fn capture_keep_separate_groups(
    store: &mut EvaluationLabelStore,
    group_a: &ImageGroup,
    group_b: &ImageGroup,
    images: &[ScannedImage],
) {
    let rep_a = group_a
        .representative_idx
        .or_else(|| group_a.member_indices.first().copied());
    let rep_b = group_b
        .representative_idx
        .or_else(|| group_b.member_indices.first().copied());
    if let (Some(ra), Some(rb)) = (rep_a, rep_b) {
        if let (Some(img_a), Some(img_b)) = (images.get(ra), images.get(rb)) {
            store.upsert(
                &img_a.stable_id(),
                &img_b.stable_id(),
                &img_a.file_name,
                &img_b.file_name,
                GroundTruthLabel::DifferentProject,
                LabelSource::ManualCorrection,
            );
        }
    }
}

/// Captures `DifferentProject` against remaining source group and `SameProject` among newly split members.
pub fn capture_split_group(
    store: &mut EvaluationLabelStore,
    source_group: &ImageGroup,
    split_indices: &[usize],
    images: &[ScannedImage],
) {
    if matches!(
        source_group.kind,
        GroupKind::FurnitureProject | GroupKind::Duplicates
    ) {
        capture_remove_from_group(store, source_group, split_indices, images);
    }
    store.record_coherent_members(
        split_indices
            .iter()
            .filter_map(|&m| images.get(m))
            .map(ScannedImage::stable_id)
            .collect(),
    );
    if split_indices.len() >= 2
        && let Some(img_anchor) = images.get(split_indices[0])
    {
        let anchor_id = img_anchor.stable_id();
        for &m in &split_indices[1..] {
            if let Some(img_m) = images.get(m) {
                store.upsert(
                    &anchor_id,
                    &img_m.stable_id(),
                    &img_anchor.file_name,
                    &img_m.file_name,
                    GroundTruthLabel::SameProject,
                    LabelSource::ManualCorrection,
                );
            }
        }
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ModeEvaluationMetrics {
    pub mode_name: String,
    pub evaluated_pairs: usize,
    pub same_project_pairs: usize,
    pub different_project_pairs: usize,
    pub true_matches: usize,
    pub true_negatives: usize,
    pub false_merges: usize,
    pub false_splits: usize,
    /// SameProject precision: TP / (TP + FP), None if no positive predictions.
    pub precision: Option<f32>,
    /// SameProject recall: TP / (TP + FN), None if no SameProject labels.
    pub recall: Option<f32>,
    /// DifferentProject precision: TN / (TN + FN), None if no negative predictions.
    pub different_project_precision: Option<f32>,
    /// FP / DifferentProjectLabels, None if no DifferentProject labels.
    pub false_merge_rate: Option<f32>,
    /// FN / SameProjectLabels, None if no SameProject labels.
    pub false_split_rate: Option<f32>,
    /// Only computed when BOTH SameProject and DifferentProject labels are present.
    pub accuracy: Option<f32>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ModeComparisonReport {
    pub deterministic: ModeEvaluationMetrics,
    pub mobilenet: Option<ModeEvaluationMetrics>,
    pub hybrid: Option<ModeEvaluationMetrics>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct GroupLevelMetrics {
    pub false_group_merges: usize,
    pub false_group_splits: usize,
    pub false_group_merge_pairs: usize,
    pub ground_truth_projects_checked: usize,
    #[serde(default)]
    pub false_group_merge_rate: Option<f32>,
    #[serde(default)]
    pub false_group_split_rate: Option<f32>,
    #[serde(default)]
    pub fragmentation_index: Option<f32>,
    #[serde(default)]
    pub fragmented_projects: usize,
    #[serde(default)]
    pub correctly_unified_projects: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AmbiguousPairRecord {
    pub image_a: String,
    pub image_b: String,
    pub display_a: String,
    pub display_b: String,
    pub deterministic_score: f32,
    pub embedding_score: Option<f32>,
    pub combined_score: f32,
    pub reason: String,
    pub human_label: Option<GroundTruthLabel>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct OnnxPolicySimulation {
    pub high_threshold: f32,
    pub low_threshold: f32,
    pub total_images: usize,
    pub images_needing_onnx: usize,
    pub onnx_inference_avoided_percent: f32,
    pub total_candidate_pairs: usize,
    pub pairs_resolved_without_onnx: usize,
    pub pair_onnx_avoided_percent: f32,
    pub evaluated_pairs: usize,
    pub false_merges: usize,
    pub false_splits: usize,
    pub false_merge_rate: Option<f32>,
    pub false_split_rate: Option<f32>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SimilarityHistogramBin {
    pub range: String,
    pub count: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ProcessingTimingsMs {
    pub scan_ms: u64,
    pub fingerprint_ms: u64,
    pub embedding_ms: u64,
    pub similarity_ms: u64,
    pub clustering_ms: u64,
    pub total_ms: u64,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ProjectIdentityMetrics {
    pub human_projects_labeled: usize,
    pub human_labeled_images: usize,
    pub correctly_recovered_human_projects: usize,
    pub fragmented_human_projects: usize,
    pub mixed_human_projects: usize,
    pub false_group_merges: usize,
    pub false_group_splits: usize,
    pub detected_outliers: usize,
    pub possible_splits: usize,
    pub very_strong_merge_suggestions: usize,
    pub strong_merge_suggestions: usize,
    pub possible_merge_suggestions: usize,
    pub average_project_purity: f32,
    pub average_project_completeness: f32,
    pub previously_good_groups_preserved: usize,
    pub previously_mixed_groups_improved: usize,
    pub known_should_merge_projects_recovered: usize,
}

/// Privacy-safe evaluation report containing zero image pixels or personal absolute paths.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PrivacySafeEvaluationReport {
    pub image_count: usize,
    pub confirmed_groups: usize,
    pub same_project_labels: usize,
    pub different_project_labels: usize,
    pub auto_grouped_count: usize,
    pub review_count: usize,
    pub ungrouped_count: usize,
    pub num_project_groups: usize,
    pub num_duplicate_groups: usize,
    #[serde(default)]
    pub num_probable_groups: usize,
    #[serde(default)]
    pub images_in_probable_groups: usize,
    #[serde(default)]
    pub possible_merge_suggestions: usize,
    pub blocked_chaining_edges: usize,
    pub cache_hit_rate: f32,
    pub onnx_enabled: bool,
    pub onnx_images_encoded: usize,
    pub approx_feature_memory_kb: usize,
    pub group_metrics: GroupLevelMetrics,
    pub onnx_policy_simulation: OnnxPolicySimulation,
    pub ambiguous_pairs: Vec<AmbiguousPairRecord>,
    pub similarity_distribution: Vec<SimilarityHistogramBin>,
    pub timings_ms: ProcessingTimingsMs,
    pub labeled_comparison: Option<ModeComparisonReport>,
    #[serde(default)]
    pub identity_metrics: Option<ProjectIdentityMetrics>,
}

fn build_index_lookup(images: &[ScannedImage]) -> HashMap<String, usize> {
    let mut map = HashMap::with_capacity(images.len() * 2);
    for (i, img) in images.iter().enumerate() {
        map.insert(img.stable_id(), i);
        map.entry(img.relative_path.clone()).or_insert(i);
    }
    map
}

pub fn evaluate_labeled_pairs(
    labels: &EvaluationLabelStore,
    images: &[ScannedImage],
    features: &[ImageFeatures],
    config: &SimilarityConfig,
    onnx_threshold: f32,
) -> ModeComparisonReport {
    let id_to_idx = build_index_lookup(images);

    let mut det_m = ModeEvaluationMetrics {
        mode_name: "Deterministic".to_string(),
        ..Default::default()
    };
    let mut mobilenet_m = ModeEvaluationMetrics {
        mode_name: "MobileNet".to_string(),
        ..Default::default()
    };
    let mut hyb_m = ModeEvaluationMetrics {
        mode_name: "Hybrid".to_string(),
        ..Default::default()
    };

    let mut det_cfg = config.clone();
    det_cfg.enable_onnx = false;

    let mut hyb_cfg = config.clone();
    hyb_cfg.enable_onnx = true;

    let mut any_onnx = false;

    for pair in &labels.pairs {
        let (Some(&ia), Some(&ib)) = (id_to_idx.get(&pair.image_a), id_to_idx.get(&pair.image_b))
        else {
            continue;
        };
        if ia == ib {
            continue;
        }

        let fa = &features[ia];
        let fb = &features[ib];

        // 1. Deterministic
        let det_score = compare_pair_direct(fa, fb, &det_cfg);
        let det_pred_same = det_score.combined >= config.same_project_threshold
            || det_score.relation.is_duplicate();
        record_outcome(&mut det_m, pair.label, det_pred_same);

        // 2. MobileNet / ONNX only
        if let Some((emb_sim, _)) = compare_embedding_only(fa, fb, onnx_threshold) {
            any_onnx = true;
            record_outcome(&mut mobilenet_m, pair.label, emb_sim >= onnx_threshold);
        }

        // 3. Hybrid
        if fa.embedding.is_some() && fb.embedding.is_some() {
            let hyb_score = compare_pair_direct(fa, fb, &hyb_cfg);
            let hyb_pred_same = hyb_score.combined >= config.same_project_threshold
                || hyb_score.relation.is_duplicate();
            record_outcome(&mut hyb_m, pair.label, hyb_pred_same);
        }
    }

    finalize_metrics(&mut det_m);
    finalize_metrics(&mut mobilenet_m);
    finalize_metrics(&mut hyb_m);

    ModeComparisonReport {
        deterministic: det_m,
        mobilenet: if any_onnx { Some(mobilenet_m) } else { None },
        hybrid: if any_onnx { Some(hyb_m) } else { None },
    }
}

/// Computes group-level FalseGroupMerges and FalseGroupSplits of the proposed groups
/// against ground-truth `SameProject` and `DifferentProject` labels.
pub fn compute_group_level_metrics(
    proposed_groups: &[ImageGroup],
    images: &[ScannedImage],
    labels: &EvaluationLabelStore,
) -> GroupLevelMetrics {
    let id_to_idx = build_index_lookup(images);

    // Map each image index to its proposed group bucket index
    let mut bucket_of = vec![usize::MAX; images.len()];
    for (gi, g) in proposed_groups.iter().enumerate() {
        for &m in &g.member_indices {
            if m < bucket_of.len() {
                bucket_of[m] = gi;
            }
        }
    }

    let mut diff_pairs_in_same_group: HashSet<usize> = HashSet::new();
    let mut false_group_merge_pairs = 0usize;

    // Union-Find over SameProject pairs to form ground-truth project components
    let mut parent: Vec<usize> = (0..images.len()).collect();
    let mut has_same_label = vec![false; images.len()];

    fn find(parent: &mut [usize], mut x: usize) -> usize {
        while parent[x] != x {
            parent[x] = parent[parent[x]];
            x = parent[x];
        }
        x
    }

    for p in &labels.pairs {
        let (Some(&ia), Some(&ib)) = (id_to_idx.get(&p.image_a), id_to_idx.get(&p.image_b)) else {
            continue;
        };
        if ia == ib {
            continue;
        }
        match p.label {
            GroundTruthLabel::DifferentProject => {
                let ba = bucket_of[ia];
                let bb = bucket_of[ib];
                if ba != usize::MAX
                    && ba == bb
                    && matches!(
                        proposed_groups[ba].kind,
                        GroupKind::FurnitureProject | GroupKind::Duplicates
                    )
                {
                    false_group_merge_pairs += 1;
                    diff_pairs_in_same_group.insert(ba);
                }
            }
            GroundTruthLabel::SameProject => {
                has_same_label[ia] = true;
                has_same_label[ib] = true;
                let ra = find(&mut parent, ia);
                let rb = find(&mut parent, ib);
                if ra != rb {
                    parent[rb] = ra;
                }
            }
        }
    }

    let mut gt_components: HashMap<usize, Vec<usize>> = HashMap::new();
    for i in 0..images.len() {
        if has_same_label[i] {
            let root = find(&mut parent, i);
            gt_components.entry(root).or_default().push(i);
        }
    }

    let mut false_group_splits = 0usize;
    let mut fragmented_projects = 0usize;
    let mut correctly_unified_projects = 0usize;
    let mut sum_fragmentation_ratios = 0.0f32;
    let mut multi_member_projects_count = 0usize;

    for members in gt_components.values() {
        if members.len() < 2 {
            continue;
        }
        multi_member_projects_count += 1;
        let mut distinct_buckets: HashSet<(usize, usize)> = HashSet::new();
        for &m in members {
            let b = bucket_of[m];
            if b != usize::MAX
                && matches!(
                    proposed_groups[b].kind,
                    GroupKind::FurnitureProject | GroupKind::Duplicates | GroupKind::CandidateGroup
                )
            {
                distinct_buckets.insert((0, b));
            } else {
                // In Review/Ungrouped/unassigned: each isolated item is a split from the project
                distinct_buckets.insert((1, m));
            }
        }
        let k = distinct_buckets.len();
        if k > 1 {
            false_group_splits += 1;
            fragmented_projects += 1;
            let ratio = (k - 1) as f32 / (members.len() - 1) as f32;
            sum_fragmentation_ratios += ratio.clamp(0.0, 1.0);
        } else {
            correctly_unified_projects += 1;
        }
    }

    let fragmentation_index = if multi_member_projects_count > 0 {
        Some(sum_fragmentation_ratios / multi_member_projects_count as f32)
    } else {
        None
    };

    let total_gt = gt_components.len();
    let false_group_merge_rate =
        (total_gt > 0).then(|| diff_pairs_in_same_group.len() as f32 / total_gt as f32);
    let false_group_split_rate =
        (total_gt > 0).then(|| false_group_splits as f32 / total_gt as f32);

    GroupLevelMetrics {
        false_group_merges: diff_pairs_in_same_group.len(),
        false_group_splits,
        false_group_merge_pairs,
        ground_truth_projects_checked: total_gt,
        false_group_merge_rate,
        false_group_split_rate,
        fragmentation_index,
        fragmented_projects,
        correctly_unified_projects,
    }
}

pub fn collect_ambiguous_pairs(
    pairs: &[PairComparison],
    images: &[ScannedImage],
    labels: &EvaluationLabelStore,
    config: &SimilarityConfig,
    onnx_threshold: f32,
) -> Vec<AmbiguousPairRecord> {
    let mut records = Vec::new();
    for p in pairs {
        let det = p.score.deterministic.combined;
        let emb = p.score.embedding;
        let comb = p.score.combined;

        let det_same = det >= config.same_project_threshold || p.score.relation.is_duplicate();
        let emb_same = emb.map(|e| e >= onnx_threshold);
        let near_threshold = (det - config.same_project_threshold).abs() <= 0.08
            || (comb - config.same_project_threshold).abs() <= 0.08
            || (det >= config.review_threshold && det < config.same_project_threshold);

        let reason = match (det_same, emb_same) {
            (true, Some(false)) => Some("DetSame_OnnxDiff"),
            (false, Some(true)) => Some("DetDiff_OnnxSame"),
            _ if near_threshold => Some("NearThreshold"),
            _ => None,
        };

        if let Some(reason_str) = reason
            && let (Some(ia), Some(ib)) = (images.get(p.i), images.get(p.j))
        {
            records.push(AmbiguousPairRecord {
                image_a: ia.stable_id(),
                image_b: ib.stable_id(),
                display_a: ia.file_name.clone(),
                display_b: ib.file_name.clone(),
                deterministic_score: det,
                embedding_score: emb,
                combined_score: comb,
                reason: reason_str.to_string(),
                human_label: labels.lookup_label(ia, ib),
            });
        }
    }

    records.sort_by(|a, b| {
        b.combined_score
            .total_cmp(&a.combined_score)
            .then_with(|| a.display_a.cmp(&b.display_a))
    });
    records.truncate(200);
    records
}

/// Evaluates the gated runtime policy:
/// `det >= high_threshold -> SameProject without ONNX`
/// `det <= low_threshold  -> Different without ONNX`
/// `low < det < high      -> consult ONNX`
pub fn simulate_gated_onnx_policy(
    images: &[ScannedImage],
    features: &[ImageFeatures],
    pairs: &[PairComparison],
    labels: &EvaluationLabelStore,
    config: &SimilarityConfig,
) -> OnnxPolicySimulation {
    let total_images = images.len();
    let needed = ambiguous_image_indices(pairs, config);
    let images_needing_onnx = needed.len();
    let onnx_inference_avoided_percent = if total_images > 0 {
        100.0 * (1.0 - (images_needing_onnx as f32 / total_images as f32))
    } else {
        0.0
    };

    let total_candidate_pairs = pairs.len();
    let pairs_resolved_without_onnx = pairs
        .iter()
        .filter(|p| {
            let d = p.score.deterministic.combined;
            d >= config.ambiguous_high || d <= config.ambiguous_low
        })
        .count();
    let pair_onnx_avoided_percent = if total_candidate_pairs > 0 {
        100.0 * (pairs_resolved_without_onnx as f32 / total_candidate_pairs as f32)
    } else {
        0.0
    };

    let id_to_idx = build_index_lookup(images);
    let mut hyb_cfg = config.clone();
    hyb_cfg.enable_onnx = true;

    let mut evaluated_pairs = 0usize;
    let mut false_merges = 0usize;
    let mut false_splits = 0usize;
    let mut same_count = 0usize;
    let mut diff_count = 0usize;

    for pair in &labels.pairs {
        let (Some(&ia), Some(&ib)) = (id_to_idx.get(&pair.image_a), id_to_idx.get(&pair.image_b))
        else {
            continue;
        };
        if ia == ib {
            continue;
        }
        evaluated_pairs += 1;

        let score = compare_pair_direct(&features[ia], &features[ib], &hyb_cfg);
        let det = score.deterministic.combined;
        let predicted_same = if det >= config.ambiguous_high {
            true
        } else if det <= config.ambiguous_low {
            false
        } else {
            score.combined >= config.same_project_threshold || score.relation.is_duplicate()
        };

        match pair.label {
            GroundTruthLabel::SameProject => {
                same_count += 1;
                if !predicted_same {
                    false_splits += 1;
                }
            }
            GroundTruthLabel::DifferentProject => {
                diff_count += 1;
                if predicted_same {
                    false_merges += 1;
                }
            }
        }
    }

    OnnxPolicySimulation {
        high_threshold: config.ambiguous_high,
        low_threshold: config.ambiguous_low,
        total_images,
        images_needing_onnx,
        onnx_inference_avoided_percent,
        total_candidate_pairs,
        pairs_resolved_without_onnx,
        pair_onnx_avoided_percent,
        evaluated_pairs,
        false_merges,
        false_splits,
        false_merge_rate: (diff_count > 0).then(|| false_merges as f32 / diff_count as f32),
        false_split_rate: (same_count > 0).then(|| false_splits as f32 / same_count as f32),
    }
}

fn record_outcome(m: &mut ModeEvaluationMetrics, label: GroundTruthLabel, predicted_same: bool) {
    m.evaluated_pairs += 1;
    match label {
        GroundTruthLabel::SameProject => {
            m.same_project_pairs += 1;
            if predicted_same {
                m.true_matches += 1;
            } else {
                m.false_splits += 1;
            }
        }
        GroundTruthLabel::DifferentProject => {
            m.different_project_pairs += 1;
            if predicted_same {
                m.false_merges += 1;
            } else {
                m.true_negatives += 1;
            }
        }
    }
}

fn finalize_metrics(m: &mut ModeEvaluationMetrics) {
    let pred_pos = m.true_matches + m.false_merges;
    let pred_neg = m.true_negatives + m.false_splits;
    m.precision = (pred_pos > 0).then(|| m.true_matches as f32 / pred_pos as f32);
    m.recall =
        (m.same_project_pairs > 0).then(|| m.true_matches as f32 / m.same_project_pairs as f32);
    m.different_project_precision =
        (pred_neg > 0).then(|| m.true_negatives as f32 / pred_neg as f32);
    m.false_merge_rate = (m.different_project_pairs > 0)
        .then(|| m.false_merges as f32 / m.different_project_pairs as f32);
    m.false_split_rate =
        (m.same_project_pairs > 0).then(|| m.false_splits as f32 / m.same_project_pairs as f32);
    m.accuracy = (m.same_project_pairs > 0 && m.different_project_pairs > 0)
        .then(|| (m.true_matches + m.true_negatives) as f32 / m.evaluated_pairs as f32);
}

pub fn calculate_project_identity_metrics(
    proposed_groups: &[ImageGroup],
    images: &[ScannedImage],
    store: &EvaluationLabelStore,
    initial_clustering: &ClusteringOutput,
) -> ProjectIdentityMetrics {
    use crate::clustering::MergeRank;
    // Evaluate SHA identities, not filenames, display group numbers, or duplicate file counts.
    let present: HashSet<String> = images.iter().map(ScannedImage::stable_id).collect();
    let group_hashes: Vec<HashSet<String>> = proposed_groups
        .iter()
        .map(|g| {
            if g.kind == GroupKind::Ungrouped {
                HashSet::new()
            } else {
                g.member_indices
                    .iter()
                    .filter_map(|&i| images.get(i))
                    .map(ScannedImage::stable_id)
                    .collect()
            }
        })
        .collect();
    let mut metric = ProjectIdentityMetrics::default();
    let mut labeled = HashSet::new();
    let mut purities = Vec::new();
    let mut completeness = Vec::new();
    for project in &store.projects {
        let members: HashSet<_> = project
            .member_hashes
            .intersection(&present)
            .cloned()
            .collect();
        if members.is_empty() {
            continue;
        }
        metric.human_projects_labeled += 1;
        labeled.extend(members.iter().cloned());
        let touched: Vec<_> = group_hashes
            .iter()
            .enumerate()
            .filter(|(_, g)| !g.is_disjoint(&members))
            .collect();
        let counts: Vec<_> = touched
            .iter()
            .map(|(_, g)| g.intersection(&members).count())
            .collect();
        let assigned: HashSet<_> = touched
            .iter()
            .flat_map(|(_, g)| g.intersection(&members).cloned())
            .collect();
        let isolated = members.len() - assigned.len();
        let largest = counts
            .iter()
            .copied()
            .max()
            .unwrap_or(0)
            .max(usize::from(isolated > 0));
        completeness.push(largest as f32 / members.len() as f32);
        // Per-project mean of each labeled SHA's fragment purity. Unlabeled identities
        // remain in the denominator; this is conservative and not proof they differ.
        let purity = members
            .iter()
            .map(|h| {
                let containing: Vec<_> = touched.iter().filter(|(_, g)| g.contains(h)).collect();
                if containing.is_empty() {
                    1.0
                } else {
                    containing
                        .iter()
                        .map(|(_, g)| g.intersection(&members).count() as f32 / g.len() as f32)
                        .sum::<f32>()
                        / containing.len() as f32
                }
            })
            .sum::<f32>()
            / members.len() as f32;
        purities.push(purity);
        let mixed = store
            .projects
            .iter()
            .filter(|p| p.id != project.id)
            .any(|other| {
                touched
                    .iter()
                    .any(|(_, g)| !g.is_disjoint(&other.member_hashes))
            });
        metric.mixed_human_projects += usize::from(mixed);
        let fragmented = touched.len() + isolated > 1;
        metric.fragmented_human_projects += usize::from(fragmented);
        metric.correctly_recovered_human_projects += usize::from(!mixed && !fragmented);
    }
    metric.human_labeled_images = labeled.len();
    metric.average_project_purity = if purities.is_empty() {
        0.0
    } else {
        purities.iter().sum::<f32>() / purities.len() as f32
    };
    metric.average_project_completeness = if completeness.is_empty() {
        0.0
    } else {
        completeness.iter().sum::<f32>() / completeness.len() as f32
    };
    metric.detected_outliers = initial_clustering.detected_outliers;
    metric.possible_splits = initial_clustering
        .possible_splits
        .iter()
        .filter(|s| s.outlier_idx.is_none())
        .count();
    for m in &initial_clustering.possible_merges {
        match m.rank {
            MergeRank::VeryStrong => metric.very_strong_merge_suggestions += 1,
            MergeRank::Strong => metric.strong_merge_suggestions += 1,
            MergeRank::Possible => metric.possible_merge_suggestions += 1,
        }
    }
    let groups = compute_group_level_metrics(proposed_groups, images, store);
    metric.false_group_merges = groups.false_group_merges;
    metric.false_group_splits = groups.false_group_splits;
    // Case-specific fields remain zero unless an explicit SHA review manifest is evaluated.
    metric
}

#[allow(clippy::too_many_arguments)]
pub fn build_evaluation_report(
    images: &[ScannedImage],
    features: &[ImageFeatures],
    initial_clustering: &ClusteringOutput,
    current_groups: &[ImageGroup],
    pairs: &[PairComparison],
    labels: &EvaluationLabelStore,
    config: &SimilarityConfig,
    cache_hit_rate: f32,
    onnx_enabled: bool,
    onnx_images_encoded: usize,
    approx_feature_memory_kb: usize,
    timings_ms: ProcessingTimingsMs,
    onnx_threshold: f32,
) -> PrivacySafeEvaluationReport {
    let mut num_project_groups = 0usize;
    let mut num_duplicate_groups = 0usize;
    let mut num_probable_groups = 0usize;
    let mut auto_grouped_count = 0usize;
    let mut review_count = 0usize;
    let mut ungrouped_count = 0usize;

    for g in &initial_clustering.groups {
        match g.kind {
            GroupKind::FurnitureProject => {
                num_project_groups += 1;
                auto_grouped_count += g.member_indices.len();
            }
            GroupKind::Duplicates => {
                num_duplicate_groups += 1;
                auto_grouped_count += g.member_indices.len();
            }
            GroupKind::CandidateGroup | GroupKind::Review => {
                num_probable_groups += 1;
                review_count += g.member_indices.len();
            }
            GroupKind::Ungrouped => ungrouped_count += g.member_indices.len(),
        }
    }

    let confirmed_groups = current_groups
        .iter()
        .filter(|g| {
            g.confirmed && matches!(g.kind, GroupKind::FurnitureProject | GroupKind::Duplicates)
        })
        .count();

    let (same_project_labels, different_project_labels) = labels.count_by_label();
    let labeled_cmp = evaluate_labeled_pairs(labels, images, features, config, onnx_threshold);
    let labeled_comparison = (labeled_cmp.deterministic.evaluated_pairs > 0).then_some(labeled_cmp);

    let group_metrics = compute_group_level_metrics(&initial_clustering.groups, images, labels);
    let onnx_policy_simulation =
        simulate_gated_onnx_policy(images, features, pairs, labels, config);
    let ambiguous_pairs = collect_ambiguous_pairs(pairs, images, labels, config, onnx_threshold);

    let mut bins = [0usize; 10];
    for p in pairs {
        let b = ((p.score.combined * 10.0).floor() as usize).min(9);
        bins[b] += 1;
    }
    let similarity_distribution = bins
        .iter()
        .enumerate()
        .map(|(i, &count)| SimilarityHistogramBin {
            range: format!("{:.1}-{:.1}", i as f32 * 0.1, (i + 1) as f32 * 0.1),
            count,
        })
        .collect();

    let identity_metrics = if !labels.projects.is_empty() {
        Some(calculate_project_identity_metrics(
            current_groups,
            images,
            labels,
            initial_clustering,
        ))
    } else {
        None
    };

    PrivacySafeEvaluationReport {
        image_count: images.len(),
        confirmed_groups,
        same_project_labels,
        different_project_labels,
        auto_grouped_count,
        review_count,
        ungrouped_count,
        num_project_groups,
        num_duplicate_groups,
        num_probable_groups,
        images_in_probable_groups: review_count,
        possible_merge_suggestions: initial_clustering.possible_merges.len(),
        blocked_chaining_edges: initial_clustering.blocked_chaining_edges,
        cache_hit_rate,
        onnx_enabled,
        onnx_images_encoded,
        approx_feature_memory_kb,
        group_metrics,
        onnx_policy_simulation,
        ambiguous_pairs,
        similarity_distribution,
        timings_ms,
        labeled_comparison,
        identity_metrics,
    }
}

pub fn format_evaluation_markdown(r: &PrivacySafeEvaluationReport) -> String {
    let fmt_opt_pct = |v: Option<f32>| match v {
        Some(x) => format!("{:.1}%", x * 100.0),
        None => "N/A".to_string(),
    };

    let mut md = String::new();
    let _ = writeln!(md, "# Furniture Sorter — Evaluation Report\n");

    if let Some(id_m) = &r.identity_metrics {
        let _ = writeln!(
            md,
            "## Project Identity & Contradiction Engine v4 Metrics\n"
        );
        let _ = writeln!(
            md,
            "- **HumanProjectsLabeled:** {}",
            id_m.human_projects_labeled
        );
        let _ = writeln!(
            md,
            "- **HumanLabeledImages:** {}",
            id_m.human_labeled_images
        );
        let _ = writeln!(
            md,
            "- **CorrectlyRecoveredHumanProjects:** {}",
            id_m.correctly_recovered_human_projects
        );
        let _ = writeln!(
            md,
            "- **FragmentedHumanProjects:** {}",
            id_m.fragmented_human_projects
        );
        let _ = writeln!(
            md,
            "- **MixedHumanProjects:** {}",
            id_m.mixed_human_projects
        );
        let _ = writeln!(
            md,
            "- **AverageProjectPurity:** {:.1}%",
            id_m.average_project_purity * 100.0
        );
        let _ = writeln!(
            md,
            "- **AverageProjectCompleteness:** {:.1}%",
            id_m.average_project_completeness * 100.0
        );
        let _ = writeln!(md, "- **DetectedOutliers:** {}", id_m.detected_outliers);
        let _ = writeln!(md, "- **PossibleSplits:** {}", id_m.possible_splits);
        let _ = writeln!(
            md,
            "- **VeryStrongMergeSuggestions:** {}",
            id_m.very_strong_merge_suggestions
        );
        let _ = writeln!(
            md,
            "- **StrongMergeSuggestions:** {}",
            id_m.strong_merge_suggestions
        );
        let _ = writeln!(
            md,
            "- **PossibleMergeSuggestions:** {}",
            id_m.possible_merge_suggestions
        );
        let _ = writeln!(
            md,
            "- **PreviouslyGoodGroupsPreserved:** {}",
            id_m.previously_good_groups_preserved
        );
        let _ = writeln!(
            md,
            "- **PreviouslyMixedGroupsImproved:** {}",
            id_m.previously_mixed_groups_improved
        );
        let _ = writeln!(
            md,
            "- **KnownShouldMergeProjectsRecovered:** {}\n",
            id_m.known_should_merge_projects_recovered
        );
    }

    let _ = writeln!(md, "## Summary\n");
    let _ = writeln!(md, "- **ImageCount:** {}", r.image_count);
    let _ = writeln!(md, "- **ConfirmedGroups:** {}", r.confirmed_groups);
    let _ = writeln!(md, "- **SameProjectLabels:** {}", r.same_project_labels);
    let _ = writeln!(
        md,
        "- **DifferentProjectLabels:** {}",
        r.different_project_labels
    );
    let _ = writeln!(md, "- **AutoGroupedCount:** {}", r.auto_grouped_count);
    let _ = writeln!(md, "- **HighConfidenceGroups:** {}", r.num_project_groups);
    let _ = writeln!(
        md,
        "- **ProbableCandidateGroups:** {}",
        r.num_probable_groups
    );
    let _ = writeln!(
        md,
        "- **ImagesInProbableGroups:** {}",
        r.images_in_probable_groups
    );
    let _ = writeln!(md, "- **DuplicateGroups:** {}", r.num_duplicate_groups);
    let _ = writeln!(md, "- **UngroupedCount:** {}", r.ungrouped_count);
    let _ = writeln!(
        md,
        "- **PossibleMergeSuggestions:** {}",
        r.possible_merge_suggestions
    );
    let _ = writeln!(
        md,
        "- **FalseGroupMerges:** {}",
        r.group_metrics.false_group_merges
    );
    let _ = writeln!(
        md,
        "- **FalseGroupSplits:** {}",
        r.group_metrics.false_group_splits
    );
    let _ = writeln!(
        md,
        "- **FalseGroupMergeRate:** {} | **FalseGroupSplitRate:** {}",
        fmt_opt_pct(r.group_metrics.false_group_merge_rate),
        fmt_opt_pct(r.group_metrics.false_group_split_rate)
    );
    let _ = writeln!(
        md,
        "- **CacheHitRate:** {:.1}% | **FeatureRAM:** {} KB",
        r.cache_hit_rate * 100.0,
        r.approx_feature_memory_kb
    );
    let _ = writeln!(
        md,
        "- **Timings:** Fingerprint {} ms | Embedding {} ms ({} imgs) | Similarity {} ms | Clustering {} ms | Total {} ms\n",
        r.timings_ms.fingerprint_ms,
        r.timings_ms.embedding_ms,
        r.onnx_images_encoded,
        r.timings_ms.similarity_ms,
        r.timings_ms.clustering_ms,
        r.timings_ms.total_ms
    );

    if let Some(cmp) = &r.labeled_comparison {
        let _ = writeln!(md, "## Pairwise Mode Comparison\n");
        let _ = writeln!(
            md,
            "| Mode | Pairs | Precision | Recall | DiffPrecision | FalseMergeRate | FalseSplitRate |"
        );
        let _ = writeln!(md, "|---|---|---|---|---|---|---|");
        for m in [
            Some(&cmp.deterministic),
            cmp.mobilenet.as_ref(),
            cmp.hybrid.as_ref(),
        ]
        .into_iter()
        .flatten()
        {
            let _ = writeln!(
                md,
                "| {} | {} | {} | {} | {} | {} | {} |",
                m.mode_name,
                m.evaluated_pairs,
                fmt_opt_pct(m.precision),
                fmt_opt_pct(m.recall),
                fmt_opt_pct(m.different_project_precision),
                fmt_opt_pct(m.false_merge_rate),
                fmt_opt_pct(m.false_split_rate),
            );
        }
        let _ = writeln!(md);
    }

    let p = &r.onnx_policy_simulation;
    let _ = writeln!(md, "## Ambiguous-Only ONNX Policy Simulation\n");
    let _ = writeln!(
        md,
        "- Region: `det <= {:.2}` (Different), `det >= {:.2}` (Same), middle -> consult ONNX",
        p.low_threshold, p.high_threshold
    );
    let _ = writeln!(
        md,
        "- **OnnxInferenceAvoidedPercent:** {:.1}% ({} / {} images skipped)",
        p.onnx_inference_avoided_percent,
        p.total_images.saturating_sub(p.images_needing_onnx),
        p.total_images
    );
    let _ = writeln!(
        md,
        "- **FalseMergeRate:** {} | **FalseSplitRate:** {}\n",
        fmt_opt_pct(p.false_merge_rate),
        fmt_opt_pct(p.false_split_rate)
    );

    let _ = writeln!(md, "## AmbiguousPairs (top {})\n", r.ambiguous_pairs.len());
    if r.ambiguous_pairs.is_empty() {
        let _ = writeln!(md, "None recorded.\n");
    } else {
        let _ = writeln!(
            md,
            "| Image A | Image B | Det | Emb | Combined | Reason | HumanLabel |"
        );
        let _ = writeln!(md, "|---|---|---|---|---|---|---|");
        for a in r.ambiguous_pairs.iter().take(40) {
            let emb_s = a
                .embedding_score
                .map(|e| format!("{e:.3}"))
                .unwrap_or_else(|| "N/A".to_string());
            let lbl_s = a
                .human_label
                .map(|l| format!("{l:?}"))
                .unwrap_or_else(|| "-".to_string());
            let _ = writeln!(
                md,
                "| {} | {} | {:.3} | {} | {:.3} | {} | {} |",
                a.display_a,
                a.display_b,
                a.deterministic_score,
                emb_s,
                a.combined_score,
                a.reason,
                lbl_s
            );
        }
    }
    md
}
