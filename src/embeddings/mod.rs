pub mod onnx;

use crate::deterministic::{DeterministicComponents, DeterministicWeights, compare_deterministic};
use crate::fingerprint::ImageFingerprint;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Clone, Debug)]
pub struct ImageFeatures {
    pub fingerprint: ImageFingerprint,
    pub embedding: Option<Vec<f32>>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum PairRelation {
    ExactDuplicate,
    NearDuplicate,
    SameProject,
    SameFamily,
    RelatedUncertain,
    Different,
}

impl PairRelation {
    pub fn ui_label(self) -> &'static str {
        match self {
            Self::ExactDuplicate => "Duplicate (Exact)",
            Self::NearDuplicate => "Duplicate (Near)",
            Self::SameProject => "Same project",
            Self::SameFamily => "Same family",
            Self::RelatedUncertain => "Possible match",
            Self::Different => "Different",
        }
    }

    pub fn is_duplicate(self) -> bool {
        matches!(self, Self::ExactDuplicate | Self::NearDuplicate)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SimilarityScore {
    pub deterministic: DeterministicComponents,
    pub embedding: Option<f32>,
    pub combined: f32,
    pub used_onnx: bool,
    pub relation: PairRelation,
    #[serde(default)]
    pub family_similarity: f32,
    #[serde(default)]
    pub project_identity: f32,
    #[serde(default)]
    pub contradiction: f32,
    #[serde(default)]
    pub hard_contradiction: bool,
}

/// Unified similarity engine interface shared across deterministic-only and
/// hybrid deterministic+ONNX scoring paths.
pub trait SimilarityEngine: Send + Sync {
    fn compare(&self, a: &ImageFeatures, b: &ImageFeatures) -> SimilarityScore;
}

#[derive(Clone, Debug)]
pub enum OnnxModelStatus {
    Available {
        model_path: PathBuf,
        runtime_path: Option<PathBuf>,
        model_size_bytes: u64,
        model_tag: String,
    },
    Unavailable {
        reason: String,
    },
}

impl OnnxModelStatus {
    pub fn is_available(&self) -> bool {
        matches!(self, Self::Available { .. })
    }

    pub fn ui_summary(&self) -> String {
        match self {
            Self::Available {
                model_path,
                model_size_bytes,
                ..
            } => {
                let mb = *model_size_bytes as f64 / (1024.0 * 1024.0);
                let name = model_path
                    .file_name()
                    .map(|n| n.to_string_lossy().into_owned())
                    .unwrap_or_else(|| "visual_encoder.onnx".to_string());
                format!("Available — {name} ({mb:.1} MB)")
            }
            Self::Unavailable { reason } => reason.clone(),
        }
    }
}

/// Cosine similarity between two L2-normalized embedding vectors, clamped to [0, 1].
pub fn embedding_cosine_similarity(a: &[f32], b: &[f32]) -> Option<f32> {
    if a.is_empty() || a.len() != b.len() {
        return None;
    }
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    Some(dot.clamp(0.0, 1.0))
}

pub fn l2_normalize(vec: &mut [f32]) {
    let norm = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 1e-8 {
        for x in vec.iter_mut() {
            *x /= norm;
        }
    }
}

pub struct DeterministicOnlyEngine {
    pub weights: DeterministicWeights,
    pub same_project_threshold: f32,
    pub review_threshold: f32,
}

impl SimilarityEngine for DeterministicOnlyEngine {
    fn compare(&self, a: &ImageFeatures, b: &ImageFeatures) -> SimilarityScore {
        let det = compare_deterministic(&a.fingerprint, &b.fingerprint, &self.weights);
        let relation = classify_pair_relation(
            &det,
            det.combined,
            self.same_project_threshold,
            self.review_threshold,
        );
        SimilarityScore {
            combined: det.combined,
            deterministic: det.clone(),
            embedding: None,
            used_onnx: false,
            relation,
            family_similarity: det.family_similarity,
            project_identity: det.project_identity,
            contradiction: det.contradiction,
            hard_contradiction: det.hard_contradiction,
        }
    }
}

pub fn classify_pair_relation(
    det: &DeterministicComponents,
    final_score: f32,
    same_project_threshold: f32,
    review_threshold: f32,
) -> PairRelation {
    if det.phash_hamming <= 2 && det.dhash_hamming <= 3 && det.color >= 0.95 && det.layout >= 0.94 {
        return PairRelation::ExactDuplicate;
    }
    if det.phash_hamming <= 8 && det.combined >= 0.90 {
        return PairRelation::NearDuplicate;
    }
    if det.hard_contradiction {
        if det.family_similarity >= 0.72 {
            return PairRelation::SameFamily;
        }
        return PairRelation::Different;
    }
    if final_score >= same_project_threshold
        && det.project_identity >= (same_project_threshold - 0.10)
        && det.contradiction <= 0.28
    {
        return PairRelation::SameProject;
    }
    if det.family_similarity >= 0.74 {
        return PairRelation::SameFamily;
    }
    if final_score >= review_threshold {
        return PairRelation::RelatedUncertain;
    }
    PairRelation::Different
}
