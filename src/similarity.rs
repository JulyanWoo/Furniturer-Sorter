use crate::deterministic::{
    DeterministicWeights, PrefilterConfig, compare_deterministic, passes_candidate_prefilter,
};
use crate::embeddings::{
    ImageFeatures, PairRelation, SimilarityEngine, SimilarityScore, classify_pair_relation,
    embedding_cosine_similarity,
};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SimilarityConfig {
    pub enable_deterministic: bool,
    pub enable_onnx: bool,
    pub weights: DeterministicWeights,
    pub prefilter: PrefilterConfig,
    /// Deterministic >= ambiguous_high resolves without ONNX.
    pub ambiguous_high: f32,
    /// Deterministic <= ambiguous_low resolves as different without ONNX.
    pub ambiguous_low: f32,
    /// Weight of deterministic layer when ONNX is consulted.
    pub deterministic_blend: f32,
    /// Weight of embedding layer when ONNX is consulted.
    pub embedding_blend: f32,
    /// Minimum combined score to qualify as SameProject.
    pub same_project_threshold: f32,
    /// Minimum combined score to enter Review instead of Ungrouped.
    pub review_threshold: f32,
}

impl Default for SimilarityConfig {
    fn default() -> Self {
        Self {
            enable_deterministic: true,
            enable_onnx: false,
            weights: DeterministicWeights::default(),
            prefilter: PrefilterConfig::default(),
            ambiguous_high: 0.88,
            ambiguous_low: 0.35,
            deterministic_blend: 0.60,
            embedding_blend: 0.40,
            same_project_threshold: 0.72,
            review_threshold: 0.56,
        }
    }
}

impl SimilarityConfig {
    pub fn normalized_hybrid_weights(&self) -> (f32, f32) {
        let sum = self.deterministic_blend + self.embedding_blend;
        if sum <= 1e-6 {
            (0.60, 0.40)
        } else {
            (self.deterministic_blend / sum, self.embedding_blend / sum)
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PairComparison {
    pub i: usize,
    pub j: usize,
    pub score: SimilarityScore,
}

pub struct HybridSimilarityEngine {
    pub config: SimilarityConfig,
}

impl SimilarityEngine for HybridSimilarityEngine {
    fn compare(&self, a: &ImageFeatures, b: &ImageFeatures) -> SimilarityScore {
        let det = compare_deterministic(&a.fingerprint, &b.fingerprint, &self.config.weights);

        let should_consult_onnx = self.config.enable_onnx
            && det.combined > self.config.ambiguous_low
            && det.combined < self.config.ambiguous_high;

        let (emb_score, combined, used_onnx) = if should_consult_onnx {
            if let (Some(va), Some(vb)) = (a.embedding.as_deref(), b.embedding.as_deref()) {
                if let Some(sim) = embedding_cosine_similarity(va, vb) {
                    let (wd, we) = self.config.normalized_hybrid_weights();
                    let hybrid = (wd * det.combined + we * sim).clamp(0.0, 1.0);
                    (Some(sim), hybrid, true)
                } else {
                    (None, det.combined, false)
                }
            } else {
                (None, det.combined, false)
            }
        } else {
            (None, det.combined, false)
        };

        let effective_identity = match emb_score {
            Some(e) => (0.75 * det.project_identity + 0.25 * e).clamp(0.0, 1.0),
            None => det.project_identity,
        };

        let relation = classify_pair_relation(
            &det,
            combined,
            self.config.same_project_threshold,
            self.config.review_threshold,
        );

        SimilarityScore {
            deterministic: det.clone(),
            embedding: emb_score,
            combined,
            used_onnx,
            relation,
            family_similarity: det.family_similarity,
            project_identity: effective_identity,
            contradiction: det.contradiction,
            hard_contradiction: det.hard_contradiction,
        }
    }
}

/// Stage 1 of similarity computation: fast prefilter + parallel deterministic comparison
/// across candidate pairs without evaluating expensive comparisons on all N^2 pairs.
pub fn compute_deterministic_candidates(
    features: &[ImageFeatures],
    config: &SimilarityConfig,
    cancel: &Arc<AtomicBool>,
) -> Result<(Vec<PairComparison>, usize, usize), String> {
    let n = features.len();
    if n < 2 {
        return Ok((Vec::new(), 0, 0));
    }

    let keep_floor = (config.review_threshold * 0.75)
        .min(config.ambiguous_low)
        .max(0.30);

    let per_i: Result<Vec<(Vec<PairComparison>, usize, usize)>, String> = (0..n)
        .into_par_iter()
        .map(|i| {
            if cancel.load(Ordering::Relaxed) {
                return Err("Similarity calculation cancelled by user.".to_string());
            }
            let a = &features[i];
            let mut local_pairs = Vec::new();
            let mut total_checked = 0usize;
            let mut prefiltered_passed = 0usize;

            for j in (i + 1)..n {
                total_checked += 1;
                let b = &features[j];
                if !passes_candidate_prefilter(&a.fingerprint, &b.fingerprint, &config.prefilter) {
                    continue;
                }
                prefiltered_passed += 1;
                let det = compare_deterministic(&a.fingerprint, &b.fingerprint, &config.weights);
                if det.combined >= keep_floor || (det.phash_hamming <= 8 && det.dhash_hamming <= 10)
                {
                    let relation = classify_pair_relation(
                        &det,
                        det.combined,
                        config.same_project_threshold,
                        config.review_threshold,
                    );
                    local_pairs.push(PairComparison {
                        i,
                        j,
                        score: SimilarityScore {
                            combined: det.combined,
                            deterministic: det.clone(),
                            embedding: None,
                            used_onnx: false,
                            relation,
                            family_similarity: det.family_similarity,
                            project_identity: det.project_identity,
                            contradiction: det.contradiction,
                            hard_contradiction: det.hard_contradiction,
                        },
                    });
                }
            }
            Ok((local_pairs, total_checked, prefiltered_passed))
        })
        .collect();

    let per_i = per_i?;
    let mut pairs = Vec::new();
    let mut total_pairs = 0usize;
    let mut prefilter_passed = 0usize;
    for (local, t, p) in per_i {
        pairs.extend(local);
        total_pairs += t;
        prefilter_passed += p;
    }
    Ok((pairs, total_pairs, prefilter_passed))
}

/// Identifies the subset of images that appear in at least one ambiguous deterministic pair
/// `(ambiguous_low, ambiguous_high)` so ONNX inference runs strictly on ambiguous items.
pub fn ambiguous_image_indices(pairs: &[PairComparison], config: &SimilarityConfig) -> Vec<usize> {
    let mut set = HashSet::new();
    for p in pairs {
        let d = p.score.deterministic.combined;
        if d > config.ambiguous_low && d < config.ambiguous_high {
            set.insert(p.i);
            set.insert(p.j);
        }
    }
    let mut out: Vec<usize> = set.into_iter().collect();
    out.sort_unstable();
    out
}

/// Stage 2: Re-scores only ambiguous candidate pairs using already-cached/extracted
/// single-image ONNX embeddings.
pub fn apply_hybrid_embeddings_to_pairs(
    pairs: &mut [PairComparison],
    features: &[ImageFeatures],
    config: &SimilarityConfig,
) {
    let (wd, we) = config.normalized_hybrid_weights();
    for pair in pairs.iter_mut() {
        let det_score = pair.score.deterministic.combined;
        if det_score <= config.ambiguous_low || det_score >= config.ambiguous_high {
            continue;
        }
        let (Some(va), Some(vb)) = (
            features[pair.i].embedding.as_deref(),
            features[pair.j].embedding.as_deref(),
        ) else {
            continue;
        };
        if let Some(emb) = embedding_cosine_similarity(va, vb) {
            let combined = (wd * det_score + we * emb).clamp(0.0, 1.0);
            pair.score.embedding = Some(emb);
            pair.score.combined = combined;
            pair.score.used_onnx = true;
            pair.score.project_identity =
                (0.75 * pair.score.deterministic.project_identity + 0.25 * emb).clamp(0.0, 1.0);
            pair.score.relation = classify_pair_relation(
                &pair.score.deterministic,
                combined,
                config.same_project_threshold,
                config.review_threshold,
            );
        }
    }
}

/// Computes pair comparison directly between any two images (for side-by-side Compare modal).
pub fn compare_pair_direct(
    a: &ImageFeatures,
    b: &ImageFeatures,
    config: &SimilarityConfig,
) -> SimilarityScore {
    let engine = HybridSimilarityEngine {
        config: config.clone(),
    };
    engine.compare(a, b)
}

/// Evaluates pure embedding-only relation (for Developer Benchmark / Evaluation comparison).
pub fn compare_embedding_only(
    a: &ImageFeatures,
    b: &ImageFeatures,
    threshold: f32,
) -> Option<(f32, PairRelation)> {
    let (Some(va), Some(vb)) = (a.embedding.as_deref(), b.embedding.as_deref()) else {
        return None;
    };
    let sim = embedding_cosine_similarity(va, vb)?;
    let rel = if sim >= threshold {
        PairRelation::SameProject
    } else if sim >= threshold - 0.12 {
        PairRelation::RelatedUncertain
    } else {
        PairRelation::Different
    };
    Some((sim, rel))
}
