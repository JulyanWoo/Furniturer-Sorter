use crate::cache::{
    CacheStore, CachedEntry, LibraryCache, is_entry_valid_for_image, normalize_cache_key,
    valid_embedding_for_image,
};
use crate::clustering::{ClusteringConfig, ClusteringOutput, cluster_images};
use crate::embeddings::onnx::{OnnxEncoder, detect_onnx_status};
use crate::embeddings::{ImageFeatures, OnnxModelStatus};
use crate::evaluation::ProcessingTimingsMs;
use crate::fingerprint::extract_fingerprint_and_hash;
use crate::scanner::{ScannedImage, scan_folder};
use crate::similarity::{
    PairComparison, SimilarityConfig, ambiguous_image_indices, apply_hybrid_embeddings_to_pairs,
    compute_deterministic_candidates,
};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::Instant;

#[derive(Clone, Debug)]
pub enum AnalysisProgress {
    Scanning { found: usize },
    AnalyzingFingerprints { done: usize, total: usize },
    RunningOnnx { done: usize, total: usize },
    CalculatingSimilarities,
    BuildingGroups,
    Ready,
}

impl AnalysisProgress {
    pub fn label(&self) -> String {
        match self {
            Self::Scanning { found } => format!("Scanning — {found} images found"),
            Self::AnalyzingFingerprints { done, total } => {
                format!("Analyzing fingerprints — {done} / {total}")
            }
            Self::RunningOnnx { done, total } => {
                format!("Running optional ONNX embeddings — {done} / {total}")
            }
            Self::CalculatingSimilarities => "Calculating similarities".to_string(),
            Self::BuildingGroups => "Building groups".to_string(),
            Self::Ready => "Ready".to_string(),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BenchmarkMetrics {
    pub images: usize,
    pub scan_ms: u64,
    pub fingerprint_ms: u64,
    pub embedding_ms: u64,
    pub similarity_ms: u64,
    pub clustering_ms: u64,
    pub total_ms: u64,
    pub cache_hits: usize,
    pub cache_misses: usize,
    pub cache_hit_rate: f32,
    pub prefilter_total_pairs: usize,
    pub prefilter_passed_pairs: usize,
    pub onnx_images_encoded: usize,
    pub embedding_dim: Option<usize>,
    pub approx_feature_memory_kb: usize,
    pub threads_used: usize,
}

#[derive(Clone, Debug)]
pub struct AnalysisResult {
    pub library_root: PathBuf,
    pub images: Vec<ScannedImage>,
    pub features: Vec<ImageFeatures>,
    pub pairs: Vec<PairComparison>,
    pub clustering: ClusteringOutput,
    pub benchmark: BenchmarkMetrics,
    pub timings_ms: ProcessingTimingsMs,
    pub onnx_status: OnnxModelStatus,
    pub warnings: Vec<String>,
}

pub fn default_worker_threads() -> usize {
    let logical = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4);
    logical.saturating_sub(2).clamp(1, 6)
}

pub fn run_analysis_pipeline<F>(
    library_root: &Path,
    sim_config: &SimilarityConfig,
    cluster_config: &ClusteringConfig,
    cache_store: &CacheStore,
    custom_onnx_path: Option<&Path>,
    worker_threads: Option<usize>,
    cancel: &Arc<AtomicBool>,
    mut on_progress: F,
) -> Result<AnalysisResult, String>
where
    F: FnMut(AnalysisProgress) + Send,
{
    let total_start = Instant::now();
    let threads = worker_threads.unwrap_or_else(default_worker_threads).max(1);
    let onnx_status = detect_onnx_status(custom_onnx_path);
    let mut warnings = Vec::new();

    // 1. Scan folder
    let scan_start = Instant::now();
    on_progress(AnalysisProgress::Scanning { found: 0 });
    let images = scan_folder(library_root, cancel, |found| {
        on_progress(AnalysisProgress::Scanning { found });
    })?;
    let scan_ms = scan_start.elapsed().as_millis() as u64;

    // 2. Load cache and extract fingerprints in bounded Rayon pool
    let fp_start = Instant::now();
    let mut lib_cache: LibraryCache = cache_store.load_library(library_root);
    let total = images.len();
    on_progress(AnalysisProgress::AnalyzingFingerprints { done: 0, total });

    let active_model_tag: Option<String> = match &onnx_status {
        OnnxModelStatus::Available { model_tag, .. } => Some(model_tag.clone()),
        _ => None,
    };

    let mut cached_fps: Vec<Option<(ImageFeatures, Option<String>)>> = vec![None; total];
    let mut misses: Vec<usize> = Vec::new();
    let mut cache_hits = 0usize;

    for (idx, img) in images.iter().enumerate() {
        let key = normalize_cache_key(&img.path);
        if let Some(entry) = lib_cache.entries.get(&key)
            && is_entry_valid_for_image(entry, img)
        {
            let emb = active_model_tag
                .as_deref()
                .and_then(|tag| valid_embedding_for_image(entry, img, tag))
                .map(|v| v.to_vec());
            cached_fps[idx] = Some((
                ImageFeatures {
                    fingerprint: entry.fingerprint.clone(),
                    embedding: emb,
                },
                entry.content_sha256.clone(),
            ));
            cache_hits += 1;
        } else {
            misses.push(idx);
        }
    }

    let cache_misses = misses.len();
    let done_counter = Arc::new(AtomicUsize::new(cache_hits));
    on_progress(AnalysisProgress::AnalyzingFingerprints {
        done: cache_hits,
        total,
    });

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .map_err(|e| format!("Failed to build Rayon pool: {e}"))?;

    let progress_cb = std::sync::Mutex::new(&mut on_progress);
    let extracted_misses: Vec<(
        usize,
        Result<(crate::fingerprint::ImageFingerprint, String), String>,
    )> = pool.install(|| {
        misses
            .into_par_iter()
            .filter_map(|idx| {
                if cancel.load(Ordering::Relaxed) {
                    return None;
                }
                let res = extract_fingerprint_and_hash(&images[idx].path);
                let done = done_counter.fetch_add(1, Ordering::Relaxed) + 1;
                if (done % 12 == 0 || done == total)
                    && let Ok(mut cb) = progress_cb.lock()
                {
                    cb(AnalysisProgress::AnalyzingFingerprints { done, total });
                }
                Some((idx, res))
            })
            .collect()
    });
    drop(progress_cb);

    if cancel.load(Ordering::Relaxed) {
        return Err("Fingerprint extraction cancelled by user.".to_string());
    }

    let mut fallback_indices = Vec::new();
    for (idx, res) in extracted_misses {
        match res {
            Ok((fp, sha)) => {
                cached_fps[idx] = Some((
                    ImageFeatures {
                        fingerprint: fp,
                        embedding: None,
                    },
                    Some(sha),
                ));
            }
            Err(err) => {
                warnings.push(format!(
                    "Skipped unreadable image {}: {err}",
                    images[idx].relative_path
                ));
                fallback_indices.push(idx);
            }
        }
    }

    // Filter out unreadable images cleanly so indices stay aligned
    let mut valid_images = Vec::with_capacity(total - fallback_indices.len());
    let mut features = Vec::with_capacity(total - fallback_indices.len());
    for (idx, maybe_feat) in cached_fps.into_iter().enumerate() {
        if let Some((feat, sha)) = maybe_feat {
            let mut img = images[idx].clone();
            img.content_sha256 = sha.clone();
            let key = normalize_cache_key(&img.path);
            lib_cache.entries.insert(
                key,
                CachedEntry {
                    size: img.size,
                    modified_secs: img.modified_secs,
                    modified_nanos: img.modified_nanos,
                    openimg_generated_sha256: img.identity_sha256().map(str::to_string),
                    content_sha256: sha,
                    fingerprint: feat.fingerprint.clone(),
                    embedding_model_tag: feat.embedding.as_ref().and(active_model_tag.clone()),
                    embedding: feat.embedding.clone(),
                },
            );
            valid_images.push(img);
            features.push(feat);
        }
    }
    on_progress(AnalysisProgress::AnalyzingFingerprints {
        done: valid_images.len(),
        total: valid_images.len(),
    });
    let fingerprint_ms = fp_start.elapsed().as_millis() as u64;

    // 3. Stage 1: Deterministic candidate prefilter + similarity scoring
    on_progress(AnalysisProgress::CalculatingSimilarities);
    let sim_start = Instant::now();
    let (mut pairs, prefilter_total_pairs, prefilter_passed_pairs) =
        pool.install(|| compute_deterministic_candidates(&features, sim_config, cancel))?;
    let mut similarity_ms = sim_start.elapsed().as_millis() as u64;

    // 4. Stage 2: Optional ONNX embeddings ONLY for ambiguous pairs
    let emb_start = Instant::now();
    let mut onnx_images_encoded = 0usize;
    let mut embedding_dim: Option<usize> = features
        .iter()
        .find_map(|f| f.embedding.as_ref().map(|v| v.len()));

    if sim_config.enable_onnx {
        if let OnnxModelStatus::Available {
            model_path,
            runtime_path: Some(rt_path),
            model_tag,
            ..
        } = &onnx_status
        {
            let target_indices = ambiguous_image_indices(&pairs, sim_config);
            let need_encode: Vec<usize> = target_indices
                .iter()
                .copied()
                .filter(|&i| features[i].embedding.is_none())
                .collect();

            if !need_encode.is_empty() {
                on_progress(AnalysisProgress::RunningOnnx {
                    done: 0,
                    total: need_encode.len(),
                });
                match OnnxEncoder::load(model_path, rt_path, model_tag.clone(), threads.min(4)) {
                    Ok(mut encoder) => {
                        for (pos, &img_idx) in need_encode.iter().enumerate() {
                            if cancel.load(Ordering::Relaxed) {
                                return Err("ONNX inference cancelled by user.".to_string());
                            }
                            match encoder.encode_image(&valid_images[img_idx].path) {
                                Ok(vec) => {
                                    embedding_dim = Some(vec.len());
                                    let key = normalize_cache_key(&valid_images[img_idx].path);
                                    if let Some(entry) = lib_cache.entries.get_mut(&key) {
                                        entry.embedding_model_tag = Some(model_tag.clone());
                                        entry.embedding = Some(vec.clone());
                                    }
                                    features[img_idx].embedding = Some(vec);
                                    onnx_images_encoded += 1;
                                }
                                Err(err) => {
                                    warnings.push(format!(
                                        "ONNX failed on {}: {err}",
                                        valid_images[img_idx].relative_path
                                    ));
                                }
                            }
                            if (pos + 1) % 4 == 0 || pos + 1 == need_encode.len() {
                                on_progress(AnalysisProgress::RunningOnnx {
                                    done: pos + 1,
                                    total: need_encode.len(),
                                });
                            }
                        }
                    }
                    Err(err) => {
                        warnings.push(format!(
                            "ONNX session unavailable ({err}); continuing in deterministic mode."
                        ));
                    }
                }
            }

            let re_sim_start = Instant::now();
            apply_hybrid_embeddings_to_pairs(&mut pairs, &features, sim_config);
            similarity_ms += re_sim_start.elapsed().as_millis() as u64;
        } else {
            warnings.push(format!(
                "ONNX requested but unavailable ({})",
                onnx_status.ui_summary()
            ));
        }
    }
    let embedding_ms = emb_start.elapsed().as_millis() as u64;

    if let Err(err) = cache_store.save_library(library_root, &lib_cache) {
        warnings.push(format!("Cache save warning: {err}"));
    }

    // 5. Clustering
    on_progress(AnalysisProgress::BuildingGroups);
    let cluster_start = Instant::now();
    let clustering = cluster_images(valid_images.len(), &features, &pairs, cluster_config);
    let clustering_ms = cluster_start.elapsed().as_millis() as u64;

    let total_ms = total_start.elapsed().as_millis() as u64;
    let total_entries = cache_hits + cache_misses;
    let cache_hit_rate = if total_entries > 0 {
        cache_hits as f32 / total_entries as f32
    } else {
        0.0
    };

    // Approximate memory used by cached fingerprints + embeddings
    let per_feat_bytes = std::mem::size_of::<ImageFeatures>()
        + (76 + 12 + 96 + 40 + 16) * std::mem::size_of::<f32>()
        + embedding_dim.unwrap_or(0) * std::mem::size_of::<f32>();
    let approx_feature_memory_kb = (valid_images.len() * per_feat_bytes) / 1024;

    let benchmark = BenchmarkMetrics {
        images: valid_images.len(),
        scan_ms,
        fingerprint_ms,
        embedding_ms,
        similarity_ms,
        clustering_ms,
        total_ms,
        cache_hits,
        cache_misses,
        cache_hit_rate,
        prefilter_total_pairs,
        prefilter_passed_pairs,
        onnx_images_encoded,
        embedding_dim,
        approx_feature_memory_kb,
        threads_used: threads,
    };

    let timings_ms = ProcessingTimingsMs {
        scan_ms,
        fingerprint_ms,
        embedding_ms,
        similarity_ms,
        clustering_ms,
        total_ms,
    };

    on_progress(AnalysisProgress::Ready);

    Ok(AnalysisResult {
        library_root: library_root.to_path_buf(),
        images: valid_images,
        features,
        pairs,
        clustering,
        benchmark,
        timings_ms,
        onnx_status,
        warnings,
    })
}
