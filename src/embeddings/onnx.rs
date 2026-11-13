use crate::embeddings::OnnxModelStatus;
#[cfg(feature = "onnx")]
use crate::embeddings::l2_normalize;
#[cfg(feature = "onnx")]
use crate::fingerprint::load_oriented_for_onnx;
use serde::Deserialize;
use std::path::{Path, PathBuf};
use std::time::UNIX_EPOCH;

#[derive(Clone, Debug, Deserialize)]
pub struct PreprocessorMeta {
    pub image_mean: Option<[f32; 3]>,
    pub image_std: Option<[f32; 3]>,
    pub size: Option<PreprocessorSize>,
    pub crop_size: Option<PreprocessorSize>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct PreprocessorSize {
    pub width: Option<u32>,
    pub height: Option<u32>,
    pub shortest_edge: Option<u32>,
}

#[derive(Clone, Debug)]
pub struct OnnxPreprocessConfig {
    pub width: u32,
    pub height: u32,
    pub mean: [f32; 3],
    pub std: [f32; 3],
}

impl Default for OnnxPreprocessConfig {
    fn default() -> Self {
        // Standard ImageNet normalization (matches DINOv2, MobileNetV3, ViT encoders)
        Self {
            width: 224,
            height: 224,
            mean: [0.485, 0.456, 0.406],
            std: [0.229, 0.224, 0.225],
        }
    }
}

pub fn default_model_candidates(custom_path: Option<&Path>) -> Vec<PathBuf> {
    let mut candidates = Vec::new();
    if let Some(p) = custom_path
        && !p.as_os_str().is_empty()
    {
        candidates.push(p.to_path_buf());
    }
    if let Ok(exe) = std::env::current_exe()
        && let Some(dir) = exe.parent()
    {
        candidates.push(dir.join("models").join("visual_encoder.onnx"));
    }
    candidates.push(PathBuf::from("models").join("visual_encoder.onnx"));
    candidates
}

pub fn find_onnx_runtime_dll(model_path: Option<&Path>) -> Option<PathBuf> {
    if let Ok(env_path) = std::env::var("ORT_DYLIB_PATH") {
        let p = PathBuf::from(env_path);
        if p.is_file() {
            return Some(p);
        }
    }
    let dll_name = if cfg!(windows) {
        "onnxruntime.dll"
    } else if cfg!(target_os = "macos") {
        "libonnxruntime.dylib"
    } else {
        "libonnxruntime.so"
    };

    let mut search_dirs = Vec::new();
    if let Some(m) = model_path
        && let Some(parent) = m.parent()
    {
        search_dirs.push(parent.to_path_buf());
    }
    if let Ok(exe) = std::env::current_exe()
        && let Some(dir) = exe.parent()
    {
        search_dirs.push(dir.to_path_buf());
        search_dirs.push(dir.join("models"));
    }
    search_dirs.push(PathBuf::from("."));
    search_dirs.push(PathBuf::from("models"));

    for dir in search_dirs {
        let candidate = dir.join(dll_name);
        if candidate.is_file() {
            return Some(candidate);
        }
    }
    None
}

pub fn detect_onnx_status(custom_model_path: Option<&Path>) -> OnnxModelStatus {
    if !cfg!(feature = "onnx") {
        return OnnxModelStatus::Unavailable {
            reason: "Unavailable — compiled without ONNX feature".to_string(),
        };
    }

    let model_path = default_model_candidates(custom_model_path)
        .into_iter()
        .find(|p| p.is_file());

    let Some(model_path) = model_path else {
        return OnnxModelStatus::Unavailable {
            reason: "Unavailable — model not installed".to_string(),
        };
    };

    let runtime_path = find_onnx_runtime_dll(Some(&model_path));
    if runtime_path.is_none() {
        return OnnxModelStatus::Unavailable {
            reason: "Unavailable — onnxruntime.dll not found alongside app or models/".to_string(),
        };
    }

    let Ok(meta) = std::fs::metadata(&model_path) else {
        return OnnxModelStatus::Unavailable {
            reason: "Unavailable — unreadable model file".to_string(),
        };
    };
    let mtime = meta
        .modified()
        .ok()
        .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let model_size_bytes = meta.len();
    let file_name = model_path
        .file_name()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "visual_encoder.onnx".to_string());
    let model_tag = format!("{file_name}:{model_size_bytes}:{mtime}");

    OnnxModelStatus::Available {
        model_path,
        runtime_path,
        model_size_bytes,
        model_tag,
    }
}

pub fn load_preprocess_config(model_path: &Path) -> OnnxPreprocessConfig {
    let mut cfg = OnnxPreprocessConfig::default();
    let Some(dir) = model_path.parent() else {
        return cfg;
    };
    let candidates = [
        model_path.with_extension("json"),
        dir.join("preprocessor_config.json"),
    ];
    for path in candidates {
        let Ok(bytes) = std::fs::read(&path) else {
            continue;
        };
        let Ok(meta) = serde_json::from_slice::<PreprocessorMeta>(&bytes) else {
            continue;
        };
        if let Some(mean) = meta.image_mean {
            cfg.mean = mean;
        }
        if let Some(std) = meta.image_std {
            cfg.std = std;
        }
        if let Some(crop) = meta.crop_size {
            if let (Some(w), Some(h)) = (crop.width, crop.height) {
                cfg.width = w;
                cfg.height = h;
            }
        } else if let Some(sz) = meta.size {
            if let (Some(w), Some(h)) = (sz.width, sz.height) {
                cfg.width = w;
                cfg.height = h;
            } else if let Some(edge) = sz.shortest_edge {
                cfg.width = edge;
                cfg.height = edge;
            }
        }
        break;
    }
    cfg
}

#[cfg(feature = "onnx")]
static ORT_ENV: std::sync::OnceLock<Result<ort::environment::Environment, String>> =
    std::sync::OnceLock::new();

#[cfg(feature = "onnx")]
fn get_or_init_env(runtime_path: &Path) -> Result<ort::environment::Environment, String> {
    let res = ORT_ENV.get_or_init(|| {
        ort::init_from(runtime_path.to_string_lossy().as_ref())
            .map_err(|e| format!("Failed to initialize ONNX Runtime: {e}"))?
            .with_telemetry(false)
            .build()
            .map_err(|e| format!("Failed to build ONNX environment: {e}"))
    });
    res.clone()
}

#[cfg(feature = "onnx")]
pub struct OnnxEncoder {
    session: ort::session::Session,
    input_name: String,
    preprocess: OnnxPreprocessConfig,
    pub model_tag: String,
}

#[cfg(feature = "onnx")]
impl OnnxEncoder {
    /// Initializes one persistent ONNX Runtime session for CPU inference.
    pub fn load(
        model_path: &Path,
        runtime_path: &Path,
        model_tag: String,
        intra_threads: usize,
    ) -> Result<Self, String> {
        let env = get_or_init_env(runtime_path)?;

        let mut preprocess = load_preprocess_config(model_path);

        let session = ort::session::Session::builder(&env)
            .map_err(|e| format!("Session builder error: {e}"))?
            .with_optimization_level(ort::session::builder::GraphOptimizationLevel::Level3)
            .map_err(|e| format!("Optimization level error: {e}"))?
            .with_intra_threads(intra_threads.max(1))
            .map_err(|e| format!("Intra-threads config error: {e}"))?
            .commit_from_file(model_path)
            .map_err(|e| format!("Failed to load ONNX model {}: {e}", model_path.display()))?;

        let input_name = session
            .inputs()
            .first()
            .map(|i| i.name().to_string())
            .unwrap_or_else(|| "pixel_values".to_string());

        if let Some(first_input) = session.inputs().first()
            && let Some(shape) = first_input.dtype().tensor_shape()
            && shape.len() == 4
        {
            let h = shape[2];
            let w = shape[3];
            if h > 0 && w > 0 {
                preprocess.height = h as u32;
                preprocess.width = w as u32;
            }
        }

        Ok(Self {
            session,
            input_name,
            preprocess,
            model_tag,
        })
    }

    pub fn encode_image(&mut self, image_path: &Path) -> Result<Vec<f32>, String> {
        let img =
            load_oriented_for_onnx(image_path, self.preprocess.width, self.preprocess.height)?;
        let rgb = img.to_rgb8();
        let w = self.preprocess.width as usize;
        let h = self.preprocess.height as usize;

        let mut tensor = ndarray::Array4::<f32>::zeros((1, 3, h, w));
        let mean = self.preprocess.mean;
        let std = self.preprocess.std;

        for (x, y, px) in rgb.enumerate_pixels() {
            tensor[[0, 0, y as usize, x as usize]] = (px[0] as f32 / 255.0 - mean[0]) / std[0];
            tensor[[0, 1, y as usize, x as usize]] = (px[1] as f32 / 255.0 - mean[1]) / std[1];
            tensor[[0, 2, y as usize, x as usize]] = (px[2] as f32 / 255.0 - mean[2]) / std[2];
        }

        let input_val = ort::value::Tensor::from_array(tensor)
            .map_err(|e| format!("Input tensor creation failed: {e}"))?;

        let outputs = self
            .session
            .run(ort::inputs![self.input_name.as_str() => input_val])
            .map_err(|e| format!("ONNX inference failed: {e}"))?;

        let preferred_keys = ["pooler_output", "image_embeds", "last_hidden_state"];
        let selected_output = preferred_keys
            .iter()
            .find_map(|k| outputs.get(*k))
            .unwrap_or(&outputs[0]);

        let (shape, data) = selected_output
            .try_extract_tensor::<f32>()
            .map_err(|e| format!("Failed to extract ONNX output tensor: {e}"))?;

        let mut embedding = extract_pooled_embedding(&shape, data)?;
        l2_normalize(&mut embedding);
        Ok(embedding)
    }
}

#[cfg(not(feature = "onnx"))]
pub struct OnnxEncoder {
    pub model_tag: String,
}

#[cfg(not(feature = "onnx"))]
impl OnnxEncoder {
    pub fn load(
        _model_path: &Path,
        _runtime_path: &Path,
        _model_tag: String,
        _intra_threads: usize,
    ) -> Result<Self, String> {
        Err("ONNX feature not compiled in this binary.".to_string())
    }

    pub fn encode_image(&mut self, _image_path: &Path) -> Result<Vec<f32>, String> {
        Err("ONNX feature not compiled in this binary.".to_string())
    }
}

pub fn extract_pooled_embedding(shape: &[i64], data: &[f32]) -> Result<Vec<f32>, String> {
    if data.is_empty() {
        return Err("ONNX output tensor is empty".to_string());
    }
    match shape.len() {
        1 => Ok(data.to_vec()),
        2 => {
            let dim = shape[1].max(0) as usize;
            if dim == 0 || data.len() < dim {
                return Err(format!("Invalid 2D ONNX output shape: {shape:?}"));
            }
            Ok(data[..dim].to_vec())
        }
        3 => {
            // (batch=1, tokens, dim): take token 0 (CLS) when ViT/DINOv2-style sequence is returned
            let dim = shape[2].max(0) as usize;
            if dim == 0 || data.len() < dim {
                return Err(format!("Invalid 3D ONNX output shape: {shape:?}"));
            }
            Ok(data[..dim].to_vec())
        }
        4 => {
            // (batch=1, C, H, W): global average pool spatial dims
            let c = shape[1].max(0) as usize;
            let h = shape[2].max(0) as usize;
            let w = shape[3].max(0) as usize;
            let hw = h * w;
            if c == 0 || hw == 0 || data.len() < c * hw {
                return Err(format!("Invalid 4D ONNX output shape: {shape:?}"));
            }
            let mut pooled = vec![0.0f32; c];
            for ch in 0..c {
                let start = ch * hw;
                let sum: f32 = data[start..start + hw].iter().sum();
                pooled[ch] = sum / hw as f32;
            }
            Ok(pooled)
        }
        _ => Err(format!("Unsupported ONNX output tensor rank: {shape:?}")),
    }
}
