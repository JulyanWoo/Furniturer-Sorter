use crate::cache::fnv1a_64;
use crate::sidecar::{SidecarInfo, load_sidecar};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::UNIX_EPOCH;
use walkdir::{DirEntry, WalkDir};

pub const OUTPUT_MARKER_FILE: &str = ".furniture_sorter_output";

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ScannedImage {
    pub path: PathBuf,
    pub file_name: String,
    pub relative_path: String,
    pub size: u64,
    pub modified_secs: i64,
    pub modified_nanos: u32,
    pub sidecar: Option<SidecarInfo>,
    #[serde(default)]
    pub content_sha256: Option<String>,
}

impl ScannedImage {
    pub fn identity_sha256(&self) -> Option<&str> {
        self.sidecar
            .as_ref()
            .and_then(|s| s.data.effective_image_sha256())
    }

    /// Stable, privacy-safe identity used for evaluation records:
    /// prefers OpenIMG `generatedSha256`, otherwise cached file SHA-256,
    /// otherwise a deterministic metadata digest (never a personal absolute path).
    pub fn stable_id(&self) -> String {
        if let Some(sha) = self.identity_sha256() {
            return format!("openimg:{}", sha.to_ascii_lowercase());
        }
        if let Some(sha) = &self.content_sha256 {
            return format!("sha256:{sha}");
        }
        let meta_key = format!(
            "{}|{}|{}|{}",
            self.file_name, self.size, self.modified_secs, self.modified_nanos
        );
        format!("meta:{:016x}", fnv1a_64(meta_key.as_bytes()))
    }
}

pub fn is_supported_image_path(path: &Path) -> bool {
    let Some(ext) = path.extension().and_then(|e| e.to_str()) else {
        return false;
    };
    matches!(
        ext.to_ascii_lowercase().as_str(),
        "jpg" | "jpeg" | "png" | "webp"
    )
}

fn should_descend(entry: &DirEntry) -> bool {
    if !entry.file_type().is_dir() {
        return true;
    }
    // Never re-scan Furniture Sorter's own output folders.
    !entry.path().join(OUTPUT_MARKER_FILE).exists()
}

pub fn scan_folder<F>(
    root: &Path,
    cancel: &Arc<AtomicBool>,
    mut on_progress: F,
) -> Result<Vec<ScannedImage>, String>
where
    F: FnMut(usize),
{
    if !root.is_dir() {
        return Err(format!(
            "Selected folder does not exist: {}",
            root.display()
        ));
    }

    let canonical_root = std::fs::canonicalize(root).unwrap_or_else(|_| root.to_path_buf());
    let mut items = Vec::new();

    for entry_res in WalkDir::new(&canonical_root)
        .follow_links(false)
        .into_iter()
        .filter_entry(should_descend)
    {
        if cancel.load(Ordering::Relaxed) {
            return Err("Scan cancelled by user.".to_string());
        }
        let Ok(entry) = entry_res else {
            continue;
        };
        if !entry.file_type().is_file() {
            continue;
        }
        let path = entry.path();
        if !is_supported_image_path(path) {
            continue;
        }

        let Ok(meta) = entry.metadata() else {
            continue;
        };
        let size = meta.len();
        let (modified_secs, modified_nanos) = match meta.modified() {
            Ok(st) => match st.duration_since(UNIX_EPOCH) {
                Ok(d) => (d.as_secs() as i64, d.subsec_nanos()),
                Err(e) => {
                    let d = e.duration();
                    (-(d.as_secs() as i64), d.subsec_nanos())
                }
            },
            Err(_) => (0, 0),
        };

        let file_name = path
            .file_name()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_default();
        let relative_path = path
            .strip_prefix(&canonical_root)
            .unwrap_or(path)
            .to_string_lossy()
            .replace('\\', "/");

        let sidecar = load_sidecar(path);
        items.push(ScannedImage {
            path: path.to_path_buf(),
            file_name,
            relative_path,
            size,
            modified_secs,
            modified_nanos,
            sidecar,
            content_sha256: None,
        });

        if items.len() % 50 == 0 {
            on_progress(items.len());
        }
    }

    on_progress(items.len());
    items.sort_by(|a, b| a.relative_path.cmp(&b.relative_path));
    Ok(items)
}
