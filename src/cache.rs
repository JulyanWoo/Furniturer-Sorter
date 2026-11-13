use crate::fingerprint::{FINGERPRINT_VERSION, ImageFingerprint};
use crate::scanner::ScannedImage;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{BufReader, BufWriter};
use std::path::{Path, PathBuf};

pub const CACHE_FORMAT_VERSION: u32 = 2;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CachedEntry {
    pub size: u64,
    pub modified_secs: i64,
    pub modified_nanos: u32,
    pub openimg_generated_sha256: Option<String>,
    pub content_sha256: Option<String>,
    pub fingerprint: ImageFingerprint,
    pub embedding_model_tag: Option<String>,
    pub embedding: Option<Vec<f32>>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LibraryCache {
    pub format_version: u32,
    pub fingerprint_version: u32,
    pub library_root: String,
    pub entries: HashMap<String, CachedEntry>,
}

impl LibraryCache {
    pub fn empty(library_root: &Path) -> Self {
        Self {
            format_version: CACHE_FORMAT_VERSION,
            fingerprint_version: FINGERPRINT_VERSION,
            library_root: library_root.to_string_lossy().into_owned(),
            entries: HashMap::new(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct CacheStore {
    root_dir: PathBuf,
}

impl CacheStore {
    pub fn default_store() -> Self {
        let base = dirs::data_local_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join("FurnitureSorter");
        Self { root_dir: base }
    }

    pub fn with_root(root_dir: PathBuf) -> Self {
        Self { root_dir }
    }

    pub fn root_dir(&self) -> &Path {
        &self.root_dir
    }

    pub fn cache_dir(&self) -> PathBuf {
        self.root_dir.join("cache")
    }

    pub fn thumbnails_dir(&self) -> PathBuf {
        self.root_dir.join("thumbnails")
    }

    pub fn eval_dir(&self) -> PathBuf {
        self.root_dir.join("evaluation")
    }

    pub fn library_cache_path(&self, library_root: &Path) -> PathBuf {
        let canonical = std::fs::canonicalize(library_root)
            .unwrap_or_else(|_| library_root.to_path_buf())
            .to_string_lossy()
            .to_ascii_lowercase();
        let hash = fnv1a_64(canonical.as_bytes());
        self.cache_dir().join(format!("lib_{hash:016x}.bin"))
    }

    pub fn load_library(&self, library_root: &Path) -> LibraryCache {
        let path = self.library_cache_path(library_root);
        let Ok(file) = File::open(&path) else {
            return LibraryCache::empty(library_root);
        };
        let reader = BufReader::new(file);
        match bincode::deserialize_from::<_, LibraryCache>(reader) {
            Ok(cache)
                if cache.format_version == CACHE_FORMAT_VERSION
                    && cache.fingerprint_version == FINGERPRINT_VERSION =>
            {
                cache
            }
            _ => LibraryCache::empty(library_root),
        }
    }

    pub fn save_library(&self, library_root: &Path, cache: &LibraryCache) -> Result<(), String> {
        fs::create_dir_all(self.cache_dir())
            .map_err(|e| format!("Create cache dir failed: {e}"))?;
        let path = self.library_cache_path(library_root);
        let tmp_path = path.with_extension("bin.tmp");
        {
            let file =
                File::create(&tmp_path).map_err(|e| format!("Create cache temp failed: {e}"))?;
            let writer = BufWriter::new(file);
            bincode::serialize_into(writer, cache)
                .map_err(|e| format!("Serialize cache failed: {e}"))?;
        }
        if path.exists() {
            let _ = fs::remove_file(&path);
        }
        fs::rename(&tmp_path, &path).map_err(|e| format!("Commit cache failed: {e}"))?;
        Ok(())
    }

    pub fn thumbnail_cache_path(&self, image: &ScannedImage) -> PathBuf {
        let key = format!(
            "{}|{}|{}|{}",
            image.path.to_string_lossy().to_ascii_lowercase(),
            image.size,
            image.modified_secs,
            image.modified_nanos
        );
        let h = fnv1a_64(key.as_bytes());
        self.thumbnails_dir().join(format!("{h:016x}.jpg"))
    }

    pub fn clear_library_cache(&self, library_root: &Path) -> Result<(), String> {
        let path = self.library_cache_path(library_root);
        if path.exists() {
            fs::remove_file(&path).map_err(|e| format!("Remove cache failed: {e}"))?;
        }
        Ok(())
    }
}

pub fn normalize_cache_key(path: &Path) -> String {
    path.to_string_lossy()
        .replace('\\', "/")
        .to_ascii_lowercase()
}

pub fn is_entry_valid_for_image(entry: &CachedEntry, img: &ScannedImage) -> bool {
    if entry.fingerprint.version != FINGERPRINT_VERSION {
        return false;
    }
    if entry.size != img.size
        || entry.modified_secs != img.modified_secs
        || entry.modified_nanos != img.modified_nanos
    {
        return false;
    }
    if let Some(sidecar_sha) = img.identity_sha256() {
        return entry.openimg_generated_sha256.as_deref() == Some(sidecar_sha);
    }
    true
}

pub fn valid_embedding_for_image<'a>(
    entry: &'a CachedEntry,
    img: &ScannedImage,
    model_tag: &str,
) -> Option<&'a [f32]> {
    if !is_entry_valid_for_image(entry, img) {
        return None;
    }
    if entry.embedding_model_tag.as_deref() != Some(model_tag) {
        return None;
    }
    entry.embedding.as_deref().filter(|v| !v.is_empty())
}

pub fn fnv1a_64(bytes: &[u8]) -> u64 {
    let mut hash: u64 = 0xcbf29ce484222325;
    for &b in bytes {
        hash ^= b as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}
