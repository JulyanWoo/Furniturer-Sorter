use crate::cache::CacheStore;
use crate::fingerprint::load_oriented_thumbnail;
use crate::scanner::ScannedImage;
use image::GenericImageView;
use std::collections::{HashMap, HashSet, VecDeque};
use std::path::PathBuf;
use std::sync::mpsc::{self, Receiver, Sender};
use std::thread;

pub const THUMB_MAX_SIDE: u32 = 256;
pub const COMPARE_MAX_SIDE: u32 = 768;
const MAX_RAM_THUMBS: usize = 180;

struct ThumbRequest {
    idx: usize,
    image: ScannedImage,
    max_side: u32,
    is_compare: bool,
}

struct ThumbResponse {
    idx: usize,
    is_compare: bool,
    color_image: Option<egui::ColorImage>,
}

/// Bounded lazy thumbnail & comparison-preview loader.
/// Decodes off the UI thread, caches <=256px JPEGs under %LocalAppData%\FurnitureSorter\thumbnails,
/// and evicts older TextureHandles in RAM above MAX_RAM_THUMBS.
pub struct ThumbnailManager {
    tx: Sender<ThumbRequest>,
    rx: Receiver<ThumbResponse>,
    textures: HashMap<usize, egui::TextureHandle>,
    lru: VecDeque<usize>,
    in_flight: HashSet<usize>,
    failed: HashSet<usize>,
    compare_textures: HashMap<usize, egui::TextureHandle>,
    compare_in_flight: HashSet<usize>,
}

impl ThumbnailManager {
    pub fn new(cache_store: CacheStore) -> Self {
        let (req_tx, req_rx) = mpsc::channel::<ThumbRequest>();
        let (res_tx, res_rx) = mpsc::channel::<ThumbResponse>();

        thread::spawn(move || {
            let _ = std::fs::create_dir_all(cache_store.thumbnails_dir());
            while let Ok(req) = req_rx.recv() {
                let img_opt =
                    load_thumb_color_image(&cache_store, &req.image, req.max_side, !req.is_compare);
                let _ = res_tx.send(ThumbResponse {
                    idx: req.idx,
                    is_compare: req.is_compare,
                    color_image: img_opt,
                });
            }
        });

        Self {
            tx: req_tx,
            rx: res_rx,
            textures: HashMap::new(),
            lru: VecDeque::new(),
            in_flight: HashSet::new(),
            failed: HashSet::new(),
            compare_textures: HashMap::new(),
            compare_in_flight: HashSet::new(),
        }
    }

    pub fn clear(&mut self) {
        self.textures.clear();
        self.lru.clear();
        self.in_flight.clear();
        self.failed.clear();
        self.compare_textures.clear();
        self.compare_in_flight.clear();
    }

    pub fn clear_compare_cache(&mut self) {
        self.compare_textures.clear();
        self.compare_in_flight.clear();
    }

    pub fn poll(&mut self, ctx: &egui::Context) -> bool {
        let mut updated = false;
        while let Ok(resp) = self.rx.try_recv() {
            updated = true;
            if resp.is_compare {
                self.compare_in_flight.remove(&resp.idx);
                if let Some(ci) = resp.color_image {
                    let handle = ctx.load_texture(
                        format!("cmp_{}", resp.idx),
                        ci,
                        egui::TextureOptions::LINEAR,
                    );
                    self.compare_textures.insert(resp.idx, handle);
                }
            } else {
                self.in_flight.remove(&resp.idx);
                if let Some(ci) = resp.color_image {
                    let handle = ctx.load_texture(
                        format!("thumb_{}", resp.idx),
                        ci,
                        egui::TextureOptions::LINEAR,
                    );
                    self.insert_lru_texture(resp.idx, handle);
                } else {
                    self.failed.insert(resp.idx);
                }
            }
        }
        updated
    }

    pub fn get_or_request_thumb(
        &mut self,
        idx: usize,
        image: &ScannedImage,
    ) -> Option<&egui::TextureHandle> {
        if self.textures.contains_key(&idx) {
            return self.textures.get(&idx);
        }
        if !self.in_flight.contains(&idx) && !self.failed.contains(&idx) {
            self.in_flight.insert(idx);
            let _ = self.tx.send(ThumbRequest {
                idx,
                image: image.clone(),
                max_side: THUMB_MAX_SIDE,
                is_compare: false,
            });
        }
        None
    }

    pub fn get_or_request_compare(
        &mut self,
        idx: usize,
        image: &ScannedImage,
    ) -> Option<&egui::TextureHandle> {
        if self.compare_textures.contains_key(&idx) {
            return self.compare_textures.get(&idx);
        }
        if !self.compare_in_flight.contains(&idx) {
            self.compare_in_flight.insert(idx);
            let _ = self.tx.send(ThumbRequest {
                idx,
                image: image.clone(),
                max_side: COMPARE_MAX_SIDE,
                is_compare: true,
            });
        }
        None
    }

    fn insert_lru_texture(&mut self, idx: usize, handle: egui::TextureHandle) {
        self.textures.insert(idx, handle);
        self.lru.retain(|&x| x != idx);
        self.lru.push_back(idx);
        while self.lru.len() > MAX_RAM_THUMBS {
            if let Some(old) = self.lru.pop_front() {
                self.textures.remove(&old);
            }
        }
    }
}

fn load_thumb_color_image(
    cache_store: &CacheStore,
    image: &ScannedImage,
    max_side: u32,
    use_disk_cache: bool,
) -> Option<egui::ColorImage> {
    let thumb_path: PathBuf = cache_store.thumbnail_cache_path(image);
    if use_disk_cache
        && thumb_path.is_file()
        && let Ok(img) = image::open(&thumb_path)
    {
        return Some(dynamic_to_color_image(img));
    }

    let loaded = load_oriented_thumbnail(&image.path, max_side).ok()?;
    if use_disk_cache {
        let _ = loaded.to_rgb8().save(&thumb_path);
    }
    Some(dynamic_to_color_image(loaded))
}

fn dynamic_to_color_image(img: image::DynamicImage) -> egui::ColorImage {
    let (w, h) = img.dimensions();
    let rgba = img.to_rgba8();
    egui::ColorImage::from_rgba_unmultiplied([w as usize, h as usize], rgba.as_raw())
}
