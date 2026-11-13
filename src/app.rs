use crate::benchmark::{
    AnalysisProgress, AnalysisResult, BenchmarkMetrics, default_worker_threads,
    run_analysis_pipeline,
};
use crate::cache::CacheStore;
use crate::clustering::{
    ClusteringConfig, ClusteringOutput, ImageGroup, PossibleMergeSuggestion, PossibleSplit,
};
use crate::embeddings::{ImageFeatures, OnnxModelStatus, onnx::detect_onnx_status};
use crate::evaluation::{
    EvaluationLabelStore, ModeComparisonReport, PrivacySafeEvaluationReport, ProcessingTimingsMs,
    load_evaluation_labels,
};
use crate::organizer::{OrganizeOptions, OrganizeReport};
use crate::scanner::ScannedImage;
use crate::similarity::{PairComparison, SimilarityConfig};
use crate::ui::{ThumbnailManager, home, review, settings};
use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver};
use std::thread;
use std::time::Duration;

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum AppScreen {
    Home,
    Analyzing,
    Review,
    Settings,
}

pub struct SessionData {
    pub library_root: PathBuf,
    pub images: Vec<ScannedImage>,
    pub features: Vec<ImageFeatures>,
    pub pairs: Vec<PairComparison>,
    pub initial_clustering: ClusteringOutput,
    pub groups: Vec<ImageGroup>,
    pub selected_group_idx: usize,
    pub selected_members: HashSet<usize>,
    pub move_target_group_idx: usize,
    pub merge_target_group_idx: usize,
    pub comparison_pair: Option<(usize, usize)>,
    pub group_comparison_pair: Option<(String, String)>,
    pub eval_labels: EvaluationLabelStore,
    pub eval_report: Option<ModeComparisonReport>,
    pub full_eval_report: Option<PrivacySafeEvaluationReport>,
    pub last_organize_report: Option<OrganizeReport>,
    pub benchmark: BenchmarkMetrics,
    pub timings_ms: ProcessingTimingsMs,
    pub blocked_chaining_edges: usize,
    pub possible_merges: Vec<PossibleMergeSuggestion>,
    pub possible_splits: Vec<PossibleSplit>,
    pub status_banner: Option<String>,
}

pub struct FurnitureSorterApp {
    pub screen: AppScreen,
    pub folder_input: String,
    pub output_dir_input: String,
    pub recent_folders: Vec<String>,
    pub sim_config: SimilarityConfig,
    pub cluster_config: ClusteringConfig,
    pub custom_onnx_path: String,
    pub worker_threads: usize,
    pub organize_options: OrganizeOptions,
    pub capture_eval_labels: bool,
    pub move_confirmation_armed: bool,
    pub pending_move_confirm_modal: bool,
    pub onnx_status: OnnxModelStatus,
    pub cache_store: CacheStore,
    pub thumbs: ThumbnailManager,
    pub session: Option<SessionData>,
    pub home_error: Option<String>,

    progress_rx: Option<Receiver<AnalysisProgress>>,
    result_rx: Option<Receiver<Result<AnalysisResult, String>>>,
    cancel_flag: Option<Arc<AtomicBool>>,
    progress_text: String,
    progress_fraction: f32,
}

impl FurnitureSorterApp {
    pub fn new() -> Self {
        let cache_store = CacheStore::default_store();
        let recent_folders = load_recent_folders(&cache_store);
        let initial_folder = recent_folders.first().cloned().unwrap_or_default();
        let output_dir_input = if initial_folder.is_empty() {
            String::new()
        } else {
            Path::new(&initial_folder)
                .join("Organized")
                .to_string_lossy()
                .into_owned()
        };

        Self {
            screen: AppScreen::Home,
            folder_input: initial_folder,
            output_dir_input,
            recent_folders,
            sim_config: SimilarityConfig::default(),
            cluster_config: ClusteringConfig::default(),
            custom_onnx_path: String::new(),
            worker_threads: default_worker_threads(),
            organize_options: OrganizeOptions::default(),
            capture_eval_labels: true,
            move_confirmation_armed: false,
            pending_move_confirm_modal: false,
            onnx_status: detect_onnx_status(None),
            thumbs: ThumbnailManager::new(cache_store.clone()),
            cache_store,
            session: None,
            home_error: None,
            progress_rx: None,
            result_rx: None,
            cancel_flag: None,
            progress_text: String::new(),
            progress_fraction: 0.0,
        }
    }

    pub fn start_analysis(&mut self) {
        let root = PathBuf::from(self.folder_input.trim());
        if !root.is_dir() {
            self.home_error = Some(format!("Folder does not exist: {}", root.display()));
            return;
        }
        self.home_error = None;
        if self.output_dir_input.trim().is_empty() {
            self.output_dir_input = root.join("Organized").to_string_lossy().into_owned();
        }

        self.push_recent_folder(root.to_string_lossy().into_owned());
        self.thumbs.clear();

        let (prog_tx, prog_rx) = mpsc::channel::<AnalysisProgress>();
        let (res_tx, res_rx) = mpsc::channel::<Result<AnalysisResult, String>>();
        let cancel = Arc::new(AtomicBool::new(false));
        let cancel_worker = Arc::clone(&cancel);

        let sim_cfg = self.sim_config.clone();
        let cluster_cfg = self.cluster_config.clone();
        let cache = self.cache_store.clone();
        let custom_onnx = if self.custom_onnx_path.trim().is_empty() {
            None
        } else {
            Some(PathBuf::from(self.custom_onnx_path.trim()))
        };
        let threads = self.worker_threads;

        thread::spawn(move || {
            let result = run_analysis_pipeline(
                &root,
                &sim_cfg,
                &cluster_cfg,
                &cache,
                custom_onnx.as_deref(),
                Some(threads),
                &cancel_worker,
                |p| {
                    let _ = prog_tx.send(p);
                },
            );
            let _ = res_tx.send(result);
        });

        self.progress_rx = Some(prog_rx);
        self.result_rx = Some(res_rx);
        self.cancel_flag = Some(cancel);
        self.progress_text = "Scanning...".to_string();
        self.progress_fraction = 0.05;
        self.screen = AppScreen::Analyzing;
    }

    fn poll_analysis(&mut self) {
        if let Some(rx) = &self.progress_rx {
            while let Ok(p) = rx.try_recv() {
                self.progress_text = p.label();
                self.progress_fraction = match p {
                    AnalysisProgress::Scanning { .. } => 0.12,
                    AnalysisProgress::AnalyzingFingerprints { done, total } => {
                        if total > 0 {
                            0.15 + 0.55 * (done as f32 / total as f32)
                        } else {
                            0.70
                        }
                    }
                    AnalysisProgress::CalculatingSimilarities => 0.78,
                    AnalysisProgress::RunningOnnx { done, total } => {
                        if total > 0 {
                            0.80 + 0.12 * (done as f32 / total as f32)
                        } else {
                            0.92
                        }
                    }
                    AnalysisProgress::VerifyingSharedDetails => 0.91,
                    AnalysisProgress::BuildingGroups => 0.96,
                    AnalysisProgress::Ready => 1.0,
                };
            }
        }

        let finished = self.result_rx.as_ref().and_then(|rx| rx.try_recv().ok());
        if let Some(res) = finished {
            self.progress_rx = None;
            self.result_rx = None;
            self.cancel_flag = None;
            match res {
                Ok(data) => {
                    let labels = load_evaluation_labels(&self.cache_store);
                    let status_banner = if data.warnings.is_empty() {
                        None
                    } else {
                        Some(data.warnings.join(" | "))
                    };
                    self.onnx_status = data.onnx_status;
                    let initial_groups = data.clustering.groups.clone();
                    let blocked_chaining_edges = data.clustering.blocked_chaining_edges;
                    self.session = Some(SessionData {
                        library_root: data.library_root,
                        images: data.images,
                        features: data.features,
                        pairs: data.pairs,
                        possible_merges: data.clustering.possible_merges.clone(),
                        possible_splits: data.clustering.possible_splits.clone(),
                        initial_clustering: data.clustering,
                        groups: initial_groups,
                        selected_group_idx: 0,
                        selected_members: HashSet::new(),
                        move_target_group_idx: 1,
                        merge_target_group_idx: 1,
                        comparison_pair: None,
                        group_comparison_pair: None,
                        eval_labels: labels,
                        eval_report: None,
                        full_eval_report: None,
                        last_organize_report: None,
                        benchmark: data.benchmark,
                        timings_ms: data.timings_ms,
                        blocked_chaining_edges,
                        status_banner,
                    });
                    self.screen = AppScreen::Review;
                }
                Err(err) => {
                    self.home_error = Some(err);
                    self.screen = AppScreen::Home;
                }
            }
        }
    }

    fn push_recent_folder(&mut self, folder: String) {
        self.recent_folders.retain(|f| f != &folder);
        self.recent_folders.insert(0, folder);
        self.recent_folders.truncate(8);
        save_recent_folders(&self.cache_store, &self.recent_folders);
    }
}

impl eframe::App for FurnitureSorterApp {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        let ctx = ui.ctx().clone();
        if self.thumbs.poll(&ctx) {
            ctx.request_repaint();
        }
        if self.screen == AppScreen::Analyzing {
            self.poll_analysis();
            ctx.request_repaint_after(Duration::from_millis(60));
        }

        match self.screen {
            AppScreen::Home => {
                egui::CentralPanel::default().show(ui, |ui| {
                    home::show_home(self, ui);
                });
            }
            AppScreen::Analyzing => {
                egui::CentralPanel::default().show(ui, |ui| {
                    ui.vertical_centered(|ui| {
                        ui.add_space(120.0);
                        ui.heading("Analyzing Furniture Library");
                        ui.add_space(16.0);
                        ui.label(egui::RichText::new(&self.progress_text).size(16.0));
                        ui.add_space(10.0);
                        ui.add(
                            egui::ProgressBar::new(self.progress_fraction)
                                .desired_width(420.0)
                                .animate(true),
                        );
                        ui.add_space(16.0);
                        if ui.button("Cancel").clicked()
                            && let Some(flag) = &self.cancel_flag
                        {
                            flag.store(true, Ordering::Relaxed);
                            self.progress_text = "Cancelling at safe point...".to_string();
                        }
                    });
                });
            }
            AppScreen::Review => {
                review::show_review(self, ui);
            }
            AppScreen::Settings => {
                settings::show_settings(self, ui);
            }
        }
    }
}

fn recent_file_path(cache_store: &CacheStore) -> PathBuf {
    cache_store.root_dir().join("recent.json")
}

fn load_recent_folders(cache_store: &CacheStore) -> Vec<String> {
    let path = recent_file_path(cache_store);
    let Ok(bytes) = std::fs::read(path) else {
        return Vec::new();
    };
    serde_json::from_slice(&bytes).unwrap_or_default()
}

fn save_recent_folders(cache_store: &CacheStore, folders: &[String]) {
    let _ = std::fs::create_dir_all(cache_store.root_dir());
    if let Ok(json) = serde_json::to_string_pretty(folders) {
        let _ = std::fs::write(recent_file_path(cache_store), json);
    }
}
