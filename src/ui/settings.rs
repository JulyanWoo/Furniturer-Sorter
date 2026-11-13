use crate::app::{AppScreen, FurnitureSorterApp};
use crate::deterministic::DeterministicWeights;
use crate::embeddings::onnx::detect_onnx_status;
use crate::evaluation::{build_evaluation_report, format_evaluation_markdown, labels_file_path};
use crate::organizer::OrganizeMode;
use crate::similarity::SimilarityConfig;

pub fn show_settings(app: &mut FurnitureSorterApp, root_ui: &mut egui::Ui) {
    let mut back = false;

    egui::Panel::top("settings_top").show(root_ui, |ui| {
        ui.horizontal(|ui| {
            ui.heading("Settings, Benchmark & Evaluation");
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if ui.button("Back").clicked() {
                    back = true;
                }
            });
        });
    });

    if back {
        app.screen = if app.session.is_some() {
            AppScreen::Review
        } else {
            AppScreen::Home
        };
        return;
    }

    egui::CentralPanel::default().show(root_ui, |ui| {
        egui::ScrollArea::vertical().show(ui, |ui| {
            ui.columns(2, |cols| {
                let left = &mut cols[0];
                left.heading("Deterministic Engine Weights");
                left.add(egui::Slider::new(&mut app.sim_config.weights.color, 0.0..=1.0).text("Color"));
                left.add(egui::Slider::new(&mut app.sim_config.weights.layout, 0.0..=1.0).text("Spatial Grid 4x4"));
                left.add(egui::Slider::new(&mut app.sim_config.weights.edges, 0.0..=1.0).text("Sobel Edges"));
                left.add(egui::Slider::new(&mut app.sim_config.weights.phash, 0.0..=1.0).text("Perceptual Hash"));
                left.add(egui::Slider::new(&mut app.sim_config.weights.texture, 0.0..=1.0).text("Texture (CS-LBP)"));
                if left.button("Reset Weights & Thresholds").clicked() {
                    app.sim_config.weights = DeterministicWeights::default();
                    let keep_det = app.sim_config.enable_deterministic;
                    let keep_onnx = app.sim_config.enable_onnx;
                    app.sim_config = SimilarityConfig::default();
                    app.sim_config.enable_deterministic = keep_det;
                    app.sim_config.enable_onnx = keep_onnx;
                }

                left.add_space(12.0);
                left.heading("Project grouping");
                left.label("Shared physical details must agree across independent views.");
                left.label("Automatic group merging is OFF. Suggestions require review.");
                left.checkbox(&mut app.cluster_config.respect_human_corrections, "Keep confirmed project corrections");
                egui::CollapsingHeader::new("Advanced group admission").show(left, |ui| {
                    let a = &mut app.cluster_config.admission;
                    ui.add(egui::Slider::new(&mut a.majority_fraction, 0.60..=1.0).text("Member agreement fraction"));
                    ui.label("Copies of the same viewpoint count as one vote.");
                });
                egui::CollapsingHeader::new("Retrieval and comparison scores").show(left, |left| {
                left.label("These scores retrieve candidates; they do not establish project membership.");
                left.add(
                    egui::Slider::new(&mut app.cluster_config.same_project_threshold, 0.50..=0.95)
                        .text("Same-project threshold"),
                );
                app.sim_config.same_project_threshold = app.cluster_config.same_project_threshold;
                left.add(
                    egui::Slider::new(&mut app.cluster_config.review_threshold, 0.40..=0.85)
                        .text("Review threshold"),
                );
                app.sim_config.review_threshold = app.cluster_config.review_threshold;
                left.add(
                    egui::Slider::new(&mut app.sim_config.ambiguous_low, 0.20..=0.65)
                        .text("ONNX ambiguous low"),
                );
                left.add(
                    egui::Slider::new(&mut app.sim_config.ambiguous_high, 0.70..=0.98)
                        .text("ONNX ambiguous high"),
                );
                left.add(
                    egui::Slider::new(&mut app.sim_config.deterministic_blend, 0.0..=1.0)
                        .text("Hybrid Deterministic weight"),
                );
                left.add(
                    egui::Slider::new(&mut app.sim_config.embedding_blend, 0.0..=1.0)
                        .text("Hybrid Embedding weight"),
                );
                });

                let right = &mut cols[1];
                right.heading("Performance, Cache & ONNX");
                let max_cpus = std::thread::available_parallelism()
                    .map(|n| n.get())
                    .unwrap_or(8);
                right.add(
                    egui::Slider::new(&mut app.worker_threads, 1..=max_cpus)
                        .text("Worker threads"),
                );
                right.label(format!(
                    "Cache directory: {}",
                    app.cache_store.root_dir().display()
                ));
                if right.button("Clear Current Library Cache").clicked()
                    && !app.folder_input.trim().is_empty()
                {
                    let _ = app
                        .cache_store
                        .clear_library_cache(std::path::Path::new(app.folder_input.trim()));
                }

                right.add_space(10.0);
                right.label("Custom ONNX model path (optional):");
                right.horizontal(|ui| {
                    ui.add(
                        egui::TextEdit::singleline(&mut app.custom_onnx_path)
                            .desired_width(240.0)
                            .hint_text("models/visual_encoder.onnx"),
                    );
                    if ui.button("Browse").clicked()
                        && let Some(file) = rfd::FileDialog::new()
                            .add_filter("ONNX Model", &["onnx"])
                            .pick_file()
                    {
                        app.custom_onnx_path = file.to_string_lossy().into_owned();
                        app.onnx_status = detect_onnx_status(Some(file.as_path()));
                    }
                    if ui.button("Refresh").clicked() {
                        let p = if app.custom_onnx_path.trim().is_empty() {
                            None
                        } else {
                            Some(std::path::Path::new(app.custom_onnx_path.trim()))
                        };
                        app.onnx_status = detect_onnx_status(p);
                    }
                });
                right.label(format!("ONNX status: {}", app.onnx_status.ui_summary()));

                right.add_space(12.0);
                right.heading("Source File Safety");
                right.checkbox(
                    &mut app.move_confirmation_armed,
                    "Unlock advanced MOVE mode (source images stay untouched in default COPY mode)",
                );
                if app.move_confirmation_armed {
                    right.horizontal(|ui| {
                        ui.radio_value(
                            &mut app.organize_options.mode,
                            OrganizeMode::Copy,
                            "COPY (Recommended)",
                        );
                        ui.radio_value(
                            &mut app.organize_options.mode,
                            OrganizeMode::Move,
                            "MOVE (Requires confirmation)",
                        );
                    });
                } else {
                    app.organize_options.mode = OrganizeMode::Copy;
                }

                right.add_space(14.0);
                right.heading("Developer Benchmark & Evaluation");
                if let Some(session) = &mut app.session {
                    let b = &session.benchmark;
                    right.label(format!(
                        "Images: {}   |   Threads: {}   |   Approx feature RAM: {} KB",
                        b.images, b.threads_used, b.approx_feature_memory_kb
                    ));
                    right.label(format!(
                        "Timings — Scan: {} ms | Fingerprints: {} ms | ONNX: {} ms ({} imgs) | Similarity: {} ms | Clustering: {} ms | Total: {} ms",
                        b.scan_ms,
                        b.fingerprint_ms,
                        b.embedding_ms,
                        b.onnx_images_encoded,
                        b.similarity_ms,
                        b.clustering_ms,
                        b.total_ms
                    ));
                    right.label(format!(
                        "Cache hits: {} / {} ({:.1}%)   |   Prefilter kept: {} / {} pairs",
                        b.cache_hits,
                        b.cache_hits + b.cache_misses,
                        b.cache_hit_rate * 100.0,
                        b.prefilter_passed_pairs,
                        b.prefilter_total_pairs
                    ));

                    right.add_space(6.0);
                    right.checkbox(
                        &mut app.capture_eval_labels,
                        "Capture ground-truth labels from manual group corrections",
                    );
                    let (same_cnt, diff_cnt) = session.eval_labels.count_by_label();
                    right.label(format!(
                        "Labels ({} SameProject, {} DifferentProject): {}",
                        same_cnt,
                        diff_cnt,
                        labels_file_path(&app.cache_store).display()
                    ));

                    right.horizontal(|ui| {
                        if ui.button("Compute Evaluation Metrics").clicked() {
                            let full = build_evaluation_report(
                                &session.images,
                                &session.features,
                                &session.initial_clustering,
                                &session.groups,
                                &session.pairs,
                                &session.eval_labels,
                                &app.sim_config,
                                session.benchmark.cache_hit_rate,
                                app.sim_config.enable_onnx,
                                session.benchmark.onnx_images_encoded,
                                session.benchmark.approx_feature_memory_kb,
                                session.timings_ms.clone(),
                                0.75,
                            );
                            session.eval_report = full.labeled_comparison.clone();
                            session.full_eval_report = Some(full);
                        }
                        if ui.button("Export evaluation_report.json + .md").clicked() {
                            let full = build_evaluation_report(
                                &session.images,
                                &session.features,
                                &session.initial_clustering,
                                &session.groups,
                                &session.pairs,
                                &session.eval_labels,
                                &app.sim_config,
                                session.benchmark.cache_hit_rate,
                                app.sim_config.enable_onnx,
                                session.benchmark.onnx_images_encoded,
                                session.benchmark.approx_feature_memory_kb,
                                session.timings_ms.clone(),
                                0.75,
                            );
                            session.eval_report = full.labeled_comparison.clone();
                            let eval_dir = app.cache_store.eval_dir();
                            let _ = std::fs::create_dir_all(&eval_dir);
                            if let Ok(json) = serde_json::to_string_pretty(&full) {
                                let _ = std::fs::write(eval_dir.join("evaluation_report.json"), json);
                                let _ = std::fs::write(
                                    eval_dir.join("evaluation_report.md"),
                                    format_evaluation_markdown(&full),
                                );
                                session.status_banner = Some(format!(
                                    "Saved evaluation_report.json & .md in {}",
                                    eval_dir.display()
                                ));
                            }
                            session.full_eval_report = Some(full);
                        }
                    });

                    let fmt_pct = |v: Option<f32>| match v {
                        Some(x) => format!("{:.1}%", x * 100.0),
                        None => "N/A".to_string(),
                    };

                    if let Some(full) = &session.full_eval_report {
                        right.label(format!(
                            "AutoGrouped: {} | Review: {} | Ungrouped: {} | ConfirmedGroups: {}",
                            full.auto_grouped_count,
                            full.review_count,
                            full.ungrouped_count,
                            full.confirmed_groups
                        ));
                        right.label(format!(
                            "Group-level: FalseGroupMerges = {} | FalseGroupSplits = {}",
                            full.group_metrics.false_group_merges,
                            full.group_metrics.false_group_splits
                        ));
                        right.label(format!(
                            "Gated ONNX Policy ({:.2}..{:.2}): ONNX Avoided {:.1}% | FalseMerge {} | FalseSplit {}",
                            full.onnx_policy_simulation.low_threshold,
                            full.onnx_policy_simulation.high_threshold,
                            full.onnx_policy_simulation.onnx_inference_avoided_percent,
                            fmt_pct(full.onnx_policy_simulation.false_merge_rate),
                            fmt_pct(full.onnx_policy_simulation.false_split_rate)
                        ));
                        right.label(format!(
                            "AmbiguousPairs logged: {}",
                            full.ambiguous_pairs.len()
                        ));
                    }

                    if let Some(rep) = &session.eval_report {
                        for m in [
                            Some(&rep.deterministic),
                            rep.mobilenet.as_ref(),
                            rep.hybrid.as_ref(),
                        ]
                        .into_iter()
                        .flatten()
                        {
                            right.label(format!(
                                "{}: Prec {} | Rec {} | FalseMerge {} ({}) | FalseSplit {} ({})",
                                m.mode_name,
                                fmt_pct(m.precision),
                                fmt_pct(m.recall),
                                fmt_pct(m.false_merge_rate),
                                m.false_merges,
                                fmt_pct(m.false_split_rate),
                                m.false_splits
                            ));
                        }
                    }
                } else {
                    right.label("Analyze a folder first to view benchmark & evaluation metrics.");
                }
            });
        });
    });
}
