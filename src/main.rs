#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use furniture_sorter::app::FurnitureSorterApp;
use furniture_sorter::benchmark::run_analysis_pipeline;
use furniture_sorter::cache::CacheStore;
use furniture_sorter::clustering::ClusteringConfig;
use furniture_sorter::evaluation::{
    build_evaluation_report, load_evaluation_labels, parse_labels_bytes,
};
use furniture_sorter::similarity::SimilarityConfig;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;

fn main() -> eframe::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.iter().any(|a| a == "--benchmark") {
        if let Err(err) = run_cli_benchmark(&args) {
            eprintln!("Benchmark failed: {err}");
            std::process::exit(1);
        }
        return Ok(());
    }

    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1280.0, 820.0])
            .with_min_inner_size([900.0, 620.0])
            .with_title("Furniture Sorter - Verified Details"),
        ..Default::default()
    };

    eframe::run_native(
        "Furniture Sorter",
        options,
        Box::new(|_cc| Ok(Box::new(FurnitureSorterApp::new()))),
    )
}

fn run_cli_benchmark(args: &[String]) -> Result<(), String> {
    let mut folder: Option<PathBuf> = None;
    let mut enable_onnx = false;
    let mut model_path: Option<PathBuf> = None;
    let mut labels_path: Option<PathBuf> = None;
    let mut report_out: Option<PathBuf> = None;
    let mut cache_dir: Option<PathBuf> = None;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--benchmark" => {
                i += 1;
                if i < args.len() {
                    folder = Some(PathBuf::from(&args[i]));
                }
            }
            "--onnx" => enable_onnx = true,
            "--model" => {
                i += 1;
                if i < args.len() {
                    model_path = Some(PathBuf::from(&args[i]));
                }
            }
            "--labels" => {
                i += 1;
                if i < args.len() {
                    labels_path = Some(PathBuf::from(&args[i]));
                }
            }
            "--report" => {
                i += 1;
                if i < args.len() {
                    report_out = Some(PathBuf::from(&args[i]));
                }
            }
            "--cache-dir" => {
                i += 1;
                if i < args.len() {
                    cache_dir = Some(PathBuf::from(&args[i]));
                }
            }
            _ => {}
        }
        i += 1;
    }

    let root = folder.ok_or_else(|| "Usage: --benchmark <folder> [--onnx] [--model <path>] [--labels <path>] [--report <path>]".to_string())?;
    let cache_store = match cache_dir {
        Some(dir) => CacheStore::with_root(dir),
        None => CacheStore::default_store(),
    };

    let mut sim_cfg = SimilarityConfig::default();
    sim_cfg.enable_onnx = enable_onnx;
    let cluster_cfg = ClusteringConfig::default();
    let cancel = Arc::new(AtomicBool::new(false));

    let res = run_analysis_pipeline(
        &root,
        &sim_cfg,
        &cluster_cfg,
        &cache_store,
        model_path.as_deref(),
        None,
        &cancel,
        |p| {
            eprintln!("[progress] {}", p.label());
        },
    )?;

    let labels_store = if let Some(lp) = labels_path {
        let raw = std::fs::read(&lp).map_err(|e| format!("Read labels failed: {e}"))?;
        parse_labels_bytes(&raw)?
    } else {
        load_evaluation_labels(&cache_store)
    };

    let report = build_evaluation_report(
        &res.images,
        &res.features,
        &res.clustering,
        &res.clustering.groups,
        &res.pairs,
        &labels_store,
        &sim_cfg,
        res.benchmark.cache_hit_rate,
        enable_onnx,
        res.benchmark.onnx_images_encoded,
        res.benchmark.approx_feature_memory_kb,
        res.timings_ms,
        0.75,
    );

    let images_summary: Vec<_> = res
        .images
        .iter()
        .map(|img| {
            serde_json::json!({
                "relative_path": &img.relative_path,
                "file_name": &img.file_name,
                "stable_id": img.stable_id(),
            })
        })
        .collect();

    let out = serde_json::json!({
        "benchmark": res.benchmark,
        "evaluation": report,
        "clustering": res.clustering,
        "images": images_summary,
        "onnx_status": res.onnx_status.ui_summary(),
        "warnings": res.warnings,
    });
    let pretty = serde_json::to_string_pretty(&out).map_err(|e| e.to_string())?;

    if let Some(rp) = report_out {
        std::fs::write(&rp, &pretty).map_err(|e| format!("Write report failed: {e}"))?;
    }
    println!("{pretty}");
    Ok(())
}
