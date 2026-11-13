use crate::app::{AppScreen, FurnitureSorterApp};
use std::path::Path;

pub fn show_home(app: &mut FurnitureSorterApp, ui: &mut egui::Ui) {
    ui.vertical_centered(|ui| {
        ui.add_space(28.0);
        ui.heading(egui::RichText::new("Furniture Sorter").size(28.0).strong());
        ui.label(
            egui::RichText::new(
                "Offline furniture visual grouping • Deterministic engine + optional ONNX embeddings",
            )
            .color(ui.visuals().weak_text_color()),
        );
        ui.add_space(22.0);
    });

    egui::Frame::group(ui.style())
        .inner_margin(egui::Margin::same(18))
        .show(ui, |ui| {
            ui.set_max_width(720.0);
            ui.label(egui::RichText::new("Folder:").strong());
            ui.horizontal(|ui| {
                ui.add_sized(
                    [540.0, 26.0],
                    egui::TextEdit::singleline(&mut app.folder_input)
                        .hint_text("Select a folder containing furniture photographs..."),
                );
                if ui.button("Browse").clicked()
                    && let Some(dir) = rfd::FileDialog::new().pick_folder()
                {
                    app.folder_input = dir.to_string_lossy().into_owned();
                    app.output_dir_input =
                        dir.join("Organized").to_string_lossy().into_owned();
                }
            });

            ui.add_space(12.0);
            ui.checkbox(
                &mut app.sim_config.enable_deterministic,
                "Deterministic similarity (pHash + Lab Color + 4x4 Layout + Sobel Edges + Texture)",
            );
            // Deterministic layer is mandatory per core product architecture
            if !app.sim_config.enable_deterministic {
                app.sim_config.enable_deterministic = true;
            }

            ui.add_space(6.0);
            let onnx_available = app.onnx_status.is_available();
            ui.horizontal(|ui| {
                ui.add_enabled_ui(onnx_available, |ui| {
                    ui.checkbox(&mut app.sim_config.enable_onnx, "ONNX semantic similarity");
                });
                let status_text = app.onnx_status.ui_summary();
                let color = if onnx_available {
                    egui::Color32::from_rgb(90, 190, 110)
                } else {
                    ui.visuals().weak_text_color()
                };
                ui.label(egui::RichText::new(format!("({status_text})")).color(color));
            });

            if !onnx_available {
                app.sim_config.enable_onnx = false;
                ui.label(
                    egui::RichText::new(
                        "Optional: place models/visual_encoder.onnx + onnxruntime.dll beside the executable to enable ONNX.",
                    )
                    .small()
                    .color(ui.visuals().weak_text_color()),
                );
            }

            ui.add_space(16.0);
            ui.horizontal(|ui| {
                let valid_dir = !app.folder_input.trim().is_empty()
                    && Path::new(app.folder_input.trim()).is_dir();
                if ui
                    .add_enabled(
                        valid_dir,
                        egui::Button::new(egui::RichText::new("Analyze").size(15.0).strong())
                            .min_size(egui::vec2(120.0, 32.0)),
                    )
                    .clicked()
                {
                    app.start_analysis();
                }
                if ui.button("Settings & Benchmark").clicked() {
                    app.screen = AppScreen::Settings;
                }
                if app.session.is_some() && ui.button("Return to Current Results").clicked() {
                    app.screen = AppScreen::Review;
                }
            });

            if let Some(banner) = &app.home_error {
                ui.add_space(10.0);
                ui.colored_label(egui::Color32::from_rgb(230, 95, 95), banner);
            }
        });

    if !app.recent_folders.is_empty() {
        ui.add_space(18.0);
        ui.label(egui::RichText::new("Recent Folders:").strong());
        ui.add_space(4.0);
        let recents = app.recent_folders.clone();
        for folder in recents {
            ui.horizontal(|ui| {
                if ui.selectable_label(false, &folder).clicked() {
                    app.folder_input = folder.clone();
                    app.output_dir_input = Path::new(&folder)
                        .join("Organized")
                        .to_string_lossy()
                        .into_owned();
                }
            });
        }
    }
}
