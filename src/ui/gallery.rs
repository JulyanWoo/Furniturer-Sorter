use crate::app::SessionData;
use crate::ui::ThumbnailManager;

pub enum GalleryAction {
    CompareTwo(usize, usize),
    CompareWithRep(usize),
}

pub fn show_gallery(
    session: &mut SessionData,
    thumbs: &mut ThumbnailManager,
    ui: &mut egui::Ui,
) -> Option<GalleryAction> {
    let mut action = None;
    let Some(group) = session.groups.get(session.selected_group_idx).cloned() else {
        ui.label("Select a group on the left.");
        return None;
    };

    ui.horizontal(|ui| {
        ui.heading(&group.name);
        ui.label(
            egui::RichText::new(format!("[{}]", group.kind.badge()))
                .color(ui.visuals().weak_text_color()),
        );
        ui.separator();
        if ui.button("Select All").clicked() {
            for &idx in &group.member_indices {
                session.selected_members.insert(idx);
            }
        }
        if ui.button("Clear Selection").clicked() {
            session.selected_members.clear();
        }
        ui.label(format!("{} selected", session.selected_members.len()));
    });
    ui.separator();

    if group.member_indices.is_empty() {
        ui.add_space(24.0);
        ui.label("No images in this group.");
        return None;
    }

    let avail_w = ui.available_width().max(240.0);
    let card_w = 176.0f32;
    let spacing = 10.0f32;
    let cols = ((avail_w + spacing) / (card_w + spacing)).floor().max(1.0) as usize;

    egui::ScrollArea::vertical()
        .auto_shrink([false, false])
        .show(ui, |ui| {
            for row_chunk in group.member_indices.chunks(cols) {
                ui.horizontal_top(|ui| {
                    for &img_idx in row_chunk {
                        let Some(img) = session.images.get(img_idx) else {
                            continue;
                        };
                        let fp = &session.features[img_idx].fingerprint;
                        let is_selected = session.selected_members.contains(&img_idx);
                        let is_rep = group.representative_idx == Some(img_idx);

                        let border_color = if is_selected {
                            egui::Color32::from_rgb(80, 165, 255)
                        } else if is_rep {
                            egui::Color32::from_rgb(235, 185, 65)
                        } else {
                            ui.visuals().widgets.noninteractive.bg_stroke.color
                        };

                        let frame = egui::Frame::group(ui.style())
                            .stroke(egui::Stroke::new(
                                if is_selected { 2.0 } else { 1.0 },
                                border_color,
                            ))
                            .inner_margin(egui::Margin::same(6));

                        frame.show(ui, |ui| {
                            ui.set_width(card_w - 14.0);
                            let thumb_box = egui::vec2(card_w - 16.0, 128.0);
                            let (rect, resp) =
                                ui.allocate_exact_size(thumb_box, egui::Sense::click());

                            if let Some(tex) = thumbs.get_or_request_thumb(img_idx, img) {
                                let tex_size = tex.size_vec2();
                                let scale = (thumb_box.x / tex_size.x.max(1.0))
                                    .min(thumb_box.y / tex_size.y.max(1.0));
                                let draw_size = tex_size * scale;
                                let draw_rect =
                                    egui::Rect::from_center_size(rect.center(), draw_size);
                                ui.painter().image(
                                    tex.id(),
                                    draw_rect,
                                    egui::Rect::from_min_max(
                                        egui::pos2(0.0, 0.0),
                                        egui::pos2(1.0, 1.0),
                                    ),
                                    egui::Color32::WHITE,
                                );
                            } else {
                                ui.painter()
                                    .rect_filled(rect, 4.0, ui.visuals().extreme_bg_color);
                                ui.painter().text(
                                    rect.center(),
                                    egui::Align2::CENTER_CENTER,
                                    "Loading...",
                                    egui::FontId::proportional(11.0),
                                    ui.visuals().weak_text_color(),
                                );
                            }

                            if resp.clicked() {
                                if is_selected {
                                    session.selected_members.remove(&img_idx);
                                } else {
                                    session.selected_members.insert(img_idx);
                                }
                            }
                            if resp.double_clicked() {
                                if let Some(rep_idx) = group.representative_idx
                                    && rep_idx != img_idx
                                {
                                    action = Some(GalleryAction::CompareTwo(rep_idx, img_idx));
                                } else {
                                    action = Some(GalleryAction::CompareWithRep(img_idx));
                                }
                            }

                            ui.horizontal(|ui| {
                                let mut chk = is_selected;
                                if ui.checkbox(&mut chk, "").changed() {
                                    if chk {
                                        session.selected_members.insert(img_idx);
                                    } else {
                                        session.selected_members.remove(&img_idx);
                                    }
                                }
                                if is_rep {
                                    ui.label(
                                        egui::RichText::new("★")
                                            .size(13.0)
                                            .strong()
                                            .color(egui::Color32::from_rgb(235, 185, 65)),
                                    )
                                    .on_hover_text("Representative image of this group");
                                }
                                if img.sidecar.is_some() {
                                    ui.label(
                                        egui::RichText::new("OpenIMG")
                                            .small()
                                            .color(egui::Color32::from_rgb(120, 200, 170)),
                                    );
                                }
                            });

                            ui.add(
                                egui::Label::new(egui::RichText::new(&img.file_name).small())
                                    .truncate(),
                            );
                            ui.label(
                                egui::RichText::new(format!("{}x{}", fp.width, fp.height))
                                    .small()
                                    .color(ui.visuals().weak_text_color()),
                            );
                        });
                    }
                });
                ui.add_space(6.0);
            }
        });

    action
}
