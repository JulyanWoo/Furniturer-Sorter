use crate::app::{AppScreen, FurnitureSorterApp, SessionData};
use crate::clustering::{GroupKind, MergeRank, PossibleMergeSuggestion};
use crate::evaluation::{
    GroundTruthLabel, LabelSource, build_evaluation_report, capture_confirm_group,
    capture_keep_separate_groups, capture_merge_groups, capture_move_members,
    capture_remove_from_group, capture_split_group, evaluate_labeled_pairs,
    format_evaluation_markdown, labels_file_path, save_evaluation_labels,
};
use crate::organizer::{OrganizeMode, organize_groups};
use crate::similarity::compare_pair_direct;
use crate::ui::gallery::{GalleryAction, show_gallery};

pub fn show_review(app: &mut FurnitureSorterApp, root_ui: &mut egui::Ui) {
    let ctx = root_ui.ctx().clone();
    let mut trigger_organize = false;
    let mut trigger_export_eval = false;
    let mut go_home = false;
    let mut go_settings = false;

    egui::Panel::top("review_top_bar").show(root_ui, |ui| {
        let Some(session) = app.session.as_ref() else {
            return;
        };
        ui.horizontal_wrapped(|ui| {
            ui.heading("Furniture Sorter");
            ui.separator();
            ui.label(
                egui::RichText::new(session.library_root.to_string_lossy().as_ref())
                    .small()
                    .color(ui.visuals().weak_text_color()),
            );
            ui.separator();
            ui.label(format!(
                "{} images • {} groups • Cache {:.0}% • {} ms",
                session.benchmark.images,
                session
                    .groups
                    .iter()
                    .filter(|g| matches!(
                        g.kind,
                        GroupKind::FurnitureProject
                            | GroupKind::CandidateGroup
                            | GroupKind::Review
                            | GroupKind::Duplicates
                    ))
                    .count(),
                session.benchmark.cache_hit_rate * 100.0,
                session.benchmark.total_ms
            ));

            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if ui.button("Home").clicked() {
                    go_home = true;
                }
                if ui.button("Settings & Benchmark").clicked() {
                    go_settings = true;
                }
            });
        });
    });

    if go_home {
        app.screen = AppScreen::Home;
        return;
    }
    if go_settings {
        app.screen = AppScreen::Settings;
        return;
    }

    // Left panel: Groups list
    egui::Panel::left("groups_panel")
        .resizable(true)
        .default_size(235.0)
        .show(root_ui, |ui| {
            let Some(session) = app.session.as_mut() else {
                return;
            };
            ui.heading("Groups");
            ui.separator();

            egui::ScrollArea::vertical().show(ui, |ui| {
                let sections = [
                    (
                        "HIGH CONFIDENCE",
                        egui::Color32::from_rgb(100, 190, 240),
                        vec![GroupKind::FurnitureProject],
                    ),
                    (
                        "PROBABLE",
                        egui::Color32::from_rgb(235, 185, 65),
                        vec![GroupKind::CandidateGroup, GroupKind::Review],
                    ),
                    (
                        "DUPLICATES",
                        egui::Color32::from_rgb(190, 140, 240),
                        vec![GroupKind::Duplicates],
                    ),
                    (
                        "UNMATCHED",
                        ui.visuals().weak_text_color(),
                        vec![GroupKind::Ungrouped],
                    ),
                ];

                for (header, color, kinds) in sections {
                    let indices: Vec<usize> = session
                        .groups
                        .iter()
                        .enumerate()
                        .filter(|(_, g)| kinds.contains(&g.kind) && !g.member_indices.is_empty())
                        .map(|(i, _)| i)
                        .collect();

                    if indices.is_empty() {
                        continue;
                    }

                    ui.add_space(6.0);
                    ui.label(egui::RichText::new(header).strong().color(color).size(12.0));
                    for idx in indices {
                        let g = &session.groups[idx];
                        let selected = idx == session.selected_group_idx;
                        let prefix = if g.confirmed { "✓ " } else { "" };
                        let text = format!("{prefix}{} ({})", g.name, g.member_indices.len());
                        let resp = ui.selectable_label(selected, text);
                        if resp.clicked() {
                            session.selected_group_idx = idx;
                            session.selected_members.clear();
                        }
                    }
                }
            });
        });

    // Right panel: Group info, manual corrections, safe COPY organization
    egui::Panel::right("inspector_panel")
        .resizable(true)
        .default_size(300.0)
        .show(root_ui, |ui| {
            let Some(session) = app.session.as_mut() else {
                return;
            };

            egui::ScrollArea::vertical().show(ui, |ui| {
                ui.heading("Group Details");
                ui.separator();

                let sel_idx = session.selected_group_idx;
                if let Some(group) = session.groups.get(sel_idx).cloned() {
                    let confidence_pct = (group.avg_similarity * 100.0).round();
                    ui.label(egui::RichText::new(&group.name).strong().size(16.0));
                    ui.label(format!("Category: {}", group.kind.badge()));
                    if group.member_indices.len() > 1 {
                        ui.label(format!("Similarity = {confidence_pct:.0}%"));
                    }
                    ui.label(format!("Members: {}", group.member_indices.len()));
                    ui.label(format!(
                        "Status: {}",
                        if group.confirmed {
                            "Confirmed"
                        } else {
                            "Unconfirmed"
                        }
                    ));

                    if let Some(rep_idx) = group.representative_idx
                        && let Some(rep_img) = session.images.get(rep_idx)
                    {
                        ui.add_space(4.0);
                        ui.label(
                            egui::RichText::new("★ Representative:")
                                .strong()
                                .color(egui::Color32::from_rgb(235, 185, 65)),
                        );
                        ui.label(egui::RichText::new(&rep_img.file_name).small());
                    }

                    if let Some((rel_name, rel_sim)) = group.nearest_related.as_ref() {
                        ui.add_space(4.0);
                        ui.label(
                            egui::RichText::new(format!(
                                "Related: {rel_name} — {:.0}%",
                                rel_sim * 100.0
                            ))
                            .color(egui::Color32::from_rgb(100, 190, 240))
                            .small(),
                        );
                    }

                    ui.add_space(10.0);
                    ui.heading("Actions");
                    ui.separator();

                    let confirm_label = if group.confirmed {
                        "Unconfirm Group"
                    } else {
                        "Confirm Group"
                    };
                    if ui.button(confirm_label).clicked() {
                        let mut captured = false;
                        if let Some(g) = session.groups.get_mut(sel_idx) {
                            g.confirmed = !g.confirmed;
                            if g.confirmed && app.capture_eval_labels {
                                capture_confirm_group(&mut session.eval_labels, g, &session.images);
                                captured = true;
                            }
                        }
                        if captured {
                            let _ = save_evaluation_labels(&app.cache_store, &session.eval_labels);
                        }
                    }

                    let selected_vec: Vec<usize> = {
                        let mut v: Vec<usize> = session.selected_members.iter().copied().collect();
                        v.sort_unstable();
                        v
                    };

                    ui.add_space(6.0);
                    ui.add_enabled_ui(selected_vec.len() == 2, |ui| {
                        if ui.button("Compare 2 Selected Images").clicked() {
                            session.comparison_pair = Some((selected_vec[0], selected_vec[1]));
                            app.thumbs.clear_compare_cache();
                        }
                    });
                    if selected_vec.len() == 1
                        && let Some(rep_idx) = group.representative_idx
                        && rep_idx != selected_vec[0]
                        && ui.button("Compare Selected vs Representative").clicked()
                    {
                        session.comparison_pair = Some((rep_idx, selected_vec[0]));
                        app.thumbs.clear_compare_cache();
                    }

                    ui.add_space(6.0);
                    ui.add_enabled_ui(selected_vec.len() == 1, |ui| {
                        if ui.button("Set Selected as Representative").clicked()
                            && let Some(g) = session.groups.get_mut(sel_idx)
                        {
                            g.representative_idx = Some(selected_vec[0]);
                        }
                    });

                    // Move selected to another group
                    if session.move_target_group_idx >= session.groups.len()
                        || session.move_target_group_idx == sel_idx
                    {
                        session.move_target_group_idx = (0..session.groups.len())
                            .find(|&i| i != sel_idx)
                            .unwrap_or(0);
                    }

                    ui.add_space(6.0);
                    ui.label("Move selected to:");
                    ui.horizontal(|ui| {
                        let target_name = session
                            .groups
                            .get(session.move_target_group_idx)
                            .map(|g| g.name.as_str())
                            .unwrap_or("-");
                        egui::ComboBox::from_id_salt("move_target_combo")
                            .selected_text(target_name)
                            .width(160.0)
                            .show_ui(ui, |ui| {
                                for (i, g) in session.groups.iter().enumerate() {
                                    if i == sel_idx {
                                        continue;
                                    }
                                    ui.selectable_value(
                                        &mut session.move_target_group_idx,
                                        i,
                                        &g.name,
                                    );
                                }
                            });
                        let can_move =
                            !selected_vec.is_empty() && session.move_target_group_idx != sel_idx;
                        if ui
                            .add_enabled(can_move, egui::Button::new("Move"))
                            .clicked()
                        {
                            let target = session.move_target_group_idx;
                            if app.capture_eval_labels
                                && let (Some(src), Some(dst)) =
                                    (session.groups.get(sel_idx), session.groups.get(target))
                            {
                                capture_move_members(
                                    &mut session.eval_labels,
                                    src,
                                    dst,
                                    &selected_vec,
                                    &session.images,
                                );
                                let _ =
                                    save_evaluation_labels(&app.cache_store, &session.eval_labels);
                            }
                            move_members_between_groups(session, sel_idx, target, &selected_vec);
                        }
                    });

                    ui.add_space(4.0);
                    ui.add_enabled_ui(!selected_vec.is_empty(), |ui| {
                        if ui.button("Create New Group from Selected").clicked() {
                            if app.capture_eval_labels
                                && let Some(src) = session.groups.get(sel_idx)
                            {
                                capture_split_group(
                                    &mut session.eval_labels,
                                    src,
                                    &selected_vec,
                                    &session.images,
                                );
                                let _ =
                                    save_evaluation_labels(&app.cache_store, &session.eval_labels);
                            }
                            create_new_group_from_selected(session, sel_idx, &selected_vec);
                        }
                        if ui.button("Split Selected Out").clicked() {
                            if app.capture_eval_labels
                                && let Some(src) = session.groups.get(sel_idx)
                            {
                                capture_split_group(
                                    &mut session.eval_labels,
                                    src,
                                    &selected_vec,
                                    &session.images,
                                );
                                let _ =
                                    save_evaluation_labels(&app.cache_store, &session.eval_labels);
                            }
                            create_new_group_from_selected(session, sel_idx, &selected_vec);
                        }
                        if ui.button("Remove Selected (to Ungrouped)").clicked()
                            && let Some(ungrouped_idx) = session
                                .groups
                                .iter()
                                .position(|g| g.kind == GroupKind::Ungrouped)
                            && ungrouped_idx != sel_idx
                        {
                            if app.capture_eval_labels
                                && let Some(src) = session.groups.get(sel_idx)
                            {
                                capture_remove_from_group(
                                    &mut session.eval_labels,
                                    src,
                                    &selected_vec,
                                    &session.images,
                                );
                                let _ =
                                    save_evaluation_labels(&app.cache_store, &session.eval_labels);
                            }
                            move_members_between_groups(
                                session,
                                sel_idx,
                                ungrouped_idx,
                                &selected_vec,
                            );
                        }
                        if ui.button("Mark Uncertain (to Review)").clicked()
                            && let Some(review_idx) = session
                                .groups
                                .iter()
                                .position(|g| g.kind == GroupKind::Review)
                            && review_idx != sel_idx
                        {
                            move_members_between_groups(
                                session,
                                sel_idx,
                                review_idx,
                                &selected_vec,
                            );
                        }
                    });

                    // Merge current group with another project group
                    let can_merge_source = matches!(
                        group.kind,
                        GroupKind::FurnitureProject
                            | GroupKind::CandidateGroup
                            | GroupKind::Review
                            | GroupKind::Duplicates
                    );
                    if can_merge_source && session.groups.len() > 1 {
                        if session.merge_target_group_idx >= session.groups.len()
                            || session.merge_target_group_idx == sel_idx
                        {
                            session.merge_target_group_idx = (0..session.groups.len())
                                .find(|&i| {
                                    i != sel_idx
                                        && matches!(
                                            session.groups[i].kind,
                                            GroupKind::FurnitureProject
                                                | GroupKind::CandidateGroup
                                                | GroupKind::Review
                                                | GroupKind::Duplicates
                                        )
                                })
                                .unwrap_or(0);
                        }
                        ui.add_space(8.0);
                        ui.label("Merge group into:");
                        ui.horizontal(|ui| {
                            let target_name = session
                                .groups
                                .get(session.merge_target_group_idx)
                                .map(|g| g.name.as_str())
                                .unwrap_or("-");
                            egui::ComboBox::from_id_salt("merge_target_combo")
                                .selected_text(target_name)
                                .width(160.0)
                                .show_ui(ui, |ui| {
                                    for (i, g) in session.groups.iter().enumerate() {
                                        if i == sel_idx {
                                            continue;
                                        }
                                        if matches!(
                                            g.kind,
                                            GroupKind::FurnitureProject
                                                | GroupKind::CandidateGroup
                                                | GroupKind::Review
                                                | GroupKind::Duplicates
                                        ) {
                                            ui.selectable_value(
                                                &mut session.merge_target_group_idx,
                                                i,
                                                &g.name,
                                            );
                                        }
                                    }
                                });
                            let target = session.merge_target_group_idx;
                            let valid_merge = target != sel_idx
                                && target < session.groups.len()
                                && matches!(
                                    session.groups[target].kind,
                                    GroupKind::FurnitureProject
                                        | GroupKind::CandidateGroup
                                        | GroupKind::Review
                                        | GroupKind::Duplicates
                                );
                            if ui
                                .add_enabled(valid_merge, egui::Button::new("Merge"))
                                .clicked()
                            {
                                if app.capture_eval_labels
                                    && let (Some(src), Some(dst)) =
                                        (session.groups.get(sel_idx), session.groups.get(target))
                                {
                                    capture_merge_groups(
                                        &mut session.eval_labels,
                                        src,
                                        dst,
                                        &session.images,
                                    );
                                    let _ = save_evaluation_labels(
                                        &app.cache_store,
                                        &session.eval_labels,
                                    );
                                }
                                merge_groups(session, sel_idx, target);
                            }
                        });
                    }
                }

                // Possible Merges Section
                let active_suggestions: Vec<(usize, PossibleMergeSuggestion)> = session
                    .possible_merges
                    .iter()
                    .enumerate()
                    .filter(|(_, m)| !m.dismissed)
                    .map(|(i, m)| (i, m.clone()))
                    .collect();

                if !active_suggestions.is_empty() {
                    ui.add_space(14.0);
                    ui.heading(format!("Possible Merges ({})", active_suggestions.len()));
                    ui.separator();

                    let mut merge_action: Option<(String, String)> = None;
                    let mut dismiss_action: Option<(String, String, usize)> = None;
                    let mut compare_action: Option<(String, String)> = None;

                    let render_item = |ui: &mut egui::Ui,
                                       suggestion: &PossibleMergeSuggestion,
                                       s_idx: usize,
                                       compare_action: &mut Option<(String, String)>,
                                       merge_action: &mut Option<(String, String)>,
                                       dismiss_action: &mut Option<(String, String, usize)>| {
                        ui.group(|ui| {
                            ui.horizontal(|ui| {
                                let badge = match suggestion.rank {
                                    MergeRank::VeryStrong => "VERY STRONG",
                                    MergeRank::Strong => "STRONG",
                                    MergeRank::Possible => "POSSIBLE",
                                };
                                let col = match suggestion.rank {
                                    MergeRank::VeryStrong => egui::Color32::from_rgb(100, 220, 140),
                                    MergeRank::Strong => egui::Color32::from_rgb(100, 190, 240),
                                    MergeRank::Possible => ui.visuals().weak_text_color(),
                                };
                                ui.label(egui::RichText::new(badge).small().color(col));
                                ui.label(
                                    egui::RichText::new(format!(
                                        "{} ↔ {}",
                                        suggestion.group_a_name, suggestion.group_b_name
                                    ))
                                    .strong(),
                                );
                            });

                            let id_pct = (suggestion.project_identity * 100.0).round();
                            let contra_pct = (suggestion.contradiction * 100.0).round();
                            ui.label(
                                egui::RichText::new(format!(
                                    "Family: {:.0}% | Identity: {id_pct:.0}% | Contradiction: {contra_pct:.0}%", suggestion.breakdown.family_similarity * 100.0
                                ))
                                .small()
                                .color(if contra_pct > 25.0 {
                                    egui::Color32::from_rgb(230, 120, 120)
                                } else {
                                    egui::Color32::from_rgb(140, 210, 150)
                                }),
                            );

                            let breakdown_str = format!(
                                "Geometry {:.0}% • Cross-member support {:.0}% • Material {:.0}%{}",
                                suggestion.breakdown.structure * 100.0,
                                suggestion.breakdown.cross_member_support * 100.0,
                                suggestion.breakdown.color * 100.0,
                                if let Some(emb) = suggestion.breakdown.embedding {
                                    format!(" • Embedding {:.0}%", emb * 100.0)
                                } else {
                                    String::new()
                                }
                            );
                            ui.label(
                                egui::RichText::new(breakdown_str)
                                    .small()
                                    .color(ui.visuals().weak_text_color()),
                            );

                            ui.horizontal(|ui| {
                                if ui.small_button("Compare").clicked() {
                                    *compare_action = Some((
                                        suggestion.group_a_id.clone(),
                                        suggestion.group_b_id.clone(),
                                    ));
                                }
                                if ui.small_button("Merge").clicked() {
                                    *merge_action = Some((
                                        suggestion.group_a_id.clone(),
                                        suggestion.group_b_id.clone(),
                                    ));
                                }
                                if ui.small_button("Dismiss").clicked() {
                                    *dismiss_action = Some((
                                        suggestion.group_a_id.clone(),
                                        suggestion.group_b_id.clone(),
                                        s_idx,
                                    ));
                                }
                            });
                        });
                    };

                    let very_strong: Vec<_> = active_suggestions
                        .iter()
                        .filter(|(_, m)| m.rank == MergeRank::VeryStrong)
                        .collect();
                    let strong: Vec<_> = active_suggestions
                        .iter()
                        .filter(|(_, m)| m.rank == MergeRank::Strong)
                        .collect();
                    let possible: Vec<_> = active_suggestions
                        .iter()
                        .filter(|(_, m)| m.rank == MergeRank::Possible)
                        .collect();

                    if !very_strong.is_empty() {
                        ui.label(
                            egui::RichText::new(format!("★ VERY STRONG ({})", very_strong.len()))
                                .strong()
                                .color(egui::Color32::from_rgb(100, 220, 140)),
                        );
                        for (s_idx, suggestion) in very_strong {
                            render_item(
                                ui,
                                suggestion,
                                *s_idx,
                                &mut compare_action,
                                &mut merge_action,
                                &mut dismiss_action,
                            );
                        }
                    }

                    if !strong.is_empty() {
                        egui::CollapsingHeader::new(format!("STRONG ({})", strong.len()))
                            .id_salt("strong_merge_suggestions").default_open(false).show(ui, |ui| {
                            for (s_idx, suggestion) in strong {
                                render_item(ui, suggestion, *s_idx, &mut compare_action,
                                    &mut merge_action, &mut dismiss_action);
                            }
                        });
                    }

                    if !possible.is_empty() {
                        ui.add_space(4.0);
                        egui::CollapsingHeader::new(format!("Advanced: POSSIBLE ({})", possible.len()))
                            .default_open(false)
                            .show(ui, |ui| {
                                for (s_idx, suggestion) in possible {
                                    render_item(
                                        ui,
                                        suggestion,
                                        *s_idx,
                                        &mut compare_action,
                                        &mut merge_action,
                                        &mut dismiss_action,
                                    );
                                }
                            });
                    }

                    if let Some((id_a, id_b)) = compare_action {
                        session.group_comparison_pair = Some((id_a, id_b));
                    }
                    if let Some((id_a, id_b, s_idx)) = dismiss_action {
                        if app.capture_eval_labels {
                            let g_a = session.groups.iter().find(|g| g.id == id_a);
                            let g_b = session.groups.iter().find(|g| g.id == id_b);
                            if let (Some(ga), Some(gb)) = (g_a, g_b) {
                                capture_keep_separate_groups(
                                    &mut session.eval_labels,
                                    ga,
                                    gb,
                                    &session.images,
                                );
                                let _ =
                                    save_evaluation_labels(&app.cache_store, &session.eval_labels);
                            }
                        }
                        if let Some(m) = session.possible_merges.get_mut(s_idx) {
                            m.dismissed = true;
                        }
                    }
                    if let Some((id_a, id_b)) = merge_action {
                        let idx_a = session.groups.iter().position(|g| g.id == id_a);
                        let idx_b = session.groups.iter().position(|g| g.id == id_b);
                        if let (Some(ia), Some(ib)) = (idx_a, idx_b) {
                            if app.capture_eval_labels
                                && let (Some(src), Some(dst)) =
                                    (session.groups.get(ib), session.groups.get(ia))
                            {
                                capture_merge_groups(
                                    &mut session.eval_labels,
                                    src,
                                    dst,
                                    &session.images,
                                );
                                let _ =
                                    save_evaluation_labels(&app.cache_store, &session.eval_labels);
                            }
                            merge_groups(session, ib, ia);
                            session
                                .possible_merges
                                .retain(|m| m.group_a_id != id_b && m.group_b_id != id_b);
                        }
                    }
                }

                // Possible Splits Section
                if !session.possible_splits.is_empty() {
                    ui.add_space(14.0);
                    ui.heading(format!(
                        "Possible Splits ({})",
                        session.possible_splits.len()
                    ));
                    ui.separator();

                    let mut remove_outlier_action: Option<(String, usize)> = None;
                    let mut dismiss_split_idx: Option<usize> = None;

                    for (ps_idx, split) in session.possible_splits.iter().enumerate().take(6) {
                        ui.group(|ui| {
                            ui.label(egui::RichText::new(&split.group_name).strong());
                            ui.label(
                                egui::RichText::new(format!(
                                    "Contradiction: {:.0}% • {}",
                                    split.cross_contradiction * 100.0,
                                    split.reason
                                ))
                                .small()
                                .color(egui::Color32::from_rgb(240, 140, 100)),
                            );

                            if let Some(outlier) = split.outlier_idx {
                                if let Some(img) = session.images.get(outlier) {
                                    ui.label(
                                        egui::RichText::new(format!(
                                            "Potential Outlier: {}",
                                            img.file_name
                                        ))
                                        .small(),
                                    );
                                }
                            }

                            ui.horizontal(|ui| {
                                if let Some(outlier) = split.outlier_idx {
                                    if ui.small_button("Remove Outlier").clicked() {
                                        remove_outlier_action =
                                            Some((split.group_id.clone(), outlier));
                                        dismiss_split_idx = Some(ps_idx);
                                    }
                                }
                                if ui.small_button("Dismiss").clicked() {
                                    dismiss_split_idx = Some(ps_idx);
                                }
                            });
                        });
                    }

                    if let Some((gid, outlier)) = remove_outlier_action {
                        if let Some(g_idx) = session.groups.iter().position(|g| g.id == gid) {
                            if let Some(ungrouped_idx) = session
                                .groups
                                .iter()
                                .position(|g| g.kind == GroupKind::Ungrouped)
                            {
                                if app.capture_eval_labels
                                    && let Some(src) = session.groups.get(g_idx)
                                {
                                    capture_remove_from_group(
                                        &mut session.eval_labels,
                                        src,
                                        &[outlier],
                                        &session.images,
                                    );
                                    let _ = save_evaluation_labels(
                                        &app.cache_store,
                                        &session.eval_labels,
                                    );
                                }
                                move_members_between_groups(
                                    session,
                                    g_idx,
                                    ungrouped_idx,
                                    &[outlier],
                                );
                            }
                        }
                    }

                    if let Some(ps_idx) = dismiss_split_idx {
                        if ps_idx < session.possible_splits.len() {
                            session.possible_splits.remove(ps_idx);
                        }
                    }
                }

                ui.add_space(16.0);
                ui.heading("Organize Files");
                ui.separator();

                ui.label("Output folder:");
                ui.horizontal(|ui| {
                    ui.add(
                        egui::TextEdit::singleline(&mut app.output_dir_input).desired_width(185.0),
                    );
                    if ui.button("Browse").clicked()
                        && let Some(dir) = rfd::FileDialog::new().pick_folder()
                    {
                        app.output_dir_input = dir.to_string_lossy().into_owned();
                    }
                });

                ui.checkbox(
                    &mut app.organize_options.include_review,
                    "Include Review folder",
                );
                ui.checkbox(
                    &mut app.organize_options.include_ungrouped,
                    "Include Ungrouped folder",
                );

                let mode_str = match app.organize_options.mode {
                    OrganizeMode::Copy => "Safe COPY (source files untouched)",
                    OrganizeMode::Move => "MOVE (advanced — removes originals after copy)",
                };
                ui.label(egui::RichText::new(format!("Mode: {mode_str}")).small());

                ui.add_space(6.0);
                if ui
                    .button(egui::RichText::new("Organize Files").strong())
                    .clicked()
                {
                    trigger_organize = true;
                }

                if let Some(report) = &session.last_organize_report {
                    ui.add_space(8.0);
                    egui::Frame::group(ui.style()).show(ui, |ui| {
                        ui.label(
                            egui::RichText::new(format!("Organization Complete ({})", report.mode))
                                .strong(),
                        );
                        ui.label(format!("Groups written: {}", report.groups_written));
                        ui.label(format!("Images copied: {}", report.files_processed));
                        ui.label(format!("Sidecars preserved: {}", report.sidecars_preserved));
                        if report.collision_renames > 0 {
                            ui.label(format!("Collision renames: {}", report.collision_renames));
                        }
                        if !report.errors.is_empty() {
                            ui.colored_label(
                                egui::Color32::from_rgb(230, 100, 100),
                                format!("Errors: {}", report.errors.len()),
                            );
                        }
                    });
                }

                if let Some(status) = &session.status_banner {
                    ui.add_space(8.0);
                    ui.label(
                        egui::RichText::new(status)
                            .small()
                            .color(egui::Color32::LIGHT_BLUE),
                    );
                }

                ui.add_space(12.0);
                egui::CollapsingHeader::new("Evaluation")
                    .default_open(false)
                    .show(ui, |ui| {
                        ui.checkbox(
                            &mut app.capture_eval_labels,
                            "Capture labels from manual corrections",
                        );
                        let (same_cnt, diff_cnt) = session.eval_labels.count_by_label();
                        ui.label(
                            egui::RichText::new(format!(
                                "Labels: {same_cnt} SameProject, {diff_cnt} DifferentProject"
                            ))
                            .small(),
                        );
                        ui.label(
                            egui::RichText::new(format!(
                                "Stored: {}",
                                labels_file_path(&app.cache_store).display()
                            ))
                            .small()
                            .color(ui.visuals().weak_text_color()),
                        );
                        if ui.button("Export evaluation_report.json + .md").clicked() {
                            trigger_export_eval = true;
                        }
                    });
            });
        });

    // Center panel: Lazy thumbnail grid
    egui::CentralPanel::default().show(root_ui, |ui| {
        let Some(session) = app.session.as_mut() else {
            return;
        };
        if let Some(action) = show_gallery(session, &mut app.thumbs, ui) {
            match action {
                GalleryAction::CompareTwo(a, b) => {
                    session.comparison_pair = Some((a, b));
                    app.thumbs.clear_compare_cache();
                }
                GalleryAction::CompareWithRep(a) => {
                    if let Some(g) = session.groups.get(session.selected_group_idx)
                        && let Some(&other) = g.member_indices.iter().find(|&&m| m != a)
                    {
                        session.comparison_pair = Some((a, other));
                        app.thumbs.clear_compare_cache();
                    }
                }
            }
        }
    });

    if trigger_organize {
        if app.organize_options.mode == OrganizeMode::Move {
            app.pending_move_confirm_modal = true;
        } else {
            execute_organize(app);
        }
    }

    if app.pending_move_confirm_modal {
        show_move_confirmation_modal(app, &ctx);
    }

    if trigger_export_eval {
        export_evaluation_report(app);
    }

    if let Some((ia, ib)) = app.session.as_ref().and_then(|s| s.comparison_pair) {
        show_comparison_modal(app, &ctx, ia, ib);
    }

    if let Some((id_a, id_b)) = app
        .session
        .as_ref()
        .and_then(|s| s.group_comparison_pair.clone())
    {
        show_group_comparison_modal(app, &ctx, &id_a, &id_b);
    }
}

fn execute_organize(app: &mut FurnitureSorterApp) {
    let out_dir = std::path::PathBuf::from(app.output_dir_input.trim());
    let Some(session) = app.session.as_mut() else {
        return;
    };
    match organize_groups(
        &out_dir,
        &session.groups,
        &session.images,
        &app.organize_options,
    ) {
        Ok(report) => {
            session.status_banner = Some(format!(
                "Organized {} files across {} groups into {}",
                report.files_processed,
                report.groups_written,
                report.output_dir.display()
            ));
            session.last_organize_report = Some(report);
        }
        Err(err) => {
            session.status_banner = Some(format!("Organize failed: {err}"));
        }
    }
}

fn show_move_confirmation_modal(app: &mut FurnitureSorterApp, ctx: &egui::Context) {
    let mut confirm = false;
    let mut cancel = false;
    egui::Window::new("Confirm MOVE Mode")
        .collapsible(false)
        .resizable(false)
        .anchor(egui::Align2::CENTER_CENTER, egui::vec2(0.0, 0.0))
        .show(ctx, |ui| {
            ui.label("MOVE mode removes source images and sidecars after verified copy.");
            ui.label("Recommended safe default is COPY.");
            ui.add_space(10.0);
            ui.horizontal(|ui| {
                if ui.button("Confirm MOVE").clicked() {
                    confirm = true;
                }
                if ui.button("Cancel").clicked() {
                    cancel = true;
                }
            });
        });

    if confirm {
        app.pending_move_confirm_modal = false;
        execute_organize(app);
    } else if cancel {
        app.pending_move_confirm_modal = false;
    }
}

fn export_evaluation_report(app: &mut FurnitureSorterApp) {
    let Some(session) = app.session.as_mut() else {
        return;
    };
    let report = build_evaluation_report(
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
    session.eval_report = report.labeled_comparison.clone();
    session.full_eval_report = Some(report.clone());

    let eval_dir = app.cache_store.eval_dir();
    if let Err(e) = std::fs::create_dir_all(&eval_dir) {
        session.status_banner = Some(format!("Failed creating evaluation dir: {e}"));
        return;
    }
    let json_path = eval_dir.join("evaluation_report.json");
    let md_path = eval_dir.join("evaluation_report.md");

    match serde_json::to_string_pretty(&report) {
        Ok(json) => {
            if let Err(e) = std::fs::write(&json_path, json) {
                session.status_banner = Some(format!("Failed writing JSON report: {e}"));
                return;
            }
            let md = format_evaluation_markdown(&report);
            let _ = std::fs::write(&md_path, md);
            session.status_banner = Some(format!(
                "Saved evaluation_report.json & .md in {}",
                eval_dir.display()
            ));
        }
        Err(e) => {
            session.status_banner = Some(format!("Report serialization error: {e}"));
        }
    }
}

fn show_comparison_modal(app: &mut FurnitureSorterApp, ctx: &egui::Context, ia: usize, ib: usize) {
    let mut close = false;
    let mut save_label: Option<GroundTruthLabel> = None;

    let Some(session) = app.session.as_mut() else {
        return;
    };
    let (Some(img_a), Some(img_b), Some(fa), Some(fb)) = (
        session.images.get(ia),
        session.images.get(ib),
        session.features.get(ia),
        session.features.get(ib),
    ) else {
        session.comparison_pair = None;
        return;
    };

    let score = compare_pair_direct(fa, fb, &app.sim_config);
    let sim_pct = (score.combined * 100.0).round();

    egui::Window::new("Visual Comparison")
        .collapsible(false)
        .resizable(true)
        .default_width(820.0)
        .show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.heading(format!("Similarity = {sim_pct:.0}%"));
                ui.separator();
                ui.label(
                    egui::RichText::new(score.relation.ui_label())
                        .strong()
                        .color(egui::Color32::from_rgb(130, 210, 145)),
                );
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if ui.button("Close").clicked() {
                        close = true;
                    }
                });
            });
            ui.separator();

            ui.columns(2, |cols| {
                for (col_idx, (idx, img)) in [(ia, img_a), (ib, img_b)].into_iter().enumerate() {
                    let ui = &mut cols[col_idx];
                    ui.label(egui::RichText::new(&img.file_name).strong());
                    ui.label(
                        egui::RichText::new(&img.relative_path)
                            .small()
                            .color(ui.visuals().weak_text_color()),
                    );
                    let box_size = egui::vec2(360.0, 270.0);
                    let (rect, _) = ui.allocate_exact_size(box_size, egui::Sense::hover());
                    if let Some(tex) = app.thumbs.get_or_request_compare(idx, img) {
                        let ts = tex.size_vec2();
                        let scale = (box_size.x / ts.x.max(1.0)).min(box_size.y / ts.y.max(1.0));
                        let draw_rect = egui::Rect::from_center_size(rect.center(), ts * scale);
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
                            "Loading preview...",
                            egui::FontId::proportional(13.0),
                            ui.visuals().weak_text_color(),
                        );
                    }
                }
            });

            ui.add_space(10.0);
            ui.heading("Component Breakdown");
            let bars = [
                ("Color (30%)", score.deterministic.color),
                ("Layout 4x4 (25%)", score.deterministic.layout),
                ("Edges (20%)", score.deterministic.edges),
                ("pHash (15%)", score.deterministic.phash),
                ("Texture (10%)", score.deterministic.texture),
            ];
            for (label, val) in bars {
                ui.horizontal(|ui| {
                    ui.add_sized([130.0, 18.0], egui::Label::new(label));
                    ui.add(
                        egui::ProgressBar::new(val.clamp(0.0, 1.0))
                            .desired_width(300.0)
                            .text(format!("{:.1}%", val * 100.0)),
                    );
                });
            }

            ui.add_space(8.0);
            ui.heading("Project Identity & Contradiction Diagnostics");
            let id_bars = [
                ("Family Similarity", score.family_similarity, egui::Color32::from_rgb(100, 180, 240)),
                ("Project Identity", score.project_identity, egui::Color32::from_rgb(120, 220, 140)),
                (
                    "Contradiction",
                    score.contradiction,
                    if score.contradiction > 0.30 {
                        egui::Color32::from_rgb(240, 100, 100)
                    } else {
                        egui::Color32::from_rgb(200, 200, 120)
                    },
                ),
                ("Topology Match", score.deterministic.topology_match, egui::Color32::from_rgb(180, 140, 240)),
            ];
            for (label, val, col) in id_bars {
                ui.horizontal(|ui| {
                    ui.add_sized([140.0, 18.0], egui::Label::new(label));
                    ui.add(
                        egui::ProgressBar::new(val.clamp(0.0, 1.0))
                            .desired_width(280.0)
                            .fill(col)
                            .text(format!("{:.1}%", val * 100.0)),
                    );
                });
            }
            if score.hard_contradiction {
                ui.colored_label(
                    egui::Color32::from_rgb(240, 80, 80),
                    "⚠ HARD CONTRADICTION DETECTED — Incompatible permanent structure or material clash",
                );
            }

            ui.add_space(6.0);
            let diag = match score.embedding {
                Some(emb) => format!(
                    "Deterministic: {:.2}   |   Embedding: {:.2}   |   Combined: {:.2}   (pHash Hamming: {}, dHash Hamming: {})",
                    score.deterministic.combined,
                    emb,
                    score.combined,
                    score.deterministic.phash_hamming,
                    score.deterministic.dhash_hamming
                ),
                None => format!(
                    "Deterministic: {:.2}   |   Embedding: N/A   |   Combined: {:.2}   (pHash Hamming: {}, dHash Hamming: {})",
                    score.deterministic.combined,
                    score.combined,
                    score.deterministic.phash_hamming,
                    score.deterministic.dhash_hamming
                ),
            };
            ui.label(egui::RichText::new(diag).monospace().small());

            ui.add_space(8.0);
            ui.horizontal(|ui| {
                ui.label("Evaluation label:");
                if ui.button("Label: SameProject").clicked() {
                    save_label = Some(GroundTruthLabel::SameProject);
                }
                if ui.button("Label: DifferentProject").clicked() {
                    save_label = Some(GroundTruthLabel::DifferentProject);
                }
                ui.label(
                    egui::RichText::new(format!(
                        "({} labeled pairs stored)",
                        session.eval_labels.pairs.len()
                    ))
                    .small()
                    .color(ui.visuals().weak_text_color()),
                );
            });
        });

    if let Some(lbl) = save_label {
        let id_a = session.images[ia].stable_id();
        let id_b = session.images[ib].stable_id();
        let disp_a = session.images[ia].file_name.clone();
        let disp_b = session.images[ib].file_name.clone();
        session.eval_labels.upsert(
            &id_a,
            &id_b,
            &disp_a,
            &disp_b,
            lbl,
            LabelSource::ManualComparison,
        );
        let _ = save_evaluation_labels(&app.cache_store, &session.eval_labels);
        session.eval_report = Some(evaluate_labeled_pairs(
            &session.eval_labels,
            &session.images,
            &session.features,
            &app.sim_config,
            0.75,
        ));
    }

    if close {
        session.comparison_pair = None;
        app.thumbs.clear_compare_cache();
    }
}

fn move_members_between_groups(
    session: &mut SessionData,
    from_idx: usize,
    to_idx: usize,
    members: &[usize],
) {
    if from_idx == to_idx || from_idx >= session.groups.len() || to_idx >= session.groups.len() {
        return;
    }
    session.groups[from_idx]
        .member_indices
        .retain(|m| !members.contains(m));
    if let Some(rep) = session.groups[from_idx].representative_idx
        && !session.groups[from_idx].member_indices.contains(&rep)
    {
        session.groups[from_idx].representative_idx =
            session.groups[from_idx].member_indices.first().copied();
    }
    for &m in members {
        if !session.groups[to_idx].member_indices.contains(&m) {
            session.groups[to_idx].member_indices.push(m);
        }
    }
    if session.groups[to_idx].representative_idx.is_none() {
        session.groups[to_idx].representative_idx =
            session.groups[to_idx].member_indices.first().copied();
    }
    session.selected_members.clear();
}

fn create_new_group_from_selected(session: &mut SessionData, from_idx: usize, members: &[usize]) {
    if members.is_empty() || from_idx >= session.groups.len() {
        return;
    }
    session.groups[from_idx]
        .member_indices
        .retain(|m| !members.contains(m));
    if let Some(rep) = session.groups[from_idx].representative_idx
        && !session.groups[from_idx].member_indices.contains(&rep)
    {
        session.groups[from_idx].representative_idx =
            session.groups[from_idx].member_indices.first().copied();
    }

    let next_num = session
        .groups
        .iter()
        .filter(|g| g.kind == GroupKind::FurnitureProject)
        .count()
        + 1;
    let id = format!("Furniture_{next_num:03}");
    let name = format!("Furniture {next_num:03}");
    let new_group = crate::clustering::ImageGroup {
        id,
        name,
        kind: GroupKind::FurnitureProject,
        representative_idx: members.first().copied(),
        member_indices: members.to_vec(),
        avg_similarity: 1.0,
        confirmed: true,
        nearest_related: None,
        representative_members: members.first().copied().into_iter().collect(),
        signature: None,
    };

    let insert_pos = session
        .groups
        .iter()
        .position(|g| {
            matches!(
                g.kind,
                GroupKind::CandidateGroup | GroupKind::Review | GroupKind::Ungrouped
            )
        })
        .unwrap_or(session.groups.len());
    session.groups.insert(insert_pos, new_group);
    session.selected_group_idx = insert_pos;
    session.selected_members.clear();
}

fn merge_groups(session: &mut SessionData, source_idx: usize, target_idx: usize) {
    if source_idx == target_idx
        || source_idx >= session.groups.len()
        || target_idx >= session.groups.len()
    {
        return;
    }
    let moved = std::mem::take(&mut session.groups[source_idx].member_indices);
    for m in moved {
        if !session.groups[target_idx].member_indices.contains(&m) {
            session.groups[target_idx].member_indices.push(m);
        }
    }
    session.groups.remove(source_idx);
    let new_sel = if source_idx < target_idx {
        target_idx - 1
    } else {
        target_idx
    };
    session.selected_group_idx = new_sel;
    session.selected_members.clear();
}

fn show_group_comparison_modal(
    app: &mut FurnitureSorterApp,
    ctx: &egui::Context,
    id_a: &str,
    id_b: &str,
) {
    let mut close = false;
    let mut do_merge = false;
    let mut do_keep_separate = false;

    let Some(session) = app.session.as_mut() else {
        return;
    };

    let idx_a = session.groups.iter().position(|g| g.id == id_a);
    let idx_b = session.groups.iter().position(|g| g.id == id_b);
    let (Some(ia), Some(ib)) = (idx_a, idx_b) else {
        session.group_comparison_pair = None;
        return;
    };

    let ga = session.groups[ia].clone();
    let gb = session.groups[ib].clone();

    let suggestion = session
        .possible_merges
        .iter()
        .find(|m| {
            (m.group_a_id == id_a && m.group_b_id == id_b)
                || (m.group_a_id == id_b && m.group_b_id == id_a)
        })
        .cloned();

    egui::Window::new(format!("Compare Groups: {} ↔ {}", ga.name, gb.name))
        .collapsible(false)
        .resizable(true)
        .default_width(880.0)
        .show(ctx, |ui| {
            ui.horizontal(|ui| {
                if let Some(sug) = &suggestion {
                    ui.heading(format!("Match Confidence: {:.0}%", sug.similarity * 100.0));
                } else {
                    ui.heading("Group Visual Comparison");
                }
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if ui.button("Close").clicked() {
                        close = true;
                    }
                });
            });
            ui.separator();

            if let Some(sug) = &suggestion {
                ui.horizontal_wrapped(|ui| {
                    ui.label(egui::RichText::new("Evidence Breakdown:").strong());
                    ui.label(format!("Structure {:.0}%", sug.breakdown.structure * 100.0));
                    ui.separator();
                    ui.label(format!(
                        "Cross-View {:.0}%",
                        sug.breakdown.cross_view * 100.0
                    ));
                    ui.separator();
                    ui.label(format!("Color/Finish {:.0}%", sug.breakdown.color * 100.0));
                    if let Some(emb) = sug.breakdown.embedding {
                        ui.separator();
                        ui.label(format!("Embedding {:.0}%", emb * 100.0));
                    }
                    if sug.breakdown.partial_match >= 0.50 {
                        ui.separator();
                        ui.label(format!(
                            "Partial Match {:.0}%",
                            sug.breakdown.partial_match * 100.0
                        ));
                    }
                    if sug.breakdown.mutual_neighbor {
                        ui.separator();
                        ui.label(
                            egui::RichText::new("★ Mutual Neighbors")
                                .color(egui::Color32::from_rgb(100, 190, 240))
                                .small(),
                        );
                    }
                });
                ui.separator();
            }

            ui.columns(2, |cols| {
                // Column 0: Group A representative views
                {
                    let ui = &mut cols[0];
                    ui.heading(format!("{} ({} images)", ga.name, ga.member_indices.len()));
                    ui.label(egui::RichText::new(format!("Category: {}", ga.kind.badge())).small());
                    ui.add_space(4.0);
                    let reps: Vec<usize> = if !ga.representative_members.is_empty() {
                        ga.representative_members.iter().copied().take(4).collect()
                    } else {
                        ga.member_indices.iter().copied().take(4).collect()
                    };

                    egui::ScrollArea::vertical()
                        .id_salt("modal_left_col")
                        .max_height(380.0)
                        .show(ui, |ui| {
                            for &img_idx in &reps {
                                if let Some(img) = session.images.get(img_idx) {
                                    ui.label(egui::RichText::new(&img.file_name).small());
                                    let box_size = egui::vec2(360.0, 160.0);
                                    let (rect, _) =
                                        ui.allocate_exact_size(box_size, egui::Sense::hover());
                                    if let Some(tex) = app.thumbs.get_or_request_thumb(img_idx, img)
                                    {
                                        let ts = tex.size_vec2();
                                        let scale = (box_size.x / ts.x.max(1.0))
                                            .min(box_size.y / ts.y.max(1.0));
                                        let draw_rect =
                                            egui::Rect::from_center_size(rect.center(), ts * scale);
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
                                        ui.painter().rect_filled(
                                            rect,
                                            4.0,
                                            ui.visuals().extreme_bg_color,
                                        );
                                    }
                                    ui.add_space(4.0);
                                }
                            }
                        });
                }

                // Column 1: Group B representative views
                {
                    let ui = &mut cols[1];
                    ui.heading(format!("{} ({} images)", gb.name, gb.member_indices.len()));
                    ui.label(egui::RichText::new(format!("Category: {}", gb.kind.badge())).small());
                    ui.add_space(4.0);
                    let reps: Vec<usize> = if !gb.representative_members.is_empty() {
                        gb.representative_members.iter().copied().take(4).collect()
                    } else {
                        gb.member_indices.iter().copied().take(4).collect()
                    };

                    egui::ScrollArea::vertical()
                        .id_salt("modal_right_col")
                        .max_height(380.0)
                        .show(ui, |ui| {
                            for &img_idx in &reps {
                                if let Some(img) = session.images.get(img_idx) {
                                    ui.label(egui::RichText::new(&img.file_name).small());
                                    let box_size = egui::vec2(360.0, 160.0);
                                    let (rect, _) =
                                        ui.allocate_exact_size(box_size, egui::Sense::hover());
                                    if let Some(tex) = app.thumbs.get_or_request_thumb(img_idx, img)
                                    {
                                        let ts = tex.size_vec2();
                                        let scale = (box_size.x / ts.x.max(1.0))
                                            .min(box_size.y / ts.y.max(1.0));
                                        let draw_rect =
                                            egui::Rect::from_center_size(rect.center(), ts * scale);
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
                                        ui.painter().rect_filled(
                                            rect,
                                            4.0,
                                            ui.visuals().extreme_bg_color,
                                        );
                                    }
                                    ui.add_space(4.0);
                                }
                            }
                        });
                }
            });

            ui.add_space(10.0);
            ui.separator();
            ui.horizontal(|ui| {
                if ui
                    .button(egui::RichText::new("Merge Groups (Same Project)").strong())
                    .clicked()
                {
                    do_merge = true;
                }
                if ui.button("Keep Separate (Different Projects)").clicked() {
                    do_keep_separate = true;
                }
                if ui.button("Dismiss Suggestion").clicked() {
                    close = true;
                }
            });
        });

    if do_merge {
        if app.capture_eval_labels {
            capture_merge_groups(&mut session.eval_labels, &gb, &ga, &session.images);
            let _ = save_evaluation_labels(&app.cache_store, &session.eval_labels);
        }
        merge_groups(session, ib, ia);
        session
            .possible_merges
            .retain(|m| m.group_a_id != id_b && m.group_b_id != id_b);
        session.group_comparison_pair = None;
    } else if do_keep_separate {
        if app.capture_eval_labels {
            capture_keep_separate_groups(&mut session.eval_labels, &ga, &gb, &session.images);
            let _ = save_evaluation_labels(&app.cache_store, &session.eval_labels);
        }
        if let Some(m) = session.possible_merges.iter_mut().find(|m| {
            (m.group_a_id == id_a && m.group_b_id == id_b)
                || (m.group_a_id == id_b && m.group_b_id == id_a)
        }) {
            m.dismissed = true;
        }
        session.group_comparison_pair = None;
    } else if close {
        if let Some(m) = session.possible_merges.iter_mut().find(|m| {
            (m.group_a_id == id_a && m.group_b_id == id_b)
                || (m.group_a_id == id_b && m.group_b_id == id_a)
        }) {
            m.dismissed = true;
        }
        session.group_comparison_pair = None;
    }
}
