use crate::clustering::{GroupKind, ImageGroup};
use crate::scanner::{OUTPUT_MARKER_FILE, ScannedImage};
use crate::sidecar::sidecar_path_for;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum OrganizeMode {
    Copy,
    Move,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OrganizeOptions {
    pub mode: OrganizeMode,
    pub include_ungrouped: bool,
    pub include_review: bool,
}

impl Default for OrganizeOptions {
    fn default() -> Self {
        Self {
            mode: OrganizeMode::Copy,
            include_ungrouped: false,
            include_review: true,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GroupMemberRecord {
    pub organized_file: String,
    pub source_path: String,
    pub size: u64,
    pub source_sha256: Option<String>,
    pub generated_sha256: Option<String>,
    pub asset_id: Option<String>,
    pub attempt_id: Option<String>,
    pub operation_id: Option<String>,
    pub automated_decision: Option<String>,
    pub sidecar_copied: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GroupMetadataFile {
    pub group_id: String,
    pub group_name: String,
    pub category: String,
    pub representative: Option<String>,
    pub average_similarity: f32,
    pub manual_confirmed: bool,
    pub members: Vec<GroupMemberRecord>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct OrganizeReport {
    pub output_dir: PathBuf,
    pub mode: String,
    pub groups_written: usize,
    pub files_processed: usize,
    pub sidecars_preserved: usize,
    pub collision_renames: usize,
    pub skipped_empty_groups: usize,
    pub errors: Vec<String>,
}

pub fn sanitize_folder_name(raw: &str) -> String {
    let cleaned: String = raw
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '_' || c == '-' {
                c
            } else if c == ' ' {
                '_'
            } else {
                '_'
            }
        })
        .collect();
    let trimmed = cleaned.trim_matches('_');
    if trimmed.is_empty() {
        "Group".to_string()
    } else {
        trimmed.to_string()
    }
}

/// Resolves a collision-safe destination file path (`name.ext`, `name_2.ext`, ...),
/// ensuring neither the image nor its `<dest>.openimg.json` sidecar already exists.
pub fn resolve_collision_safe_path(
    dir: &Path,
    file_name: &str,
    has_sidecar: bool,
) -> (PathBuf, bool) {
    let candidate = dir.join(file_name);
    let sidecar_ok = !has_sidecar || !sidecar_path_for(&candidate).exists();
    if !candidate.exists() && sidecar_ok {
        return (candidate, false);
    }

    let p = Path::new(file_name);
    let stem = p
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "image".to_string());
    let ext = p
        .extension()
        .map(|s| format!(".{}", s.to_string_lossy()))
        .unwrap_or_default();

    let mut suffix = 2usize;
    loop {
        let next_name = format!("{stem}_{suffix}{ext}");
        let next_path = dir.join(next_name);
        let next_sidecar_ok = !has_sidecar || !sidecar_path_for(&next_path).exists();
        if !next_path.exists() && next_sidecar_ok {
            return (next_path, true);
        }
        suffix += 1;
    }
}

pub fn organize_groups(
    output_dir: &Path,
    groups: &[ImageGroup],
    images: &[ScannedImage],
    options: &OrganizeOptions,
) -> Result<OrganizeReport, String> {
    fs::create_dir_all(output_dir).map_err(|e| {
        format!(
            "Failed to create output directory {}: {e}",
            output_dir.display()
        )
    })?;

    let marker_path = output_dir.join(OUTPUT_MARKER_FILE);
    let _ = fs::write(&marker_path, b"FurnitureSorter output directory\n");

    let mut report = OrganizeReport {
        output_dir: output_dir.to_path_buf(),
        mode: match options.mode {
            OrganizeMode::Copy => "COPY".to_string(),
            OrganizeMode::Move => "MOVE".to_string(),
        },
        ..Default::default()
    };

    for group in groups {
        if group.kind == GroupKind::Ungrouped && !options.include_ungrouped {
            continue;
        }
        if matches!(group.kind, GroupKind::Review | GroupKind::CandidateGroup)
            && !options.include_review
        {
            continue;
        }
        if group.member_indices.is_empty() {
            report.skipped_empty_groups += 1;
            continue;
        }

        let folder_name = sanitize_folder_name(&group.id);
        let group_dir = output_dir.join(&folder_name);
        if let Err(e) = fs::create_dir_all(&group_dir) {
            report
                .errors
                .push(format!("Create dir {} failed: {e}", group_dir.display()));
            continue;
        }

        let mut records = Vec::with_capacity(group.member_indices.len());
        let mut representative_rel_name: Option<String> = None;

        for &idx in &group.member_indices {
            let Some(img) = images.get(idx) else {
                continue;
            };
            let has_sidecar = img.sidecar.is_some();
            let (dest_path, collided) =
                resolve_collision_safe_path(&group_dir, &img.file_name, has_sidecar);
            if collided {
                report.collision_renames += 1;
            }

            if let Err(e) = fs::copy(&img.path, &dest_path) {
                report.errors.push(format!(
                    "Copy {} -> {} failed: {e}",
                    img.path.display(),
                    dest_path.display()
                ));
                continue;
            }

            let mut sidecar_copied = false;
            if let Some(sidecar) = &img.sidecar {
                let dest_sidecar = sidecar_path_for(&dest_path);
                if let Err(e) = fs::copy(&sidecar.path, &dest_sidecar) {
                    report.errors.push(format!(
                        "Copy sidecar {} -> {} failed: {e}",
                        sidecar.path.display(),
                        dest_sidecar.display()
                    ));
                } else {
                    sidecar_copied = true;
                    report.sidecars_preserved += 1;
                }
            }

            if options.mode == OrganizeMode::Move {
                let copied_ok = fs::metadata(&dest_path)
                    .map(|m| m.len() == img.size)
                    .unwrap_or(false);
                let canonical_src = fs::canonicalize(&img.path).ok();
                let canonical_dst = fs::canonicalize(&dest_path).ok();
                if copied_ok && canonical_src != canonical_dst {
                    if let Err(e) = fs::remove_file(&img.path) {
                        report
                            .errors
                            .push(format!("Remove source {} failed: {e}", img.path.display()));
                    } else if sidecar_copied && let Some(sidecar) = &img.sidecar {
                        let _ = fs::remove_file(&sidecar.path);
                    }
                }
            }

            let dest_name = dest_path
                .file_name()
                .map(|s| s.to_string_lossy().into_owned())
                .unwrap_or_else(|| img.file_name.clone());

            if group.representative_idx == Some(idx) {
                representative_rel_name = Some(dest_name.clone());
            }

            let sc = img.sidecar.as_ref().map(|s| &s.data);
            records.push(GroupMemberRecord {
                organized_file: dest_name,
                source_path: img.path.to_string_lossy().into_owned(),
                size: img.size,
                source_sha256: sc.and_then(|d| d.source_sha256.clone()),
                generated_sha256: sc.and_then(|d| d.effective_image_sha256().map(str::to_string)),
                asset_id: sc.and_then(|d| d.asset_id.clone()),
                attempt_id: sc.and_then(|d| d.attempt_id.clone()),
                operation_id: sc.and_then(|d| d.operation_id.clone()),
                automated_decision: sc.and_then(|d| d.automated_decision.clone()),
                sidecar_copied,
            });
            report.files_processed += 1;
        }

        let group_meta = GroupMetadataFile {
            group_id: group.id.clone(),
            group_name: group.name.clone(),
            category: group.kind.badge().to_string(),
            representative: representative_rel_name,
            average_similarity: group.avg_similarity,
            manual_confirmed: group.confirmed,
            members: records,
        };

        let meta_path = group_dir.join("group.json");
        match serde_json::to_string_pretty(&group_meta) {
            Ok(json) => {
                if let Err(e) = fs::write(&meta_path, json) {
                    report
                        .errors
                        .push(format!("Write {} failed: {e}", meta_path.display()));
                } else {
                    report.groups_written += 1;
                }
            }
            Err(e) => report
                .errors
                .push(format!("Serialize group.json failed: {e}")),
        }
    }

    Ok(report)
}
