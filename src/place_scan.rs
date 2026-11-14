//! Recursive project discovery and readable place-file extraction.
use crate::models::ScanOptions;
use crate::{
    image_scan,
    policy::Policy,
    report::{AssetReport, ScanSummary},
    text_scan,
};
use anyhow::{Context, Result};
use std::{
    collections::BTreeSet,
    fs,
    path::{Path, PathBuf},
};
use walkdir::WalkDir;
fn extension(path: &Path) -> String {
    path.extension()
        .unwrap_or_default()
        .to_string_lossy()
        .to_lowercase()
}
fn supported(path: &Path) -> bool {
    matches!(
        extension(path).as_str(),
        "png"
            | "jpg"
            | "jpeg"
            | "webp"
            | "luau"
            | "lua"
            | "json"
            | "txt"
            | "rbxl"
            | "rbxlx"
            | "rbxm"
            | "rbxmx"
    )
}
/// Extract readable UTF-8 spans, preserving text on both sides of binary bytes.
pub fn readable_spans(bytes: &[u8]) -> String {
    String::from_utf8_lossy(bytes)
        .split(|c: char| c == '\u{fffd}' || (c.is_control() && c != '\n' && c != '\t' && c != '\r'))
        .filter(|s| s.chars().count() >= 3)
        .collect::<Vec<_>>()
        .join("\n")
}
/// Scan one supported file; failures include its path.
pub fn scan_file(path: &Path, policy: &Policy) -> Result<AssetReport> {
    scan_file_with_options(path, policy, ScanOptions::default())
}
pub fn scan_file_with_options(
    path: &Path,
    policy: &Policy,
    options: ScanOptions,
) -> Result<AssetReport> {
    let bytes = fs::read(path).with_context(|| format!("Reading {}", path.display()))?;
    let ext = extension(path);
    let image = matches!(ext.as_str(), "png" | "jpg" | "jpeg" | "webp");
    let mut report = AssetReport::new(
        path.display().to_string(),
        &bytes,
        if image {
            "image"
        } else if ext.starts_with("rbx") {
            "place"
        } else {
            "text"
        },
    );
    if image {
        image_scan::scan_image_with_options(&bytes, policy, &mut report, options)
            .with_context(|| format!("Scanning {}", path.display()))?;
    } else {
        let text = readable_spans(&bytes);
        if ext == "json" {
            text_scan::scan_json(&text, policy, &mut report);
        } else {
            text_scan::scan_text(
                &text,
                matches!(ext.as_str(), "luau" | "lua"),
                policy,
                &mut report,
            );
        }
        if ext.starts_with("rbx") {
            text_scan::scan_script(&text, policy, &mut report);
        }
    }
    Ok(report)
}
/// Scan a single file or recursively discover supported project assets.
pub fn scan(path: &Path, policy: &Policy) -> Result<ScanSummary> {
    scan_paths(&[path.to_path_buf()], policy, ScanOptions::default())
}
/// Scan multiple roots, deduplicating overlapping paths in deterministic order.
pub fn scan_paths(roots: &[PathBuf], policy: &Policy, options: ScanOptions) -> Result<ScanSummary> {
    anyhow::ensure!(!roots.is_empty(), "At least one scan path is required");
    let mut paths = BTreeSet::new();
    for path in roots {
        anyhow::ensure!(
            path.exists(),
            "Scan path does not exist: {}",
            path.display()
        );
        if path.is_file() {
            anyhow::ensure!(supported(path), "Unsupported file type: {}", path.display());
            paths.insert(fs::canonicalize(path)?);
        } else {
            for entry in WalkDir::new(path)
                .into_iter()
                .filter_entry(|e| ![".git", "target"].iter().any(|s| e.file_name() == *s))
            {
                let entry = entry?;
                if entry.file_type().is_file() && supported(entry.path()) {
                    paths.insert(fs::canonicalize(entry.path())?);
                }
            }
        }
    }
    Ok(ScanSummary::new(
        paths
            .iter()
            .map(|p| scan_file_with_options(p, policy, options))
            .collect::<Result<_>>()?,
    ))
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn binary_spans() {
        let text = readable_spans(b"\0TextBox\0https://evil.gg\xffhello");
        assert!(text.contains("TextBox"));
        assert!(text.contains("evil.gg"));
    }
}
