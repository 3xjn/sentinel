//! Command-line entry point for local screening and artifact verification.
use anyhow::{ensure, Context, Result};
use clap::{Parser, Subcommand};
use sentinel::{
    adapter, place_scan,
    policy::{Policy, Profile},
    scrub::{self, Rect, ScrubMode},
};
use std::{
    fs,
    io::Write,
    path::PathBuf,
    process::ExitCode,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Mutex,
    },
};

#[derive(Parser)]
#[command(
    name = "sentinel",
    version,
    about = "Offline neural asset screening and port verification"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}
#[derive(Subcommand)]
enum Command {
    /// Scan a file or recursively scan a project. Exit 1 when flags are found.
    Scan {
        #[arg(required = true)]
        paths: Vec<PathBuf>,
        #[arg(long)]
        no_models: bool,
        #[arg(long)]
        fast: bool,
        #[arg(long)]
        only_flagged: bool,
        #[arg(long)]
        json: bool,
        #[arg(long)]
        jsonl: bool,
        #[arg(long, value_enum, default_value = "high-recall")]
        profile: Profile,
        #[arg(long, value_name = "CATEGORY=SCORE")]
        threshold: Vec<String>,
    },
    /// Scrub rectangles while preserving every pixel in kept rectangles.
    Scrub {
        input: PathBuf,
        #[arg(short, long)]
        output: PathBuf,
        #[arg(long="rect",value_parser=parse_rect,required=true,value_name="X,Y,W,H")]
        rects: Vec<Rect>,
        #[arg(long="keep",value_parser=parse_rect,value_name="X,Y,W,H")]
        kept: Vec<Rect>,
        #[arg(long, value_enum, default_value = "auto")]
        mode: ScrubMode,
    },
    /// Verify staged port artifacts and optional cloud hash references.
    VerifyPort {
        root: PathBuf,
        #[arg(long)]
        cloud_manifest: Option<PathBuf>,
        #[arg(long)]
        json: bool,
    },
    /// Generate a Luau policy table from an audit JSON file.
    EmitPolicy {
        audit: PathBuf,
        #[arg(short, long)]
        output: PathBuf,
    },
    /// Inspect, initialize, or download configured ONNX models.
    Models {
        #[command(subcommand)]
        action: ModelsCommand,
    },
}
#[derive(Subcommand)]
enum ModelsCommand {
    /// List configured models and their local install status.
    List {
        #[arg(long)]
        json: bool,
    },
    /// Write the default models.toml configuration file.
    Init {
        #[arg(long)]
        force: bool,
    },
    /// Download any missing enabled models into the sentinel model directory.
    Install {
        #[arg(long)]
        force: bool,
    },
}
fn parse_rect(text: &str) -> Result<Rect, String> {
    let parts = text
        .split(',')
        .map(str::parse::<u32>)
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| e.to_string())?;
    if parts.len() != 4 || parts[2] == 0 || parts[3] == 0 {
        return Err("Expected X,Y,W,H with positive width and height".into());
    }
    Ok(Rect {
        x: parts[0],
        y: parts[1],
        w: parts[2],
        h: parts[3],
    })
}
fn scan(
    paths: Vec<PathBuf>,
    no_models: bool,
    fast: bool,
    only_flagged: bool,
    json: bool,
    jsonl: bool,
    profile: Profile,
    thresholds: Vec<String>,
) -> Result<bool> {
    let mut policy = Policy::new(profile);
    for entry in thresholds {
        let (category, score) = entry
            .split_once('=')
            .context("Threshold must be CATEGORY=SCORE")?;
        policy.set(
            category.parse().map_err(anyhow::Error::msg)?,
            score.parse()?,
        )?;
    }
    let flagged_count = AtomicUsize::new(0);
    let out_lock = Mutex::new(());
    let mut summary = place_scan::scan_paths_with_progress(
        &paths,
        &policy,
        sentinel::models::ScanOptions { no_models, fast },
        |done, total, asset| {
            let hit = asset.blocked || asset.flag_count > 0;
            let flagged = if hit {
                flagged_count.fetch_add(1, Ordering::Relaxed) + 1
            } else {
                flagged_count.load(Ordering::Relaxed)
            };
            let _guard = out_lock.lock().unwrap();
            if jsonl {
                if !only_flagged || hit {
                    if let Ok(line) = serde_json::to_string(asset) {
                        println!("{line}");
                    }
                }
            } else if !json && asset.blocked {
                eprint!("\r{:60}\r", "");
                println!("{}: {} flags", asset.path, asset.flag_count);
                for f in &asset.flags {
                    println!(
                        "  {:?} {} {:.2}: {}",
                        f.category, f.detector, f.score, f.evidence
                    );
                }
            }
            if total > 1 && (hit || done == total || done % 10 == 0) {
                eprint!("\rScanning: {done}/{total} ({flagged} flagged)");
                let _ = std::io::stderr().flush();
            }
        },
    )?;
    if summary.total_assets > 1 {
        eprintln!();
    }
    if only_flagged {
        summary.assets.retain(|a| a.blocked || a.flag_count > 0);
    }
    if jsonl {
        return Ok(summary.blocked_assets > 0);
    }
    if json {
        println!("{}", serde_json::to_string_pretty(&summary)?);
    } else {
        println!(
            "{} assets, {} blocked, {} flags",
            summary.total_assets, summary.blocked_assets, summary.total_flags
        );
    }
    Ok(summary.blocked_assets > 0)
}
fn scrub_file(
    input: PathBuf,
    output: PathBuf,
    rects: Vec<Rect>,
    kept: Vec<Rect>,
    mode: ScrubMode,
) -> Result<bool> {
    ensure!(
        output
            .extension()
            .is_some_and(|s| s.eq_ignore_ascii_case("png")),
        "Scrub output must be PNG to preserve pixel values and transparency"
    );
    let mut img = image::open(&input)?.to_rgba8();
    for r in rects.iter().chain(&kept) {
        ensure!(
            r.x.checked_add(r.w).is_some_and(|x| x <= img.width())
                && r.y.checked_add(r.h).is_some_and(|y| y <= img.height()),
            "Rectangle outside image: {r:?}"
        );
    }
    let overlaps = scrub::find_rect_overlaps(&rects, &kept);
    if !overlaps.is_empty() {
        eprintln!(
            "{} protected overlaps; kept pixels will be preserved",
            overlaps.len()
        );
    }
    scrub::scrub_image(&mut img, &rects, &kept, mode);
    img.save(&output)?;
    println!("Wrote {}", output.display());
    Ok(false)
}
fn run() -> Result<bool> {
    match Cli::parse().command {
        Command::Scan {
            paths,
            no_models,
            fast,
            only_flagged,
            json,
            jsonl,
            profile,
            threshold,
        } => scan(
            paths,
            no_models,
            fast,
            only_flagged,
            json,
            jsonl,
            profile,
            threshold,
        ),
        Command::Scrub {
            input,
            output,
            rects,
            kept,
            mode,
        } => scrub_file(input, output, rects, kept, mode),
        Command::VerifyPort {
            root,
            cloud_manifest,
            json,
        } => {
            let report = adapter::verify_port(&root, cloud_manifest.as_deref())?;
            if json {
                println!("{}", serde_json::to_string_pretty(&report)?);
            } else {
                for v in &report.violations {
                    eprintln!("{v}");
                }
                println!(
                    "{} ports, {} PNG pages, {} blank regions, {} violations",
                    report.games_checked,
                    report.pngs_checked,
                    report.blank_regions_checked,
                    report.violations.len()
                );
            }
            Ok(!report.violations.is_empty())
        }
        Command::EmitPolicy { audit, output } => {
            let value = serde_json::from_slice(&fs::read(audit)?)?;
            fs::write(&output, adapter::emit_policy(&value))?;
            println!("Wrote {}", output.display());
            Ok(false)
        }
        Command::Models { action } => {
            match action {
                ModelsCommand::List { json } => {
                    let list = sentinel::models::list_models()?;
                    if json {
                        println!("{}", serde_json::to_string_pretty(&list)?);
                    } else {
                        for m in list {
                            let state = if !m.enabled {
                                "disabled"
                            } else if m.installed {
                                "installed"
                            } else {
                                "missing"
                            };
                            let size = m
                                .bytes
                                .map(|b| format!(" ({:.1} MB)", b as f64 / (1024.0 * 1024.0)))
                                .unwrap_or_default();
                            println!("{:<18} {:<10} {}{} -> {}", m.id, state, m.file, size, m.path);
                        }
                    }
                }
                ModelsCommand::Init { force } => {
                    let path = sentinel::models::init_config(force)?;
                    println!("Wrote {}", path.display());
                }
                ModelsCommand::Install { force } => {
                    let installed = sentinel::models::install_models(force)?;
                    if installed.is_empty() {
                        println!("All enabled models are already installed.");
                    } else {
                        for (id, path, bytes) in installed {
                            println!(
                                "Installed {} ({:.1} MB) -> {}",
                                id,
                                bytes as f64 / (1024.0 * 1024.0),
                                path.display()
                            );
                        }
                    }
                }
            }
            Ok(false)
        }
    }
}
fn main() -> ExitCode {
    match run() {
        Ok(false) => ExitCode::SUCCESS,
        Ok(true) => ExitCode::from(1),
        Err(error) => {
            eprintln!("sentinel: {error:#}");
            ExitCode::from(2)
        }
    }
}
