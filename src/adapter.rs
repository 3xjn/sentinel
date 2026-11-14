//! Artifact-based pre-upload checks and engine-independent Luau policy emission.
use crate::{report::sha256, scrub::Rect, text_scan};
use anyhow::{ensure, Context, Result};
use serde::Serialize;
use serde_json::Value;
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::{Path, PathBuf},
};

/// Port verification ledger; violations make the CLI exit with status one.
#[derive(Debug, Default, Serialize)]
pub struct Verification {
    pub games_checked: usize,
    pub pngs_checked: usize,
    pub blank_regions_checked: usize,
    pub violations: Vec<String>,
}
fn read(path: &Path) -> Result<Value> {
    serde_json::from_slice(&fs::read(path).with_context(|| format!("Reading {}", path.display()))?)
        .with_context(|| format!("Parsing {}", path.display()))
}
fn array<'a>(v: &'a Value, key: &str) -> Result<&'a Vec<Value>> {
    v[key]
        .as_array()
        .with_context(|| format!("Missing array {key}"))
}
fn ids(v: &Value, key: &str) -> BTreeSet<u64> {
    v[key]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(Value::as_u64)
        .collect()
}
fn number(v: &Value) -> Result<u32> {
    u32::try_from(v.as_u64().context("Missing unsigned coordinate")?)
        .context("Coordinate exceeds u32")
}
fn rect(v: &Value, keys: [&str; 4]) -> Result<Rect> {
    Ok(Rect {
        x: number(&v[keys[0]])?,
        y: number(&v[keys[1]])?,
        w: number(&v[keys[2]])?,
        h: number(&v[keys[3]])?,
    })
}
fn sprite_rect(v: &Value) -> Result<Rect> {
    rect(v, ["rectX", "rectY", "rectW", "rectH"])
}
fn safe_join(root: &Path, relative: &str) -> Result<PathBuf> {
    let path = Path::new(relative);
    ensure!(
        !path.is_absolute()
            && path.components().all(|p| matches!(
                p,
                std::path::Component::Normal(_) | std::path::Component::CurDir
            )),
        "Artifact path must stay relative: {relative}"
    );
    let base = root.canonicalize()?;
    let resolved = root
        .join(path)
        .canonicalize()
        .with_context(|| format!("Resolving {}", root.join(path).display()))?;
    ensure!(
        resolved.starts_with(base),
        "Artifact path escapes root: {relative}"
    );
    Ok(resolved)
}
fn uses_scrubbed(value: &Value, scrubbed: &BTreeSet<u64>) -> bool {
    match value {
        Value::Object(fields) => fields.iter().any(|(key, child)| {
            (["charId", "characterId", "character_id", "char_id"].contains(&key.as_str())
                && child.as_u64().is_some_and(|id| scrubbed.contains(&id)))
                || uses_scrubbed(child, scrubbed)
        }),
        Value::Array(items) => items.iter().any(|v| uses_scrubbed(v, scrubbed)),
        _ => false,
    }
}
fn unsafe_visible(text: &str) -> bool {
    let low = text.to_lowercase();
    text_scan::unsafe_text(text)
        || [
            "@",
            "papalouie",
            "coolmath",
            "mochi",
            "myspace",
            "itunes",
            "amazon",
            "google",
            "android",
            "iphone",
            "ipad",
            "ipod",
            "kindle",
            "merch",
            "more games",
            "walkthrough",
            "on this site",
            "play papa",
            "play the game now",
            "available for all sites",
            "when pizzas attack",
        ]
        .iter()
        .any(|s| low.contains(s))
}
fn verify_findings(audit: &Value, result: &mut Verification) -> Result<BTreeSet<u64>> {
    let hidden = ids(audit, "hide_character_ids");
    let controls = ids(audit, "preserved_control_containers");
    let retained = ids(audit, "retained_promo_containers");
    let blank = ids(audit, "blank_text_ids");
    for f in array(audit, "findings")? {
        if f["category"] == "visible_text" && f["text"].as_str().is_some_and(unsafe_visible) {
            let covered = f["auto_blank"] == true
                && f["char_id"].as_u64().is_some_and(|id| blank.contains(&id));
            if !covered {
                result
                    .violations
                    .push(format!("Unblanked unsafe visible_text: {f}"));
            }
        }
        if f["category"] == "symbol_or_instance" && f["auto_hide"] == true {
            let covered = f["char_id"].as_u64().is_some_and(|id| {
                hidden.contains(&id) || controls.contains(&id) || retained.contains(&id)
            });
            if !covered {
                result.violations.push(format!(
                    "Promo node lacks hidden/preserved/retained coverage: {f}"
                ));
            }
        }
    }
    Ok(hidden.union(&blank).copied().collect())
}
struct Layout {
    output: PathBuf,
    audit: PathBuf,
    plan: PathBuf,
    metadata: PathBuf,
    repairs: PathBuf,
    mirror: Option<PathBuf>,
}
fn catalog_layout(root: &Path, slug: &str) -> Layout {
    Layout {
        output: root.join(format!("build/GeneratedGames/{slug}")),
        audit: root.join(format!("build/tos-audit/{slug}.json")),
        plan: root.join(format!("build/packaging/{slug}/plan.json")),
        metadata: root.join(format!("build/packaging/{slug}/metadata.json")),
        repairs: root.join(format!("build/recipe-repairs/{slug}/repairs.json")),
        mirror: Some(root.join(format!(
            "build/StudioContent/content/papas2luau/{slug}/assets"
        ))),
    }
}
fn single_layout(root: &Path) -> Layout {
    let slug = root.file_name().unwrap_or_default().to_string_lossy();
    if let Some(repo) = root.parent().and_then(Path::parent).and_then(Path::parent) {
        let candidate = catalog_layout(repo, &slug);
        if candidate.audit.exists() {
            return candidate;
        }
    }
    Layout {
        output: root.into(),
        audit: root.join("audit.json"),
        plan: root.join("plan.json"),
        metadata: root.join("metadata.json"),
        repairs: root.join("repairs.json"),
        mirror: None,
    }
}
fn record(result: &mut Verification, label: &str, check: Result<()>) {
    if let Err(error) = check {
        result.violations.push(format!("{label}: {error:#}"));
    }
}
type BlankChecks = BTreeMap<String, Vec<(Rect, Vec<Rect>)>>;
fn register_blanks(plan: &Value, metadata: &Value, checks: &mut BlankChecks) -> Result<()> {
    let pages = array(plan, "pages")?;
    let atlas = number(&plan["atlas_size"])?;
    for region in array(metadata, "blank_regions")? {
        let source = number(&region["page"])?;
        let map = &plan["mapping"][source.to_string()];
        let page = number(&map["page"])?;
        let registered = pages
            .iter()
            .find(|p| p["pageIndex"] == page)
            .context("Blank region references unregistered page")?;
        let expected = format!("assets/atlas_{atlas}_{page}.png");
        let path = registered["path"].as_str().context("Missing page path")?;
        ensure!(
            Path::new(path).file_name() == Path::new(&expected).file_name(),
            "Unexpected atlas page path"
        );
        let mut r = rect(region, ["x", "y", "w", "h"])?;
        r.x = r.x.checked_add(number(&map["x"])?).context("x overflow")?;
        r.y = r.y.checked_add(number(&map["y"])?).context("y overflow")?;
        checks.entry(path.into()).or_default().push((r, vec![]));
    }
    Ok(())
}
fn register_repairs(
    layout: &Layout,
    plan: &Value,
    scrubbed: &BTreeSet<u64>,
    checks: &mut BlankChecks,
    result: &mut Verification,
) -> Result<()> {
    for variant in plan["recipe_repairs"].as_array().into_iter().flatten() {
        if uses_scrubbed(variant, scrubbed) {
            result
                .violations
                .push("Active recipe repair references scrubbed character".into());
        }
    }
    if !layout.repairs.exists() {
        return Ok(());
    }
    let repair = read(&layout.repairs)?;
    let variants = array(&repair, "variants")?;
    let pages = array(plan, "pages")?;
    for v in variants.iter().filter(|v| uses_scrubbed(v, scrubbed)) {
        let sprite = &v["sprite"];
        let Some(page) = pages.iter().find(|p| p["pageIndex"] == sprite["page"]) else {
            continue;
        };
        let kept = variants
            .iter()
            .filter(|k| k["sprite"]["page"] == sprite["page"] && !uses_scrubbed(k, scrubbed))
            .map(|k| sprite_rect(&k["sprite"]))
            .collect::<Result<Vec<_>>>()?;
        checks
            .entry(
                page["path"]
                    .as_str()
                    .context("Missing repair page path")?
                    .into(),
            )
            .or_default()
            .push((sprite_rect(sprite)?, kept));
    }
    Ok(())
}
fn verify_transparent(img: &image::RgbaImage, r: Rect, kept: &[Rect]) -> Result<()> {
    ensure!(
        r.x.checked_add(r.w).is_some_and(|x| x <= img.width())
            && r.y.checked_add(r.h).is_some_and(|y| y <= img.height()),
        "Blank rectangle outside image: {r:?}"
    );
    for y in r.y..r.y + r.h {
        for x in r.x..r.x + r.w {
            if !kept.iter().any(|k| k.contains(x, y)) {
                ensure!(
                    img.get_pixel(x, y)[3] == 0,
                    "Nontransparent blank pixel ({x},{y})"
                );
            }
        }
    }
    Ok(())
}
fn verify_pixels(layout: &Layout, checks: BlankChecks, result: &mut Verification) {
    for (path, regions) in checks {
        let check = (|| {
            let img = image::open(safe_join(&layout.output, &path)?)?.to_rgba8();
            for (r, kept) in regions {
                result.blank_regions_checked += 1;
                record(
                    result,
                    &format!("{path} {r:?}"),
                    verify_transparent(&img, r, &kept),
                );
            }
            Ok(())
        })();
        record(result, &path, check);
    }
}
fn staged_refs(output: &Path) -> Result<String> {
    let mut refs = String::new();
    let directory = output.join("AssetManifest_staged");
    ensure!(directory.is_dir(), "Missing AssetManifest_staged directory");
    for item in fs::read_dir(directory)? {
        let path = item?.path();
        if path.extension().is_some_and(|s| s == "luau") {
            refs.push_str(&fs::read_to_string(path)?);
        }
    }
    ensure!(!refs.is_empty(), "Empty staged manifests");
    Ok(refs)
}
fn verify_pages(
    layout: &Layout,
    plan: &Value,
    active: &mut BTreeSet<String>,
    result: &mut Verification,
) -> Result<()> {
    let mut names = BTreeSet::new();
    for page in array(plan, "pages")? {
        let relative = page["path"].as_str().context("Missing page path")?;
        let name = Path::new(relative)
            .file_name()
            .context("Missing page filename")?
            .to_string_lossy()
            .into_owned();
        names.insert(name);
        let check = (|| {
            let bytes = fs::read(safe_join(&layout.output, relative)?)?;
            let hash = sha256(&bytes);
            active.insert(hash.clone());
            result.pngs_checked += 1;
            ensure!(
                page["sha256"].as_str() == Some(hash.as_str()),
                "Page SHA-256 mismatch"
            );
            Ok(())
        })();
        record(result, relative, check);
    }
    let refs = staged_refs(&layout.output)?;
    let mut directories = vec![layout.output.join("assets")];
    if let Some(mirror) = &layout.mirror {
        if mirror.exists() {
            directories.push(mirror.clone());
        }
    }
    for dir in directories {
        for entry in fs::read_dir(dir)? {
            let path = entry?.path();
            if !path
                .extension()
                .is_some_and(|s| s.eq_ignore_ascii_case("png"))
            {
                continue;
            }
            let name = path.file_name().unwrap().to_string_lossy();
            let referenced = names.contains(name.as_ref()) || refs.contains(name.as_ref());
            if !referenced {
                result
                    .violations
                    .push(format!("Unreferenced PNG {}", path.display()));
            } else {
                let hash = sha256(&fs::read(&path)?);
                let staged = layout.output.join("assets").join(name.as_ref());
                if path == staged {
                    active.insert(hash);
                } else {
                    let expected = sha256(&fs::read(&staged)?);
                    if hash != expected {
                        result.violations.push(format!(
                            "Studio mirror PNG differs from active PNG: {}",
                            path.display()
                        ));
                    }
                }
            }
        }
    }
    Ok(())
}
fn verify_game(
    layout: &Layout,
    active: &mut BTreeSet<String>,
    result: &mut Verification,
) -> Result<()> {
    result.games_checked += 1;
    let audit = read(&layout.audit)?;
    let stage = read(&layout.output.join("staging-report.json"))?;
    ensure!(
        stage["roblox_safe"] == true,
        "Staging report does not declare roblox_safe"
    );
    let plan = read(&layout.plan)?;
    let metadata = read(&layout.metadata)?;
    let scrubbed = verify_findings(&audit, result)?;
    let mut checks = BlankChecks::new();
    register_blanks(&plan, &metadata, &mut checks)?;
    register_repairs(layout, &plan, &scrubbed, &mut checks, result)?;
    let page_check = verify_pages(layout, &plan, active, result);
    record(result, "Pages and manifests", page_check);
    verify_pixels(layout, checks, result);
    Ok(())
}
/// Verify a single artifact bundle, staged game, or multi-game catalog.
pub fn verify_port(root: &Path, cloud_manifest: Option<&Path>) -> Result<Verification> {
    ensure!(root.is_dir(), "Port root must be a directory");
    let mut result = Verification::default();
    let mut active = BTreeSet::new();
    let layouts = if root.join("games.json").exists() {
        let catalog = read(&root.join("games.json"))?;
        array(&catalog, "games")?
            .iter()
            .map(|g| {
                let slug = g["slug"].as_str().context("Missing game slug")?;
                ensure!(
                    !slug.is_empty()
                        && slug
                            .chars()
                            .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-'),
                    "Invalid game slug"
                );
                Ok(catalog_layout(root, slug))
            })
            .collect::<Result<Vec<_>>>()?
    } else {
        vec![single_layout(root)]
    };
    ensure!(!layouts.is_empty(), "Empty game catalog");
    for layout in layouts {
        let label = layout.output.display().to_string();
        let check = verify_game(&layout, &mut active, &mut result);
        record(&mut result, &label, check);
    }
    let launcher = root.join("build/StudioContent/content/papas2luau/arcade_icons_1024.png");
    if launcher.exists() {
        active.insert(sha256(&fs::read(launcher)?));
    }
    if let Some(path) = cloud_manifest {
        let manifest = read(path)?;
        let hashes = manifest["by_sha256"]
            .as_object()
            .context("Missing by_sha256 map")?;
        for hash in hashes.keys() {
            if !active.contains(hash) {
                result
                    .violations
                    .push(format!("Cloud manifest stale by_sha256 hash {hash}"));
            }
        }
    }
    Ok(result)
}
fn luau(value: &Value) -> String {
    match value {
        Value::Null => "nil".into(),
        Value::Bool(v) => v.to_string(),
        Value::Number(v) => v.to_string(),
        Value::String(v) => {
            let mut s = String::from("\"");
            for c in v.chars() {
                match c {
                    '"' => s.push_str("\\\""),
                    '\\' => s.push_str("\\\\"),
                    '\n' => s.push_str("\\n"),
                    '\r' => s.push_str("\\r"),
                    '\t' => s.push_str("\\t"),
                    c if c.is_control() => {
                        for b in c.to_string().as_bytes() {
                            s.push_str(&format!("\\{b:03}"));
                        }
                    }
                    c => s.push(c),
                }
            }
            s.push('"');
            s
        }
        Value::Array(v) => format!("{{{}}}", v.iter().map(luau).collect::<Vec<_>>().join(",")),
        Value::Object(v) => format!(
            "{{{}}}",
            v.iter()
                .map(|(k, v)| format!("[{}]={}", luau(&Value::String(k.clone())), luau(v)))
                .collect::<Vec<_>>()
                .join(",")
        ),
    }
}
fn id_table(audit: &Value, key: &str) -> String {
    format!(
        "{{{}}}",
        ids(audit, key)
            .iter()
            .map(|id| format!("[{id}]=true"))
            .collect::<Vec<_>>()
            .join(",")
    )
}
/// Emit membership tables for ghosted nodes and a scrubbed-rectangle list.
pub fn emit_policy(audit: &Value) -> String {
    let rects = audit
        .get("scrubbedRects")
        .or_else(|| audit.get("blank_regions"))
        .cloned()
        .unwrap_or(Value::Array(vec![]));
    format!("--!optimize 2\nreturn {{\n    hiddenNodeIds = {},\n    preservedControls = {},\n    retainedContainers = {},\n    scrubbedRects = {},\n}}\n",id_table(audit,"hide_character_ids"),id_table(audit,"preserved_control_containers"),id_table(audit,"retained_promo_containers"),luau(&rects))
}
#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    #[test]
    fn policy_tables() {
        let emitted = emit_policy(
            &json!({"hide_character_ids":[1,2],"preserved_control_containers":[3],"retained_promo_containers":[4],"blank_regions":[{"x":2,"y":3,"w":4,"h":5}]}),
        );
        assert!(emitted.starts_with("--!optimize 2\n"));
        for s in [
            "hiddenNodeIds",
            "preservedControls",
            "retainedContainers",
            "scrubbedRects",
            "[1]=true",
            "[3]=true",
        ] {
            assert!(emitted.contains(s));
        }
    }
    #[test]
    fn findings_coverage() {
        let mut r = Verification::default();
        verify_findings(&json!({"findings":[{"category":"visible_text","text":"Upgrade Shop","auto_blank":false},{"category":"visible_text","text":"discord.gg/hello","auto_blank":false},{"category":"symbol_or_instance","char_id":5,"auto_hide":true}],"preserved_control_containers":[5]}),&mut r).unwrap();
        assert_eq!(r.violations.len(), 1);
    }
    #[test]
    fn repair_recursion() {
        assert!(uses_scrubbed(
            &json!({"pose":{"children":[{"charId":7}]}}),
            &BTreeSet::from([7])
        ));
    }
}
