//! Model discovery, modular ONNX classifier registry, and process-wide lazy pure-Rust inference engines.
use crate::policy::Category;
use anyhow::{ensure, Context, Result};
use ocrs::{OcrEngine, OcrEngineParams};
use rten::Model;
use rten_tensor::{prelude::*, NdTensor};
use serde::{Deserialize, Serialize};
use std::{
    fs,
    io::Write,
    path::{Path, PathBuf},
    sync::OnceLock,
};

#[derive(Clone, Copy, Debug, Default)]
/// Model-backed scanning is enabled unless explicitly disabled.
pub struct ScanOptions {
    pub no_models: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TensorLayout {
    Nchw,
    Nhwc,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ColorOrder {
    Rgb,
    Bgr,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ResizeMode {
    Exact,
    Crop256Center224,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Postprocess {
    Softmax,
    Sigmoid,
    ProbIndex,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OcrConfig {
    pub detection_file: String,
    pub detection_url: Option<String>,
    pub recognition_file: String,
    pub recognition_url: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ClassifierSpec {
    pub id: String,
    pub file: String,
    pub url: Option<String>,
    pub category: Category,
    #[serde(default = "default_true")]
    pub enabled: bool,
    pub layout: TensorLayout,
    pub color: ColorOrder,
    #[serde(default = "default_resize")]
    pub resize: ResizeMode,
    pub width: u32,
    pub height: u32,
    pub mean: [f32; 3],
    pub std: [f32; 3],
    #[serde(default = "default_scale")]
    pub scale: f32,
    pub postprocess: Postprocess,
    #[serde(default)]
    pub target_index: usize,
    #[serde(default = "default_min_crop")]
    pub min_crop_size: u32,
}

fn default_true() -> bool {
    true
}
fn default_resize() -> ResizeMode {
    ResizeMode::Exact
}
fn default_scale() -> f32 {
    255.0
}
fn default_min_crop() -> u32 {
    48
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ClipConfig {
    #[serde(default = "default_true")]
    pub enabled: bool,
    pub file: String,
    pub url: Option<String>,
    #[serde(default = "default_clip_min_crop")]
    pub min_crop_size: u32,
    #[serde(default = "default_clip_temperature")]
    pub temperature: f32,
    #[serde(default = "default_clip_margin")]
    pub safe_margin: f32,
    #[serde(default)]
    pub disabled_prompts: Vec<String>,
}

impl Default for ClipConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            file: "clip-vision.onnx".into(),
            url: Some(
                "https://huggingface.co/Xenova/clip-vit-base-patch32/resolve/main/onnx/vision_model.onnx"
                    .into(),
            ),
            min_crop_size: default_clip_min_crop(),
            temperature: default_clip_temperature(),
            safe_margin: default_clip_margin(),
            disabled_prompts: Vec::new(),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ModelsConfig {
    pub ocr: OcrConfig,
    #[serde(default)]
    pub classifiers: Vec<ClassifierSpec>,
    #[serde(default)]
    pub clip: ClipConfig,
}

fn default_clip_min_crop() -> u32 {
    192
}
fn default_clip_temperature() -> f32 {
    80.0
}
fn default_clip_margin() -> f32 {
    0.050
}

impl Default for ModelsConfig {
    fn default() -> Self {
        Self {
            ocr: OcrConfig {
                detection_file: "text-detection.onnx".into(),
                detection_url: Some(
                    "https://ocrs-models.s3-accelerate.amazonaws.com/text-detection.onnx".into(),
                ),
                recognition_file: "text-recognition.onnx".into(),
                recognition_url: Some(
                    "https://ocrs-models.s3-accelerate.amazonaws.com/text-recognition.onnx".into(),
                ),
            },
            classifiers: vec![
                ClassifierSpec {
                    id: "onnx_nsfw".into(),
                    file: "open-nsfw.onnx".into(),
                    url: Some(
                        "https://huggingface.co/bluefoxcreation/open-nsfw/resolve/main/open-nsfw.onnx"
                            .into(),
                    ),
                    category: Category::SexualNudity,
                    enabled: true,
                    layout: TensorLayout::Nhwc,
                    color: ColorOrder::Bgr,
                    resize: ResizeMode::Crop256Center224,
                    width: 224,
                    height: 224,
                    mean: [104.0, 117.0, 123.0],
                    std: [1.0, 1.0, 1.0],
                    scale: 1.0,
                    postprocess: Postprocess::ProbIndex,
                    target_index: 1,
                    min_crop_size: 48,
                },
                ClassifierSpec {
                    id: "marqo_nsfw".into(),
                    file: "marqo-nsfw-384.onnx".into(),
                    url: Some(
                        "https://huggingface.co/KanariKanaru/nsfw-image-detection-384-onnx/resolve/main/model.onnx"
                            .into(),
                    ),
                    category: Category::SexualNudity,
                    enabled: true,
                    layout: TensorLayout::Nchw,
                    color: ColorOrder::Rgb,
                    resize: ResizeMode::Exact,
                    width: 384,
                    height: 384,
                    mean: [0.5, 0.5, 0.5],
                    std: [0.5, 0.5, 0.5],
                    scale: 255.0,
                    postprocess: Postprocess::Softmax,
                    target_index: 0,
                    min_crop_size: 192,
                },
            ],
            clip: ClipConfig::default(),
        }
    }
}

pub struct LoadedClassifier {
    pub spec: ClassifierSpec,
    pub model: Model,
}

#[derive(Clone, Debug, Serialize)]
pub struct ModelStatus {
    pub id: String,
    pub kind: String,
    pub file: String,
    pub path: String,
    pub enabled: bool,
    pub installed: bool,
    pub bytes: Option<u64>,
    pub category: Option<Category>,
    pub url: Option<String>,
}

pub fn model_dir() -> PathBuf {
    if let Some(dir) = std::env::var_os("SENTINEL_MODEL_DIR") {
        return dir.into();
    }
    let user_dir = if cfg!(windows) {
        std::env::var_os("LOCALAPPDATA").map(|p| PathBuf::from(p).join("sentinel/models"))
    } else {
        std::env::var_os("HOME").map(|p| PathBuf::from(p).join(".local/share/sentinel/models"))
    };
    user_dir
        .filter(|p| p.is_dir())
        .unwrap_or_else(|| PathBuf::from("models"))
}

fn default_install_dir() -> PathBuf {
    if let Some(dir) = std::env::var_os("SENTINEL_MODEL_DIR") {
        return dir.into();
    }
    if cfg!(windows) {
        std::env::var_os("LOCALAPPDATA")
            .map(|p| PathBuf::from(p).join("sentinel/models"))
            .unwrap_or_else(|| PathBuf::from("models"))
    } else {
        std::env::var_os("HOME")
            .map(|p| PathBuf::from(p).join(".local/share/sentinel/models"))
            .unwrap_or_else(|| PathBuf::from("models"))
    }
}

pub fn config_path() -> Option<PathBuf> {
    if let Some(path) = std::env::var_os("SENTINEL_MODELS_CONFIG") {
        return Some(path.into());
    }
    let in_model_dir = model_dir().join("models.toml");
    if in_model_dir.is_file() {
        return Some(in_model_dir);
    }
    let local = PathBuf::from("models.toml");
    if local.is_file() {
        return Some(local);
    }
    None
}

pub fn load_config() -> Result<ModelsConfig> {
    match config_path() {
        Some(path) => {
            let text = fs::read_to_string(&path)
                .with_context(|| format!("Reading models config {}", path.display()))?;
            toml::from_str(&text)
                .with_context(|| format!("Parsing models config {}", path.display()))
        }
        None => Ok(ModelsConfig::default()),
    }
}

pub fn init_config(force: bool) -> Result<PathBuf> {
    let dir = default_install_dir();
    fs::create_dir_all(&dir)?;
    let path = std::env::var_os("SENTINEL_MODELS_CONFIG")
        .map(PathBuf::from)
        .unwrap_or_else(|| dir.join("models.toml"));
    if !path.exists() || force {
        let content = toml::to_string_pretty(&ModelsConfig::default())?;
        fs::write(&path, content)?;
    }
    Ok(path)
}

pub fn list_models() -> Result<Vec<ModelStatus>> {
    let cfg = load_config()?;
    let dir = model_dir();
    let mut out = Vec::new();
    for (id, file, url) in [
        ("ocr_detection", &cfg.ocr.detection_file, &cfg.ocr.detection_url),
        ("ocr_recognition", &cfg.ocr.recognition_file, &cfg.ocr.recognition_url),
    ] {
        let p = dir.join(file);
        let meta = fs::metadata(&p).ok();
        out.push(ModelStatus {
            id: id.into(),
            kind: "ocr".into(),
            file: file.clone(),
            path: p.display().to_string(),
            enabled: true,
            installed: meta.is_some(),
            bytes: meta.map(|m| m.len()),
            category: None,
            url: url.clone(),
        });
    }
    for c in cfg.classifiers {
        let p = dir.join(&c.file);
        let meta = fs::metadata(&p).ok();
        out.push(ModelStatus {
            id: c.id,
            kind: "classifier".into(),
            file: c.file,
            path: p.display().to_string(),
            enabled: c.enabled,
            installed: meta.is_some(),
            bytes: meta.map(|m| m.len()),
            category: Some(c.category),
            url: c.url,
        });
    }
    let clip_p = dir.join(&cfg.clip.file);
    let clip_meta = fs::metadata(&clip_p).ok();
    out.push(ModelStatus {
        id: "clip_vision".into(),
        kind: "clip".into(),
        file: cfg.clip.file,
        path: clip_p.display().to_string(),
        enabled: cfg.clip.enabled,
        installed: clip_meta.is_some(),
        bytes: clip_meta.map(|m| m.len()),
        category: None,
        url: cfg.clip.url,
    });
    Ok(out)
}

fn download_to(url: &str, dest: &Path) -> Result<u64> {
    if let Some(parent) = dest.parent() {
        fs::create_dir_all(parent)?;
    }
    let resp = ureq::get(url)
        .call()
        .with_context(|| format!("GET {url}"))?;
    let tmp = dest.with_extension("onnx.tmp");
    let mut reader = resp.into_reader();
    let mut file = fs::File::create(&tmp)?;
    let bytes = std::io::copy(&mut reader, &mut file)?;
    file.flush()?;
    drop(file);
    fs::rename(&tmp, dest)?;
    Ok(bytes)
}

pub fn install_models(force: bool) -> Result<Vec<(String, PathBuf, u64)>> {
    let cfg_path = init_config(false)?;
    let cfg: ModelsConfig = toml::from_str(&fs::read_to_string(&cfg_path)?)?;
    let dir = default_install_dir();
    fs::create_dir_all(&dir)?;

    let mut targets: Vec<(String, String, Option<String>)> = vec![
        (
            "ocr_detection".into(),
            cfg.ocr.detection_file,
            cfg.ocr.detection_url,
        ),
        (
            "ocr_recognition".into(),
            cfg.ocr.recognition_file,
            cfg.ocr.recognition_url,
        ),
    ];
    for c in cfg.classifiers.into_iter().filter(|c| c.enabled) {
        targets.push((c.id, c.file, c.url));
    }
    if cfg.clip.enabled {
        targets.push(("clip_vision".into(), cfg.clip.file, cfg.clip.url));
    }

    let mut installed = Vec::new();
    for (id, file, url) in targets {
        let dest = dir.join(&file);
        if dest.is_file() && !force {
            continue;
        }
        let url = url.with_context(|| {
            format!(
                "Model '{id}' ({}) is missing and has no download url in {}",
                dest.display(),
                cfg_path.display()
            )
        })?;
        let bytes = download_to(&url, &dest)?;
        installed.push((id, dest, bytes));
    }
    Ok(installed)
}

pub(crate) fn help() -> String {
    format!(
        "Models in {} must be readable and compatible. Run `sentinel models install` to download defaults, or set SENTINEL_MODEL_DIR / SENTINEL_MODELS_CONFIG. Default sources: text-detection.onnx (https://ocrs-models.s3-accelerate.amazonaws.com/text-detection.onnx), text-recognition.onnx (https://ocrs-models.s3-accelerate.amazonaws.com/text-recognition.onnx), open-nsfw.onnx (https://huggingface.co/bluefoxcreation/open-nsfw/resolve/main/open-nsfw.onnx), marqo-nsfw-384.onnx (https://huggingface.co/KanariKanaru/nsfw-image-detection-384-onnx/resolve/main/model.onnx). Use --no-models only for explicit lightweight scanning.",
        model_dir().display()
    )
}

fn cached<T>(
    cell: &'static OnceLock<std::result::Result<T, String>>,
    init: impl FnOnce() -> Result<T>,
) -> Result<&'static T> {
    cell.get_or_init(|| init().map_err(|e| format!("{e:#}. {}", help())))
        .as_ref()
        .map_err(|e| anyhow::anyhow!("{e}"))
}

static OCR: OnceLock<std::result::Result<OcrEngine, String>> = OnceLock::new();
static CLASSIFIERS: OnceLock<std::result::Result<Vec<LoadedClassifier>, String>> = OnceLock::new();
static CLIP_VISION: OnceLock<std::result::Result<Option<(ClipConfig, Model)>, String>> = OnceLock::new();

pub fn ocr() -> Result<&'static OcrEngine> {
    cached(&OCR, || {
        let cfg = load_config()?;
        let dir = model_dir();
        Ok(OcrEngine::new(OcrEngineParams {
            detection_model: Some(Model::load_file(dir.join(&cfg.ocr.detection_file))?),
            recognition_model: Some(Model::load_file(dir.join(&cfg.ocr.recognition_file))?),
            ..Default::default()
        })?)
    })
}

fn validate_classifier_input(path: &Path, spec: &ClassifierSpec) -> Result<()> {
    let graph = rten_onnx::onnx::ModelProto::parse_file(fs::File::open(path)?)?
        .graph
        .context("Missing ONNX graph")?;
    let input = graph.input.first().context("Missing ONNX input")?;
    let tensor = input
        .r#type
        .as_ref()
        .and_then(|t| t.tensor_type.as_ref())
        .context("Expected tensor input")?;
    let shape = tensor.shape.as_ref().context("Missing input shape")?;
    let expected = match spec.layout {
        TensorLayout::Nchw => [Some(3), Some(i64::from(spec.height)), Some(i64::from(spec.width))],
        TensorLayout::Nhwc => [Some(i64::from(spec.height)), Some(i64::from(spec.width)), Some(3)],
    };
    ensure!(
        shape.dim.len() == 4
            && shape.dim[1..]
                .iter()
                .map(|d| d.dim_value)
                .eq(expected),
        "Classifier '{}' expected {:?} {:?}, got {:?}",
        spec.id,
        spec.layout,
        expected,
        shape
    );
    Ok(())
}

pub fn classifiers() -> Result<&'static [LoadedClassifier]> {
    cached(&CLASSIFIERS, || {
        let cfg = load_config()?;
        let dir = model_dir();
        let mut loaded = Vec::new();
        for spec in cfg.classifiers.into_iter().filter(|c| c.enabled) {
            let path = dir.join(&spec.file);
            validate_classifier_input(&path, &spec)
                .with_context(|| format!("Validating classifier '{}' ({})", spec.id, path.display()))?;
            let model = Model::load_file(&path)
                .with_context(|| format!("Loading classifier '{}' ({})", spec.id, path.display()))?;
            loaded.push(LoadedClassifier { spec, model });
        }
        Ok(loaded)
    })
    .map(Vec::as_slice)
}

pub fn run_classifier(loaded: &LoadedClassifier, img: &image::RgbImage) -> Result<f32> {
    let spec = &loaded.spec;
    ensure!(
        img.width() > 0 && img.height() > 0,
        "Classifier '{}' needs a nonempty image",
        spec.id
    );
    let (w, h) = (spec.width as usize, spec.height as usize);
    let prepared = match spec.resize {
        ResizeMode::Exact => {
            image::imageops::resize(img, spec.width, spec.height, image::imageops::FilterType::CatmullRom)
        }
        ResizeMode::Crop256Center224 => {
            ensure!(
                spec.width == 224 && spec.height == 224,
                "Crop256Center224 requires 224x224"
            );
            let resized = image::imageops::resize(img, 256, 256, image::imageops::FilterType::Triangle);
            image::imageops::crop_imm(&resized, 16, 16, 224, 224).to_image()
        }
    };

    let mut tensor = match spec.layout {
        TensorLayout::Nchw => NdTensor::zeros([1, 3, h, w]),
        TensorLayout::Nhwc => NdTensor::zeros([1, h, w, 3]),
    };

    for y in 0..h {
        for x in 0..w {
            let p = prepared.get_pixel(x as u32, y as u32);
            for c in 0..3 {
                let raw = match spec.color {
                    ColorOrder::Rgb => p[c],
                    ColorOrder::Bgr => p[2 - c],
                };
                let val = (f32::from(raw) / spec.scale - spec.mean[c]) / spec.std[c];
                match spec.layout {
                    TensorLayout::Nchw => tensor[[0, c, y, x]] = val,
                    TensorLayout::Nhwc => tensor[[0, y, x, c]] = val,
                }
            }
        }
    }

    let output = loaded
        .model
        .run_one(tensor.view().into(), None)
        .with_context(|| format!("Inference failed for '{}'", spec.id))?;
    let values = output
        .as_tensor_view::<f32>()
        .with_context(|| format!("Expected float output for '{}'", spec.id))?;
    let data: Vec<_> = values.iter().copied().collect();
    ensure!(
        spec.target_index < data.len() && data.iter().all(|v| v.is_finite()),
        "Invalid output for '{}': {:?}",
        spec.id,
        data
    );

    match spec.postprocess {
        Postprocess::Softmax => {
            let max_v = data.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let exps: Vec<f32> = data.iter().map(|v| (v - max_v).exp()).collect();
            let sum: f32 = exps.iter().sum();
            Ok(exps[spec.target_index] / sum)
        }
        Postprocess::Sigmoid => {
            let z = data[spec.target_index];
            Ok(1.0 / (1.0 + (-z).exp()))
        }
        Postprocess::ProbIndex => {
            ensure!(
                data.iter().all(|p| (0.0..=1.0).contains(p)),
                "Invalid probability output for '{}': {:?}",
                spec.id,
                data
            );
            Ok(data[spec.target_index])
        }
    }
}

pub fn nsfw_score(img: &image::RgbImage) -> Result<f32> {
    let c = classifiers()?
        .iter()
        .find(|c| c.spec.id == "onnx_nsfw")
        .context("Classifier 'onnx_nsfw' is not enabled")?;
    run_classifier(c, img)
}

pub fn marqo_nsfw_score(img: &image::RgbImage) -> Result<f32> {
    let c = classifiers()?
        .iter()
        .find(|c| c.spec.id == "marqo_nsfw")
        .context("Classifier 'marqo_nsfw' is not enabled")?;
    run_classifier(c, img)
}

pub fn clip_vision() -> Result<Option<&'static (ClipConfig, Model)>> {
    cached(&CLIP_VISION, || {
        let cfg = load_config()?;
        if !cfg.clip.enabled {
            return Ok(None);
        }
        let path = model_dir().join(&cfg.clip.file);
        let model = Model::load_file(&path)
            .with_context(|| format!("Loading CLIP vision model ({})", path.display()))?;
        Ok(Some((cfg.clip, model)))
    })
    .map(Option::as_ref)
}

pub fn clip_scores(img: &image::RgbImage) -> Result<Vec<(&'static str, Category, &'static str, f32)>> {
    let Some((cfg, model)) = clip_vision()? else {
        return Ok(Vec::new());
    };
    ensure!(
        img.width() > 0 && img.height() > 0,
        "CLIP vision needs a nonempty image"
    );
    let resized = image::imageops::resize(img, 224, 224, image::imageops::FilterType::CatmullRom);
    let mean = [0.48145466f32, 0.4578275, 0.40821073];
    let std = [0.26862954f32, 0.26130258, 0.27577711];
    let mut tensor = NdTensor::zeros([1, 3, 224, 224]);
    for y in 0..224 {
        for x in 0..224 {
            let p = resized.get_pixel(x, y);
            for c in 0..3 {
                tensor[[0, c, y as usize, x as usize]] =
                    (f32::from(p[c]) / 255.0 - mean[c]) / std[c];
            }
        }
    }
    let output = model
        .run_one(tensor.view().into(), None)
        .context("CLIP vision inference")?;
    let mut iv: Vec<f32> = output
        .as_tensor_view::<f32>()
        .context("Expected float CLIP image embedding")?
        .iter()
        .copied()
        .collect();
    ensure!(iv.len() == 512, "Expected 512-d CLIP embedding, got {}", iv.len());
    let norm = iv.iter().map(|x| x * x).sum::<f32>().sqrt();
    ensure!(norm > 0.0 && norm.is_finite(), "Invalid CLIP embedding norm");
    for x in &mut iv {
        *x /= norm;
    }

    let dot = |a: &[f32], b: &[f32; 512]| -> f32 { a.iter().zip(b.iter()).map(|(x, y)| x * y).sum() };
    let max_safe = crate::clip_prompts::CLIP_PROMPTS
        .iter()
        .filter(|p| p.category.is_none())
        .map(|p| dot(&iv, &p.embed))
        .fold(f32::NEG_INFINITY, f32::max);

    let mut out = Vec::new();
    for p in crate::clip_prompts::CLIP_PROMPTS {
        let Some(cat) = p.category else { continue };
        if cfg.disabled_prompts.iter().any(|d| d == p.id) {
            continue;
        }
        let sim = dot(&iv, &p.embed);
        let prob = 1.0 / (1.0 + ((max_safe + cfg.safe_margin - sim) * cfg.temperature).exp());
        out.push((p.id, cat, p.description, prob));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn corrupt_model_error_includes_setup() {
        static BROKEN: OnceLock<std::result::Result<Model, String>> = OnceLock::new();
        let err = cached(&BROKEN, || Ok(Model::load(vec![0, 1, 2, 3])?))
            .err()
            .unwrap()
            .to_string();
        for expected in [
            "text-detection.onnx",
            "text-recognition.onnx",
            "open-nsfw.onnx",
            "marqo-nsfw-384.onnx",
            "sentinel models install",
            "--no-models",
        ] {
            assert!(err.contains(expected), "{err}");
        }
    }
    #[test]
    fn default_config_roundtrips_toml() {
        let raw = toml::to_string_pretty(&ModelsConfig::default()).unwrap();
        let parsed: ModelsConfig = toml::from_str(&raw).unwrap();
        assert_eq!(parsed.classifiers.len(), 2);
        assert_eq!(parsed.classifiers[0].id, "onnx_nsfw");
        assert_eq!(parsed.classifiers[1].id, "marqo_nsfw");
    }
    #[test]
    fn inspect_graph_and_blank_score() {
        let score =
            nsfw_score(&image::RgbImage::from_pixel(224, 224, image::Rgb([255; 3]))).unwrap();
        assert!((0.0..0.30).contains(&score));
        let colorful = image::RgbImage::from_fn(224, 224, |x, y| {
            image::Rgb([x as u8, y as u8, (x ^ y) as u8])
        });
        assert!((0.0..=1.0).contains(&nsfw_score(&colorful).unwrap()));
        let marqo_white =
            marqo_nsfw_score(&image::RgbImage::from_pixel(384, 384, image::Rgb([255; 3]))).unwrap();
        assert!((0.0..0.30).contains(&marqo_white));
        assert!((0.0..=1.0).contains(&marqo_nsfw_score(&colorful).unwrap()));
    }
}
