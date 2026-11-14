<div align="center">
<img src="assets/icon.png" width="96" alt="sentinel icon" />

# sentinel

Local asset screening, protected image scrubbing, and pre-upload port verification in pure Rust.
</div>

sentinel flags text, scripts, images, and readable place-file content for review under categories aligned with Roblox Community Standards. It runs offline and records evidence, thresholds, content hashes, and image bounds in a JSON ledger.

## Setup

Download a prebuilt binary from [GitHub Releases](https://github.com/3xjn/sentinel/releases) (`sentinel-windows-x86_64.exe`, `sentinel-linux-x86_64`, `sentinel-macos-aarch64`, `sentinel-macos-x86_64`), or install from source with a current stable Rust toolchain:

```bash
cargo install --git https://github.com/3xjn/sentinel
```

Or build locally:

```bash
cargo build --release --bin sentinel
```

The executable is `sentinel` (`target/release/sentinel` or `target\release\sentinel.exe` on Windows). Image scanning uses neural OCR (`ocrs` + `rten`), QR payload decoding (`rqrr`), OpenNSFW ONNX inference, Marqo NSFW-384 ViT inference, and CLIP ViT-B/32 zero-shot screening in RTen. All inference runs locally in Rust; there are no C/C++ inference libraries or cloud moderation calls.

Install these models before scanning images:

| Filename | Download |
| --- | --- |
| `text-detection.onnx` | [ocrs DBNet](https://ocrs-models.s3-accelerate.amazonaws.com/text-detection.onnx) |
| `text-recognition.onnx` | [ocrs CRNN](https://ocrs-models.s3-accelerate.amazonaws.com/text-recognition.onnx) |
| `open-nsfw.onnx` | [OpenNSFW](https://huggingface.co/bluefoxcreation/open-nsfw/resolve/main/open-nsfw.onnx) |
| `marqo-nsfw-384.onnx` | [Marqo NSFW-384](https://huggingface.co/KanariKanaru/nsfw-image-detection-384-onnx/resolve/main/model.onnx) |
| `clip-vision.onnx` | [CLIP ViT-B/32 Vision](https://huggingface.co/Xenova/clip-vit-base-patch32/resolve/main/onnx/vision_model.onnx) |

Model discovery uses `SENTINEL_MODEL_DIR` first, then `%LOCALAPPDATA%\sentinel\models` on Windows (`$HOME/.local/share/sentinel/models` elsewhere), then `./models` if the user model directory does not exist. An explicit environment override is authoritative. Models initialize lazily and are cached once per process, including across library scans. Missing, incompatible, or corrupt models fail the scan with setup guidance. Models are never downloaded automatically during `scan`.

To download the default models automatically:

```bash
sentinel models install
```

To inspect installed models or write a `models.toml` you can customize with additional ONNX classifiers:

```bash
sentinel models list
sentinel models init
```

You can point `SENTINEL_MODELS_CONFIG` at a custom `models.toml` (or edit `<model_dir>/models.toml`) to enable/disable models, tune `[clip]` (`temperature`, `safe_margin`, `disabled_prompts`), or add extra `[[classifiers]]` entries without recompiling.

CLIP zero-shot screening (`clip-vision.onnx`) covers Roblox Community Standards image categories out of the box: drugs & paraphernalia, alcohol, tobacco/vaping, gambling, graphic gore/corpses, realistic firearms/weapons, self-harm, hate symbols, obscene gestures, nudity/suggestive content, and offsite/QR indicators.

To use a custom model directory:

```bash
export SENTINEL_MODEL_DIR=/path/to/models
```

## Use

```bash
sentinel scan ./assets ./other-assets/badge.png --json
sentinel scan ./assets --no-models --json
sentinel scan ./game --profile balanced --threshold offsite_link=0.25
sentinel scrub sheet.png -o clean.png --rect 20,40,180,60 --keep 150,40,20,20
sentinel verify-port ./ports/catalog --cloud-manifest ./ports/catalog/cloud-assets.json
sentinel emit-policy audit.json -o Policies.luau
```

`scan` accepts one or more files or directories, deduplicates overlapping paths, and recursively discovers PNG, JPEG, WebP, Lua/Luau, JSON, TXT, and Roblox place/model files. The default `high-recall` profile favors review; `balanced` raises each threshold by 0.20. Use repeated `--threshold category=score` options to override `offsite_link`, `gore_violence`, `sexual_nudity`, `profanity_slurs`, `regulated_goods`, or `unfiltered_text_input`.

Image OCR scans light and dark alpha composites and upscaled eligible sprite islands. Recognized lines pass through text rules and fuzzy offsite-brand/domain matching; flags retain the recognized string and line bounds. QR scanning attempts normal and inverted grayscale grids, records decoded payloads (or undecodable-grid evidence), and applies text rules to decoded payloads. Existing PNG metadata, badge, tone-cluster, and QR finder heuristics also run.

OpenNSFW uses the supplied model's NHWC `[1,224,224,3]` input: resize to 256 square, center crop to 224, convert RGB to BGR, and subtract `[104,117,123]`. It evaluates the full image and, on images larger than 256 pixels in either dimension, alpha-connected sprite islands at least 48 pixels wide and tall. `onnx_nsfw` flags retain the NSFW probability and evaluated region. The high-recall sexual-nudity threshold remains `0.30`.

`--no-models` explicitly disables neural OCR and OpenNSFW. QR decoding, metadata, text, and visual heuristics still run. This mode provides reduced screening coverage; a clean result is not equivalent to a full scan. Library callers can use `models::ScanOptions` with `place_scan::scan_paths`, `scan_file_with_options`, or `image_scan::scan_image_with_options`; existing `scan` and `scan_image` calls use models by default.

`scrub` accepts repeated `--rect` and `--keep` rectangles as `x,y,width,height`. Kept pixels are protected even when rectangles overlap. The default `--mode auto` samples the two-pixel clean perimeter, clears regions with at least half transparent samples, and otherwise interpolates background colors from clean cardinal neighbors. `transparent` and `inpaint` force either strategy. Output must be PNG.

`verify-port` checks a catalog with `games.json` and `build/{tos-audit,GeneratedGames,packaging,recipe-repairs}` artifacts. It also accepts an individual `build/GeneratedGames/<slug>` directory, or a standalone bundle containing `audit.json`, `staging-report.json`, `plan.json`, `metadata.json`, optional `repairs.json`, `AssetManifest_staged/*.luau`, and `assets/`. It verifies unsafe visible-text blanking, node coverage, page hashes, transparent blank regions, repair variants, PNG references, and optional cloud hash freshness. `--json` emits its verification ledger.

`emit-policy` produces a Luau module starting with `--!optimize 2`. `hiddenNodeIds`, `preservedControls`, and `retainedContainers` are numeric membership tables; `scrubbedRects` comes from the audit's `blank_regions` or `scrubbedRects` list. Runtimes must consume these tables to ghost UI nodes while retaining script lookups.

Exit codes: **0** means no findings, **1** means review flags or verification violations, and **2** means an input or processing error.

## Limits

These checks support human review and do not provide moderation approval. Text lists and filtering proximity can produce false positives and miss contextual or obfuscated content. Place scans extract readable spans; compressed Roblox chunks are not fully parsed. OCR can misread stylized, tiny, rotated, or overlapping text; fuzzy matching favors recall and can flag innocent fragments. OpenNSFW is an older binary classifier, not a general safety model: it can miss cartoon or contextual sexual content and misclassify ordinary artwork. It does not detect gore, weapons, or other policy categories. Sprite crops use alpha connectivity, so touching sprites and opaque sheets can merge; resizing and center cropping can lose details. QR decoding can fail on damaged or unusual codes. Image tone clusters, finder patterns, and badge geometry can miss small or unusual content and mistake ordinary artwork for unsafe material. Multi-scale windows and connected sprite regions improve localization but do not establish meaning. PNG text decompression is bounded to 8 MiB per chunk.

Port verification checks the supplied artifact contract and current pixels; it cannot prove the audit was complete or that a runtime enforces its policies. Review flagged assets and inspect final runtime behavior before uploading.

## Checks

```bash
cargo test --release
```
