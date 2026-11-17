//! Neural OCR and vision, QR decoding, pixel heuristics, and PNG metadata.
use crate::{
    policy::{Category, Policy},
    report::AssetReport,
    scrub::Rect,
    text_scan,
};
use anyhow::{ensure, Context, Result};
use image::{Rgba, RgbaImage};
use rten_imageproc::BoundingRect;
use std::collections::VecDeque;
fn dark(p: &Rgba<u8>) -> bool {
    p[3] > 200 && p[0] < 60 && p[1] < 60 && p[2] < 60
}
fn white(p: &Rgba<u8>) -> bool {
    p[3] > 200 && p[0] > 235 && p[1] > 235 && p[2] > 235
}
fn social_blue(p: &Rgba<u8>) -> bool {
    p[3] > 200
        && [[59, 89, 152], [29, 161, 242]]
            .iter()
            .any(|color| (0..3).all(|i| i16::from(p[i]).abs_diff(color[i]) <= 14))
}
fn blood(p: &Rgba<u8>) -> bool {
    p[3] > 200
        && p[0] >= 95
        && p[0] < 235
        && u16::from(p[0]) > u16::from(p[1]) * 2
        && u16::from(p[0]) > u16::from(p[2]) * 2
        && p[1] < 75
        && p[2] < 85
}
fn skin(p: &Rgba<u8>) -> bool {
    p[3] > 200
        && p[0] > 160
        && p[1] > 95
        && p[2] > 60
        && p[0] > p[1]
        && p[1] > p[2]
        && p[0] - p[2] > 35
        && p[0] - p[1] < 100
}
fn bounds(r: Rect) -> Option<[u32; 4]> {
    Some([r.x, r.y, r.w, r.h])
}
fn component_regions(img: &RgbaImage, predicate: impl Fn(&Rgba<u8>) -> bool) -> Vec<(Rect, usize)> {
    let (w, h) = img.dimensions();
    let mut seen = vec![false; w as usize * h as usize];
    let mut regions = Vec::new();
    for y in 0..h {
        for x in 0..w {
            let index = (y * w + x) as usize;
            if seen[index] || !predicate(img.get_pixel(x, y)) {
                continue;
            }
            seen[index] = true;
            let mut queue = VecDeque::from([(x, y)]);
            let (mut lx, mut ly, mut rx, mut by, mut count) = (x, y, x, y, 0);
            while let Some((px, py)) = queue.pop_front() {
                lx = lx.min(px);
                ly = ly.min(py);
                rx = rx.max(px);
                by = by.max(py);
                count += 1;
                for (nx, ny) in [
                    (px.wrapping_sub(1), py),
                    (px + 1, py),
                    (px, py.wrapping_sub(1)),
                    (px, py + 1),
                ] {
                    if nx >= w || ny >= h {
                        continue;
                    }
                    let i = (ny * w + nx) as usize;
                    if !seen[i] && predicate(img.get_pixel(nx, ny)) {
                        seen[i] = true;
                        queue.push_back((nx, ny));
                    }
                }
            }
            regions.push((
                Rect {
                    x: lx,
                    y: ly,
                    w: rx - lx + 1,
                    h: by - ly + 1,
                },
                count,
            ));
        }
    }
    regions
}
fn regions(img: &RgbaImage) -> Vec<Rect> {
    let (w, h) = img.dimensions();
    let mut out = vec![Rect { x: 0, y: 0, w, h }];
    out.extend(
        component_regions(img, |p| p[3] > 0)
            .into_iter()
            .filter(|(r, n)| *n >= 64 && r.w >= 8 && r.h >= 8)
            .map(|(r, _)| r),
    );
    for predicate in [dark as fn(&Rgba<u8>) -> bool, social_blue] {
        out.extend(
            component_regions(img, predicate)
                .into_iter()
                .filter(|(r, n)| *n >= 128 && r.w >= 24 && r.h >= 18)
                .map(|(r, _)| r),
        );
    }
    if w > 256 || h > 256 {
        for (size, step) in [(128, 64), (256, 128)] {
            for y in (0..h).step_by(step) {
                for x in (0..w).step_by(step) {
                    out.push(Rect {
                        x,
                        y,
                        w: size.min(w - x),
                        h: size.min(h - y),
                    });
                }
            }
        }
    }
    out.sort_by_key(|r| (r.x, r.y, r.w, r.h));
    out.dedup();
    out
}
fn ratio(img: &RgbaImage, r: Rect, predicate: impl Fn(&Rgba<u8>) -> bool) -> f32 {
    let mut count = 0;
    for y in r.y..r.y + r.h {
        for x in r.x..r.x + r.w {
            if predicate(img.get_pixel(x, y)) {
                count += 1;
            }
        }
    }
    count as f32 / (r.w as f32 * r.h as f32).max(1.)
}
fn visual_badge(img: &RgbaImage, r: Rect) -> f32 {
    if r.w < 24 || r.h < 18 {
        return 0.;
    }
    let aspect = r.w as f32 / r.h as f32;
    let white_ratio = ratio(img, r, white);
    if (2.0..=6.5).contains(&aspect)
        && r.h <= 180
        && r.w >= 70
        && ratio(img, r, dark) > 0.72
        && (0.025..0.26).contains(&white_ratio)
    {
        let inner = Rect {
            x: r.x + r.w / 8,
            y: r.y + r.h / 4,
            w: r.w * 3 / 4,
            h: r.h / 2,
        };
        if ratio(img, inner, white) > 0.045 {
            return 0.85;
        }
    }
    if (0.85..=1.18).contains(&aspect) && r.w <= 256 && (0.035..0.30).contains(&white_ratio) {
        let blue = ratio(img, r, social_blue);
        if blue > 0.65 {
            let inner = Rect {
                x: r.x + r.w / 5,
                y: r.y + r.h / 5,
                w: r.w * 3 / 5,
                h: r.h * 3 / 5,
            };
            if ratio(img, inner, white) > 0.08 {
                return 0.9;
            }
        }
    }
    0.
}
fn pattern(runs: &[(bool, u32, u32)]) -> bool {
    if runs.len() != 5 || !runs[0].0 || runs[1].0 || !runs[2].0 || runs[3].0 || !runs[4].0 {
        return false;
    }
    let unit = runs.iter().map(|r| r.2).sum::<u32>() as f32 / 7.;
    unit >= 1.
        && runs
            .iter()
            .zip([1., 1., 3., 1., 1.])
            .all(|(r, n)| (r.2 as f32 - unit * n).abs() <= unit * 0.6)
}
fn finder_line(img: &RgbaImage, fixed: u32, vertical: bool) -> Vec<(u32, u32)> {
    let length = if vertical { img.height() } else { img.width() };
    let mut runs: Vec<(bool, u32, u32)> = Vec::new();
    let mut found = Vec::new();
    for i in 0..length {
        let p = if vertical {
            img.get_pixel(fixed, i)
        } else {
            img.get_pixel(i, fixed)
        };
        let black = dark(p);
        let clean = black || white(p);
        if !clean {
            collect_finders(&runs, &mut found);
            runs.clear();
            continue;
        }
        if let Some(last) = runs.last_mut() {
            if clean && last.0 == black {
                last.2 += 1;
                continue;
            }
        }
        runs.push((black, i, 1));
    }
    collect_finders(&runs, &mut found);
    found
}
fn collect_finders(runs: &[(bool, u32, u32)], found: &mut Vec<(u32, u32)>) {
    found.extend(
        runs.windows(5)
            .filter(|r| pattern(r))
            .map(|r| (r[0].1, r.iter().map(|v| v.2).sum())),
    );
}
fn qr_regions(img: &RgbaImage) -> Vec<Rect> {
    let mut out = Vec::new();
    for y in 0..img.height() {
        for (x, size) in finder_line(img, y, false) {
            let cx = x + size / 2;
            if let Some((top, height)) = finder_line(img, cx, true)
                .into_iter()
                .find(|(top, h)| y >= *top && y < top + h && h.abs_diff(size) <= size / 3 + 1)
            {
                let r = Rect {
                    x,
                    y: top,
                    w: size,
                    h: height,
                };
                if !out.iter().any(|old: &Rect| old.intersection(&r).is_some()) {
                    out.push(r);
                }
            }
        }
    }
    out
}
/// Decode PNG/JPEG/WebP and append localized neural and heuristic evidence.
pub fn scan_image(bytes: &[u8], policy: &Policy, report: &mut AssetReport) -> Result<()> {
    scan_image_with_options(bytes, policy, report, crate::models::ScanOptions::default())
}
pub fn scan_image_with_options(
    bytes: &[u8],
    policy: &Policy,
    report: &mut AssetReport,
    options: crate::models::ScanOptions,
) -> Result<()> {
    let img = image::load_from_memory(bytes)?.to_rgba8();
    if options.allows(Category::OffsiteLink) {
        decode_qr(&img, policy, report);
    }
    if !options.no_models {
        scan_neural(&img, policy, report, options).with_context(crate::models::help)?;
    }
    if !options.nudity_only() {
        for text in png_text(bytes)? {
            text_scan::scan_text(&text, false, policy, report);
        }
    }
    if options.allows(Category::OffsiteLink) {
        for r in qr_regions(&img) {
            report.record(
                policy,
                Category::OffsiteLink,
                "qr_finder",
                0.95,
                "Orthogonal 1:1:3:1:1 finder patterns",
                bounds(r),
            );
        }
    }
    for (predicate, category, detector, min_area, min_ratio) in [
        (
            blood as fn(&Rgba<u8>) -> bool,
            Category::GoreViolence,
            "visual_gore",
            8192,
            0.65,
        ),
        (skin, Category::SexualNudity, "visual_nsfw", 12000, 0.72),
    ] {
        if !options.allows(category) {
            continue;
        }
        for (r, area) in component_regions(&img, predicate) {
            let density = area as f32 / (r.w as f32 * r.h as f32);
            let compact_blood = detector == "visual_gore" && area >= 1024 && density >= 0.90;
            let score = if (area >= min_area && density >= min_ratio) || compact_blood {
                0.5 + 0.4 * density
            } else {
                0.
            };
            if score > 0. {
                report.record(
                    policy,
                    category,
                    detector,
                    score,
                    format!("{area} connected tone pixels; density {density:.3}"),
                    bounds(r),
                );
            }
        }
    }
    if options.allows(Category::OffsiteLink) {
        let mut badges: Vec<Rect> = Vec::new();
        for r in regions(&img) {
            let score = visual_badge(&img, r);
            if score > 0. && !badges.iter().any(|b| b.intersection(&r).is_some()) {
                badges.push(r);
                report.record(
                    policy,
                    Category::OffsiteLink,
                    "visual_badge",
                    score,
                    "Badge geometry, background color and white inner text/glyph",
                    bounds(r),
                );
            }
        }
    }
    Ok(())
}
fn nul(data: &[u8]) -> Result<usize> {
    data.iter()
        .position(|b| *b == 0)
        .context("Malformed PNG text field")
}
/// Extract PNG text, including bounded pure-Rust zlib decompression.
pub fn png_text(bytes: &[u8]) -> Result<Vec<String>> {
    if !bytes.starts_with(b"\x89PNG\r\n\x1a\n") {
        return Ok(vec![]);
    }
    let mut pos = 8;
    let mut out = Vec::new();
    while pos + 12 <= bytes.len() {
        let len = u32::from_be_bytes(bytes[pos..pos + 4].try_into()?) as usize;
        let end = pos
            .checked_add(12)
            .and_then(|p| p.checked_add(len))
            .context("PNG chunk overflow")?;
        ensure!(end <= bytes.len(), "Truncated PNG chunk");
        let tag = &bytes[pos + 4..pos + 8];
        let data = &bytes[pos + 8..pos + 8 + len];
        let decoded = match tag {
            b"tEXt" => Some(data.to_vec()),
            b"zTXt" => {
                let n = nul(data)?;
                ensure!(data.get(n + 1) == Some(&0), "Unsupported PNG compression");
                let mut text = data[..n].to_vec();
                text.push(b' ');
                text.extend(inflate(&data[n + 2..])?);
                Some(text)
            }
            b"iTXt" => {
                let n = nul(data)?;
                ensure!(data.len() > n + 3, "Truncated iTXt");
                let compressed = data[n + 1];
                ensure!(
                    compressed <= 1 && data[n + 2] == 0,
                    "Invalid iTXt compression"
                );
                let rest = &data[n + 3..];
                let language = nul(rest)?;
                let rest = &rest[language + 1..];
                let translated = nul(rest)?;
                let payload = &rest[translated + 1..];
                Some(if compressed == 1 {
                    inflate(payload)?
                } else {
                    payload.to_vec()
                })
            }
            _ => None,
        };
        if let Some(text) = decoded {
            out.push(String::from_utf8_lossy(&text).replace('\0', " "));
        }
        pos = end;
        if tag == b"IEND" {
            break;
        }
    }
    Ok(out)
}
struct Bits<'a> {
    data: &'a [u8],
    pos: usize,
}
impl Bits<'_> {
    fn read(&mut self, n: usize) -> Result<u32> {
        let mut v = 0;
        for i in 0..n {
            let b = *self.data.get(self.pos / 8).context("Truncated DEFLATE")?;
            v |= u32::from((b >> (self.pos % 8)) & 1) << i;
            self.pos += 1;
        }
        Ok(v)
    }
}
struct Huffman(Vec<(u32, usize, usize)>);
impl Huffman {
    fn new(lengths: &[u32]) -> Result<Self> {
        let mut counts = [0u32; 16];
        for &n in lengths {
            ensure!(n <= 15, "Invalid Huffman length");
            if n > 0 {
                counts[n as usize] += 1;
            }
        }
        let mut next = [0u32; 16];
        let mut code = 0;
        for bits in 1..16 {
            code = (code + counts[bits - 1]) << 1;
            ensure!(
                code + counts[bits] <= 1 << bits,
                "Oversubscribed Huffman tree"
            );
            next[bits] = code;
        }
        let mut out = Vec::new();
        for (symbol, &n) in lengths.iter().enumerate() {
            if n > 0 {
                let c = next[n as usize];
                next[n as usize] += 1;
                out.push((c.reverse_bits() >> (32 - n), n as usize, symbol));
            }
        }
        Ok(Self(out))
    }
    fn decode(&self, b: &mut Bits) -> Result<usize> {
        let mut code = 0;
        for n in 1..=15 {
            code |= b.read(1)? << (n - 1);
            if let Some(v) = self.0.iter().find(|v| v.0 == code && v.1 == n) {
                return Ok(v.2);
            }
        }
        anyhow::bail!("Invalid Huffman code")
    }
}
fn dynamic(b: &mut Bits) -> Result<(Huffman, Huffman)> {
    let literals = b.read(5)? as usize + 257;
    let distances = b.read(5)? as usize + 1;
    let count = b.read(4)? as usize + 4;
    // RFC 1951 code-length alphabet order.
    let order = [
        16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15,
    ];
    let mut lens = [0; 19];
    for &i in &order[..count] {
        lens[i] = b.read(3)?;
    }
    let tree = Huffman::new(&lens)?;
    let mut all = Vec::new();
    while all.len() < literals + distances {
        let symbol = tree.decode(b)?;
        match symbol {
            0..=15 => all.push(symbol as u32),
            16..=18 => {
                let (n, value) = match symbol {
                    16 => (
                        b.read(2)? + 3,
                        *all.last().context("Missing previous code length")?,
                    ),
                    17 => (b.read(3)? + 3, 0),
                    _ => (b.read(7)? + 11, 0),
                };
                ensure!(
                    all.len() + n as usize <= literals + distances,
                    "Code length overflow"
                );
                all.extend(std::iter::repeat_n(value, n as usize));
            }
            _ => unreachable!(),
        }
    }
    Ok((
        Huffman::new(&all[..literals])?,
        Huffman::new(&all[literals..])?,
    ))
}
fn inflate(data: &[u8]) -> Result<Vec<u8>> {
    ensure!(
        data.len() >= 6
            && data[0] & 15 == 8
            && (u16::from(data[0]) * 256 + u16::from(data[1])) % 31 == 0
            && data[1] & 32 == 0,
        "Invalid zlib header"
    );
    let mut b = Bits {
        data: &data[2..data.len() - 4],
        pos: 0,
    };
    let mut out = Vec::new();
    loop {
        let final_block = b.read(1)?;
        let kind = b.read(2)?;
        if kind == 0 {
            b.pos = b.pos.div_ceil(8) * 8;
            let n = b.read(16)?;
            ensure!(n ^ b.read(16)? == 65535, "Invalid stored block");
            for _ in 0..n {
                out.push(b.read(8)? as u8);
            }
        } else {
            let (lit, dist) = match kind {
                1 => {
                    let mut l = vec![8; 288];
                    l[144..256].fill(9);
                    l[256..280].fill(7);
                    (Huffman::new(&l)?, Huffman::new(&[5; 32])?)
                }
                2 => dynamic(&mut b)?,
                _ => anyhow::bail!("Reserved DEFLATE block"),
            };
            decode_block(&mut b, &lit, &dist, &mut out)?;
        }
        ensure!(out.len() <= 8 * 1024 * 1024, "PNG text exceeds 8 MiB limit");
        if final_block == 1 {
            break;
        }
    }
    let (mut a, mut c) = (1u32, 0u32);
    for v in &out {
        a = (a + u32::from(*v)) % 65521;
        c = (c + a) % 65521;
    }
    ensure!(
        (c << 16) | a == u32::from_be_bytes(data[data.len() - 4..].try_into()?),
        "Invalid zlib checksum"
    );
    Ok(out)
}
fn decode_block(b: &mut Bits, lit: &Huffman, dist: &Huffman, out: &mut Vec<u8>) -> Result<()> {
    const LENGTH: [u32; 29] = [
        3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31, 35, 43, 51, 59, 67, 83, 99, 115,
        131, 163, 195, 227, 258,
    ];
    const EXTRA: [usize; 29] = [
        0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0,
    ];
    const DIST: [u32; 30] = [
        1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193, 257, 385, 513, 769, 1025, 1537,
        2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577,
    ];
    loop {
        ensure!(out.len() <= 8 * 1024 * 1024, "PNG text exceeds limit");
        match lit.decode(b)? {
            n @ 0..=255 => out.push(n as u8),
            256 => return Ok(()),
            n @ 257..=285 => {
                let n = n - 257;
                let length = LENGTH[n] + b.read(EXTRA[n])?;
                let d = dist.decode(b)?;
                ensure!(d < 30, "Invalid distance code");
                let distance = (DIST[d] + b.read(if d < 4 { 0 } else { d / 2 - 1 })?) as usize;
                ensure!(distance <= out.len(), "Distance exceeds output");
                for _ in 0..length {
                    out.push(out[out.len() - distance]);
                }
            }
            _ => anyhow::bail!("Invalid length code"),
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    fn ledger(img: &RgbaImage) -> AssetReport {
        let mut bytes = std::io::Cursor::new(Vec::new());
        img.write_to(&mut bytes, image::ImageFormat::Png).unwrap();
        let mut r = AssetReport::new("test".into(), bytes.get_ref(), "image");
        scan_image(bytes.get_ref(), &Policy::default(), &mut r).unwrap();
        r
    }
    #[test]
    fn qr_finder() {
        let mut img = RgbaImage::from_pixel(90, 90, Rgba([255; 4]));
        for y in 10..80 {
            for x in 10..80 {
                let (a, b) = ((x - 10) / 10, (y - 10) / 10);
                if a == 0
                    || a == 6
                    || b == 0
                    || b == 6
                    || ((2..=4).contains(&a) && (2..=4).contains(&b))
                {
                    img.put_pixel(x, y, Rgba([0, 0, 0, 255]));
                }
            }
        }
        assert!(ledger(&img).flags.iter().any(|f| f.detector == "qr_finder"));
    }
    #[test]
    fn tone_clusters() {
        for (color, detector) in [
            ([180, 15, 25, 255], "visual_gore"),
            ([225, 170, 130, 255], "visual_nsfw"),
        ] {
            let img = RgbaImage::from_pixel(128, 128, Rgba(color));
            assert!(ledger(&img).flags.iter().any(|f| f.detector == detector));
        }
    }
    #[test]
    fn badges() {
        for social in [false, true] {
            let mut img = RgbaImage::from_pixel(
                if social { 64 } else { 160 },
                64,
                Rgba(if social {
                    [59, 89, 152, 255]
                } else {
                    [15, 15, 15, 255]
                }),
            );
            for y in 20..42 {
                for x in 25..40 {
                    img.put_pixel(x, y, Rgba([255; 4]));
                }
            }
            assert!(ledger(&img)
                .flags
                .iter()
                .any(|f| f.detector == "visual_badge"));
        }
    }
    #[test]
    fn zlib_stored() {
        let bytes = [120, 1, 1, 3, 0, 252, 255, 97, 98, 99, 2, 77, 1, 39];
        assert_eq!(inflate(&bytes).unwrap(), b"abc");
    }
    #[test]
    fn compressed_metadata() {
        let fixed = [
            120, 156, 203, 72, 205, 201, 201, 87, 40, 207, 47, 202, 73, 1, 0, 26, 11, 4, 93,
        ];
        assert_eq!(inflate(&fixed).unwrap(), b"hello world");
        let dynamic = [
            120, 156, 237, 202, 177, 13, 0, 16, 16, 64, 209, 85, 76, 112, 102, 18, 39, 23, 9, 33,
            20, 103, 124, 141, 206, 10, 191, 123, 197, 211, 186, 243, 88, 42, 102, 177, 156, 212,
            103, 43, 193, 221, 229, 89, 242, 232, 65, 41, 20, 10, 133, 66, 161, 80, 40, 20, 10,
            229, 47, 23, 238, 131, 161, 91,
        ];
        let expected = "discord.gg/example www.example.com ".repeat(90);
        assert_eq!(inflate(&dynamic).unwrap(), expected.as_bytes());
        for tag in [b"zTXt", b"iTXt"] {
            let mut data = if tag == b"zTXt" {
                b"comment\0\0".to_vec()
            } else {
                b"comment\0\x01\0en\0Description\0".to_vec()
            };
            data.extend(dynamic);
            let mut png = b"\x89PNG\r\n\x1a\n".to_vec();
            png.extend((data.len() as u32).to_be_bytes());
            png.extend(tag);
            png.extend(data);
            png.extend([0; 4]);
            assert!(png_text(&png).unwrap()[0].contains(&expected));
        }
        assert!(inflate(&fixed[..fixed.len() - 1]).is_err());
    }
    #[test]
    fn small_badge_on_opaque_sheet() {
        let mut img = RgbaImage::from_pixel(1024, 1024, Rgba([250, 250, 250, 255]));
        for y in 300..364 {
            for x in 300..364 {
                img.put_pixel(x, y, Rgba([29, 161, 242, 255]));
            }
        }
        for y in 320..346 {
            for x in 325..340 {
                img.put_pixel(x, y, Rgba([255; 4]));
            }
        }
        let r = ledger(&img);
        assert!(r
            .flags
            .iter()
            .any(|f| f.detector == "visual_badge" && f.rect == Some([300, 300, 64, 64])));
    }
}

fn composite(img: &RgbaImage, background: u8) -> image::RgbImage {
    image::RgbImage::from_fn(img.width(), img.height(), |x, y| {
        let p = img.get_pixel(x, y);
        image::Rgb(std::array::from_fn(|c| {
            ((u32::from(p[c]) * u32::from(p[3])
                + u32::from(background) * (255 - u32::from(p[3]))
                + 127)
                / 255) as u8
        }))
    })
}
fn decode_qr(img: &RgbaImage, policy: &Policy, report: &mut AssetReport) {
    let rgb = composite(img, 255);
    let mut luma = image::DynamicImage::ImageRgb8(rgb).into_luma8();
    for _ in 0..2 {
        let mut prepared = rqrr::PreparedImage::prepare(luma.clone());
        for grid in prepared.detect_grids() {
            let min_x = grid.bounds.iter().map(|p| p.x).min().unwrap().max(0) as u32;
            let min_y = grid.bounds.iter().map(|p| p.y).min().unwrap().max(0) as u32;
            let max_x = grid.bounds.iter().map(|p| p.x).max().unwrap().max(0) as u32;
            let max_y = grid.bounds.iter().map(|p| p.y).max().unwrap().max(0) as u32;
            let rect = Some([min_x, min_y, max_x - min_x + 1, max_y - min_y + 1]);
            let payload = grid.decode().ok().map(|(_, text)| text);
            report.record(
                policy,
                Category::OffsiteLink,
                "qr_code",
                0.99,
                payload
                    .as_deref()
                    .unwrap_or("QR code grid detected (undecodable payload)"),
                rect,
            );
            if let Some(text) = payload {
                text_scan::scan_text(&text, false, policy, report);
            }
        }
        image::imageops::invert(&mut luma);
    }
}
fn edit_distance(a: &str, b: &str) -> usize {
    let mut row: Vec<usize> = (0..=b.len()).collect();
    for (i, ac) in a.bytes().enumerate() {
        let mut prev = row[0];
        row[0] = i + 1;
        for (j, bc) in b.bytes().enumerate() {
            let old = row[j + 1];
            row[j + 1] = (prev + usize::from(ac != bc)).min(row[j] + 1).min(old + 1);
            prev = old;
        }
    }
    row[b.len()]
}
fn fuzzy_offsite(text: &str) -> bool {
    let lower = text.to_ascii_lowercase();
    let joined: String = lower.chars().filter(|c| !c.is_whitespace()).collect();
    if [".com", ".net", ".org", ".gg", "itch.io"]
        .iter()
        .any(|s| joined.contains(s))
    {
        return true;
    }
    // OCR often drops the tiny dot entirely, especially in reversed lettering.
    let tokens: Vec<_> = lower
        .split(|c: char| !c.is_ascii_alphanumeric())
        .filter(|s| !s.is_empty())
        .collect();
    if tokens.windows(2).any(|pair| {
        pair[0].len() >= 4 && ["com", "net", "org", "gg", "c0m", "corn"].contains(&pair[1])
    }) {
        return true;
    }
    let compact: String = lower.chars().filter(char::is_ascii_alphanumeric).collect();
    [
        "jayisgames",
        "flipline",
        "newgrounds",
        "armorgames",
        "kongregate",
        "googleplay",
        "appstore",
        "amazon",
        "facebook",
        "twitter",
        "discord",
        "youtube",
        "steam",
        "itchio",
    ]
    .iter()
    .any(|brand| {
        let tolerance = if brand.len() >= 8 { 2 } else { 1 };
        if compact.contains(brand) || edit_distance(&compact, brand) <= tolerance {
            return true;
        }
        if tokens
            .windows(2)
            .any(|pair| edit_distance(&format!("{}{}", pair[0], pair[1]), brand) <= tolerance)
        {
            return true;
        }
        // Require a substantial fragment: short random OCR noise is not a brand.
        lower
            .split(|c: char| !c.is_ascii_alphanumeric())
            .filter(|s| s.len() >= 5)
            .any(|token| {
                edit_distance(token, brand) <= tolerance
                    || (token.len() >= 6 && brand.contains(token))
            })
    })
}
fn scan_ocr(
    rgb: &image::RgbImage,
    offset: (u32, u32),
    scale: f32,
    policy: &Policy,
    report: &mut AssetReport,
) -> Result<()> {
    use ocrs::TextItem;
    let engine = crate::models::ocr()?;
    let source = ocrs::ImageSource::from_bytes(rgb.as_raw(), rgb.dimensions())?;
    let input = engine.prepare_input(source)?;
    let words = engine.detect_words(&input)?;
    let lines = engine.find_text_lines(&input, &words);
    for line in engine.recognize_text(&input, &lines)?.into_iter().flatten() {
        let text = line.to_string();
        let r = line.rotated_rect().bounding_rect();
        let x = (r.left().max(0.) / scale).floor() as u32 + offset.0;
        let y = (r.top().max(0.) / scale).floor() as u32 + offset.1;
        let right = (r.right().max(0.) / scale).ceil() as u32 + offset.0;
        let bottom = (r.bottom().max(0.) / scale).ceil() as u32 + offset.1;
        let rect = Some([x, y, right - x, bottom - y]);
        let mut text_report = AssetReport::new(String::new(), &[], "text");
        text_scan::scan_text(&text, false, policy, &mut text_report);
        for flag in text_report.flags {
            report.record(policy, flag.category, "ocr", flag.score, &text, rect);
        }
        if fuzzy_offsite(&text)
            && !report.flags.iter().any(|f| {
                f.detector == "ocr" && f.rect == rect && f.category == Category::OffsiteLink
            })
        {
            report.record(policy, Category::OffsiteLink, "ocr", 0.85, &text, rect);
        }
        #[cfg(test)]
        eprintln!("OCR {rect:?}: {text}");
    }
    Ok(())
}
fn tile_has_detail(crop: &image::RgbImage) -> bool {
    let step_x = (crop.width() / 16).max(1);
    let step_y = (crop.height() / 16).max(1);
    let mut count = 0u64;
    let mut sum = [0f64; 3];
    let mut sum_sq = [0f64; 3];
    for y in (0..crop.height()).step_by(step_y as usize) {
        for x in (0..crop.width()).step_by(step_x as usize) {
            let p = crop.get_pixel(x, y);
            for c in 0..3 {
                let v = f64::from(p[c]);
                sum[c] += v;
                sum_sq[c] += v * v;
            }
            count += 1;
        }
    }
    if count < 2 {
        return true;
    }
    let n = count as f64;
    let max_var = (0..3)
        .map(|c| (sum_sq[c] / n) - (sum[c] / n).powi(2))
        .fold(0.0f64, f64::max);
    max_var >= 64.0
}

fn scan_neural(
    img: &RgbaImage,
    policy: &Policy,
    report: &mut AssetReport,
    options: crate::models::ScanOptions,
) -> Result<()> {
    // Validate all engines before beginning work; failures must never become clean scans.
    let run_ocr = !options.nudity_only();
    if run_ocr {
        crate::models::ocr()?;
    }
    let classifiers = crate::models::classifiers()?;
    let clip_cfg = if options.nudity_only() {
        None
    } else {
        crate::models::clip_vision()?.map(|(c, _)| c)
    };
    let light = composite(img, 255);
    if run_ocr {
        scan_ocr(&light, (0, 0), 1., policy, report)?;
        if img.pixels().any(|p| p[3] < 255) {
            scan_ocr(&composite(img, 0), (0, 0), 1., policy, report)?;
        }
    }
    let full = Rect {
        x: 0,
        y: 0,
        w: img.width(),
        h: img.height(),
    };
    let mut candidates = vec![full];
    if !options.fast && (img.width() > 256 || img.height() > 256) {
        let islands: Vec<Rect> = component_regions(img, |p| p[3] > 0)
            .into_iter()
            .filter(|(r, _)| r.w >= 48 && r.h >= 48 && *r != full)
            .map(|(r, _)| r)
            .collect();
        if islands.is_empty() {
            // Opaque image with no alpha islands: add a 3x3 coarse grid (45%, >= 192px)
            // plus a 50%-overlap fine grid (~28%, >= 160px) so small embedded graphics
            // never get split across tile seams.
            let cw = ((img.width() as f32 * 0.45).round() as u32).max(192).min(img.width());
            let ch = ((img.height() as f32 * 0.45).round() as u32).max(192).min(img.height());
            let sx = img.width().saturating_sub(cw);
            let sy = img.height().saturating_sub(ch);
            for gy in 0..3u32 {
                for gx in 0..3u32 {
                    let r = Rect { x: (sx * gx) / 2, y: (sy * gy) / 2, w: cw, h: ch };
                    if r != full && !candidates.contains(&r) {
                        candidates.push(r);
                    }
                }
            }
            let fw = ((img.width() as f32 * 0.28).round() as u32).max(160).min(img.width());
            let fh = ((img.height() as f32 * 0.28).round() as u32).max(160).min(img.height());
            let step_x = (fw / 2).max(1);
            let step_y = (fh / 2).max(1);
            let mut y = 0u32;
            loop {
                let ry = y.min(img.height().saturating_sub(fh));
                let mut x = 0u32;
                loop {
                    let rx = x.min(img.width().saturating_sub(fw));
                    let r = Rect { x: rx, y: ry, w: fw, h: fh };
                    if r != full && !candidates.contains(&r) {
                        candidates.push(r);
                    }
                    if rx + fw >= img.width() { break; }
                    x += step_x;
                }
                if ry + fh >= img.height() { break; }
                y += step_y;
            }
        } else {
            candidates.extend(islands);
        }
    }

    use rayon::prelude::*;
    type Finding = (Category, String, f32, String, Option<[u32; 4]>);
    let tile_results: Vec<Result<(Vec<Finding>, Option<image::RgbImage>)>> = candidates
        .par_iter()
        .map(|&r| {
            let crop = image::imageops::crop_imm(&light, r.x, r.y, r.w, r.h).to_image();
            if r != full && !tile_has_detail(&crop) {
                return Ok((Vec::new(), None));
            }
            let mut findings = Vec::new();
            for classifier in classifiers {
                if !options.allows(classifier.spec.category) {
                    continue;
                }
                let min_crop = classifier.spec.min_crop_size;
                if r == full || (r.w >= min_crop && r.h >= min_crop) {
                    let score = crate::models::run_classifier(classifier, &crop)?;
                    findings.push((
                        classifier.spec.category,
                        classifier.spec.id.clone(),
                        score,
                        format!("{} probability {score:.6}", classifier.spec.id),
                        bounds(r),
                    ));
                }
            }
            if let Some(cfg) = clip_cfg {
                if r == full || (r.w >= cfg.min_crop_size && r.h >= cfg.min_crop_size) {
                    for (id, cat, desc, prob) in crate::models::clip_scores(&crop)? {
                        if options.allows(cat) {
                            findings.push((
                                cat,
                                id.to_string(),
                                prob,
                                format!("CLIP {desc} ({prob:.6})"),
                                bounds(r),
                            ));
                        }
                    }
                }
            }
            let ocr_crop = if run_ocr && r != full && (r.w < 400 || r.h < 120) {
                Some(crop)
            } else {
                None
            };
            Ok((findings, ocr_crop))
        })
        .collect();

    for (r, res) in candidates.into_iter().zip(tile_results) {
        let (findings, ocr_crop) = res?;
        for (cat, detector, score, evidence, rect) in findings {
            report.record(policy, cat, &detector, score, evidence, rect);
        }
        // Upscale small atlas islands so text remains legible to DBNet.
        if let Some(crop) = ocr_crop {
            let enlarged = image::imageops::resize(
                &crop,
                r.w * 2,
                r.h * 2,
                image::imageops::FilterType::CatmullRom,
            );
            scan_ocr(&enlarged, (r.x, r.y), 2., policy, report)?;
            let dark_crop = image::imageops::crop_imm(img, r.x, r.y, r.w, r.h).to_image();
            let enlarged = image::imageops::resize(
                &composite(&dark_crop, 0),
                r.w * 2,
                r.h * 2,
                image::imageops::FilterType::CatmullRom,
            );
            scan_ocr(&enlarged, (r.x, r.y), 2., policy, report)?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod neural_tests {
    use super::*;
    #[test]
    fn fuzzy_ocr_fragments_and_typos() {
        for text in [
            "JAY1SGAMES",
            "GoogIe Play",
            "download on the GoogIe Play",
            "Fliplne",
            "newground",
            "discord",
            "WWW. EXAMPLE. COM",
            "example COM",
            "app st0re",
            "stean",
        ] {
            assert!(fuzzy_offsite(text), "Missed {text}");
        }
        for text in [
            "BUILDING",
            "FINISHING",
            "CUSTOMERS",
            "TOTAL CUSTOMER",
            "UPGRADES",
            "GRILLING",
        ] {
            assert!(!fuzzy_offsite(text), "Unexpected match: {text}");
        }
    }
    fn scan_pixels(img: &RgbaImage) -> AssetReport {
        let mut bytes = std::io::Cursor::new(Vec::new());
        img.write_to(&mut bytes, image::ImageFormat::Png).unwrap();
        let mut report = AssetReport::new("pixels".into(), bytes.get_ref(), "image");
        scan_image(bytes.get_ref(), &Policy::default(), &mut report).unwrap();
        report
    }
    #[test]
    fn bitmap_url_is_read_by_neural_ocr() {
        use font8x8::UnicodeFonts;
        let text = "WWW.EXAMPLE.COM";
        let mut img = RgbaImage::from_pixel(40 + text.len() as u32 * 24, 80, Rgba([255; 4]));
        for (i, c) in text.chars().enumerate() {
            let glyph = font8x8::BASIC_FONTS.get(c).unwrap();
            for (y, row) in glyph.iter().enumerate() {
                for x in 0..8 {
                    if row & (1 << x) != 0 {
                        for sy in 0..3 {
                            for sx in 0..3 {
                                img.put_pixel(
                                    20 + i as u32 * 24 + x * 3 + sx,
                                    25 + y as u32 * 3 + sy,
                                    Rgba([0, 0, 0, 255]),
                                );
                            }
                        }
                    }
                }
            }
        }
        let original = img.clone();
        // Opaque and transparent backgrounds, with both black and white lettering.
        for transparent in [false, true] {
            for inverted in [false, true] {
                img = original.clone();
                for p in img.pixels_mut() {
                    let background = p[0] == 255;
                    if inverted {
                        p[0] = 255 - p[0];
                        p[1] = 255 - p[1];
                        p[2] = 255 - p[2];
                    }
                    if transparent && background {
                        p[3] = 0;
                    }
                }
                let report = scan_pixels(&img);
                assert!(
                    report.flags.iter().any(|f| f.detector == "ocr"
                        && f.category == Category::OffsiteLink
                        && f.rect.is_some()
                        && f.evidence.to_ascii_lowercase().contains("example")),
                    "transparent={transparent} inverted={inverted}: {report:?}"
                );
            }
        }
    }
    #[test]
    fn real_qr_payload_and_inverted_payload() {
        let url = "https://example.com/sentinel";
        let code = qrcode::QrCode::new(url).unwrap();
        let size = code.width() as u32;
        let mut img = RgbaImage::from_pixel((size + 8) * 6, (size + 8) * 6, Rgba([255; 4]));
        for y in 0..size {
            for x in 0..size {
                if code[(x as usize, y as usize)] == qrcode::Color::Dark {
                    for sy in 0..6 {
                        for sx in 0..6 {
                            img.put_pixel((x + 4) * 6 + sx, (y + 4) * 6 + sy, Rgba([0, 0, 0, 255]));
                        }
                    }
                }
            }
        }
        for _ in 0..2 {
            let mut report = AssetReport::new("qr".into(), &[], "image");
            decode_qr(&img, &Policy::default(), &mut report);
            assert!(
                report
                    .flags
                    .iter()
                    .any(|f| f.detector == "qr_code" && f.evidence == url && f.rect.is_some()),
                "{report:?}"
            );
            image::imageops::invert(&mut img);
        }
    }
    #[test]
    fn ui_atlas_is_clean() {
        use font8x8::UnicodeFonts;
        // Synthesize a 1024x1024 transparent sprite atlas with UI buttons and labels
        // so the test is self-contained and exercises multi-island OCR + NSFW gating.
        let mut img = RgbaImage::from_pixel(1024, 1024, Rgba([0, 0, 0, 0]));
        let islands = [
            (2u32, 2u32, 156u32, 190u32, [46u8, 125, 50, 255], "SHOP"),
            (162, 2, 116, 71, [21, 101, 192, 255], "UPGRADES"),
            (282, 2, 87, 87, [239, 108, 0, 255], "PLAY"),
            (373, 2, 74, 66, [106, 27, 154, 255], "SAVE"),
            (527, 2, 93, 89, [0, 131, 143, 255], "MENU"),
        ];
        for (ix, iy, iw, ih, bg, label) in islands {
            for y in iy..(iy + ih) {
                for x in ix..(ix + iw) {
                    img.put_pixel(x, y, Rgba(bg));
                }
            }
            for (ci, ch) in label.chars().enumerate() {
                if let Some(glyph) = font8x8::BASIC_FONTS.get(ch) {
                    for (row, bits) in glyph.iter().enumerate() {
                        for col in 0..8 {
                            if (bits >> col) & 1 == 1 {
                                let px = ix + 8 + (ci as u32) * 10 + col as u32;
                                let py = iy + 16 + row as u32;
                                if px < ix + iw && py < iy + ih {
                                    img.put_pixel(px, py, Rgba([255, 255, 255, 255]));
                                }
                            }
                        }
                    }
                }
            }
        }
        let mut bytes = Vec::new();
        image::DynamicImage::ImageRgba8(img)
            .write_to(&mut std::io::Cursor::new(&mut bytes), image::ImageFormat::Png)
            .unwrap();
        let mut report = AssetReport::new("ui_atlas.png".into(), &bytes, "image");
        scan_image(&bytes, &Policy::default(), &mut report).unwrap();
        assert!(!report.blocked, "{report:#?}");
    }
}
