//! Background-aware scrubbing with immutable protected rectangles.
use clap::ValueEnum;
use image::{Rgba, RgbaImage};
use serde::{Deserialize, Serialize};
/// An axis-aligned pixel rectangle, with exclusive right/bottom edges.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Rect {
    pub x: u32,
    pub y: u32,
    pub w: u32,
    pub h: u32,
}
impl Rect {
    /// Test containment without overflowing coordinates.
    pub fn contains(&self, x: u32, y: u32) -> bool {
        x >= self.x && y >= self.y && x - self.x < self.w && y - self.y < self.h
    }
    /// Return the positive-area intersection.
    pub fn intersection(&self, other: &Self) -> Option<Self> {
        let x = self.x.max(other.x);
        let y = self.y.max(other.y);
        let right = self
            .x
            .saturating_add(self.w)
            .min(other.x.saturating_add(other.w));
        let bottom = self
            .y
            .saturating_add(self.h)
            .min(other.y.saturating_add(other.h));
        (right > x && bottom > y).then(|| Self {
            x,
            y,
            w: right - x,
            h: bottom - y,
        })
    }
}
/// A scrub/kept collision, including indices and intersection.
#[derive(Debug, Clone, Serialize)]
pub struct Overlap {
    pub scrub_index: usize,
    pub kept_index: usize,
    pub intersection: Rect,
}
/// Find all positive-area collisions between scrub and protected regions.
pub fn find_rect_overlaps(scrub_rects: &[Rect], kept_rects: &[Rect]) -> Vec<Overlap> {
    scrub_rects
        .iter()
        .enumerate()
        .flat_map(|(i, s)| {
            kept_rects.iter().enumerate().filter_map(move |(j, k)| {
                s.intersection(k).map(|intersection| Overlap {
                    scrub_index: i,
                    kept_index: j,
                    intersection,
                })
            })
        })
        .collect()
}
/// Background strategy; auto chooses transparency from a two-pixel perimeter.
#[derive(Debug, Clone, Copy, Default, ValueEnum)]
pub enum ScrubMode {
    #[default]
    Auto,
    Transparent,
    Inpaint,
}
fn inside(rects: &[Rect], x: u32, y: u32) -> bool {
    rects.iter().any(|r| r.contains(x, y))
}
fn perimeter(img: &RgbaImage, r: Rect, all: &[Rect]) -> Vec<(u32, u32)> {
    let mut out = Vec::new();
    for y in r.y.saturating_sub(2)..r.y.saturating_add(r.h).saturating_add(2).min(img.height()) {
        for x in r.x.saturating_sub(2)..r.x.saturating_add(r.w).saturating_add(2).min(img.width()) {
            if !inside(all, x, y) {
                out.push((x, y));
            }
        }
    }
    out
}
fn directional_samples(img: &RgbaImage, x: u32, y: u32, all: &[Rect]) -> Vec<(u32, u32)> {
    let mut out = Vec::with_capacity(4);
    for (dx, dy) in [(-1, 0), (1, 0), (0, -1), (0, 1)] {
        let (mut px, mut py) = (i64::from(x) + dx, i64::from(y) + dy);
        while px >= 0 && py >= 0 && px < i64::from(img.width()) && py < i64::from(img.height()) {
            if !inside(all, px as u32, py as u32) {
                out.push((px as u32, py as u32));
                break;
            }
            px += dx;
            py += dy;
        }
    }
    out
}
fn interpolate(img: &RgbaImage, x: u32, y: u32, all: &[Rect]) -> Rgba<u8> {
    let mut sum = [0.; 4];
    let mut weight = 0.;
    for (px, py) in directional_samples(img, x, y, all) {
        let distance = (f64::from(x) - f64::from(px)).hypot(f64::from(y) - f64::from(py));
        let w = 1. / distance.max(1.);
        weight += w;
        for (c, v) in sum.iter_mut().zip(img.get_pixel(px, py).0) {
            *c += f64::from(v) * w;
        }
    }
    if weight == 0. {
        return Rgba([0, 0, 0, 0]);
    }
    Rgba(sum.map(|v| (v / weight).round().clamp(0., 255.) as u8))
}
/// Scrub using original clean pixels. Protected pixels are never written.
pub fn scrub_image(
    img: &mut RgbaImage,
    scrub_rects: &[Rect],
    kept_rects: &[Rect],
    mode: ScrubMode,
) {
    let source = img.clone();
    for &r in scrub_rects {
        let samples = perimeter(&source, r, scrub_rects);
        let transparent = match mode {
            ScrubMode::Transparent => true,
            ScrubMode::Inpaint => false,
            ScrubMode::Auto => {
                !samples.is_empty()
                    && samples
                        .iter()
                        .filter(|&&(x, y)| source.get_pixel(x, y)[3] == 0)
                        .count()
                        * 2
                        >= samples.len()
            }
        };
        for y in r.y.min(img.height())..r.y.saturating_add(r.h).min(img.height()) {
            for x in r.x.min(img.width())..r.x.saturating_add(r.w).min(img.width()) {
                if !inside(kept_rects, x, y) {
                    img.put_pixel(
                        x,
                        y,
                        if transparent {
                            Rgba([0, 0, 0, 0])
                        } else {
                            interpolate(&source, x, y, scrub_rects)
                        },
                    );
                }
            }
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn backgrounds_and_protection() {
        let r = Rect {
            x: 3,
            y: 3,
            w: 4,
            h: 4,
        };
        let k = Rect {
            x: 4,
            y: 4,
            w: 1,
            h: 1,
        };
        for bg in [[0, 0, 0, 0], [255, 255, 255, 255], [37, 62, 91, 255]] {
            let mut img = RgbaImage::from_pixel(10, 10, Rgba(bg));
            for y in 3..7 {
                for x in 3..7 {
                    img.put_pixel(x, y, Rgba([255, 0, 0, 255]));
                }
            }
            scrub_image(&mut img, &[r], &[k], ScrubMode::Auto);
            assert_eq!(img.get_pixel(3, 3).0, bg);
            assert_eq!(img.get_pixel(4, 4).0, [255, 0, 0, 255]);
        }
        let overlaps = find_rect_overlaps(&[r], &[k]);
        assert_eq!(overlaps.len(), 1);
        assert_eq!(overlaps[0].intersection, k);
    }
    #[test]
    fn gradient_and_adjacent_scrubs() {
        let mut img = RgbaImage::new(20, 10);
        for (x, _, p) in img.enumerate_pixels_mut() {
            *p = Rgba([(x * 10) as u8; 4]);
        }
        let rects = [
            Rect {
                x: 5,
                y: 2,
                w: 5,
                h: 6,
            },
            Rect {
                x: 10,
                y: 2,
                w: 5,
                h: 6,
            },
        ];
        scrub_image(&mut img, &rects, &[], ScrubMode::Inpaint);
        assert!(img.get_pixel(7, 5)[0] < img.get_pixel(12, 5)[0]);
    }
}
