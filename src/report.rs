//! Serializable high-recall flag ledger.
use crate::policy::{Category, Policy};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// One detector observation meeting its policy threshold.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Flag {
    pub category: Category,
    pub detector: String,
    pub score: f32,
    pub threshold: f32,
    pub evidence: String,
    pub rect: Option<[u32; 4]>,
}
/// Findings and content digest for one asset.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssetReport {
    pub path: String,
    pub sha256: String,
    pub kind: String,
    pub flag_count: usize,
    pub max_score: f32,
    pub blocked: bool,
    pub category_scores: BTreeMap<Category, f32>,
    pub flags: Vec<Flag>,
}
impl AssetReport {
    /// Start an empty ledger.
    pub fn new(path: String, bytes: &[u8], kind: &str) -> Self {
        Self {
            path,
            sha256: sha256(bytes),
            kind: kind.into(),
            flag_count: 0,
            max_score: 0.,
            blocked: false,
            category_scores: BTreeMap::new(),
            flags: vec![],
        }
    }
    /// Record a bounded score; retain evidence when it meets the threshold.
    pub fn record(
        &mut self,
        policy: &Policy,
        category: Category,
        detector: &str,
        score: f32,
        evidence: impl Into<String>,
        rect: Option<[u32; 4]>,
    ) {
        let score = if score.is_finite() {
            score.clamp(0., 1.)
        } else {
            0.
        };
        self.max_score = self.max_score.max(score);
        self.category_scores
            .entry(category)
            .and_modify(|v| *v = v.max(score))
            .or_insert(score);
        let threshold = policy.threshold(category);
        if score > 0. && score >= threshold {
            self.flags.push(Flag {
                category,
                detector: detector.into(),
                score,
                threshold,
                evidence: evidence.into(),
                rect,
            });
            self.flag_count = self.flags.len();
            self.blocked = true;
        }
    }
}
/// Aggregate result for a file or recursive project scan.
#[derive(Debug, Serialize, Deserialize)]
pub struct ScanSummary {
    pub assets: Vec<AssetReport>,
    pub total_assets: usize,
    pub blocked_assets: usize,
    pub total_flags: usize,
}
impl ScanSummary {
    /// Compute totals from asset ledgers.
    pub fn new(assets: Vec<AssetReport>) -> Self {
        Self {
            total_assets: assets.len(),
            blocked_assets: assets.iter().filter(|a| a.blocked).count(),
            total_flags: assets.iter().map(|a| a.flag_count).sum(),
            assets,
        }
    }
}
/// Compute a lowercase SHA-256 digest.
pub fn sha256(bytes: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    format!("{:x}", Sha256::digest(bytes))
}
