//! Categories and configurable review thresholds.
use clap::ValueEnum;
use serde::{Deserialize, Serialize};
use std::{collections::BTreeMap, str::FromStr};

/// Review categories aligned with Roblox Community Standards.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Category {
    OffsiteLink,
    GoreViolence,
    SexualNudity,
    ProfanitySlurs,
    RegulatedGoods,
    UnfilteredTextInput,
}
impl FromStr for Category {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        serde_json::from_value(serde_json::Value::String(s.into()))
            .map_err(|_| format!("Unknown category: {s}"))
    }
}
/// Built-in threshold presets; high recall favors human review.
#[derive(Debug, Clone, Copy, Default, ValueEnum)]
pub enum Profile {
    #[default]
    HighRecall,
    Balanced,
}
/// Category thresholds, inclusive at the boundary.
#[derive(Debug, Clone)]
pub struct Policy {
    pub thresholds: BTreeMap<Category, f32>,
}
impl Default for Policy {
    fn default() -> Self {
        Self::new(Profile::HighRecall)
    }
}
impl Policy {
    /// Construct a preset.
    pub fn new(profile: Profile) -> Self {
        use Category::*;
        let mut thresholds: BTreeMap<_, _> = [
            (OffsiteLink, 0.20),
            (ProfanitySlurs, 0.20),
            (SexualNudity, 0.30),
            (GoreViolence, 0.40),
            (RegulatedGoods, 0.35),
            (UnfilteredTextInput, 0.25),
        ]
        .into();
        if matches!(profile, Profile::Balanced) {
            for score in thresholds.values_mut() {
                *score += 0.20;
            }
        }
        Self { thresholds }
    }
    /// Set a finite score between zero and one.
    pub fn set(&mut self, category: Category, score: f32) -> anyhow::Result<()> {
        anyhow::ensure!(
            score.is_finite() && (0.0..=1.0).contains(&score),
            "Threshold must be between 0 and 1"
        );
        self.thresholds.insert(category, score);
        Ok(())
    }
    /// Return the inclusive review threshold.
    pub fn threshold(&self, category: Category) -> f32 {
        self.thresholds[&category]
    }
}
