//! Text, URL, and script-input heuristics.
use crate::{
    policy::{Category, Policy},
    report::AssetReport,
};
use regex::Regex;
use std::sync::LazyLock;
static URL: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)\b(?:https?://|rbxasset(?:id)?://|www\.)[^\s<>\x22']+|\b[a-z0-9][a-z0-9-]*(?:\.[a-z0-9-]+)*\.(?:com|net|org|gg|io|tv|co|me|biz)\b(?:/[^\s<>\x22']*)?").unwrap()
});
static LITERALS: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r#"(?s)"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*'|\[(=*)\[.*?\](?:=*)\]"#).unwrap()
});
const BRANDS: &[&str] = &[
    "google play",
    "app store",
    "amazon appstore",
    "steam",
    "itch.io",
    "newgrounds",
    "armorgames",
    "kongregate",
    "jayisgames",
    "flipline",
    "poki",
    "crazygames",
    "miniclip",
    "addictinggames",
    "discord",
    "twitter",
    "facebook",
    "instagram",
    "youtube",
    "tiktok",
    "patreon",
    "ko-fi",
    "paypal",
];
const APIS: &[&str] = &[
    "navigatetourl",
    "sendtourl",
    "geturl",
    "externalinterface",
    "application.openurl",
    "os.shell_open",
    "url_open",
    "process.start",
    "openbrowserwindow",
];
const TERMS: &[(Category, &[&str])] = &[
    (
        Category::ProfanitySlurs,
        &[
            "fuck", "shit", "bitch", "asshole", "nigger", "faggot", "cunt",
        ],
    ),
    (
        Category::GoreViolence,
        &[
            "gore",
            "dismember",
            "decapitat",
            "entrails",
            "bloodbath",
            "guts spilling",
            "graphic violence",
        ],
    ),
    (
        Category::SexualNudity,
        &[
            "porn",
            "nsfw",
            "nude",
            "nudity",
            "hentai",
            "xxx",
            "explicit sex",
        ],
    ),
    (
        Category::RegulatedGoods,
        &[
            "cocaine",
            "heroin",
            "methamphetamine",
            "cannabis",
            "marijuana",
            "vape",
            "tobacco",
            "cigarette",
            "vodka",
            "whiskey",
            "alcohol",
            "nicotine",
            "fentanyl",
            "firearm",
            "ammunition",
        ],
    ),
];
fn allowed_url(s: &str) -> bool {
    let low = s.to_lowercase();
    if low.starts_with("rbxasset://") || low.starts_with("rbxassetid://") {
        return true;
    }
    let host = low
        .trim_start_matches("https://")
        .trim_start_matches("http://")
        .split(['/', ':', '?', '#'])
        .next()
        .unwrap_or("");
    ["roblox.com", "rbxcdn.com"]
        .iter()
        .any(|d| host == *d || host.ends_with(&format!(".{d}")))
}
fn normalized(s: &str) -> String {
    s.to_lowercase()
        .chars()
        .filter_map(|c| match c {
            '0' => Some('o'),
            '1' | '!' => Some('i'),
            '3' => Some('e'),
            '4' | '@' => Some('a'),
            '5' | '$' => Some('s'),
            '7' => Some('t'),
            c if c.is_ascii_alphabetic() => Some(c),
            _ => None,
        })
        .collect()
}
/// Whether visible UI text contains an outbound or unsafe content cue.
pub fn unsafe_text(text: &str) -> bool {
    let mut report = AssetReport::new(String::new(), &[], "text");
    scan_text(text, false, &Policy::default(), &mut report);
    report.blocked
}
/// Scan content; scripts also inspect APIs and nearby input-filter hints.
pub fn scan_text(text: &str, script: bool, policy: &Policy, report: &mut AssetReport) {
    let literals: Vec<_> = LITERALS
        .find_iter(text)
        .map(|m| decode_literal(m.as_str()))
        .collect();
    let content = if script && !literals.is_empty() {
        literals.join("\n")
    } else {
        text.into()
    };
    let low = content.to_lowercase();
    for m in URL.find_iter(&content) {
        if !allowed_url(m.as_str()) {
            report.record(
                policy,
                Category::OffsiteLink,
                "external_url",
                0.95,
                m.as_str(),
                None,
            );
        }
    }
    for term in BRANDS {
        if low.contains(term) {
            report.record(
                policy,
                Category::OffsiteLink,
                "brand_social",
                0.75,
                *term,
                None,
            );
        }
    }
    let norm = normalized(&content);
    for (category, terms) in TERMS {
        for term in *terms {
            if norm.contains(&normalized(term)) {
                report.record(policy, *category, "text_term", 0.85, *term, None);
            }
        }
    }
    if script {
        scan_script(text, policy, report);
    }
}
/// Decode JSON string values before scanning escapes and embedded domains.
pub fn scan_json(text: &str, policy: &Policy, report: &mut AssetReport) {
    let Ok(value) = serde_json::from_str::<serde_json::Value>(text) else {
        scan_text(text, false, policy, report);
        return;
    };
    let mut pending = vec![&value];
    let mut strings = Vec::new();
    while let Some(value) = pending.pop() {
        match value {
            serde_json::Value::String(s) => strings.push(s.as_str()),
            serde_json::Value::Array(items) => pending.extend(items),
            serde_json::Value::Object(fields) => {
                strings.extend(fields.keys().map(String::as_str));
                pending.extend(fields.values());
            }
            _ => {}
        }
    }
    scan_text(&strings.join("\n"), false, policy, report);
}
fn decode_literal(text: &str) -> String {
    serde_json::from_str::<String>(text).unwrap_or_else(|_| {
        text.trim_matches(['\'', '"'])
            .replace("\\'", "'")
            .replace("\\\"", "\"")
            .replace("\\/", "/")
    })
}
pub(crate) fn scan_script(text: &str, policy: &Policy, report: &mut AssetReport) {
    let low = text.to_lowercase();
    for api in APIS {
        if low.contains(api) {
            report.record(
                policy,
                Category::OffsiteLink,
                "outbound_api",
                0.90,
                *api,
                None,
            );
        }
    }
    for input in ["textbox", "defineedittext", "inputfield", "lineedit"] {
        for (pos, _) in low.match_indices(input) {
            // Character boundaries are preserved by slicing the original ASCII byte window lossily.
            let bytes = low.as_bytes();
            let near = String::from_utf8_lossy(
                &bytes[pos.saturating_sub(512)..(pos + 512).min(bytes.len())],
            );
            if !["textservice", "filterstringasync"]
                .iter()
                .any(|s| near.contains(s))
            {
                report.record(
                    policy,
                    Category::UnfilteredTextInput,
                    "editable_input",
                    0.8,
                    input,
                    None,
                );
            }
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    fn scan(s: &str, script: bool) -> AssetReport {
        let mut r = AssetReport::new("test".into(), s.as_bytes(), "text");
        scan_text(s, script, &Policy::default(), &mut r);
        r
    }
    #[test]
    fn urls_and_safe_chains() {
        for s in [
            "https://evil.com",
            "join us at example.gg",
            "www.foo.tv",
            "itch.io",
            "discord",
            "google play",
            "Application.OpenURL('x')",
        ] {
            assert!(scan(s, true).blocked, "{s}");
        }
        for s in [
            "https://roblox.com/a",
            "https://cdn.rbxcdn.com/a",
            "rbxasset://textures/a",
            "rbxasset://textures/example.com",
            "rbxassetid://123",
            "math.min(1,2); string.format('hi'); table.insert(t,1)",
        ] {
            assert!(!scan(s, true).blocked, "{s}");
        }
        assert!(scan("https://roblox.com.evil.com", false).blocked);
    }
    #[test]
    fn content_and_input() {
        for (s, c) in [
            ("f.u.c.k", Category::ProfanitySlurs),
            ("sh1t", Category::ProfanitySlurs),
            ("dismembered entrails", Category::GoreViolence),
            ("NSFW nudity", Category::SexualNudity),
            ("buy cocaine", Category::RegulatedGoods),
            ("Instance.new('TextBox')", Category::UnfilteredTextInput),
        ] {
            assert!(scan(s, true).flags.iter().any(|f| f.category == c), "{s}");
        }
        assert!(!scan("TextBox TextService FilterStringAsync", true).blocked);
    }
    #[test]
    fn escaped_json_and_luau_strings() {
        let mut r = AssetReport::new("a.json".into(), &[], "text");
        scan_json(
            r#"{"message":"ex\u0061mple.com"}"#,
            &Policy::default(),
            &mut r,
        );
        assert!(r.flags.iter().any(|f| f.detector == "external_url"));
        assert!(scan(r#"print("https:\/\/example.com")"#, true).blocked);
        let mut r = AssetReport::new("a.rbxlx".into(), &[], "place");
        scan_text(
            r#"<Item class="TextBox"><string name="Text">example.gg</string></Item>"#,
            false,
            &Policy::default(),
            &mut r,
        );
        scan_script("TextBox", &Policy::default(), &mut r);
        assert!(r.flags.iter().any(|f| f.category == Category::OffsiteLink));
        assert!(r
            .flags
            .iter()
            .any(|f| f.category == Category::UnfilteredTextInput));
    }
}
