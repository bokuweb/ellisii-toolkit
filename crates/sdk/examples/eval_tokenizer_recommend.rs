//! Run 8 で集めた 6 corpus に対して
//! [`ellisii_jp_tokenizer_core::recommend_tokenizer`] を回し、
//! 各 corpus の signals (英字比率 / zenkaku digit / kanji digit) と recommended
//! tokenizer を表で出す。後段の "auto facade が Run 8 の経験則と一致しているか"
//! のサニティチェック。
//!
//! 実行:
//! ```sh
//! cargo run -p ellisii-sdk --example eval_tokenizer_recommend --release
//! ```
//!
//! 期待: delarocha 辞書 (`~/Library/.../models/delarocha/system.dic.zst`) が
//! 存在する限り **全 corpus で MorphemeNfkc が選ばれる** (= Run 8 の defensible
//! default と一致する)。

use ellisii_jp_tokenizer_core::recommend_tokenizer;
use serde::Deserialize;
use std::path::PathBuf;

#[derive(Debug, Deserialize)]
struct CorpusEntry {
    #[allow(dead_code)]
    doc_id: String,
    #[serde(default)]
    caption: String,
    text: String,
}

fn fixtures_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .join("rag/tests/fixtures/eval")
}

fn delarocha_dict() -> Option<PathBuf> {
    let home = std::env::var("HOME").ok()?;
    let p = PathBuf::from(home)
        .join("Library/Application Support/ellisii/models/delarocha/system.dic.zst");
    p.is_file().then_some(p)
}

fn main() -> anyhow::Result<()> {
    let root = fixtures_root();
    let dela = delarocha_dict();
    let dict_available = dela.is_some();
    eprintln!(
        "fixtures root: {}\ndelarocha dict: {}",
        root.display(),
        match &dela {
            Some(p) => p.display().to_string(),
            None => "<missing>".to_string(),
        }
    );

    let corpora = [
        "jp-workplace-regs",
        "jp-civil-law-hard",
        "jp-cs-wiki-hard",
        "jp-tokkyo-hou",
        "jp-labor-law",
        "yokohama",
    ];

    println!(
        "\n=== recommend_tokenizer across {} corpora (dict_available={}) ===",
        corpora.len(),
        dict_available
    );
    println!(
        "{:<22} {:>9} {:>9} {:>9} {:>9}  recommended",
        "corpus", "chars", "en_ratio", "zen_digit", "kanji"
    );

    for name in corpora {
        let path = root.join(name).join("corpus.json");
        if !path.is_file() {
            println!(
                "{:<22}  (skip: corpus.json not found at {})",
                name,
                path.display()
            );
            continue;
        }
        let raw = std::fs::read_to_string(&path)?;
        let corpus: Vec<CorpusEntry> = serde_json::from_str(&raw)?;
        let samples: Vec<String> = corpus
            .iter()
            .map(|e| {
                if e.caption.is_empty() {
                    e.text.clone()
                } else {
                    format!("({})\n{}", e.caption, e.text)
                }
            })
            .collect();
        let refs: Vec<&str> = samples.iter().map(|s| s.as_str()).collect();
        let (pick, sig) = recommend_tokenizer(refs.iter().copied(), dict_available);
        println!(
            "{:<22} {:>9} {:>9.3} {:>9} {:>9}  {:?}",
            name, sig.total_chars, sig.en_ratio, sig.has_zenkaku_digit, sig.has_kanji_digit, pick,
        );
    }

    Ok(())
}
