//! **法令系 corpus 用 静的 thesaurus による caption enrichment**。
//!
//! Run 12c の LLM Doc2Query (~5s/chunk) に対し、**pure string match で
//! <1ms/chunk** で paraphrase gap を埋めるアプローチ。`crates/rag/data/
//! jp-law-thesaurus.json` の 100+ 法律ターム → シナリオ/同義語 マッピングを
//! 使い、chunk の caption / body 先頭に key (法律ターム) が出現したら
//! scenarios + synonyms を caption に append する。
//!
//! 法令/法律/例規/規程/特許/税法 系では LLM 不要で高品質な enrichment が
//! 期待できる (Run 12d を参照)。LLM 合成は本辞書に無い term だけにフォール
//! バックする想定。
//!
//! 実行 (LLM 不要):
//! ```sh
//! ELLISII_EVAL_FIXTURE=jp-civil-law-hard \
//!   cargo run -p ellisii-sdk --example enrich_with_thesaurus --release
//! ```
//!
//! 出力先: `crates/rag/tests/fixtures/eval/<fixture>/corpus_dict.json`

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::time::Instant;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Deserialize, Serialize)]
struct CorpusEntry {
    doc_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    parent_id: Option<String>,
    #[serde(default)]
    title: String,
    #[serde(default)]
    caption: String,
    text: String,
}

#[derive(Debug, Clone, Deserialize)]
struct ThesaurusEntry {
    #[allow(dead_code)]
    category: String,
    #[serde(default)]
    synonyms: Vec<String>,
    #[serde(default)]
    scenarios: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct Thesaurus {
    #[allow(dead_code)]
    name: String,
    #[allow(dead_code)]
    #[serde(default)]
    comment: String,
    entries: BTreeMap<String, ThesaurusEntry>,
}

fn fixture_dir() -> PathBuf {
    let name =
        std::env::var("ELLISII_EVAL_FIXTURE").unwrap_or_else(|_| "jp-civil-law-hard".to_string());
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .join("rag/tests/fixtures/eval")
        .join(name)
}

fn thesaurus_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .join("rag/data/jp-law-thesaurus.json")
}

/// caption + body 先頭 200 文字を走査して、辞書 key が **部分文字列として
/// 出現する** ものを集める。長い key を優先 (overlap で短い key が紛れ込ま
/// ないように)。1 chunk あたり最大 N 個まで。
fn match_terms<'a>(entry: &CorpusEntry, thesaurus: &'a Thesaurus) -> Vec<&'a str> {
    let probe = format!(
        "{} {}",
        entry.caption,
        entry.text.chars().take(200).collect::<String>()
    );
    let mut keys: Vec<&str> = thesaurus
        .entries
        .keys()
        .map(String::as_str)
        .filter(|k| probe.contains(*k))
        .collect();
    // 長い key を先頭に (例: "法定相続分" を "相続" より優先)
    keys.sort_by_key(|k| std::cmp::Reverse(k.chars().count()));
    // 1 chunk あたり 3 key までに絞る (caption が肥大化しないように)
    keys.truncate(3);
    keys
}

fn main() -> anyhow::Result<()> {
    let dir = fixture_dir();
    let corpus: Vec<CorpusEntry> =
        serde_json::from_str(&std::fs::read_to_string(dir.join("corpus.json"))?)?;
    eprintln!(
        "fixture: {}\ncorpus: {} chunks",
        dir.display(),
        corpus.len()
    );

    let thesaurus_p = thesaurus_path();
    let thesaurus: Thesaurus = serde_json::from_str(&std::fs::read_to_string(&thesaurus_p)?)?;
    eprintln!(
        "thesaurus: {} ({} entries)",
        thesaurus_p.display(),
        thesaurus.entries.len()
    );

    let t0 = Instant::now();
    let mut out: Vec<CorpusEntry> = Vec::with_capacity(corpus.len());
    let mut matched_count = 0usize;
    let mut total_phrases = 0usize;
    for e in &corpus {
        let hit_keys = match_terms(e, &thesaurus);
        if !hit_keys.is_empty() {
            matched_count += 1;
        }
        let mut additions: Vec<String> = Vec::new();
        for k in &hit_keys {
            if let Some(te) = thesaurus.entries.get(*k) {
                additions.extend(te.synonyms.iter().cloned());
                additions.extend(te.scenarios.iter().cloned());
            }
        }
        // 重複排除
        additions.sort();
        additions.dedup();
        total_phrases += additions.len();

        let new_caption = if additions.is_empty() {
            e.caption.clone()
        } else if e.caption.is_empty() {
            additions.join(", ")
        } else {
            format!("{} ｜ シナリオ: {}", e.caption, additions.join(", "))
        };
        out.push(CorpusEntry {
            doc_id: e.doc_id.clone(),
            parent_id: e.parent_id.clone(),
            title: e.title.clone(),
            caption: new_caption,
            text: e.text.clone(),
        });
    }
    let dt = t0.elapsed();

    eprintln!(
        "\n=== Enrichment summary ===\nchunks:           {}\nmatched (≥1 key): {}\ntotal phrases:    {}\nelapsed:          {:.2} ms  (= {:.2} µs/chunk)",
        corpus.len(),
        matched_count,
        total_phrases,
        dt.as_secs_f64() * 1000.0,
        dt.as_micros() as f64 / corpus.len() as f64,
    );

    let out_path = dir.join("corpus_dict.json");
    std::fs::write(&out_path, serde_json::to_string_pretty(&out)?)?;
    eprintln!("wrote {}", out_path.display());

    Ok(())
}
