//! Build a sparse-body version of a fixture corpus by stripping v8 thesaurus
//! synonym strings from each chunk's body.
//!
//! Goal: 「現実 sparse 法令 corpus を想定した enrichment の真の上限」 を計測する
//! ための corpus を機械的に生成する。Run 12z (medical) / 12aa (finance) /
//! 12cc (workplace-regs, yokohama) で Python script (`/tmp/build_sparse.py`)
//! を使っていたものを toolkit 内に Rust example として恒久化。
//!
//! 各 chunk について:
//!
//! 1. caption + body 先頭 300 chars を probe にして [`LawThesaurus`] から
//!    matched key を最大 3 件 (長い順) 抽出 — `enrich_chunks` と同じロジック
//! 2. matched 各 key の `synonyms` を body から `—` に置換
//! 3. 連続 `—` をまとめる
//!
//! 使い方:
//!
//! ```sh
//! cargo run -p ellisii-sdk --example eval_build_sparse_corpus -- \
//!     jp-tokkyo-hou jp-tokkyo-hou-sparse
//! ```
//!
//! 第1引数 = 入力 fixture name、第2引数 = 出力 fixture name (省略時は
//! 入力名 + "-sparse")。fixture root は
//! `crates/rag/tests/fixtures/eval/` 固定。golden.json は同じものを
//! コピー (name と _note のみ書き換え)。

use std::fs;
use std::path::PathBuf;

use ellisii_jp_law_thesaurus::LawThesaurus;
use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Deserialize, Serialize)]
struct CorpusEntry {
    doc_id: String,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    parent_id: Option<String>,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    title: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    caption: String,
    text: String,
}

const PROBE_BODY_CHARS: usize = 300;
const MAX_KEYS_PER_CHUNK: usize = 3;
const FILLER: &str = "—";

fn fixture_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .join("rag/tests/fixtures/eval")
}

fn main() -> anyhow::Result<()> {
    let mut args = std::env::args().skip(1);
    let in_name = args.next().ok_or_else(|| {
        anyhow::anyhow!("usage: eval_build_sparse_corpus <in_fixture> [out_fixture]")
    })?;
    let out_name = args.next().unwrap_or_else(|| format!("{in_name}-sparse"));

    let root = fixture_root();
    let in_dir = root.join(&in_name);
    let out_dir = root.join(&out_name);
    fs::create_dir_all(&out_dir)?;

    let corpus_json = fs::read_to_string(in_dir.join("corpus.json"))?;
    let corpus: Vec<CorpusEntry> = serde_json::from_str(&corpus_json)?;

    let thes = LawThesaurus::bundled();
    // BTreeMap iteration is alphabetical; we want longest-first for key match.
    // entry_count() is the only access we have, so we re-load via from_json_str
    // pattern: simpler to just enumerate all entries via reflection on the
    // bundled JSON. We re-parse the bundled JSON to get key list with synonyms.
    let bundled: Value = serde_json::from_str(include_str!(
        "../../jp-law-thesaurus/data/jp-law-thesaurus.json"
    ))?;
    let entries_obj = bundled["entries"]
        .as_object()
        .ok_or_else(|| anyhow::anyhow!("bundled.entries is not an object"))?;
    let mut keys_with_syns: Vec<(String, Vec<String>)> = entries_obj
        .iter()
        .filter_map(|(k, v)| {
            let obj = v.as_object()?;
            let syns: Vec<String> = obj
                .get("synonyms")?
                .as_array()?
                .iter()
                .filter_map(|s| s.as_str().map(str::to_string))
                .collect();
            Some((k.clone(), syns))
        })
        .collect();
    // Sort by key char count descending (longest first, mirrors matched_keys).
    keys_with_syns.sort_by_key(|(k, _)| std::cmp::Reverse(k.chars().count()));

    let mut total_stripped = 0usize;
    let mut chunks_touched = 0usize;
    let mut out_corpus = Vec::with_capacity(corpus.len());
    for ent in corpus {
        let probe: String = (ent.caption.clone() + &ent.text)
            .chars()
            .take(PROBE_BODY_CHARS)
            .collect();
        let mut matched: Vec<&Vec<String>> = Vec::new();
        for (k, syns) in &keys_with_syns {
            if probe.contains(k) {
                matched.push(syns);
                if matched.len() >= MAX_KEYS_PER_CHUNK {
                    break;
                }
            }
        }
        let mut body = ent.text.clone();
        let mut stripped_here = 0;
        for syns in matched {
            let mut sorted = syns.clone();
            sorted.sort_by_key(|s| std::cmp::Reverse(s.chars().count()));
            for s in sorted {
                // Skip empty / same-as-caption to avoid wiping the caption term.
                if s.is_empty() || s == ent.caption {
                    continue;
                }
                if body.contains(&s) {
                    body = body.replace(&s, FILLER);
                    stripped_here += 1;
                }
            }
        }
        // Collapse repeated fillers and surrounding punctuation noise.
        while body.contains("——") {
            body = body.replace("——", "—");
        }
        body = body.replace("—。", "。").replace("。—", "。");
        total_stripped += stripped_here;
        if stripped_here > 0 {
            chunks_touched += 1;
        }
        out_corpus.push(CorpusEntry { text: body, ..ent });
    }

    let pretty = serde_json::to_string_pretty(&out_corpus)? + "\n";
    fs::write(out_dir.join("corpus.json"), pretty)?;

    // Copy + adjust golden.
    let golden_src = fs::read_to_string(in_dir.join("golden.json"))?;
    let mut golden: Value = serde_json::from_str(&golden_src)?;
    if let Some(name) = golden.get("name").and_then(|v| v.as_str()) {
        let new_name = if name.ends_with("-sparse") {
            name.to_string()
        } else {
            format!("{name}-sparse")
        };
        golden["name"] = Value::String(new_name);
    }
    let note = format!(
        "Auto-generated by `eval_build_sparse_corpus` (Run 12dd): {} の sparse 版。\
         body から caption に matched する v8 thesaurus entries の synonyms を \
         `—` に置換、enrichment の synonym bridge 効果を直接抽出。",
        in_name
    );
    golden["_note"] = Value::String(note);
    let golden_pretty = serde_json::to_string_pretty(&golden)? + "\n";
    fs::write(out_dir.join("golden.json"), golden_pretty)?;

    eprintln!(
        "{} → {}: {} chunks, stripped {} substrings across {} chunks (thesaurus: {}, {} entries)",
        in_name,
        out_name,
        out_corpus.len(),
        total_stripped,
        chunks_touched,
        thes.name(),
        thes.entry_count(),
    );
    Ok(())
}
