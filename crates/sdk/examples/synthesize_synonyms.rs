//! **Phase 1 doc2query**: 各 chunk から「同義 / 関連法律ターム / 関連シナリオ
//! 表現」を LLM (gemma-4-E2B 既定) で 5-10 個列挙し、caption に enrich する。
//!
//! Run 12b で「シナリオ → 法律ターム の semantic bridge は LlmRewriter
//! (query 側) では埋まらない」と判明したため、**index 側で caption を richer
//! にする** ことで橋渡しする。本ハーネスは:
//!
//! - corpus.json を読む
//! - 各 chunk について LLM 1 呼出で 5-7 個の同義語/類義表現を生成
//! - 元 caption に prepend / append して `corpus_synth.json` に書き出す
//! - **timing を全 chunk について計測**して slow なら opt-in 化判断
//!
//! 実行 (gemma-4-E2B 既定; 環境変数 `ELLISII_SYNTH_LLM` で override 可):
//! ```sh
//! ELLISII_EVAL_FIXTURE=jp-civil-law-hard \
//!   cargo run -p ellisii-sdk --features llamacpp \
//!     --example synthesize_synonyms --release
//! ```
//!
//! 出力先: `crates/rag/tests/fixtures/eval/<fixture>/corpus_synth.json`

use std::path::PathBuf;
use std::sync::{Arc, Mutex};
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

fn home() -> PathBuf {
    PathBuf::from(std::env::var("HOME").expect("HOME"))
}
fn default_llm() -> PathBuf {
    home().join("Library/Application Support/ellisii/models/gemma-4-E2B-it-IQ4_XS.gguf")
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

fn build_prompt(entry: &CorpusEntry) -> String {
    // 法律タームではなく **シナリオ表現** を生成させるのが Phase 1.5 の眼目。
    // body は 240 文字までを context にする。
    let body_preview: String = entry.text.chars().take(240).collect();
    let cap = if entry.caption.is_empty() {
        "<no caption>".to_string()
    } else {
        entry.caption.clone()
    };
    format!(
        "以下は日本の規程・条文の 1 章です。\n\n\
         caption: {cap}\n\
         本文: {body_preview}\n\n\
         この条文が現実で適用される **日常用語のシナリオ** を 3 つだけ、\
         半角カンマ区切りで 1 行に列挙してください。\n\
         - 必ず「日常語のフレーズ」で書くこと (法律タームを使わない)\n\
         - 例 (虚偽表示): 「税逃れのために売買契約書だけ作る, 偽装離婚で財産を隠す, 強制執行を避けるために通謀して名義変更」\n\
         - 例 (公序良俗): 「違法薬物の売買, 賭博の借金契約, 殺人の依頼契約」\n\
         - 例 (法定相続分): 「夫が亡くなった遺産の分け方, 配偶者と子の遺産配分, 兄弟姉妹だけの相続割合」\n\
         出力は 1 行のシナリオ列挙のみ。説明・前置きは禁止。\n"
    )
}

/// LLM 出力から先頭 1 行 (= カンマ区切り列挙) を取り出して同義語ベクタにする。
fn parse_synonyms(raw: &str) -> Vec<String> {
    let line = raw.lines().find(|l| !l.trim().is_empty()).unwrap_or("");
    line.split([',', '、', '，'])
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty() && s.chars().count() <= 30)
        .take(10)
        .collect()
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> anyhow::Result<()> {
    #[cfg(not(feature = "llamacpp"))]
    {
        anyhow::bail!("build with --features llamacpp");
    }
    #[cfg(feature = "llamacpp")]
    return run().await;
}

#[cfg(feature = "llamacpp")]
async fn run() -> anyhow::Result<()> {
    use ellisii_llm_core::{LlmBackend, LlmRequest, ModelFamily};
    use ellisii_llm_llamacpp::{LlamaConfig, LlamaCppBackend};

    let dir = fixture_dir();
    let corpus: Vec<CorpusEntry> =
        serde_json::from_str(&std::fs::read_to_string(dir.join("corpus.json"))?)?;
    eprintln!(
        "fixture: {}\ncorpus: {} chunks",
        dir.display(),
        corpus.len()
    );

    let gguf = std::env::var("ELLISII_SYNTH_LLM")
        .map(PathBuf::from)
        .unwrap_or_else(|_| default_llm());
    if !gguf.is_file() {
        anyhow::bail!("LLM not found: {} (set ELLISII_SYNTH_LLM)", gguf.display());
    }
    eprintln!("LLM: {}", gguf.display());

    let cfg = LlamaConfig::new(gguf, ModelFamily::Gemma4);
    let llm = Arc::new(LlamaCppBackend::load(cfg).map_err(|e| anyhow::anyhow!("load: {e}"))?);

    let mut out: Vec<CorpusEntry> = Vec::with_capacity(corpus.len());
    let t_total = Instant::now();
    let mut per_chunk_secs: Vec<f32> = Vec::with_capacity(corpus.len());
    for (i, e) in corpus.iter().enumerate() {
        let prompt = build_prompt(e);
        let buf = Arc::new(Mutex::new(String::new()));
        let b = Arc::clone(&buf);
        let req = LlmRequest {
            system: "あなたは日本法令に詳しい assistant です。出力は指示通り簡潔に。".into(),
            history: vec![],
            user: prompt,
            max_tokens: 160,
            temperature: 0.3,
        };
        let t0 = Instant::now();
        llm.generate_stream(
            req,
            Box::new(move |tok| {
                b.lock().unwrap().push_str(&tok);
            }),
        )
        .await?;
        let dt = t0.elapsed().as_secs_f32();
        per_chunk_secs.push(dt);
        let raw = buf.lock().unwrap().clone();
        let synonyms = parse_synonyms(&raw);

        let new_caption = if synonyms.is_empty() {
            e.caption.clone()
        } else if e.caption.is_empty() {
            synonyms.join(", ")
        } else {
            format!("{} ｜ シナリオ: {}", e.caption, synonyms.join(", "))
        };
        eprintln!(
            "  [{}/{}] {} ({:.2}s)  caption→ {}",
            i + 1,
            corpus.len(),
            e.doc_id,
            dt,
            new_caption.chars().take(60).collect::<String>()
        );

        out.push(CorpusEntry {
            doc_id: e.doc_id.clone(),
            parent_id: e.parent_id.clone(),
            title: e.title.clone(),
            caption: new_caption,
            text: e.text.clone(),
        });
    }
    let total = t_total.elapsed().as_secs_f32();
    per_chunk_secs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = per_chunk_secs[per_chunk_secs.len() / 2];
    let max = per_chunk_secs.last().copied().unwrap_or(0.0);
    let mean = per_chunk_secs.iter().sum::<f32>() / per_chunk_secs.len() as f32;
    eprintln!(
        "\n=== Timing summary ===\nchunks: {}\ntotal:  {:.1}s  ({:.1} min)\nmean:   {:.2}s/chunk\nmedian: {:.2}s/chunk\nmax:    {:.2}s/chunk",
        corpus.len(),
        total,
        total / 60.0,
        mean,
        median,
        max,
    );

    let out_path = dir.join("corpus_synth.json");
    std::fs::write(&out_path, serde_json::to_string_pretty(&out)?)?;
    eprintln!("wrote {}", out_path.display());

    Ok(())
}
