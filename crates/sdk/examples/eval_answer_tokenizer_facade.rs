//! `eval_workplace_regs_answer` の **マルチ corpus + tokenizer A/B 版**。
//!
//! Run 8 で delarocha は retrieval (recall / nDCG / MRR) を確実に押し上げる
//! ことが分かったが、その向上が **answer pass rate** にまで乗るかは未確認だった
//! (Run 5 は cap rerank only で計測済み)。本ハーネスは:
//!
//! - `ELLISII_EVAL_FIXTURE` で指定した corpus の `answer_golden.json` を読む
//! - bigram baseline / delarocha+NFKC (sqlite, in-memory) の 2 ストアを構築
//! - 同じ gemma-4-E4B + cap rerank=on で full RAG を回し、`must_include` の
//!   全部分文字列が answer に揃えば pass とみなす
//! - bigram vs delarocha の pass rate / 失敗 query の差分を出す
//!
//! 実行 (gemma-4-E4B + delarocha 必須):
//! ```sh
//! ELLISII_EVAL_FIXTURE=jp-cs-wiki-hard \
//!   cargo run -p ellisii-sdk \
//!     --features static-jp,llamacpp,delarocha \
//!     --example eval_answer_tokenizer_facade --release
//! ```

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use ellisii_core::Chunk;
use ellisii_jp_tokenizer_bigram::CharBigramTokenizer;
use ellisii_jp_tokenizer_core::JpTokenizer;
use ellisii_jp_tokenizer_nfkc::NfkcTokenizer;
use ellisii_sdk::{AskOptions, Ellisii};
use ellisii_store_core::VectorStore;
use ellisii_store_sqlite::SqliteStore;
use serde::Deserialize;
use uuid::Uuid;

#[derive(Debug, Deserialize)]
struct CorpusEntry {
    doc_id: String,
    #[serde(default)]
    #[allow(dead_code)]
    parent_id: Option<String>,
    #[serde(default)]
    #[allow(dead_code)]
    title: String,
    #[serde(default)]
    caption: String,
    text: String,
}

#[derive(Debug, Deserialize)]
struct AnswerItem {
    query: String,
    must_include: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct AnswerSet {
    #[allow(dead_code)]
    name: String,
    items: Vec<AnswerItem>,
}

fn home() -> PathBuf {
    PathBuf::from(std::env::var("HOME").expect("HOME"))
}
fn embed_dir() -> PathBuf {
    home().join("Library/Application Support/ellisii/models/static-embedding-japanese")
}
fn gemma_e4b() -> PathBuf {
    home().join("Library/Application Support/ellisii/models/gemma-4-E4B-it-IQ4_XS.gguf")
}
fn delarocha_dict() -> PathBuf {
    home().join("Library/Application Support/ellisii/models/delarocha/system.dic.zst")
}
fn fixture_dir() -> PathBuf {
    let name =
        std::env::var("ELLISII_EVAL_FIXTURE").unwrap_or_else(|_| "jp-cs-wiki-hard".to_string());
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .join("rag/tests/fixtures/eval")
        .join(name)
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> anyhow::Result<()> {
    #[cfg(not(all(feature = "static-jp", feature = "llamacpp", feature = "delarocha")))]
    {
        anyhow::bail!("build with --features static-jp,llamacpp,delarocha");
    }
    #[cfg(all(feature = "static-jp", feature = "llamacpp", feature = "delarocha"))]
    return run().await;
}

#[cfg(all(feature = "static-jp", feature = "llamacpp", feature = "delarocha"))]
async fn run() -> anyhow::Result<()> {
    use ellisii_jp_tokenizer_delarocha::DelarochaTokenizer;
    use ellisii_llm_core::ModelFamily;

    let dir = fixture_dir();
    let corpus: Vec<CorpusEntry> =
        serde_json::from_str(&std::fs::read_to_string(dir.join("corpus.json"))?)?;
    let gold: AnswerSet =
        serde_json::from_str(&std::fs::read_to_string(dir.join("answer_golden.json"))?)?;
    eprintln!(
        "fixture: {}\ncorpus:  {} chunks\nanswer:  {} items",
        dir.display(),
        corpus.len(),
        gold.items.len()
    );

    let nb = Uuid::new_v4();
    let src = Uuid::new_v4();
    let mut chunks: Vec<Chunk> = Vec::with_capacity(corpus.len());
    let mut texts: Vec<String> = Vec::with_capacity(corpus.len());
    let mut id_map: HashMap<Uuid, String> = HashMap::new();
    for (i, e) in corpus.iter().enumerate() {
        let cid = Uuid::new_v4();
        id_map.insert(cid, e.doc_id.clone());
        let txt = if e.caption.is_empty() {
            e.text.clone()
        } else {
            format!("({})\n{}", e.caption, e.text)
        };
        chunks.push(Chunk {
            id: cid,
            source_id: src,
            ord: i as u32,
            text: txt.clone(),
            heading_path: vec![e.doc_id.clone()],
            page: None,
            bbox: None,
            summary: None,
        });
        texts.push(txt);
    }

    let embed = embed_dir();
    let dela_path = delarocha_dict();
    let gguf = gemma_e4b();
    if !gguf.is_file() {
        anyhow::bail!("gemma-4-E4B-it-IQ4_XS.gguf not found");
    }
    if !dela_path.is_file() {
        anyhow::bail!("delarocha system.dic.zst not found");
    }
    eprintln!("embed:    {}", embed.display());
    eprintln!("delarocha:{}", dela_path.display());
    eprintln!("LLM:      {}", gguf.display());

    let dim = 1024;
    let bigram: Arc<dyn JpTokenizer> = Arc::new(CharBigramTokenizer::new());
    let dela_nfkc: Arc<dyn JpTokenizer> = Arc::new(NfkcTokenizer::new(Arc::new(
        DelarochaTokenizer::from_path(&dela_path)
            .map_err(|e| anyhow::anyhow!("load delarocha: {e}"))?,
    )));
    let store_bigram: Arc<dyn VectorStore> =
        Arc::new(SqliteStore::open_in_memory_with_tokenizer(dim, bigram)?);
    let store_dela: Arc<dyn VectorStore> =
        Arc::new(SqliteStore::open_in_memory_with_tokenizer(dim, dela_nfkc)?);

    let bigram_engine = Ellisii::builder()
        .with_embedder_static_jp(&embed)?
        .with_store(store_bigram.clone())
        .with_llm_llamacpp(&gguf, ModelFamily::Gemma4)?
        .with_notebook_id(nb)
        .build()?;
    let embs = bigram_engine.embedder().embed(&texts).await?;
    store_bigram.upsert(nb, &chunks, &embs).await?;
    store_dela.upsert(nb, &chunks, &embs).await?;
    let dela_engine = Ellisii::builder()
        .with_embedder_static_jp(&embed)?
        .with_store(store_dela.clone())
        .with_llm_llamacpp(&gguf, ModelFamily::Gemma4)?
        .with_notebook_id(nb)
        .build()?;

    let mut results: Vec<(String, bool, bool, String, String)> = Vec::new();
    let total = gold.items.len();
    for (i, item) in gold.items.iter().enumerate() {
        let (b_ok, b_ans) = run_one(&bigram_engine, &item.query, &item.must_include).await?;
        let (d_ok, d_ans) = run_one(&dela_engine, &item.query, &item.must_include).await?;
        let marker = match (b_ok, d_ok) {
            (true, true) => "= both pass",
            (false, false) => "✗ both fail",
            (false, true) => "↑ delarocha rescues",
            (true, false) => "↓ delarocha breaks (regression)",
        };
        eprintln!("  [{}/{}] {}  | {}", i + 1, total, marker, item.query);
        results.push((item.query.clone(), b_ok, d_ok, b_ans, d_ans));
    }

    let b_pass = results.iter().filter(|r| r.1).count();
    let d_pass = results.iter().filter(|r| r.2).count();
    let rescues = results.iter().filter(|r| !r.1 && r.2).count();
    let breakages = results.iter().filter(|r| r.1 && !r.2).count();
    println!(
        "\nbigram pass:    {} / {} = {:.3}",
        b_pass,
        total,
        b_pass as f32 / total as f32
    );
    println!(
        "delarocha pass: {} / {} = {:.3}",
        d_pass,
        total,
        d_pass as f32 / total as f32
    );
    println!(
        "rescues (b ✗ → d ✓): {}    breakages (b ✓ → d ✗): {}",
        rescues, breakages
    );

    println!("\n=== Rescue cases (bigram failed, delarocha passed) ===");
    for (q, b, d, _b_ans, d_ans) in &results {
        if !*b && *d {
            println!("query : {}", q);
            let preview: String = d_ans.chars().take(120).collect();
            println!(
                "answer: {}{}",
                preview,
                if d_ans.len() > preview.len() {
                    "..."
                } else {
                    ""
                }
            );
            println!();
        }
    }
    println!("\n=== Regression cases (bigram passed, delarocha failed) ===");
    for (q, b, d, b_ans, _d_ans) in &results {
        if *b && !*d {
            println!("query : {}", q);
            let preview: String = b_ans.chars().take(120).collect();
            println!(
                "bigram answer: {}{}",
                preview,
                if b_ans.len() > preview.len() {
                    "..."
                } else {
                    ""
                }
            );
            println!();
        }
    }

    Ok(())
}

#[cfg(all(feature = "static-jp", feature = "llamacpp", feature = "delarocha"))]
async fn run_one(
    ellisii: &Ellisii,
    query: &str,
    must_include: &[String],
) -> anyhow::Result<(bool, String)> {
    let buf = Arc::new(Mutex::new(String::new()));
    let cloned = Arc::clone(&buf);
    let opts = AskOptions {
        top_k: 5,
        semantic_weight: 0.5,
        caption_rerank: true,
        max_tokens: 256,
        temperature: 0.0,
        route_by_intent: false,
        ..Default::default()
    };
    let _ = ellisii
        .ask(query, opts, move |tok| {
            cloned.lock().unwrap().push_str(&tok);
        })
        .await?;
    let ans = buf.lock().unwrap().clone();
    let ok = must_include.iter().all(|s| ans.contains(s.as_str()));
    Ok((ok, ans))
}
