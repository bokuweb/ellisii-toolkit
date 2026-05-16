//! `eval_workplace_regs` の answer-level eval。
//!
//! retrieval 層 (recall@k) ではなく「LLM が最終的に正しい数値/語を答えに含めたか」を
//! 計測する。Run 3 で確認した CE rerank ベスト構成 (top_n=10 w=0.5) + gemma-4-E4B
//! で full RAG を回し、`answer_golden.json` の `must_include` 必須語がすべて answer に
//! 含まれていれば pass とみなす。
//!
//! 実行 (gemma-4-E4B + open-provence ONNX 必須):
//! ```sh
//! cargo run -p ellisii-sdk \
//!   --features static-jp,llamacpp,provence-onnx \
//!   --example eval_workplace_regs_answer --release
//! ```
//!
//! 結果は `docs/eval/recall-evals.md` jp-workplace-regs セクションに
//! 「Run 5 (answer-level)」として追記する。

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use ellisii_core::Chunk;
use ellisii_sdk::{AskOptions, Ellisii};
use serde::Deserialize;
use uuid::Uuid;

#[derive(Debug, Deserialize)]
struct CorpusEntry {
    doc_id: String,
    #[allow(dead_code)]
    parent_id: String,
    #[allow(dead_code)]
    title: String,
    caption: String,
    text: String,
}

#[derive(Debug, Deserialize)]
struct AnswerGoldenItem {
    query: String,
    must_include: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct AnswerGoldenSet {
    #[allow(dead_code)]
    name: String,
    items: Vec<AnswerGoldenItem>,
}

fn home() -> PathBuf {
    PathBuf::from(std::env::var("HOME").expect("HOME"))
}
fn embed_dir() -> PathBuf {
    home().join("Library/Application Support/ellisii/models/static-embedding-japanese")
}
fn provence_dir() -> PathBuf {
    home().join("Library/Application Support/ellisii/models/open-provence")
}
fn gemma_e4b() -> PathBuf {
    home().join("Library/Application Support/ellisii/models/gemma-4-E4B-it-IQ4_XS.gguf")
}
fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .join("rag/tests/fixtures/eval/jp-workplace-regs")
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> anyhow::Result<()> {
    #[cfg(not(all(feature = "static-jp", feature = "llamacpp", feature = "provence-onnx")))]
    {
        anyhow::bail!("build with --features static-jp,llamacpp,provence-onnx");
    }
    #[cfg(all(feature = "static-jp", feature = "llamacpp", feature = "provence-onnx"))]
    return run().await;
}

#[cfg(all(feature = "static-jp", feature = "llamacpp", feature = "provence-onnx"))]
async fn run() -> anyhow::Result<()> {
    use ellisii_llm_core::ModelFamily;

    let dir = fixture_dir();
    let corpus: Vec<CorpusEntry> =
        serde_json::from_str(&std::fs::read_to_string(dir.join("corpus.json"))?)?;
    let gold: AnswerGoldenSet =
        serde_json::from_str(&std::fs::read_to_string(dir.join("answer_golden.json"))?)?;
    eprintln!(
        "corpus: {} chunks, answer-golden: {} items",
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
    let prov = provence_dir();
    let gguf = gemma_e4b();
    if !gguf.is_file() {
        anyhow::bail!("gemma-4-E4B-it-IQ4_XS.gguf not found");
    }
    if !prov.exists() {
        anyhow::bail!("open-provence ONNX not found");
    }
    eprintln!("embed:    {}", embed.display());
    eprintln!("provence: {}", prov.display());
    eprintln!("LLM:      {}", gguf.display());

    let ellisii = Ellisii::builder()
        .with_embedder_static_jp(&embed)?
        .with_store_memory()
        .with_compressor_provence_onnx(&prov, 0.20)?
        .with_llm_llamacpp(&gguf, ModelFamily::Gemma4)?
        .with_notebook_id(nb)
        .build()?;

    let embs = ellisii.embedder().embed(&texts).await?;
    ellisii.store().upsert(nb, &chunks, &embs).await?;

    println!(
        "\n=== jp-workplace-regs: answer-level eval (Run 5, gemma-4-E4B + cap+CE) ===\n\
         pass = all must_include 部分文字列 ∈ answer"
    );

    let mut pass = 0usize;
    let mut details: Vec<(String, bool, Vec<String>, String)> = Vec::new();
    let total = gold.items.len();
    for (i, item) in gold.items.iter().enumerate() {
        let buf = Arc::new(Mutex::new(String::new()));
        let buf_cloned = Arc::clone(&buf);
        let opts = AskOptions {
            top_k: 5,
            semantic_weight: 0.5,
            caption_rerank: true,
            ce_rerank_top_n: 10,
            ce_rerank_weight: 0.5,
            max_tokens: 256,
            temperature: 0.0,
            route_by_intent: false,
            ..Default::default()
        };
        let t0 = Instant::now();
        let _ = ellisii
            .ask(&item.query, opts, move |tok| {
                buf_cloned.lock().unwrap().push_str(&tok);
            })
            .await?;
        let answer = buf.lock().unwrap().clone();
        let dur = t0.elapsed();

        let missing: Vec<String> = item
            .must_include
            .iter()
            .filter(|s| !answer.contains(s.as_str()))
            .cloned()
            .collect();
        let ok = missing.is_empty();
        if ok {
            pass += 1;
        }
        eprintln!(
            "  [{}/{}] {} {} ({:.1}s)",
            i + 1,
            total,
            if ok { "✓" } else { "✗" },
            item.query,
            dur.as_secs_f32()
        );
        if !ok {
            eprintln!("       missing={:?}", missing);
        }
        details.push((item.query.clone(), ok, missing, answer));
    }

    println!(
        "\nanswer pass rate: {} / {} = {:.3}",
        pass,
        total,
        pass as f32 / total as f32
    );

    println!("\n=== Failed items ===");
    for (q, ok, missing, ans) in &details {
        if *ok {
            continue;
        }
        println!("query   : {}", q);
        println!("missing : {:?}", missing);
        let preview: String = ans.chars().take(160).collect();
        println!(
            "answer  : {}{}",
            preview,
            if ans.len() > preview.len() { "..." } else { "" }
        );
        println!();
    }

    Ok(())
}
