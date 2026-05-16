//! `eval_workplace_regs` + Provence ONNX cross-encoder rerank。
//!
//! Run 1 (cap only) baseline = recall@5 0.950 を、CE rerank (`open-provence`
//! ONNX) でどこまで押せるかを top_n / weight で sweep する。
//!
//! 実行 (open-provence ONNX が必須):
//! ```sh
//! cargo run -p ellisii-sdk \
//!   --features static-jp,provence-onnx \
//!   --example eval_workplace_regs_ce --release
//! ```
//!
//! 結果は ellisii の `docs/eval/recall-evals.md` jp-workplace-regs セクションに
//! 「Run 3 (Provence CE rerank)」として追記する。

use std::collections::HashMap;
use std::path::PathBuf;

use ellisii_core::Chunk;
use ellisii_rag::eval::{summarize, GoldenSet};
use ellisii_sdk::{Ellisii, SearchOptions};
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

fn home() -> PathBuf {
    PathBuf::from(std::env::var("HOME").expect("HOME"))
}
fn embed_dir() -> PathBuf {
    home().join("Library/Application Support/ellisii/models/static-embedding-japanese")
}
fn provence_dir() -> PathBuf {
    home().join("Library/Application Support/ellisii/models/open-provence")
}
fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .join("rag/tests/fixtures/eval/jp-workplace-regs")
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> anyhow::Result<()> {
    #[cfg(not(all(feature = "static-jp", feature = "provence-onnx")))]
    {
        anyhow::bail!("build with --features static-jp,provence-onnx");
    }
    #[cfg(all(feature = "static-jp", feature = "provence-onnx"))]
    return run().await;
}

#[cfg(all(feature = "static-jp", feature = "provence-onnx"))]
async fn run() -> anyhow::Result<()> {
    let dir = fixture_dir();
    let corpus: Vec<CorpusEntry> =
        serde_json::from_str(&std::fs::read_to_string(dir.join("corpus.json"))?)?;
    let gold: GoldenSet =
        GoldenSet::from_json_str(&std::fs::read_to_string(dir.join("golden.json"))?)?;
    eprintln!(
        "corpus: {} chunks, golden: {} ({} items)",
        corpus.len(),
        gold.name,
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
    if !prov.exists() {
        anyhow::bail!("open-provence ONNX not found at {}", prov.display());
    }
    eprintln!("embed:    {}", embed.display());
    eprintln!("provence: {}", prov.display());

    // baseline (no CE) と CE-enabled の 2 つの Ellisii を作る。
    let baseline = Ellisii::builder()
        .with_embedder_static_jp(&embed)?
        .with_store_memory()
        .with_notebook_id(nb)
        .build()?;
    let ce = Ellisii::builder()
        .with_embedder_static_jp(&embed)?
        .with_store_memory()
        .with_notebook_id(nb)
        .with_compressor_provence_onnx(&prov, 0.20)?
        .build()?;

    let embs = baseline.embedder().embed(&texts).await?;
    baseline.store().upsert(nb, &chunks, &embs).await?;
    ce.store().upsert(nb, &chunks, &embs).await?;

    println!(
        "\n=== jp-workplace-regs: Provence CE rerank sweep (Run 3, k=5) ===\n\
         baseline = caption rerank only"
    );
    println!(
        "{:<32} {:<10} {:<10} {:<10} {:<10}",
        "variant", "recall", "hit", "ndcg", "mrr"
    );

    // baseline
    let base = run_pairs(&baseline, &gold, &id_map, 5, 0, 0.0).await?;
    let s = summarize(&base, 5);
    println!(
        "{:<32} {:<10.3} {:<10.3} {:<10.3} {:<10.3}",
        "cap only (k=5)", s.recall_at_k, s.hit_at_k, s.ndcg_at_k, s.mrr
    );

    let configs: &[(&str, usize, f32)] = &[
        ("cap+CE top_n=10 w=0.3", 10, 0.3),
        ("cap+CE top_n=10 w=0.5", 10, 0.5),
        ("cap+CE top_n=10 w=0.7", 10, 0.7),
        ("cap+CE top_n=10 w=1.0", 10, 1.0),
        ("cap+CE top_n=20 w=0.5", 20, 0.5),
        ("cap+CE top_n=30 w=0.5", 30, 0.5),
    ];
    for &(label, top_n, w) in configs {
        let pairs = run_pairs(&ce, &gold, &id_map, 5, top_n, w).await?;
        let s = summarize(&pairs, 5);
        println!(
            "{:<32} {:<10.3} {:<10.3} {:<10.3} {:<10.3}",
            label, s.recall_at_k, s.hit_at_k, s.ndcg_at_k, s.mrr
        );
    }

    // Targeted failure inspection: 法定休日 / 出張中の労働時間
    println!("\n=== Targeted failures at k=5 (cap+CE top_n=20 w=0.5) ===");
    for q in ["法定休日は何曜日", "出張中の労働時間はどう扱われるか"] {
        let item = gold.items.iter().find(|i| i.query == q).unwrap();
        let hits = ce
            .search(
                q,
                SearchOptions {
                    top_k: 5,
                    semantic_weight: 0.5,
                    caption_rerank: true,
                    ce_rerank_top_n: 20,
                    ce_rerank_weight: 0.5,
                    ..Default::default()
                },
            )
            .await?;
        let pred: Vec<String> = hits
            .iter()
            .filter_map(|h| id_map.get(&h.chunk.id).cloned())
            .collect();
        println!("  {}  expected={:?}  top5={:?}", q, item.relevant, pred);
    }

    Ok(())
}

#[cfg(all(feature = "static-jp", feature = "provence-onnx"))]
async fn run_pairs(
    ellisii: &Ellisii,
    gold: &GoldenSet,
    id_map: &HashMap<Uuid, String>,
    k: usize,
    ce_top_n: usize,
    ce_weight: f32,
) -> ellisii_core::Result<Vec<(Vec<String>, Vec<String>)>> {
    let mut pairs = Vec::with_capacity(gold.items.len());
    for item in &gold.items {
        let hits = ellisii
            .search(
                &item.query,
                SearchOptions {
                    top_k: k,
                    semantic_weight: 0.5,
                    caption_rerank: true,
                    ce_rerank_top_n: ce_top_n,
                    ce_rerank_weight: ce_weight,
                    ..Default::default()
                },
            )
            .await?;
        let pred: Vec<String> = hits
            .iter()
            .filter_map(|h| id_map.get(&h.chunk.id).cloned())
            .collect();
        pairs.push((pred, item.relevant.clone()));
    }
    Ok(pairs)
}
