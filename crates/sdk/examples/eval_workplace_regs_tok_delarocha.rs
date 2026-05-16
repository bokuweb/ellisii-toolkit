//! `eval_workplace_regs_tok` の delarocha 版。
//!
//! Run 4 では vaporetto (点予測) を bigram と比較したが、本 Run は
//! [delarocha](https://github.com/bokuweb/delarocha) (Vibrato 互換 MeCab 形態素)
//! を `store-sqlite` の FTS5 tokenizer として A/B する。
//!
//! delarocha の方が辞書ベースなので語境界が安定し、特に「法定休日 / 出張中」
//! のような熟語境界で bigram より上振れすることを期待する。
//!
//! 実行 (delarocha system.dic.zst 必須):
//! ```sh
//! cargo run -p ellisii-sdk --features static-jp,delarocha \
//!   --example eval_workplace_regs_tok_delarocha --release
//! ```
//!
//! 結果は `docs/eval/recall-evals.md` jp-workplace-regs セクション
//! 「Run 6 (FTS5 tokenizer A/B — delarocha)」として追記する。

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use ellisii_core::Chunk;
use ellisii_jp_tokenizer_bigram::CharBigramTokenizer;
use ellisii_jp_tokenizer_core::JpTokenizer;
use ellisii_rag::eval::{summarize, GoldenSet};
use ellisii_sdk::{Ellisii, SearchOptions};
use ellisii_store_core::VectorStore;
use ellisii_store_sqlite::SqliteStore;
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
fn delarocha_dict() -> PathBuf {
    home().join("Library/Application Support/ellisii/models/delarocha/system.dic.zst")
}
fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .join("rag/tests/fixtures/eval/jp-workplace-regs")
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> anyhow::Result<()> {
    #[cfg(not(all(feature = "static-jp", feature = "delarocha")))]
    {
        anyhow::bail!("build with --features static-jp,delarocha");
    }
    #[cfg(all(feature = "static-jp", feature = "delarocha"))]
    return run().await;
}

#[cfg(all(feature = "static-jp", feature = "delarocha"))]
async fn run() -> anyhow::Result<()> {
    use ellisii_jp_tokenizer_delarocha::DelarochaTokenizer;

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
    eprintln!("embed: {}", embed.display());
    let dela_path = delarocha_dict();
    if !dela_path.is_file() {
        anyhow::bail!(
            "delarocha system.dic.zst not found at {}",
            dela_path.display()
        );
    }
    eprintln!("delarocha: {}", dela_path.display());

    let dim = 1024;
    let bigram: Arc<dyn JpTokenizer> = Arc::new(CharBigramTokenizer::new());
    let store_bigram = Arc::new(SqliteStore::open_in_memory_with_tokenizer(dim, bigram)?);

    let dela_tok: Arc<dyn JpTokenizer> = Arc::new(
        DelarochaTokenizer::from_path(&dela_path)
            .map_err(|e| anyhow::anyhow!("load delarocha: {e}"))?,
    );
    let store_dela: Arc<dyn VectorStore> =
        Arc::new(SqliteStore::open_in_memory_with_tokenizer(dim, dela_tok)?);

    let build = |store: Arc<dyn VectorStore>| -> anyhow::Result<Ellisii> {
        Ok(Ellisii::builder()
            .with_embedder_static_jp(&embed)?
            .with_store(store)
            .with_notebook_id(nb)
            .build()?)
    };

    let ellisii_bigram = build(store_bigram.clone())?;
    let embs = ellisii_bigram.embedder().embed(&texts).await?;
    store_bigram.upsert(nb, &chunks, &embs).await?;
    store_dela.upsert(nb, &chunks, &embs).await?;
    let ellisii_dela = build(store_dela.clone())?;

    println!("\n=== jp-workplace-regs: FTS5 tokenizer A/B — delarocha (Run 6, k=5) ===");
    println!(
        "{:<28} {:<10} {:<10} {:<10} {:<10}",
        "variant", "recall", "hit", "ndcg", "mrr"
    );

    for w in [0.0_f32, 0.25, 0.5, 0.75, 1.0] {
        for (label, eng) in [
            (format!("bigram   cap w={:.2}", w), &ellisii_bigram),
            (format!("delarocha cap w={:.2}", w), &ellisii_dela),
        ] {
            let pairs = run_pairs(eng, &gold, &id_map, w, true, 5).await?;
            let s = summarize(&pairs, 5);
            println!(
                "{:<28} {:<10.3} {:<10.3} {:<10.3} {:<10.3}",
                label, s.recall_at_k, s.hit_at_k, s.ndcg_at_k, s.mrr
            );
        }
    }

    println!("\n=== Targeted failures @ k=5 (weight=0.5, cap=on) ===");
    for q in ["法定休日は何曜日", "出張中の労働時間はどう扱われるか"] {
        let item = gold.items.iter().find(|i| i.query == q).unwrap();
        println!("  {}  expected={:?}", q, item.relevant);
        for (label, eng) in [("bigram   ", &ellisii_bigram), ("delarocha", &ellisii_dela)] {
            let hits = eng.search(q, opts(0.5, 5)).await?;
            let pred: Vec<String> = hits
                .iter()
                .filter_map(|h| id_map.get(&h.chunk.id).cloned())
                .collect();
            println!("    {} top5={:?}", label, pred);
        }
    }

    Ok(())
}

#[cfg(all(feature = "static-jp", feature = "delarocha"))]
fn opts(semantic_weight: f32, top_k: usize) -> SearchOptions {
    SearchOptions {
        top_k,
        semantic_weight,
        caption_rerank: true,
        ..Default::default()
    }
}

#[cfg(all(feature = "static-jp", feature = "delarocha"))]
async fn run_pairs(
    ellisii: &Ellisii,
    gold: &GoldenSet,
    id_map: &HashMap<Uuid, String>,
    semantic_weight: f32,
    caption_rerank: bool,
    k: usize,
) -> ellisii_core::Result<Vec<(Vec<String>, Vec<String>)>> {
    let mut pairs = Vec::with_capacity(gold.items.len());
    for item in &gold.items {
        let hits = ellisii
            .search(
                &item.query,
                SearchOptions {
                    top_k: k,
                    semantic_weight,
                    caption_rerank,
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
