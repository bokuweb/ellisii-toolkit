//! Run 12i (SDK 統合) の A/B 実測ハーネス。
//!
//! Run 12d で eval 例ツール経由で 4/4 #1 rescue が確認できた静的辞書
//! enrichment を、**production パス (ellisii-jp-law-thesaurus crate +
//! CaptionEnricher trait)** 経由で同条件で走らせ、同じ rescue が再現するかを
//! 検証する。
//!
//! 既存 `eval_retrieval_dump.rs` と同様に corpus.json を chunk として手で
//! upsert するが、本ハーネスは:
//!
//! - baseline (raw chunks)
//! - enriched (`LawThesaurus::bundled().enrich_chunks(...)` を通したもの)
//!
//! の 2 store を bigram (Run 12d で最良) 1 つで比較し、hit@5 / MRR / 4
//! 取り逃しクエリ (94 / 90 / 192 / 900) の rank を表で出す。
//!
//! 使い方:
//! ```sh
//! cargo run -p ellisii-sdk --features static-jp \
//!   --example eval_civil_law_enrichment_ab --release
//! ```

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use ellisii_core::Chunk;
use ellisii_jp_law_thesaurus::LawThesaurus;
use ellisii_jp_tokenizer_bigram::CharBigramTokenizer;
use ellisii_jp_tokenizer_core::JpTokenizer;
use ellisii_rag::eval::{hit_at_k, reciprocal_rank, GoldenSet};
use ellisii_sdk::{Ellisii, SearchOptions};
use ellisii_store_core::VectorStore;
use ellisii_store_sqlite::SqliteStore;
use serde::Deserialize;
use uuid::Uuid;

#[derive(Debug, Deserialize, Clone)]
struct CorpusEntry {
    doc_id: String,
    #[serde(default)]
    caption: String,
    text: String,
}

fn home() -> PathBuf {
    PathBuf::from(std::env::var("HOME").expect("HOME"))
}
fn embed_dir() -> PathBuf {
    home().join("Library/Application Support/ellisii/models/static-embedding-japanese")
}
fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .join("rag/tests/fixtures/eval/jp-civil-law-hard")
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> anyhow::Result<()> {
    #[cfg(not(feature = "static-jp"))]
    {
        anyhow::bail!("build with --features static-jp");
    }
    #[cfg(feature = "static-jp")]
    return run().await;
}

#[cfg(feature = "static-jp")]
async fn run() -> anyhow::Result<()> {
    let dir = fixture_dir();
    let corpus: Vec<CorpusEntry> =
        serde_json::from_str(&std::fs::read_to_string(dir.join("corpus.json"))?)?;
    let gold: GoldenSet =
        GoldenSet::from_json_str(&std::fs::read_to_string(dir.join("golden.json"))?)?;
    eprintln!(
        "fixture: jp-civil-law-hard ({} chunks, {} queries)",
        corpus.len(),
        gold.items.len()
    );

    let nb = Uuid::new_v4();
    let src = Uuid::new_v4();
    let mut chunks_base: Vec<Chunk> = Vec::with_capacity(corpus.len());
    let mut id_map: HashMap<Uuid, String> = HashMap::new();
    for (i, e) in corpus.iter().enumerate() {
        let cid = Uuid::new_v4();
        id_map.insert(cid, e.doc_id.clone());
        let txt = if e.caption.is_empty() {
            e.text.clone()
        } else {
            format!("({})\n{}", e.caption, e.text)
        };
        chunks_base.push(Chunk {
            id: cid,
            source_id: src,
            ord: i as u32,
            text: txt,
            heading_path: vec![e.doc_id.clone()],
            page: None,
            bbox: None,
            summary: None,
        });
    }

    // enrichment は本物の crate / trait 経路で実施 (= Ingestor が
    // ingest_one_inner 内で呼ぶのと同じ呼び出し)。
    let thes = Arc::new(LawThesaurus::bundled());
    let mut chunks_enriched = chunks_base.clone();
    let t0 = Instant::now();
    let n_enriched = thes.enrich_chunks(&mut chunks_enriched);
    let enrich_elapsed = t0.elapsed();
    eprintln!(
        "thesaurus: {} ({} entries), enriched {}/{} chunks in {:.2?}",
        thes.name(),
        thes.entry_count(),
        n_enriched,
        chunks_enriched.len(),
        enrich_elapsed,
    );

    let embed = embed_dir();
    let dim = 1024;
    let tok_base: Arc<dyn JpTokenizer> = Arc::new(CharBigramTokenizer::new());
    let tok_enr: Arc<dyn JpTokenizer> = Arc::new(CharBigramTokenizer::new());
    let store_base: Arc<dyn VectorStore> =
        Arc::new(SqliteStore::open_in_memory_with_tokenizer(dim, tok_base)?);
    let store_enr: Arc<dyn VectorStore> =
        Arc::new(SqliteStore::open_in_memory_with_tokenizer(dim, tok_enr)?);

    let engine_base = Ellisii::builder()
        .with_embedder_static_jp(&embed)?
        .with_store(store_base.clone())
        .with_notebook_id(nb)
        .build()?;
    let engine_enr = Ellisii::builder()
        .with_embedder_static_jp(&embed)?
        .with_store(store_enr.clone())
        .with_notebook_id(nb)
        .build()?;

    // 別々に embed して別 store に入れる。
    let texts_base: Vec<String> = chunks_base.iter().map(|c| c.text.clone()).collect();
    let texts_enr: Vec<String> = chunks_enriched.iter().map(|c| c.text.clone()).collect();
    let embs_base = engine_base.embedder().embed(&texts_base).await?;
    let embs_enr = engine_enr.embedder().embed(&texts_enr).await?;
    store_base.upsert(nb, &chunks_base, &embs_base).await?;
    store_enr.upsert(nb, &chunks_enriched, &embs_enr).await?;

    // 4 hard-fail queries (Run 12d で baseline 取り逃した 4 件)。
    let hard = ["minpou-94", "minpou-90", "minpou-192", "minpou-900"];

    let opts = SearchOptions {
        top_k: 5,
        semantic_weight: 0.5,
        caption_rerank: true,
        ..Default::default()
    };

    let mut pairs_base: Vec<(Vec<String>, Vec<String>)> = Vec::new();
    let mut pairs_enr: Vec<(Vec<String>, Vec<String>)> = Vec::new();
    let mut hard_ranks: Vec<(String, String, Option<usize>, Option<usize>)> = Vec::new();

    for item in &gold.items {
        let hits_b = engine_base.search(&item.query, opts.clone()).await?;
        let hits_e = engine_enr.search(&item.query, opts.clone()).await?;
        let pred_b: Vec<String> = hits_b
            .iter()
            .map(|h| id_map.get(&h.chunk.id).cloned().unwrap_or_default())
            .collect();
        let pred_e: Vec<String> = hits_e
            .iter()
            .map(|h| id_map.get(&h.chunk.id).cloned().unwrap_or_default())
            .collect();

        if hard.iter().any(|h| item.relevant.iter().any(|r| r == h)) {
            let rank_b = item
                .relevant
                .iter()
                .find_map(|r| pred_b.iter().position(|p| p == r))
                .map(|i| i + 1);
            let rank_e = item
                .relevant
                .iter()
                .find_map(|r| pred_e.iter().position(|p| p == r))
                .map(|i| i + 1);
            hard_ranks.push((
                item.relevant.first().cloned().unwrap_or_default(),
                item.query.clone(),
                rank_b,
                rank_e,
            ));
        }

        pairs_base.push((pred_b, item.relevant.clone()));
        pairs_enr.push((pred_e, item.relevant.clone()));
    }

    let hit5_b: f32 = pairs_base
        .iter()
        .map(|(p, r)| hit_at_k(p, r, 5))
        .sum::<f32>()
        / pairs_base.len() as f32;
    let hit5_e: f32 = pairs_enr
        .iter()
        .map(|(p, r)| hit_at_k(p, r, 5))
        .sum::<f32>()
        / pairs_enr.len() as f32;
    let mrr_b: f32 = pairs_base
        .iter()
        .map(|(p, r)| reciprocal_rank(p, r))
        .sum::<f32>()
        / pairs_base.len() as f32;
    let mrr_e: f32 = pairs_enr
        .iter()
        .map(|(p, r)| reciprocal_rank(p, r))
        .sum::<f32>()
        / pairs_enr.len() as f32;

    println!(
        "\n=== Aggregate (n={}, k=5, bigram, cap rerank=on) ===",
        pairs_base.len()
    );
    println!("| variant            | hit@5 | MRR    |");
    println!("|--------------------|------:|-------:|");
    println!("| baseline           | {:.3} | {:.3} |", hit5_b, mrr_b);
    println!(
        "| **enriched (v5)**  | **{:.3}** | **{:.3}** |",
        hit5_e, mrr_e
    );

    println!("\n=== Hard-fail rescues (Run 12d 残 4 件) ===");
    println!(
        "| expected   | query                                            | baseline | enriched |"
    );
    println!(
        "|------------|--------------------------------------------------|:--------:|:--------:|"
    );
    for (exp, q, rb, re) in &hard_ranks {
        let fmt = |r: &Option<usize>| match r {
            Some(n) if *n <= 5 => format!("#{n}"),
            Some(_) => "—".to_string(),
            None => "—".to_string(),
        };
        let q_short: String = q.chars().take(48).collect();
        println!(
            "| {} | {:<48} | {:^8} | {:^8} |",
            exp,
            q_short,
            fmt(rb),
            fmt(re)
        );
    }

    Ok(())
}
