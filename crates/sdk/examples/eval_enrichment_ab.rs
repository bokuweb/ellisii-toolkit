//! 任意 fixture (corpus.json + golden.json) に対して、
//! `LawThesaurus::bundled()` enrichment ON/OFF を A/B 計測する汎用ハーネス。
//!
//! Run 12j で civil-law-hard 専用に作ったハーネス
//! (`eval_civil_law_enrichment_ab.rs`) を一般化したもの。fixture を
//! env var で切替えるだけで他法令系 corpus にも適用できる。
//!
//! 使い方:
//! ```sh
//! ELLISII_EVAL_FIXTURE=yokohama \
//!   cargo run -p ellisii-sdk --features static-jp \
//!     --example eval_enrichment_ab --release
//! ```
//!
//! Run 12k で yokohama / jp-labor-law / jp-cs-wiki-hard / jp-workplace-regs /
//! jp-tokkyo-hou に対して順に走らせ、辞書未マッチが多い corpus でも net で
//! 退行しないこと (= do-no-harm) を確認する。

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
    /// 親 doc id。golden の `relevant` が parent 粒度で書かれている fixture
    /// (例: jp-manual) では `parent_id.unwrap_or(doc_id)` を id_map の値に
    /// 使うことで parent-level relevance を素直に評価できる。
    #[serde(default)]
    parent_id: Option<String>,
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
fn fixture_dir() -> (String, PathBuf) {
    let name =
        std::env::var("ELLISII_EVAL_FIXTURE").unwrap_or_else(|_| "jp-civil-law-hard".to_string());
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .join("rag/tests/fixtures/eval")
        .join(&name);
    (name, p)
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
    let (name, dir) = fixture_dir();
    let corpus: Vec<CorpusEntry> =
        serde_json::from_str(&std::fs::read_to_string(dir.join("corpus.json"))?)?;
    let gold: GoldenSet =
        GoldenSet::from_json_str(&std::fs::read_to_string(dir.join("golden.json"))?)?;
    eprintln!(
        "fixture: {} ({} chunks, {} queries)",
        name,
        corpus.len(),
        gold.items.len()
    );

    // golden の relevant が doc_id 粒度か parent_id 粒度か auto-detect
    // (eval_fixtures.rs と同じロジック)。誤検出回避のため doc_id 優先。
    let use_parent_id = {
        use std::collections::HashSet;
        let doc_ids: HashSet<&str> = corpus.iter().map(|e| e.doc_id.as_str()).collect();
        let parent_ids: HashSet<&str> = corpus
            .iter()
            .filter_map(|e| e.parent_id.as_deref())
            .collect();
        let golden_relevant: HashSet<&str> = gold
            .items
            .iter()
            .flat_map(|i| i.relevant.iter().map(|s| s.as_str()))
            .collect();
        !golden_relevant.is_empty()
            && !golden_relevant.iter().all(|r| doc_ids.contains(r))
            && golden_relevant.iter().all(|r| parent_ids.contains(r))
    };

    let nb = Uuid::new_v4();
    let src = Uuid::new_v4();
    let mut chunks_base: Vec<Chunk> = Vec::with_capacity(corpus.len());
    let mut id_map: HashMap<Uuid, String> = HashMap::new();
    for (i, e) in corpus.iter().enumerate() {
        let cid = Uuid::new_v4();
        let rel_id = if use_parent_id {
            e.parent_id.clone().unwrap_or_else(|| e.doc_id.clone())
        } else {
            e.doc_id.clone()
        };
        id_map.insert(cid, rel_id);
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

    let thes = Arc::new(LawThesaurus::bundled());
    let mut chunks_enriched = chunks_base.clone();
    let t0 = Instant::now();
    let n_enriched = thes.enrich_chunks(&mut chunks_enriched);
    eprintln!(
        "thesaurus: {} ({} entries), enriched {}/{} chunks ({:.0}%) in {:.2?}",
        thes.name(),
        thes.entry_count(),
        n_enriched,
        chunks_enriched.len(),
        100.0 * n_enriched as f32 / chunks_enriched.len() as f32,
        t0.elapsed(),
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

    let texts_base: Vec<String> = chunks_base.iter().map(|c| c.text.clone()).collect();
    let texts_enr: Vec<String> = chunks_enriched.iter().map(|c| c.text.clone()).collect();
    let embs_base = engine_base.embedder().embed(&texts_base).await?;
    let embs_enr = engine_enr.embedder().embed(&texts_enr).await?;
    store_base.upsert(nb, &chunks_base, &embs_base).await?;
    store_enr.upsert(nb, &chunks_enriched, &embs_enr).await?;

    for cap in [true, false] {
        let opts = SearchOptions {
            top_k: 5,
            semantic_weight: 0.5,
            caption_rerank: cap,
            ..Default::default()
        };

        let mut pairs_base: Vec<(Vec<String>, Vec<String>)> = Vec::new();
        let mut pairs_enr: Vec<(Vec<String>, Vec<String>)> = Vec::new();
        let mut swaps_rescue: Vec<(String, String)> = Vec::new();
        let mut swaps_regress: Vec<(String, String)> = Vec::new();

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

            let hit_b = hit_at_k(&pred_b, &item.relevant, 5) > 0.5;
            let hit_e = hit_at_k(&pred_e, &item.relevant, 5) > 0.5;
            let exp = item.relevant.first().cloned().unwrap_or_default();
            match (hit_b, hit_e) {
                (false, true) => swaps_rescue.push((exp, item.query.clone())),
                (true, false) => swaps_regress.push((exp, item.query.clone())),
                _ => {}
            }

            pairs_base.push((pred_b, item.relevant.clone()));
            pairs_enr.push((pred_e, item.relevant.clone()));
        }

        let n = pairs_base.len() as f32;
        let hit5 = |pairs: &[(Vec<String>, Vec<String>)]| -> f32 {
            pairs.iter().map(|(p, r)| hit_at_k(p, r, 5)).sum::<f32>() / n
        };
        let mrr = |pairs: &[(Vec<String>, Vec<String>)]| -> f32 {
            pairs
                .iter()
                .map(|(p, r)| reciprocal_rank(p, r))
                .sum::<f32>()
                / n
        };
        let (hit5_b, hit5_e) = (hit5(&pairs_base), hit5(&pairs_enr));
        let (mrr_b, mrr_e) = (mrr(&pairs_base), mrr(&pairs_enr));

        println!(
            "\n=== {} (n={}, k=5, bigram, cap rerank={}) ===",
            name,
            pairs_base.len(),
            if cap { "on" } else { "off" }
        );
        println!("| variant            | hit@5 | MRR    |");
        println!("|--------------------|------:|-------:|");
        println!("| baseline           | {:.3} | {:.3} |", hit5_b, mrr_b);
        println!(
            "| **enriched (v5)**  | **{:.3}** | **{:.3}** |",
            hit5_e, mrr_e
        );
        println!(
            "| Δ                  | {:+.3} | {:+.3} |",
            hit5_e - hit5_b,
            mrr_e - mrr_b
        );

        println!(
            "rescue (baseline ✗ → enriched ✓): {} / regress: {}",
            swaps_rescue.len(),
            swaps_regress.len()
        );
        for (exp, q) in &swaps_rescue {
            let q: String = q.chars().take(60).collect();
            println!("  + {exp}  {q}");
        }
        for (exp, q) in &swaps_regress {
            let q: String = q.chars().take(60).collect();
            println!("  - {exp}  {q}");
        }
    }

    Ok(())
}
