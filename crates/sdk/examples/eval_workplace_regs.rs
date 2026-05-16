//! 日本の社内規程 (就業規則 / パートタイマー就業規則 / リフレッシュ休暇取扱規程 /
//! 育児・介護休業規程) を corpus とした recall 評価ハーネス。
//!
//! 公的に公開されている規程例 (4 ドキュメント, 175 chunks) を `corpus.json` に
//! 1 条 = 1 chunk で並べ、人手で作った golden Q&A (n=40) に対して
//! recall@K / hit@K / nDCG@K / MRR を計測する。yokohama 市税条例と同じ
//! caption 付きフォーマット (`(caption)\n第N条 …`) を採用しているので
//! caption rerank の効きを別 corpus でも検証できる。
//!
//! 実行:
//! ```sh
//! cargo run -p ellisii-sdk --features static-jp \
//!   --example eval_workplace_regs --release
//! ```
//!
//! 結果は `docs/eval/recall-evals.md` の「就業規則系コーパス」セクションに
//! 追記する。

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
fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .join("rag/tests/fixtures/eval/jp-workplace-regs")
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
        "corpus: {} chunks, golden: {} ({} items)",
        corpus.len(),
        gold.name,
        gold.items.len()
    );

    // doc_id → uuid 両方向で持って、結果評価時に逆引きする。
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

    let ellisii = Ellisii::builder()
        .with_embedder_static_jp(&embed)?
        .with_store_memory()
        .with_notebook_id(nb)
        .build()?;
    let embs = ellisii.embedder().embed(&texts).await?;
    ellisii.store().upsert(nb, &chunks, &embs).await?;

    // signals
    println!("\n=== Corpus / query signals ===");
    let density = ellisii.caption_density().await?;
    let para = ellisii.corpus_paraphrase_score().await?;
    let q_strs: Vec<&str> = gold.items.iter().map(|i| i.query.as_str()).collect();
    let q_specific = ellisii_rag::specific_query_ratio(&q_strs);
    let q_body_recall = ellisii.query_body_literal_match(&q_strs).await?;
    println!(
        "jp-workplace-regs density={:.3} paraphrase={:.3} q_specific={:.3} q_body_recall={:.3}",
        density, para, q_specific, q_body_recall
    );

    // 1) weight x top_k sweep, caption_rerank=off
    println!("\n=== Baseline: semantic_weight × top_k (caption_rerank=false) ===");
    println!(
        "{:<8} {:<6} {:<10} {:<10} {:<10} {:<10}",
        "weight", "k", "recall", "hit", "ndcg", "mrr"
    );
    let weights = [0.0_f32, 0.25, 0.5, 0.75, 1.0];
    let ks = [1usize, 3, 5, 10];
    let kmax = *ks.iter().max().unwrap();
    for &w in &weights {
        let pairs = run_pairs(&ellisii, &gold, &id_map, w, false, kmax, 0, 0.0).await?;
        for &k in &ks {
            let s = summarize(&trim(&pairs, k), k);
            println!(
                "{:<8.2} {:<6} {:<10.3} {:<10.3} {:<10.3} {:<10.3}",
                w, k, s.recall_at_k, s.hit_at_k, s.ndcg_at_k, s.mrr
            );
        }
    }

    // 2) caption rerank A/B (weight=0.5)
    println!("\n=== Caption rerank A/B (weight=0.5) ===");
    println!(
        "{:<20} {:<10} {:<10} {:<10} {:<10}",
        "variant", "recall", "hit", "ndcg", "mrr"
    );
    for &k in &ks {
        let off = run_pairs(&ellisii, &gold, &id_map, 0.5, false, k, 0, 0.0).await?;
        let on = run_pairs(&ellisii, &gold, &id_map, 0.5, true, k, 0, 0.0).await?;
        let s_off = summarize(&off, k);
        let s_on = summarize(&on, k);
        println!(
            "{:<20} {:<10.3} {:<10.3} {:<10.3} {:<10.3}",
            format!("off (k={})", k),
            s_off.recall_at_k,
            s_off.hit_at_k,
            s_off.ndcg_at_k,
            s_off.mrr
        );
        println!(
            "{:<20} {:<10.3} {:<10.3} {:<10.3} {:<10.3}",
            format!("on  (k={})", k),
            s_on.recall_at_k,
            s_on.hit_at_k,
            s_on.ndcg_at_k,
            s_on.mrr
        );
    }

    // 3) auto_adjust_weight A/B (weight base=0.5, caption_rerank=on)
    println!("\n=== auto_adjust_weight A/B (cap=on, base=0.5) ===");
    println!(
        "{:<20} {:<10} {:<10} {:<10} {:<10}",
        "variant", "recall", "hit", "ndcg", "mrr"
    );
    for &k in &[1usize, 5, 10] {
        let mut off = Vec::with_capacity(gold.items.len());
        let mut on = Vec::with_capacity(gold.items.len());
        for item in &gold.items {
            for (acc, auto) in [(&mut off, false), (&mut on, true)] {
                let hits = ellisii
                    .search(
                        &item.query,
                        SearchOptions {
                            top_k: k,
                            semantic_weight: 0.5,
                            caption_rerank: true,
                            auto_adjust_weight: auto,
                            ..Default::default()
                        },
                    )
                    .await?;
                let pred: Vec<String> = hits
                    .iter()
                    .filter_map(|h| id_map.get(&h.chunk.id).cloned())
                    .collect();
                acc.push((pred, item.relevant.clone()));
            }
        }
        let s_off = summarize(&off, k);
        let s_on = summarize(&on, k);
        println!(
            "{:<20} {:<10.3} {:<10.3} {:<10.3} {:<10.3}",
            format!("auto=off (k={})", k),
            s_off.recall_at_k,
            s_off.hit_at_k,
            s_off.ndcg_at_k,
            s_off.mrr
        );
        println!(
            "{:<20} {:<10.3} {:<10.3} {:<10.3} {:<10.3}",
            format!("auto=on  (k={})", k),
            s_on.recall_at_k,
            s_on.hit_at_k,
            s_on.ndcg_at_k,
            s_on.mrr
        );
    }

    // 4) failure diagnosis at k=5 (caption on)
    println!("\n=== Failure diagnosis (cap=on, weight=0.5, k=5) ===");
    let pairs = run_pairs(&ellisii, &gold, &id_map, 0.5, true, 5, 0, 0.0).await?;
    let mut misses = 0usize;
    for (item, (pred, expected)) in gold.items.iter().zip(pairs.iter()) {
        let ok = expected.iter().any(|e| pred.iter().any(|p| p == e));
        if !ok {
            misses += 1;
            println!(
                "  miss: \"{}\"  expected={:?}  top5={:?}",
                item.query, expected, pred
            );
        }
    }
    println!("  {}/{} miss", misses, gold.items.len());

    Ok(())
}

#[cfg(feature = "static-jp")]
fn trim(pairs: &[(Vec<String>, Vec<String>)], k: usize) -> Vec<(Vec<String>, Vec<String>)> {
    pairs
        .iter()
        .map(|(p, r)| (p.iter().take(k).cloned().collect(), r.clone()))
        .collect()
}

#[cfg(feature = "static-jp")]
async fn run_pairs(
    ellisii: &Ellisii,
    gold: &GoldenSet,
    id_map: &HashMap<Uuid, String>,
    semantic_weight: f32,
    caption_rerank: bool,
    k: usize,
    multi_variants: usize,
    variant_caption_filter: f32,
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
                    multi_query_max_variants: multi_variants,
                    multi_query_variant_weight: 0.7,
                    variant_caption_filter_threshold: variant_caption_filter,
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
