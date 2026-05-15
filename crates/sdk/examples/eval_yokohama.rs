//! 横浜市市税条例 (既に index 済み) に対する recall / RAG 精度評価。
//!
//! 既存の `~/Library/Application Support/ellisii/ellisii.db` を SDK ([`Ellisii`]) 経由で開き、
//! 手作りの golden Q&A について caption rerank の on/off で recall を比較する。
//!
//! 実行:
//! ```sh
//! cargo run -p ellisii-sdk --features static-jp --example eval_yokohama --release
//! ```
//!
//! このハーネスを更新したら `docs/eval/recall-evals.md` も追記して結果を残すこと。

use std::path::PathBuf;
use std::sync::Arc;

use async_trait::async_trait;
use ellisii_core::Result as ECResult;
use ellisii_query_rewriter_core::{QueryRewriter, RewrittenQueries};
use ellisii_rag::eval::{summarize, EvalSummary, GoldenItem, GoldenSet};
use ellisii_sdk::{Ellisii, SearchOptions};
use uuid::Uuid;

/// LLM の代わりに、横浜市市税条例向けに手作りした synonym 表で variant を生やす
/// 決定的 rewriter。本番では `ellisii_query_rewriter_llm::LlmRewriter` を使う想定で、
/// ここでの結果は「LLM が完璧に近い言い換えを返したらどこまで上がるか」の参考値。
struct YokohamaSynonymRewriter;

#[async_trait]
impl QueryRewriter for YokohamaSynonymRewriter {
    async fn rewrite(&self, query: &str, max_variants: usize) -> ECResult<RewrittenQueries> {
        let table: &[(&str, &[&str])] = &[
            ("温泉", &["入湯税の税率", "入湯税"]),
            ("入湯", &["入湯税の税率"]),
            ("市たばこ", &["たばこ税の税率"]),
            ("個人市民税", &["所得割の税率", "均等割の税率"]),
            ("利率", &["延滞金", "割合"]),
            ("普通税", &["市税として課する普通税", "市民税 固定資産税 軽自動車税"]),
            ("徴税吏員", &["市長 委任 市職員"]),
            ("課税の根拠", &["地方税法 賦課徴収"]),
            ("均等割", &["均等割の税率", "均等割の税率の軽減"]),
            ("いつ", &["納期"]),
        ];
        let mut variants: Vec<String> = Vec::new();
        for (key, vs) in table {
            if query.contains(key) {
                for v in *vs {
                    if variants.len() >= max_variants {
                        break;
                    }
                    if !variants.iter().any(|x| x == v) {
                        variants.push((*v).to_string());
                    }
                }
            }
            if variants.len() >= max_variants {
                break;
            }
        }
        Ok(RewrittenQueries {
            original: query.to_string(),
            variants,
        })
    }
}

const NOTEBOOK_ID: &str = "95339065-df88-4ee7-82c1-e11c587250e4";

fn home() -> PathBuf {
    PathBuf::from(std::env::var("HOME").expect("HOME"))
}
fn db_path() -> PathBuf {
    home().join("Library/Application Support/ellisii/ellisii.db")
}
fn model_dir() -> PathBuf {
    home().join("Library/Application Support/ellisii/models/static-embedding-japanese")
}

/// 横浜市市税条例 (notebook=95339065…) 用の golden Q&A。chunk_id は ord から逆引きした
/// uuid を直書き。fixture は `crates/sdk/tests/fixtures/eval/yokohama/golden.json`
/// に配置して regression test と共有する。条例の章立てを変えると id がずれるので、
/// 再 index 後は `docs/eval/recall-evals.md` の chunk-id 引き直し snippet で更新する。
fn golden() -> Vec<GoldenItem> {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/eval/yokohama/golden.json");
    let raw = std::fs::read_to_string(&path).expect("read yokohama golden fixture");
    let set = GoldenSet::from_json_str(&raw).expect("parse yokohama golden fixture");
    eprintln!("golden: {} ({} items)", set.name, set.items.len());
    set.items
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
    let db = db_path();
    let model = model_dir();
    eprintln!("DB:    {}", db.display());
    eprintln!("Model: {}", model.display());

    let dim = 1024;
    let ellisii = Arc::new(
        Ellisii::builder()
            .with_embedder_static_jp(&model)?
            .with_store_sqlite(&db, dim)?
            .with_notebook_id(Uuid::parse_str(NOTEBOOK_ID)?)
            .build()?,
    );

    let gold = golden();
    eprintln!("queries: {}", gold.len());

    // 0) Corpus / query signals
    //    Run 18 (paraphrase) / Run 20 (hypothesis 訂正) / Run 21 (query 側 signal) を参照。
    //    corpus signal だけでは rewriter ROI を予測できないと判明したため、
    //    クエリ側 specific_query_ratio を主軸の判断材料として出力する。
    println!("\n=== Corpus / query signals (Run 18 / 20 / 21 / 22) ===");
    let density = ellisii.caption_density().await?;
    let para = ellisii.corpus_paraphrase_score().await?;
    let q_strs: Vec<&str> = gold.iter().map(|i| i.query.as_str()).collect();
    let q_specific = ellisii_rag::specific_query_ratio(&q_strs);
    let q_body_recall = ellisii.query_body_literal_match(&q_strs).await?;
    let recommendation = if q_specific >= 0.5 {
        "rewriter≈OFF (specific 偏重)"
    } else if q_body_recall >= 0.7 {
        "rewriter≈OFF (literal lookup, body recall 高)"
    } else if q_specific < 0.3 && q_body_recall < 0.4 {
        "rewriter ON (paraphrase ROI 期待)"
    } else {
        "mix (per-query gate に委ねる)"
    };
    println!(
        "yokohama 市税条例           density={:.3}  paraphrase={:.3}  q_specific={:.3}  q_body_recall={:.3}  → {}",
        density, para, q_specific, q_body_recall, recommendation
    );

    // 1) semantic_weight × top_k のスイープ (caption_rerank なし) — 旧仕様の挙動。
    println!("\n=== Baseline: semantic_weight × top_k (caption_rerank=false) ===");
    println!("{:<8} {:<6} {:<10} {:<10} {:<10} {:<10}", "weight", "k", "recall", "hit", "ndcg", "mrr");
    let weights = [0.0_f32, 0.25, 0.5, 0.75, 1.0];
    let k_values = [1usize, 3, 5, 10];
    for w in weights {
        let pairs = run_queries(&ellisii, &gold, w, false, *k_values.iter().max().unwrap()).await?;
        for k in k_values {
            let s = summarize(&pairs, k);
            print_row(w, k, &s);
        }
    }

    // 2) caption rerank on/off (weight=0.5)
    println!("\n=== Caption rerank A/B (weight=0.5) ===");
    println!("{:<28} {:<10} {:<10} {:<10} {:<10}", "variant", "recall", "hit", "ndcg", "mrr");
    for k in [1usize, 3, 5, 10] {
        let off = run_queries(&ellisii, &gold, 0.5, false, k).await?;
        let on = run_queries(&ellisii, &gold, 0.5, true, k).await?;
        let s_off = summarize(&off, k);
        let s_on = summarize(&on, k);
        println!("{:<28} {:<10.3} {:<10.3} {:<10.3} {:<10.3}", format!("off (k={})", k), s_off.recall_at_k, s_off.hit_at_k, s_off.ndcg_at_k, s_off.mrr);
        println!("{:<28} {:<10.3} {:<10.3} {:<10.3} {:<10.3}", format!("on  (k={})", k), s_on.recall_at_k, s_on.hit_at_k, s_on.ndcg_at_k, s_on.mrr);
    }

    // 3) caption + CE rerank (provence-onnx) sweep。`provence-onnx` feature 必須。
    #[cfg(feature = "provence-onnx")]
    {
        let provence_dir = home().join("Library/Application Support/ellisii/models/open-provence");
        if provence_dir.exists() {
            println!("\n=== Caption rerank + CE rerank (provence-onnx) sweep ===");
            let ellisii_ce = Ellisii::builder()
                .with_embedder_static_jp(&model)?
                .with_store_sqlite(&db, dim)?
                .with_notebook_id(Uuid::parse_str(NOTEBOOK_ID)?)
                .with_compressor_provence_onnx(&provence_dir, 0.20)?
                .build()?;
            println!(
                "{:<32} {:<10} {:<10} {:<10} {:<10}",
                "variant", "recall", "hit", "ndcg", "mrr"
            );
            // baseline (cap only) at k=5
            let baseline = run_queries(&ellisii, &gold, 0.5, true, 5).await?;
            let s = summarize(&baseline, 5);
            println!(
                "{:<32} {:<10.3} {:<10.3} {:<10.3} {:<10.3}",
                "cap only (k=5)", s.recall_at_k, s.hit_at_k, s.ndcg_at_k, s.mrr
            );
            for &(label, top_n, w) in &[
                ("cap+CE top_n=10 w=0.3", 10usize, 0.3_f32),
                ("cap+CE top_n=10 w=0.5", 10, 0.5),
                ("cap+CE top_n=10 w=0.7", 10, 0.7),
                ("cap+CE top_n=10 w=1.0", 10, 1.0),
                ("cap+CE top_n=20 w=0.5", 20, 0.5),
                ("cap+CE top_n=30 w=0.5", 30, 0.5),
            ] {
                let mut pairs: Vec<(Vec<String>, Vec<String>)> = Vec::with_capacity(gold.len());
                for item in &gold {
                    let hits = ellisii_ce
                        .search(
                            &item.query,
                            SearchOptions {
                                top_k: 5,
                                semantic_weight: 0.5,
                                caption_rerank: true,
                                ce_rerank_top_n: top_n,
                                ce_rerank_weight: w,
                                ..Default::default()
                            },
                        )
                        .await?;
                    let pred: Vec<String> = hits.iter().map(|h| h.chunk.id.to_string()).collect();
                    pairs.push((pred, item.relevant.clone()));
                }
                let s = summarize(&pairs, 5);
                println!(
                    "{:<32} {:<10.3} {:<10.3} {:<10.3} {:<10.3}",
                    label, s.recall_at_k, s.hit_at_k, s.ndcg_at_k, s.mrr
                );
            }
        } else {
            eprintln!(
                "[skip CE rerank section] provence model not found at {}",
                provence_dir.display()
            );
        }
    }

    // 4) caption + multi-query (synonym rewriter, simulating LLM)
    println!("\n=== Caption rerank + multi-query (synonym, weight=0.5) ===");
    let dim2 = 1024;
    let ellisii_mq = Ellisii::builder()
        .with_embedder_static_jp(&model)?
        .with_store_sqlite(&db, dim2)?
        .with_notebook_id(Uuid::parse_str(NOTEBOOK_ID)?)
        .with_query_rewriter(Arc::new(YokohamaSynonymRewriter))
        .build()?;
    println!("{:<28} {:<10} {:<10} {:<10} {:<10}", "variant", "recall", "hit", "ndcg", "mrr");
    for k in [1usize, 3, 5, 10] {
        let cap_only = run_queries(&ellisii, &gold, 0.5, true, k).await?;
        let mut mq_pairs: Vec<(Vec<String>, Vec<String>)> = Vec::with_capacity(gold.len());
        for item in &gold {
            let hits = ellisii_mq
                .search(
                    &item.query,
                    SearchOptions {
                        top_k: k,
                        semantic_weight: 0.5,
                        caption_rerank: true,
                        multi_query_max_variants: 3,
                        multi_query_variant_weight: 0.7,
                        ..Default::default()
                    },
                )
                .await?;
            let pred: Vec<String> = hits.iter().map(|h| h.chunk.id.to_string()).collect();
            mq_pairs.push((pred, item.relevant.clone()));
        }
        let s_cap = summarize(&cap_only, k);
        let s_mq = summarize(&mq_pairs, k);
        println!(
            "{:<28} {:<10.3} {:<10.3} {:<10.3} {:<10.3}",
            format!("cap only (k={})", k),
            s_cap.recall_at_k, s_cap.hit_at_k, s_cap.ndcg_at_k, s_cap.mrr
        );
        println!(
            "{:<28} {:<10.3} {:<10.3} {:<10.3} {:<10.3}",
            format!("cap+mq   (k={})", k),
            s_mq.recall_at_k, s_mq.hit_at_k, s_mq.ndcg_at_k, s_mq.mrr
        );
    }

    // 4) auto_adjust_weight A/B (caption rerank=on, weight base=0.5)
    println!("\n=== auto_adjust_weight A/B (caption rerank=on, base=0.5) ===");
    println!("{:<28} {:<10} {:<10} {:<10} {:<10}", "variant", "recall", "hit", "ndcg", "mrr");
    for k in [1usize, 5, 10] {
        // baseline: auto_adjust_weight=false (現状の固定 0.5)
        let mut off_pairs: Vec<(Vec<String>, Vec<String>)> = Vec::with_capacity(gold.len());
        let mut on_pairs: Vec<(Vec<String>, Vec<String>)> = Vec::with_capacity(gold.len());
        for item in &gold {
            let off = ellisii
                .search(
                    &item.query,
                    SearchOptions {
                        top_k: k,
                        semantic_weight: 0.5,
                        caption_rerank: true,
                        auto_adjust_weight: false,
                        ..Default::default()
                    },
                )
                .await?;
            let on = ellisii
                .search(
                    &item.query,
                    SearchOptions {
                        top_k: k,
                        semantic_weight: 0.5,
                        caption_rerank: true,
                        auto_adjust_weight: true,
                        ..Default::default()
                    },
                )
                .await?;
            off_pairs.push((off.iter().map(|h| h.chunk.id.to_string()).collect(), item.relevant.clone()));
            on_pairs.push((on.iter().map(|h| h.chunk.id.to_string()).collect(), item.relevant.clone()));
        }
        let s_off = summarize(&off_pairs, k);
        let s_on = summarize(&on_pairs, k);
        println!("{:<28} {:<10.3} {:<10.3} {:<10.3} {:<10.3}", format!("auto=off (k={})", k), s_off.recall_at_k, s_off.hit_at_k, s_off.ndcg_at_k, s_off.mrr);
        println!("{:<28} {:<10.3} {:<10.3} {:<10.3} {:<10.3}", format!("auto=on  (k={})", k), s_on.recall_at_k, s_on.hit_at_k, s_on.ndcg_at_k, s_on.mrr);
    }

    // 5) per-query (caption_rerank=on, k=5)
    println!("\n=== Per-query (caption_rerank=on, k=5) ===");
    let pairs = run_queries(&ellisii, &gold, 0.5, true, 5).await?;
    for (i, ((pred, rel), item)) in pairs.iter().zip(gold.iter()).enumerate() {
        let rr = ellisii_rag::eval::reciprocal_rank(pred, rel);
        let hit = ellisii_rag::eval::hit_at_k(pred, rel, 5);
        let rank = pred.iter().position(|p| rel.contains(p)).map(|x| x + 1);
        println!(
            "  [{:>2}] hit@5={:>3.0}% rr={:.2} rank={:?}  «{}»",
            i + 1,
            hit * 100.0,
            rr,
            rank,
            item.query,
        );
    }
    Ok(())
}

fn print_row(w: f32, k: usize, s: &EvalSummary) {
    println!(
        "{:<8.2} {:<6} {:<10.3} {:<10.3} {:<10.3} {:<10.3}",
        w, k, s.recall_at_k, s.hit_at_k, s.ndcg_at_k, s.mrr
    );
}

#[cfg(feature = "static-jp")]
async fn run_queries(
    ellisii: &Ellisii,
    gold: &[GoldenItem],
    semantic_weight: f32,
    caption_rerank: bool,
    top_k: usize,
) -> ellisii_core::Result<Vec<(Vec<String>, Vec<String>)>> {
    let mut out = Vec::with_capacity(gold.len());
    for item in gold {
        let hits = ellisii
            .search(
                &item.query,
                SearchOptions {
                    top_k,
                    semantic_weight,
                    caption_rerank,
                    ..Default::default()
                },
            )
            .await?;
        let pred: Vec<String> = hits.iter().map(|h| h.chunk.id.to_string()).collect();
        out.push((pred, item.relevant.clone()));
    }
    Ok(out)
}
