//! ベースライン (rewriter なし) を k=3 / k=5 / k=10 で計測する。
//!
//! 目的: real_multi_query_e4b で recall@10=1.0 (天井) が出てしまい、
//! 改善の余地を測れない問題を切り分ける。k を狭めれば天井が下がるので、
//! 実際にどこまで余地があるかを確認する。
//!
//! `#[ignore]`。`cargo test -p ellisii-rag-eval-cli --test baseline_low_k -- --ignored --nocapture`

use ellisii_rag::{eval::GoldenSet, MultiQueryOptions};
use ellisii_rag_eval_cli::{
    run_eval_with_options, Backend, CorpusEntry, EmbedderKind, EvalOptions,
};
use std::path::PathBuf;

fn locate_static_jp() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("ELLISII_STATIC_JP_DIR") {
        let pb = PathBuf::from(p);
        if pb.is_dir() {
            return Some(pb);
        }
    }
    if let Some(home) = std::env::var_os("HOME") {
        let mac = PathBuf::from(&home)
            .join("Library/Application Support/ellisii/models/static-embedding-japanese");
        if mac.is_dir() {
            return Some(mac);
        }
    }
    None
}

fn load_fixture(domain: &str) -> (Vec<CorpusEntry>, GoldenSet) {
    let base = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .join("rag/tests/fixtures/eval")
        .join(domain);
    let corpus: Vec<CorpusEntry> =
        serde_json::from_str(&std::fs::read_to_string(base.join("corpus.json")).unwrap()).unwrap();
    let golden: GoldenSet =
        serde_json::from_str(&std::fs::read_to_string(base.join("golden.json")).unwrap()).unwrap();
    (corpus, golden)
}

async fn measure_at_k(domain: &str, embedder_dir: &PathBuf) {
    let (corpus, golden) = load_fixture(domain);
    let queries = golden.items.len();
    println!("\n=== {domain} (semantic=0.75, queries={queries}) ===");
    println!("  k    recall  hit    nDCG   MRR     headroom");
    for k in [3usize, 5, 10] {
        let embedder = EmbedderKind::StaticJp {
            model_dir: embedder_dir.clone(),
        }
        .build()
        .expect("load static-jp");
        let opts = EvalOptions {
            backend: Backend::Sqlite,
            embedder,
            weights: vec![0.75],
            k,
            judge: None,
            rewriter: None,
            multi: MultiQueryOptions::default(),
        };
        let rows = run_eval_with_options(&opts, &corpus, &golden)
            .await
            .unwrap();
        let s = &rows[0].summary;
        let headroom = if s.recall_at_k >= 0.999 {
            "saturated"
        } else {
            "ROOM"
        };
        println!(
            "  {:<4} {:.3}   {:.3}  {:.3}  {:.3}   {}",
            k, s.recall_at_k, s.hit_at_k, s.ndcg_at_k, s.mrr, headroom
        );
    }
}

#[tokio::test]
#[ignore]
async fn civil_law_low_k() {
    let static_jp = locate_static_jp().expect("static-jp model not present");
    measure_at_k("jp-civil-law", &static_jp).await;
}

#[tokio::test]
#[ignore]
async fn cs_wiki_low_k() {
    let static_jp = locate_static_jp().expect("static-jp model not present");
    measure_at_k("jp-cs-wiki", &static_jp).await;
}

#[tokio::test]
#[ignore]
async fn civil_law_hard_low_k() {
    let static_jp = locate_static_jp().expect("static-jp model not present");
    measure_at_k("jp-civil-law-hard", &static_jp).await;
}

#[tokio::test]
#[ignore]
async fn sql_antipatterns_low_k() {
    let static_jp = locate_static_jp().expect("static-jp model not present");
    measure_at_k("sql-antipatterns", &static_jp).await;
}

#[tokio::test]
#[ignore]
async fn jp_patents_low_k() {
    let static_jp = locate_static_jp().expect("static-jp model not present");
    measure_at_k("jp-patents", &static_jp).await;
}

#[tokio::test]
#[ignore]
async fn cs_wiki_hard_low_k() {
    let static_jp = locate_static_jp().expect("static-jp model not present");
    measure_at_k("jp-cs-wiki-hard", &static_jp).await;
}
