//! `embed-static-jp` を実際にロードして 民法 / CS golden を eval する。
//!
//! デフォルトで `#[ignore]`。ローカルにモデルが配置されている場合のみ
//! `cargo test -p ellisii-rag-eval-cli -- --ignored` で実行する。
//!
//! モデル探索順:
//! 1. `ELLISII_STATIC_JP_DIR` 環境変数
//! 2. `~/Library/Application Support/ellisii/models/static-embedding-japanese/` (macOS)
//! 3. `$XDG_DATA_HOME/ellisii/models/static-embedding-japanese/` (Linux)

use ellisii_rag::eval::GoldenSet;
use ellisii_rag_eval_cli::{
    run_eval_with_options, Backend, CorpusEntry, EmbedderKind, EvalOptions,
};
use std::path::PathBuf;

fn locate_model_dir() -> Option<PathBuf> {
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
    if let Some(xdg) = std::env::var_os("XDG_DATA_HOME") {
        let lin = PathBuf::from(&xdg).join("ellisii/models/static-embedding-japanese");
        if lin.is_dir() {
            return Some(lin);
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

#[tokio::test]
#[ignore]
async fn static_jp_vector_only_civil_law() {
    let model_dir = locate_model_dir().expect("static-jp model not present");
    let kind = EmbedderKind::StaticJp { model_dir };
    let embedder = kind.build().expect("load static-jp");
    println!("static-jp dim = {}", embedder.dim());

    let (corpus, golden) = load_fixture("jp-civil-law");
    let opts = EvalOptions {
        backend: Backend::Sqlite,
        embedder,
        weights: vec![0.0, 0.25, 0.5, 0.75, 1.0],
        k: 10,
        judge: None,
        rewriter: None,
        multi: ellisii_rag::MultiQueryOptions::default(),
    };
    let rows = run_eval_with_options(&opts, &corpus, &golden)
        .await
        .unwrap();
    println!("\n=== static-jp on jp-civil-law (sqlite, k=10) ===");
    for r in &rows {
        println!(
            "  semantic={:.2}  recall={:.3}  hit={:.3}  nDCG={:.3}  MRR={:.3}",
            r.semantic,
            r.summary.recall_at_k,
            r.summary.hit_at_k,
            r.summary.ndcg_at_k,
            r.summary.mrr
        );
    }
    // 真 embedder では vector-only でも recall@10 ≥ 0.7 を期待。
    let vec_only = &rows
        .iter()
        .find(|r| (r.semantic - 1.0).abs() < 1e-6)
        .unwrap()
        .summary;
    assert!(
        vec_only.recall_at_k >= 0.7,
        "static-jp vector recall@10 too low: {}",
        vec_only.recall_at_k
    );
}

#[tokio::test]
#[ignore]
async fn static_jp_vector_only_cs_wiki() {
    let model_dir = locate_model_dir().expect("static-jp model not present");
    let embedder = EmbedderKind::StaticJp { model_dir }.build().unwrap();

    let (corpus, golden) = load_fixture("jp-cs-wiki");
    let opts = EvalOptions {
        backend: Backend::Sqlite,
        embedder,
        weights: vec![0.0, 0.25, 0.5, 0.75, 1.0],
        k: 10,
        judge: None,
        rewriter: None,
        multi: ellisii_rag::MultiQueryOptions::default(),
    };
    let rows = run_eval_with_options(&opts, &corpus, &golden)
        .await
        .unwrap();
    println!("\n=== static-jp on jp-cs-wiki (sqlite, k=10) ===");
    for r in &rows {
        println!(
            "  semantic={:.2}  recall={:.3}  hit={:.3}  nDCG={:.3}  MRR={:.3}",
            r.semantic,
            r.summary.recall_at_k,
            r.summary.hit_at_k,
            r.summary.ndcg_at_k,
            r.summary.mrr
        );
    }
}
