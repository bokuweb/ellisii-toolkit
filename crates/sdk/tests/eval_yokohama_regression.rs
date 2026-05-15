//! 横浜市市税条例 (web index) の golden Q&A に対する recall の **退行ガード**。
//!
//! このテストは外部 DB (`~/Library/Application Support/ellisii/ellisii.db`) と
//! `static-embedding-japanese` モデルを必要とするので、**`#[ignore]` 付き**で
//! 通常 CI からは外す (= 開発者のローカル / dedicated eval CI でだけ走らせる)。
//!
//! 実行:
//! ```sh
//! cargo test -p ellisii-sdk --features static-jp \
//!   --test eval_yokohama_regression -- --ignored --nocapture
//! ```
//!
//! 失敗条件:
//! - DB に必要な notebook が存在しない (skip 扱いで pass)
//! - `caption_rerank=true` で hit@5 が `MIN_HIT_AT_5` を下回る
//!
//! 詳細な手順 / 改善履歴は `docs/eval/recall-evals.md` を参照。

#![cfg(feature = "static-jp")]

use std::path::PathBuf;

use ellisii_rag::eval::{summarize, GoldenSet};
use ellisii_sdk::{Ellisii, SearchOptions};
use uuid::Uuid;

const NOTEBOOK_ID: &str = "95339065-df88-4ee7-82c1-e11c587250e4";
const SOURCE_ID: &str = "057975fc-61c6-4990-9f63-54e8679b963c";
/// `eval_yokohama` の caption rerank=on ハーネスでの実測下限を踏まえた閾値。
/// golden を増やしたら底上げを検討する。
const MIN_HIT_AT_5: f32 = 0.80;
const MIN_MRR: f32 = 0.65;

fn home() -> PathBuf {
    PathBuf::from(std::env::var("HOME").expect("HOME"))
}
fn db_path() -> PathBuf {
    home().join("Library/Application Support/ellisii/ellisii.db")
}
fn model_dir() -> PathBuf {
    home().join("Library/Application Support/ellisii/models/static-embedding-japanese")
}

fn load_golden() -> GoldenSet {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/eval/yokohama/golden.json");
    let raw = std::fs::read_to_string(&path).expect("read golden fixture");
    GoldenSet::from_json_str(&raw).expect("parse golden fixture")
}

#[tokio::test(flavor = "current_thread")]
#[ignore]
async fn yokohama_recall_does_not_regress() {
    let db = db_path();
    let model = model_dir();
    if !db.exists() || !model.exists() {
        eprintln!(
            "[skip] yokohama eval: missing fixture\n  db:    {}\n  model: {}",
            db.display(),
            model.display()
        );
        return;
    }

    let dim = 1024;
    let ellisii = match Ellisii::builder()
        .with_embedder_static_jp(&model)
        .and_then(|b| b.with_store_sqlite(&db, dim))
        .and_then(|b| {
            Ok(b.with_notebook_id(Uuid::parse_str(NOTEBOOK_ID).expect("notebook uuid")))
        })
        .and_then(|b| b.build())
    {
        Ok(e) => e,
        Err(e) => {
            eprintln!("[skip] yokohama eval: build failed: {e}");
            return;
        }
    };

    let source_id = Uuid::parse_str(SOURCE_ID).unwrap();
    let n_chunks = ellisii.store().count_chunks(source_id).await.unwrap_or(0);
    if n_chunks == 0 {
        eprintln!("[skip] yokohama eval: source {SOURCE_ID} not present in DB");
        return;
    }
    eprintln!("yokohama source has {n_chunks} chunks");

    let g = load_golden();
    eprintln!("golden: {} ({} items)", g.name, g.items.len());

    let mut pairs: Vec<(Vec<String>, Vec<String>)> = Vec::with_capacity(g.items.len());
    for item in &g.items {
        let hits = ellisii
            .search(
                &item.query,
                SearchOptions {
                    top_k: 5,
                    semantic_weight: 0.5,
                    caption_rerank: true,
                    ..Default::default()
                },
            )
            .await
            .expect("search");
        let pred: Vec<String> = hits.iter().map(|h| h.chunk.id.to_string()).collect();
        pairs.push((pred, item.relevant.clone()));
    }
    let s = summarize(&pairs, 5);
    eprintln!(
        "yokohama eval (k=5, caption_rerank=on): hit={:.3} mrr={:.3} ndcg={:.3} recall={:.3}",
        s.hit_at_k, s.mrr, s.ndcg_at_k, s.recall_at_k
    );

    assert!(
        s.hit_at_k >= MIN_HIT_AT_5,
        "hit@5 regressed: {:.3} < {:.3}",
        s.hit_at_k,
        MIN_HIT_AT_5
    );
    assert!(
        s.mrr >= MIN_MRR,
        "MRR regressed: {:.3} < {:.3}",
        s.mrr,
        MIN_MRR
    );
}
