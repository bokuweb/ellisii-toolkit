//! `SearchOptions::max_chunks_per_source` (MMR-lite) の配線テスト。
//!
//! 動機: production の chunker は 1 source (= 1 取り込み doc) を複数 chunk に分割する。
//! 同一 source の似た chunk が top-K を埋めると、別 source の重要 chunk が押し出される。
//! `max_chunks_per_source` で「同一 source は最大 N 件まで」と制限することで上位の
//! source 多様性を確保する。fixture corpora は 1 doc = 1 source の構造のため
//! `eval_fixtures_regression` では metrics 上 no-op になるが、本テストは
//! 「複数 chunk が同一 source を持つ」状態を意図的に作って end-to-end 配線を検証する。

use async_trait::async_trait;
use ellisii_core::{Chunk, Result};
use ellisii_embed_core::Embedder;
use ellisii_sdk::{Ellisii, SearchOptions};
use ellisii_store_core::VectorStore as _;
use std::sync::Arc;
use uuid::Uuid;

/// 全 chunk に同じベクトルを返す embedder。retrieval バイアスを潰し、
/// 「dedup の上限ロジックがそのまま見える」状況を作る。
struct FixedEmbedder;

#[async_trait]
impl Embedder for FixedEmbedder {
    fn dim(&self) -> usize {
        4
    }
    async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        Ok(texts.iter().map(|_| vec![1.0, 0.0, 0.0, 0.0]).collect())
    }
}

async fn setup(nb: Uuid, source_a: Uuid, source_b: Uuid) -> Arc<Ellisii> {
    let ellisii = Arc::new(
        Ellisii::builder()
            .with_embedder(Arc::new(FixedEmbedder))
            .with_store_memory()
            .with_notebook_id(nb)
            .build()
            .unwrap(),
    );
    // source_a から 3 chunk、source_b から 3 chunk。本文は noise filter (>=25 chars) を通す。
    let mk = |sid: Uuid, ord: u32, tag: &str| Chunk {
        id: Uuid::new_v4(),
        source_id: sid,
        ord,
        text: format!(
            "{tag} 検索クエリにマッチする本文を十分な長さで含めて noise filter を確実に通す。"
        ),
        heading_path: vec![],
        page: None,
        bbox: None,
        summary: None,
    };
    let chunks = vec![
        mk(source_a, 0, "A1"),
        mk(source_a, 1, "A2"),
        mk(source_a, 2, "A3"),
        mk(source_b, 0, "B1"),
        mk(source_b, 1, "B2"),
        mk(source_b, 2, "B3"),
    ];
    let texts: Vec<String> = chunks.iter().map(|c| c.text.clone()).collect();
    let embs = ellisii.embedder().embed(&texts).await.unwrap();
    ellisii.store().upsert(nb, &chunks, &embs).await.unwrap();
    ellisii
}

#[tokio::test]
async fn max_chunks_per_source_zero_is_passthrough() {
    let nb = Uuid::new_v4();
    let sa = Uuid::new_v4();
    let sb = Uuid::new_v4();
    let ellisii = setup(nb, sa, sb).await;

    let hits = ellisii
        .search(
            "検索クエリ",
            SearchOptions {
                top_k: 10,
                caption_rerank: false,
                max_chunks_per_source: 0,
                ..Default::default()
            },
        )
        .await
        .unwrap();
    assert_eq!(hits.len(), 6, "no dedup: all 6 chunks returned");
}

#[tokio::test]
async fn max_chunks_per_source_one_keeps_one_per_source() {
    let nb = Uuid::new_v4();
    let sa = Uuid::new_v4();
    let sb = Uuid::new_v4();
    let ellisii = setup(nb, sa, sb).await;

    let hits = ellisii
        .search(
            "検索クエリ",
            SearchOptions {
                top_k: 10,
                caption_rerank: false,
                max_chunks_per_source: 1,
                ..Default::default()
            },
        )
        .await
        .unwrap();
    assert_eq!(hits.len(), 2, "dedup=1: at most 1 per source → 2 sources × 1 = 2");
    // 2 つの source がそれぞれちょうど 1 件ずつ
    let mut counts: std::collections::HashMap<Uuid, usize> = std::collections::HashMap::new();
    for h in &hits {
        *counts.entry(h.chunk.source_id).or_insert(0) += 1;
    }
    assert_eq!(counts.get(&sa).copied().unwrap_or(0), 1);
    assert_eq!(counts.get(&sb).copied().unwrap_or(0), 1);
}

#[tokio::test]
async fn max_chunks_per_source_two_keeps_two_per_source() {
    let nb = Uuid::new_v4();
    let sa = Uuid::new_v4();
    let sb = Uuid::new_v4();
    let ellisii = setup(nb, sa, sb).await;

    let hits = ellisii
        .search(
            "検索クエリ",
            SearchOptions {
                top_k: 10,
                caption_rerank: false,
                max_chunks_per_source: 2,
                ..Default::default()
            },
        )
        .await
        .unwrap();
    assert_eq!(hits.len(), 4, "dedup=2: 2 sources × 2 = 4");
}

#[tokio::test]
async fn max_chunks_per_source_above_population_is_no_op() {
    let nb = Uuid::new_v4();
    let sa = Uuid::new_v4();
    let sb = Uuid::new_v4();
    let ellisii = setup(nb, sa, sb).await;

    let hits = ellisii
        .search(
            "検索クエリ",
            SearchOptions {
                top_k: 10,
                caption_rerank: false,
                max_chunks_per_source: 100,
                ..Default::default()
            },
        )
        .await
        .unwrap();
    assert_eq!(hits.len(), 6, "limit far above population is no-op");
}
