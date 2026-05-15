//! `SearchOptions::auto_max_chunks_per_source` の挙動検証。
//!
//! Multi-source notebook (source_count >= 3) で auto=true なら dedup1 が自動適用、
//! 単独/2 source の小さな notebook では no-op、明示 `max_chunks_per_source` は常に優先。

use async_trait::async_trait;
use ellisii_core::{Chunk, Result};
use ellisii_embed_core::Embedder;
use ellisii_sdk::{Ellisii, SearchOptions};
use ellisii_store_core::VectorStore as _;
use std::sync::Arc;
use uuid::Uuid;

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

fn chunk(text: &str, source_id: Uuid, ord: u32) -> Chunk {
    Chunk {
        id: Uuid::new_v4(),
        source_id,
        ord,
        text: format!(
            "{text} 検索クエリ ヒット 用の本文を十分な長さで含めて noise filter を確実に通す。"
        ),
        heading_path: vec![],
        page: None,
        bbox: None,
        summary: None,
    }
}

async fn build(nb: Uuid) -> Ellisii {
    Ellisii::builder()
        .with_embedder(Arc::new(FixedEmbedder))
        .with_store_memory()
        .with_notebook_id(nb)
        .build()
        .unwrap()
}

async fn upsert(ellisii: &Ellisii, nb: Uuid, cs: Vec<Chunk>) {
    let texts: Vec<String> = cs.iter().map(|c| c.text.clone()).collect();
    let embs = ellisii.embedder().embed(&texts).await.unwrap();
    ellisii.store().upsert(nb, &cs, &embs).await.unwrap();
}

/// 3 source × 3 chunk/source = 9 chunks の multi-source notebook。
/// auto=true で dedup=1 が自動適用 → 上位 3 件が 3 source から 1 件ずつ。
#[tokio::test]
async fn auto_applies_dedup_when_source_count_exceeds_threshold() {
    let nb = Uuid::new_v4();
    let ellisii = build(nb).await;
    let (s1, s2, s3) = (Uuid::new_v4(), Uuid::new_v4(), Uuid::new_v4());
    let cs = vec![
        chunk("alpha1 検索", s1, 0),
        chunk("alpha2 検索", s1, 1),
        chunk("alpha3 検索", s1, 2),
        chunk("beta1 検索", s2, 0),
        chunk("beta2 検索", s2, 1),
        chunk("beta3 検索", s2, 2),
        chunk("gamma1 検索", s3, 0),
        chunk("gamma2 検索", s3, 1),
        chunk("gamma3 検索", s3, 2),
    ];
    upsert(&ellisii, nb, cs).await;

    assert_eq!(ellisii.source_count().await.unwrap(), 3);

    let hits = ellisii
        .search(
            "検索",
            SearchOptions {
                top_k: 3,
                caption_rerank: false,
                auto_heading_rerank: false,
                max_chunks_per_source: 0,
                auto_max_chunks_per_source: true,
                ..Default::default()
            },
        )
        .await
        .unwrap();
    assert_eq!(hits.len(), 3, "expected 3 hits");
    let mut sources: Vec<Uuid> = hits.iter().map(|h| h.chunk.source_id).collect();
    sources.sort();
    sources.dedup();
    assert_eq!(sources.len(), 3, "dedup=1 auto should yield 3 distinct sources");
}

/// 1 source notebook (source_count=1 < 3) で auto は no-op。
/// auto=true でも auto=false と完全に同じ rank / score を返す。
#[tokio::test]
async fn auto_no_op_when_source_count_below_threshold() {
    let nb = Uuid::new_v4();
    let ellisii = build(nb).await;
    let s1 = Uuid::new_v4();
    let cs = vec![
        chunk("a 検索", s1, 0),
        chunk("b 検索", s1, 1),
        chunk("c 検索", s1, 2),
    ];
    upsert(&ellisii, nb, cs).await;
    assert_eq!(ellisii.source_count().await.unwrap(), 1);

    let mk = |auto: bool| SearchOptions {
        top_k: 3,
        caption_rerank: false,
        max_chunks_per_source: 0,
        auto_max_chunks_per_source: auto,
        ..Default::default()
    };
    let hits_off = ellisii.search("検索", mk(false)).await.unwrap();
    let hits_auto = ellisii.search("検索", mk(true)).await.unwrap();
    let ids_off: Vec<Uuid> = hits_off.iter().map(|h| h.chunk.id).collect();
    let ids_auto: Vec<Uuid> = hits_auto.iter().map(|h| h.chunk.id).collect();
    assert_eq!(ids_off, ids_auto, "auto should not change ranking when source_count<3");
    for (a, b) in hits_off.iter().zip(hits_auto.iter()) {
        assert!((a.score - b.score).abs() < 1e-6);
    }
}

/// 明示的 `max_chunks_per_source > 0` は auto より常に優先される。
#[tokio::test]
async fn explicit_max_overrides_auto() {
    let nb = Uuid::new_v4();
    let ellisii = build(nb).await;
    let (s1, s2, s3) = (Uuid::new_v4(), Uuid::new_v4(), Uuid::new_v4());
    let cs = vec![
        chunk("a1 検索", s1, 0),
        chunk("a2 検索", s1, 1),
        chunk("b1 検索", s2, 0),
        chunk("b2 検索", s2, 1),
        chunk("c1 検索", s3, 0),
        chunk("c2 検索", s3, 1),
    ];
    upsert(&ellisii, nb, cs).await;
    assert_eq!(ellisii.source_count().await.unwrap(), 3);

    // 明示 max=2 → auto はトリガしても上書きされない
    let hits = ellisii
        .search(
            "検索",
            SearchOptions {
                top_k: 10,
                caption_rerank: false,
                max_chunks_per_source: 2,
                auto_max_chunks_per_source: true,
                ..Default::default()
            },
        )
        .await
        .unwrap();
    // 3 source × 上限 2 = 6 件残るはず (源 6 件すべて)
    assert_eq!(hits.len(), 6, "explicit max=2 should yield 6 chunks (2 × 3 sources)");
}

/// source_count キャッシュが新規 ingest 後に invalidate される。
#[tokio::test]
async fn source_count_cache_invalidated_on_ingest() {
    let nb = Uuid::new_v4();
    let ellisii = build(nb).await;
    let s1 = Uuid::new_v4();
    upsert(&ellisii, nb, vec![chunk("a", s1, 0)]).await;
    assert_eq!(ellisii.source_count().await.unwrap(), 1);

    // index_chunks で新 source を追加 → cache が破棄されて 2 になるはず
    let s2 = Uuid::new_v4();
    ellisii.index_chunks(vec![chunk("b", s2, 0)]).await.unwrap();
    assert_eq!(ellisii.source_count().await.unwrap(), 2);
}
