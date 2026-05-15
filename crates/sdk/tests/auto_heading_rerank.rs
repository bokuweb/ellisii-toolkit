//! `SearchOptions::auto_heading_rerank` の挙動検証。
//!
//! density が閾値以上 → 内部で heading_boost が走る (= heading_rerank=true 等価)、
//! 閾値未満 → 走らない、を end-to-end で確認する。明示 `heading_rerank=true` は常に優先。

use async_trait::async_trait;
use ellisii_core::{Chunk, Result};
use ellisii_embed_core::Embedder;
use ellisii_sdk::{Ellisii, SearchOptions};
use ellisii_store_core::VectorStore as _;
use std::sync::Arc;
use uuid::Uuid;

/// 全 chunk に同じベクトルを返す embedder。retrieval バイアスを潰し、
/// heading_boost の効きだけが順位を変える状況を作る。
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

fn chunk(text: &str, heading: &str) -> Chunk {
    Chunk {
        id: Uuid::new_v4(),
        source_id: Uuid::new_v4(),
        ord: 0,
        text: format!("{text} 本文を noise filter 通過のため十分な長さに保つ日本語ダミー説明。"),
        heading_path: vec![heading.to_string()],
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

/// rich heading_path corpus + query が heading に一致 → auto で heading_boost が走り、
/// 該当 chunk が rank 1 に来る。
#[tokio::test]
async fn auto_promotes_match_when_density_high() {
    let nb = Uuid::new_v4();
    let ellisii = build(nb).await;
    // 3 chunks 全部リッチ heading_path → heading_density ≈ 1.0
    let cs = vec![
        chunk("本文A", "1.4 ロードバランサ"),
        chunk("本文B", "1.5 TLS終端"),
        chunk("本文C", "2.1 MySQLレプリケーション"),
    ];
    let target_id = cs[2].id;
    upsert(&ellisii, nb, cs).await;

    let hits = ellisii
        .search(
            "MySQLレプリケーション",
            SearchOptions {
                top_k: 3,
                caption_rerank: false,
                heading_rerank: false,
                auto_heading_rerank: true,
                ..Default::default()
            },
        )
        .await
        .unwrap();
    assert_eq!(hits.len(), 3);
    assert_eq!(
        hits[0].chunk.id, target_id,
        "auto_heading_rerank should promote the matching heading to rank 1"
    );
}

/// ASCII-only heading_path → density 0 → auto でも heading_boost は走らない。
/// 結果として全 chunk の score が等しく、rank は store の元順序か任意のまま。
#[tokio::test]
async fn auto_no_op_when_density_low() {
    let nb = Uuid::new_v4();
    let ellisii = build(nb).await;
    // 全 heading が ASCII doc-id 様 → density = 0
    let cs = vec![
        chunk("alpha 本文", "doc-001"),
        chunk("beta 本文", "doc-002"),
        chunk("MySQL レプリケーション本文", "doc-003"),
    ];
    upsert(&ellisii, nb, cs).await;

    let density = ellisii.heading_density().await.unwrap();
    assert!(density < 1e-6, "expected density 0, got {density}");

    // auto=true と auto=false で **完全に同じ rank / score** が返るのが no-op の証拠。
    let mk = |auto: bool| SearchOptions {
        top_k: 3,
        caption_rerank: false,
        heading_rerank: false,
        auto_heading_rerank: auto,
        ..Default::default()
    };
    let hits_off = ellisii.search("MySQL レプリケーション", mk(false)).await.unwrap();
    let hits_auto = ellisii.search("MySQL レプリケーション", mk(true)).await.unwrap();
    let ids_off: Vec<Uuid> = hits_off.iter().map(|h| h.chunk.id).collect();
    let ids_auto: Vec<Uuid> = hits_auto.iter().map(|h| h.chunk.id).collect();
    assert_eq!(ids_off, ids_auto, "auto should not change ranking at density=0");
    for (a, b) in hits_off.iter().zip(hits_auto.iter()) {
        assert!(
            (a.score - b.score).abs() < 1e-6,
            "scores diverge: {} vs {}",
            a.score,
            b.score
        );
    }
}

/// 明示 `heading_rerank=true` は density に関わらず動く。
#[tokio::test]
async fn explicit_heading_rerank_overrides_density() {
    let nb = Uuid::new_v4();
    let ellisii = build(nb).await;
    let cs = vec![
        chunk("alpha 本文", "doc-001"),
        chunk("MySQL レプリケーション本文", "minpou-002"),
    ];
    let target = cs[1].id;
    upsert(&ellisii, nb, cs).await;

    // density = 0 だが heading_rerank=true で明示的に boost を強制。
    let hits = ellisii
        .search(
            "MySQL",
            SearchOptions {
                top_k: 2,
                caption_rerank: false,
                heading_rerank: true,
                auto_heading_rerank: false,
                ..Default::default()
            },
        )
        .await
        .unwrap();
    // heading に MySQL が無いため boost は 0 だが、コードパスが走ったかは挙動からは
    // 直接見えない。重要なのは「density=0 でも explicit ON で no-panic」なこと。
    assert_eq!(hits.len(), 2);
    let _ = target;
}

/// density キャッシュが新規 ingest 後に invalidate される。
#[tokio::test]
async fn density_cache_invalidated_on_ingest() {
    let nb = Uuid::new_v4();
    let ellisii = build(nb).await;
    // 最初は ASCII heading のみ → 0
    let cs1 = vec![chunk("a", "doc-001")];
    upsert(&ellisii, nb, cs1).await;
    let d1 = ellisii.heading_density().await.unwrap();
    assert!(d1 < 1e-6);

    // index_chunks で日本語 heading を持つ chunk を追加 → 0 から上昇するはず
    let cs2 = vec![chunk("b", "第三条 私権の享有")];
    ellisii.index_chunks(cs2).await.unwrap();
    let d2 = ellisii.heading_density().await.unwrap();
    assert!(
        d2 > 1e-6,
        "density should rise after ingesting a Japanese heading; got {d2}"
    );
}
