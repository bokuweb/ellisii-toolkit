//! `Ellisii::caption_density()` の挙動検証。InMemoryStore + 手作り chunk で
//! `(captioned chunks) / (all chunks)` の比率が期待通り返ることを確認する。

use async_trait::async_trait;
use ellisii_core::{Chunk, Result};
use ellisii_embed_core::Embedder;
use ellisii_sdk::Ellisii;
use ellisii_store_core::VectorStore;
use std::sync::Arc;
use uuid::Uuid;

struct ConstEmbedder;
#[async_trait]
impl Embedder for ConstEmbedder {
    fn dim(&self) -> usize {
        4
    }
    async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        Ok(texts.iter().map(|_| vec![1.0, 0.0, 0.0, 0.0]).collect())
    }
}

fn chunk(text: &str) -> Chunk {
    Chunk {
        id: Uuid::new_v4(),
        source_id: Uuid::nil(),
        ord: 0,
        text: text.to_string(),
        heading_path: vec![],
        page: None,
        bbox: None,
        summary: None,
    }
}

#[tokio::test]
async fn empty_notebook_returns_zero() {
    let nb = Uuid::new_v4();
    let ellisii = Ellisii::builder()
        .with_embedder(Arc::new(ConstEmbedder))
        .with_store_memory()
        .with_notebook_id(nb)
        .build()
        .unwrap();
    assert!(ellisii.caption_density().await.unwrap() < 1e-6);
}

#[tokio::test]
async fn fully_captioned_corpus_is_one() {
    let nb = Uuid::new_v4();
    let ellisii = Ellisii::builder()
        .with_embedder(Arc::new(ConstEmbedder))
        .with_store_memory()
        .with_notebook_id(nb)
        .build()
        .unwrap();
    let cs = vec![
        chunk("(入湯税の税率)\n第123条 入湯税は…"),
        chunk("(都市計画税の税率)\n第132条 都市計画税は…"),
        chunk("(事業所税の税率)\n第129条の5 事業所税は…"),
    ];
    let texts: Vec<String> = cs.iter().map(|c| c.text.clone()).collect();
    let embs = ellisii.embedder().embed(&texts).await.unwrap();
    ellisii.store().upsert(nb, &cs, &embs).await.unwrap();
    let d = ellisii.caption_density().await.unwrap();
    assert!((d - 1.0).abs() < 1e-6, "expected 1.0, got {d}");
}

#[tokio::test]
async fn mixed_corpus_returns_proportion() {
    let nb = Uuid::new_v4();
    let ellisii = Ellisii::builder()
        .with_embedder(Arc::new(ConstEmbedder))
        .with_store_memory()
        .with_notebook_id(nb)
        .build()
        .unwrap();
    // 4 chunks total, 2 with leading caption
    let cs = vec![
        chunk("(入湯税の税率)\n第123条 入湯税は…"),
        chunk("(都市計画税の税率)\n第132条 都市計画税は…"),
        chunk("第3条 市税として課する普通税は…"),
        chunk("付則 この条例は…"),
    ];
    let texts: Vec<String> = cs.iter().map(|c| c.text.clone()).collect();
    let embs = ellisii.embedder().embed(&texts).await.unwrap();
    ellisii.store().upsert(nb, &cs, &embs).await.unwrap();
    let d = ellisii.caption_density().await.unwrap();
    assert!((d - 0.5).abs() < 1e-6, "expected 0.5, got {d}");
}

#[tokio::test]
async fn density_is_per_notebook() {
    // 2 つの notebook を 1 つの ellisii instance で扱うのは実用ケースではないので、
    // 別 instance に分けて、それぞれの notebook で正しく density が出ることを確認する。
    let nb_caption = Uuid::new_v4();
    let nb_no_caption = Uuid::new_v4();

    let e1 = Ellisii::builder()
        .with_embedder(Arc::new(ConstEmbedder))
        .with_store_memory()
        .with_notebook_id(nb_caption)
        .build()
        .unwrap();
    let cs1 = vec![chunk("(あ)\n本文"), chunk("(い)\n本文")];
    let texts1: Vec<String> = cs1.iter().map(|c| c.text.clone()).collect();
    let embs1 = e1.embedder().embed(&texts1).await.unwrap();
    e1.store().upsert(nb_caption, &cs1, &embs1).await.unwrap();
    assert!((e1.caption_density().await.unwrap() - 1.0).abs() < 1e-6);

    let e2 = Ellisii::builder()
        .with_embedder(Arc::new(ConstEmbedder))
        .with_store_memory()
        .with_notebook_id(nb_no_caption)
        .build()
        .unwrap();
    let cs2 = vec![chunk("第1条 本文"), chunk("第2条 本文")];
    let texts2: Vec<String> = cs2.iter().map(|c| c.text.clone()).collect();
    let embs2 = e2.embedder().embed(&texts2).await.unwrap();
    e2.store().upsert(nb_no_caption, &cs2, &embs2).await.unwrap();
    assert!(e2.caption_density().await.unwrap() < 1e-6);
}
