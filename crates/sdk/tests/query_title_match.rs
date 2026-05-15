//! `Ellisii::query_title_match()` の挙動検証。Run 26 で導入した
//! caption_synthesis ROI signal の SDK 経由 API。

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

fn chunk_with_heading(text: &str, heading: &[&str]) -> Chunk {
    Chunk {
        id: Uuid::new_v4(),
        source_id: Uuid::nil(),
        ord: 0,
        text: text.to_string(),
        heading_path: heading.iter().map(|s| s.to_string()).collect(),
        page: None,
        bbox: None,
        summary: None,
    }
}

#[tokio::test]
async fn query_title_match_zero_for_empty_inputs() {
    let nb = Uuid::new_v4();
    let e = Ellisii::builder()
        .with_embedder(Arc::new(ConstEmbedder))
        .with_store_memory()
        .with_notebook_id(nb)
        .build()
        .unwrap();
    // empty queries
    let qs: Vec<&str> = vec![];
    assert_eq!(e.query_title_match(&qs).await.unwrap(), 0.0);

    // empty corpus
    let qs2 = vec!["query"];
    assert_eq!(e.query_title_match(&qs2).await.unwrap(), 0.0);
}

#[tokio::test]
async fn query_title_match_high_when_query_directly_matches_title() {
    let nb = Uuid::new_v4();
    let e = Ellisii::builder()
        .with_embedder(Arc::new(ConstEmbedder))
        .with_store_memory()
        .with_notebook_id(nb)
        .build()
        .unwrap();
    let cs = vec![
        chunk_with_heading(
            "ACID 本文 1。文字数を稼ぐためにダミー文を加える。",
            &["wiki", "ACID"],
        ),
        chunk_with_heading(
            "B 木 本文 1。これも noise filter を通すために本文を入れる。",
            &["wiki", "B木"],
        ),
        chunk_with_heading(
            "DNS の解説本文。同じく十分な長さを確保しておく。",
            &["wiki", "Domain Name System"],
        ),
    ];
    let texts: Vec<String> = cs.iter().map(|c| c.text.clone()).collect();
    let embs = e.embedder().embed(&texts).await.unwrap();
    e.store().upsert(nb, &cs, &embs).await.unwrap();

    let qs = vec!["ACID とは何か", "B木の特性"];
    let r = e.query_title_match(&qs).await.unwrap();
    assert!(r > 0.3, "expected high match, got {r}");
}

#[tokio::test]
async fn query_title_match_low_for_paraphrase_queries() {
    let nb = Uuid::new_v4();
    let e = Ellisii::builder()
        .with_embedder(Arc::new(ConstEmbedder))
        .with_store_memory()
        .with_notebook_id(nb)
        .build()
        .unwrap();
    // タイトルから完全 paraphrase したクエリ — Run 25 の jp-cs-wiki-hard 風
    let cs = vec![
        chunk_with_heading(
            "ACID の解説本文をここに置く。長さを確保。",
            &["wiki", "ACID"],
        ),
        chunk_with_heading("B 木 の解説本文。同じく十分な長さで。", &["wiki", "B木"]),
    ];
    let texts: Vec<String> = cs.iter().map(|c| c.text.clone()).collect();
    let embs = e.embedder().embed(&texts).await.unwrap();
    e.store().upsert(nb, &cs, &embs).await.unwrap();

    let qs = vec![
        "トランザクション処理の信頼性を保証する性質を 4 つ挙げよ",
        "ブロック単位のランダムアクセスで利用される木構造とは",
    ];
    let r = e.query_title_match(&qs).await.unwrap();
    assert!(r < 0.3, "expected low match, got {r}");
}

#[tokio::test]
async fn query_title_match_zero_when_heading_path_empty() {
    // 全 chunk の heading_path が空 → タイトル抽出不可で 0.0
    let nb = Uuid::new_v4();
    let e = Ellisii::builder()
        .with_embedder(Arc::new(ConstEmbedder))
        .with_store_memory()
        .with_notebook_id(nb)
        .build()
        .unwrap();
    let cs = vec![Chunk {
        id: Uuid::new_v4(),
        source_id: Uuid::nil(),
        ord: 0,
        text: "(タイトル付きチャンク)\n本文をここに入れる。十分な長さで noise filter を通す。"
            .into(),
        heading_path: vec![],
        page: None,
        bbox: None,
        summary: None,
    }];
    let texts: Vec<String> = cs.iter().map(|c| c.text.clone()).collect();
    let embs = e.embedder().embed(&texts).await.unwrap();
    e.store().upsert(nb, &cs, &embs).await.unwrap();

    let qs = vec!["タイトル付き"];
    let r = e.query_title_match(&qs).await.unwrap();
    assert_eq!(r, 0.0, "no heading_path → 0.0, got {r}");
}
