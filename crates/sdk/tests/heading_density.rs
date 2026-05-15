//! `Ellisii::heading_density()` の挙動検証。InMemoryStore + 手作り chunk で
//! 「heading_path に **日本語タイトル相当** が入っているか」の比率が
//! 期待通り返ることを確認する。

use async_trait::async_trait;
use ellisii_core::{Chunk, Result};
use ellisii_embed_core::Embedder;
use ellisii_sdk::Ellisii;
use ellisii_store_core::VectorStore as _;
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

fn chunk_with_heading(heading: Vec<&str>) -> Chunk {
    Chunk {
        id: Uuid::new_v4(),
        source_id: Uuid::nil(),
        ord: 0,
        text: "本文 noise filter を通すのに十分な長さの日本語テキストをここに入れて 25 文字以上にする。".to_string(),
        heading_path: heading.into_iter().map(String::from).collect(),
        page: None,
        bbox: None,
        summary: None,
    }
}

async fn build(nb: Uuid) -> Ellisii {
    Ellisii::builder()
        .with_embedder(Arc::new(ConstEmbedder))
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

#[tokio::test]
async fn empty_notebook_returns_zero() {
    let nb = Uuid::new_v4();
    let ellisii = build(nb).await;
    assert!(ellisii.heading_density().await.unwrap() < 1e-6);
}

#[tokio::test]
async fn ascii_only_headings_are_zero() {
    // doc-id 様の ASCII heading は分子に入らない。
    let nb = Uuid::new_v4();
    let ellisii = build(nb).await;
    let cs = vec![
        chunk_with_heading(vec!["minpou-94"]),
        chunk_with_heading(vec!["wiki-foo"]),
        chunk_with_heading(vec!["doc-12345"]),
    ];
    upsert(&ellisii, nb, cs).await;
    let d = ellisii.heading_density().await.unwrap();
    assert!(d < 1e-6, "ASCII-only headings should give 0, got {d}");
}

#[tokio::test]
async fn rich_japanese_headings_are_one() {
    // 8 文字以上 + 非 ASCII を含む heading は全て分子に入る (Run 62 で 4 → 8 に refine)。
    let nb = Uuid::new_v4();
    let ellisii = build(nb).await;
    let cs = vec![
        chunk_with_heading(vec!["1.4 ロードバランサ"]), // 11 chars
        chunk_with_heading(vec!["第三条 私権の享有"]),  // 9 chars
        chunk_with_heading(vec!["（入湯税の税率について）"]), // 12 chars
    ];
    upsert(&ellisii, nb, cs).await;
    let d = ellisii.heading_density().await.unwrap();
    assert!((d - 1.0).abs() < 1e-6, "expected 1.0, got {d}");
}

#[tokio::test]
async fn mixed_corpus_returns_proportion() {
    // 4 chunks: 2 がリッチ (>=8 chars + 非 ASCII)、2 が短いまたは ASCII のみ → density = 0.5
    let nb = Uuid::new_v4();
    let ellisii = build(nb).await;
    let cs = vec![
        chunk_with_heading(vec!["1.4 ロードバランサ"]), // 11 chars → pass
        chunk_with_heading(vec!["第三条 私権の享有"]),  // 9 chars → pass
        chunk_with_heading(vec!["minpou-94"]),          // ASCII → fail
        chunk_with_heading(vec!["第一条目"]),           // 4 chars (< 8) → fail
    ];
    upsert(&ellisii, nb, cs).await;
    let d = ellisii.heading_density().await.unwrap();
    assert!((d - 0.5).abs() < 1e-6, "expected 0.5, got {d}");
}

#[tokio::test]
async fn short_jp_headings_below_threshold() {
    // 8 文字未満は分子に入らない (Run 62)。短い topic-name (e.g. "ACID") は除外。
    let nb = Uuid::new_v4();
    let ellisii = build(nb).await;
    let cs = vec![
        chunk_with_heading(vec!["あ"]),                // 1 char
        chunk_with_heading(vec!["いろは"]),            // 3 chars
        chunk_with_heading(vec!["第一条目"]),          // 4 chars
        chunk_with_heading(vec!["第三条 私権の享有"]), // 9 chars → 通る
    ];
    upsert(&ellisii, nb, cs).await;
    let d = ellisii.heading_density().await.unwrap();
    // 4 chunks 中 1 つが条件を満たす → 0.25
    assert!((d - 0.25).abs() < 1e-3, "expected 0.25, got {d}");
}

/// Run 62 の確認: jp-cs-wiki-hard 系の短い topic-name title はリッチ判定しない。
#[tokio::test]
async fn topic_name_titles_are_filtered_out() {
    let nb = Uuid::new_v4();
    let ellisii = build(nb).await;
    // 全部 8 文字未満 (旧閾値 4 では 1.0、新閾値 8 では 0 を期待)
    let cs = vec![
        chunk_with_heading(vec!["ACID"]), // 4 chars + ASCII → fail (ASCII)
        chunk_with_heading(vec!["TLS"]),  // 3 + ASCII → fail
        chunk_with_heading(vec!["B木"]),  // 2 chars → fail
        chunk_with_heading(vec!["セマフォ"]), // 4 chars → fail (< 8)
        chunk_with_heading(vec!["デッドロック"]), // 6 chars → fail (< 8)
        chunk_with_heading(vec!["関係の正規化"]), // 6 chars → fail (< 8)
    ];
    upsert(&ellisii, nb, cs).await;
    let d = ellisii.heading_density().await.unwrap();
    assert!(d < 1e-6, "expected 0, got {d}");
}

#[tokio::test]
async fn empty_heading_path_is_excluded() {
    // heading_path が空の chunk は all_headings に出てこない (store の実装) ため
    // 分子にも入らない。分母 (chunk 数) には入るので密度は下がる。
    let nb = Uuid::new_v4();
    let ellisii = build(nb).await;
    let cs = vec![
        chunk_with_heading(vec!["第三条 私権の享有"]),
        chunk_with_heading(vec![]), // empty heading_path
    ];
    upsert(&ellisii, nb, cs).await;
    let d = ellisii.heading_density().await.unwrap();
    assert!((d - 0.5).abs() < 1e-6, "expected 0.5, got {d}");
}
