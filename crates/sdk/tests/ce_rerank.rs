//! `Ellisii::search` の CE rerank 配線テスト。Provence ONNX を直接使わず、
//! 小さな手書き `ContextCompressor` モックでスコアブレンドの挙動を検証する。

use async_trait::async_trait;
use ellisii_core::{Chunk, Result, SearchHit};
use ellisii_embed_core::Embedder;
use ellisii_provence_core::{CompressedContext, ContextCompressor};
use ellisii_query_rewriter_core::{QueryRewriter, RewrittenQueries};
use ellisii_sdk::{Ellisii, SearchOptions};
use ellisii_store_core::VectorStore;
use std::sync::Arc;
use uuid::Uuid;

/// 任意の variant を吐く scripted rewriter。`skip_ce_when_rewriting` の挙動検証用。
struct ScriptedRewriter {
    variant: String,
}

#[async_trait]
impl QueryRewriter for ScriptedRewriter {
    async fn rewrite(&self, query: &str, _max: usize) -> Result<RewrittenQueries> {
        Ok(RewrittenQueries {
            original: query.to_string(),
            variants: vec![self.variant.clone()],
        })
    }
}

/// 全テキストに同じベクトルを返す embedder (テスト用)。vector 検索では cos 類似度が
/// 全 chunk で同じになり、結果として全件 pool に入る。CE rerank の挙動だけを見たい
/// テストでは、retrieval バイアスを潰すために常に均一スコアにする方が見通しが良い。
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

/// 「passage に <BOOST> が含まれていれば 1.0、そうでなければ 0.0」を返す compressor モック。
/// CE rerank が boosted passage を上位に押し上げるかを検証する。
struct BoostCompressor;

#[async_trait]
impl ContextCompressor for BoostCompressor {
    fn is_active(&self) -> bool {
        true
    }
    async fn compress(&self, _query: &str, text: &str) -> Result<CompressedContext> {
        Ok(CompressedContext {
            kept_text: text.to_string(),
            original_chars: text.chars().count(),
            kept_chars: text.chars().count(),
            sentences: Vec::new(),
        })
    }
    async fn score_passages(&self, _query: &str, passages: &[String]) -> Result<Vec<f32>> {
        Ok(passages
            .iter()
            .map(|p| if p.contains("<BOOST>") { 1.0 } else { 0.0 })
            .collect())
    }
}

#[tokio::test]
async fn ce_rerank_promotes_boosted_passage_when_enabled() {
    let nb = Uuid::new_v4();
    let ellisii = Ellisii::builder()
        .with_embedder(Arc::new(FixedEmbedder))
        .with_store_memory()
        .with_compressor(Arc::new(BoostCompressor))
        .with_notebook_id(nb)
        .build()
        .unwrap();

    // 4 つの chunk を index。<BOOST> 付きは 1 つだけ。
    let chunks = vec![
        Chunk {
            id: Uuid::new_v4(),
            source_id: Uuid::nil(),
            ord: 0,
            text: "あ普通のチャンク 1。本文として十分な長さの説明をここに入れて noise filter の最低 25 文字を確実に超える。".to_string(),
            heading_path: vec![],
            page: None,
            bbox: None,
            summary: None,
        },
        Chunk {
            id: Uuid::new_v4(),
            source_id: Uuid::nil(),
            ord: 1,
            text: "い普通のチャンク 2。同じくダミー本文を加えて 25 文字以上にしておく。これでひっかからない。".to_string(),
            heading_path: vec![],
            page: None,
            bbox: None,
            summary: None,
        },
        Chunk {
            id: Uuid::new_v4(),
            source_id: Uuid::nil(),
            ord: 2,
            text: "う boosted <BOOST> chunk と日本語ダミーをくっつけて noise filter を通す本文にしておく。".to_string(),
            heading_path: vec![],
            page: None,
            bbox: None,
            summary: None,
        },
        Chunk {
            id: Uuid::new_v4(),
            source_id: Uuid::nil(),
            ord: 3,
            text: "え普通のチャンク 4。これも noise filter を通すためにダミー文を加えて 25 文字以上の本文にする。".to_string(),
            heading_path: vec![],
            page: None,
            bbox: None,
            summary: None,
        },
    ];
    let boosted_id = chunks[2].id;
    let store = ellisii.store();
    let texts: Vec<String> = chunks.iter().map(|c| c.text.clone()).collect();
    let embedder = ellisii.embedder();
    let embs = embedder.embed(&texts).await.unwrap();
    store.upsert(nb, &chunks, &embs).await.unwrap();

    // CE rerank=off: vec 検索の素直な順位 (boosted は最上位ではない可能性高い)
    let hits_off = ellisii
        .search(
            "クエリ",
            SearchOptions {
                top_k: 4,
                semantic_weight: 0.5,
                caption_rerank: false,
                ce_rerank_top_n: 0,
                ..Default::default()
            },
        )
        .await
        .unwrap();
    assert!(!hits_off.is_empty());

    // CE rerank=on: boosted passage が rank 1 に来るはず
    let hits_on = ellisii
        .search(
            "クエリ",
            SearchOptions {
                top_k: 4,
                semantic_weight: 0.5,
                caption_rerank: false,
                ce_rerank_top_n: 4,
                ce_rerank_weight: 1.0, // pure CE
                ..Default::default()
            },
        )
        .await
        .unwrap();
    assert!(!hits_on.is_empty(), "CE rerank should not eat all results");
    assert_eq!(
        hits_on[0].chunk.id, boosted_id,
        "boosted passage should be rank 1 with pure CE rerank"
    );
}

#[tokio::test]
async fn ce_rerank_top_n_zero_is_no_op() {
    let nb = Uuid::new_v4();
    let ellisii = Ellisii::builder()
        .with_embedder(Arc::new(FixedEmbedder))
        .with_store_memory()
        .with_compressor(Arc::new(BoostCompressor))
        .with_notebook_id(nb)
        .build()
        .unwrap();
    let chunks = vec![Chunk {
        id: Uuid::new_v4(),
        source_id: Uuid::nil(),
        ord: 0,
        text: "あ <BOOST> チャンク。noise filter を通すために十分な本文をここに加えておく。".into(),
        heading_path: vec![],
        page: None,
        bbox: None,
        summary: None,
    }];
    let texts: Vec<String> = chunks.iter().map(|c| c.text.clone()).collect();
    let embs = ellisii.embedder().embed(&texts).await.unwrap();
    ellisii.store().upsert(nb, &chunks, &embs).await.unwrap();

    let hits = ellisii
        .search(
            "あ",
            SearchOptions {
                top_k: 4,
                semantic_weight: 0.5,
                caption_rerank: false,
                ce_rerank_top_n: 0,
                ..Default::default()
            },
        )
        .await
        .unwrap();
    assert_eq!(hits.len(), 1);
    // CE が無効なら score は元の RRF score (BoostCompressor は呼ばれない)
    assert!(hits[0].score > 0.0);
}

#[tokio::test]
async fn ce_rerank_without_compressor_is_no_op() {
    // compressor 未設定の場合、ce_rerank_top_n > 0 でも安全に passthrough
    let nb = Uuid::new_v4();
    let ellisii = Ellisii::builder()
        .with_embedder(Arc::new(FixedEmbedder))
        .with_store_memory()
        .with_notebook_id(nb)
        .build()
        .unwrap();
    let chunks = vec![Chunk {
        id: Uuid::new_v4(),
        source_id: Uuid::nil(),
        ord: 0,
        text:
            "あ チャンク。noise filter を通すために本文を追加する必要があるので長めに書いておく。"
                .into(),
        heading_path: vec![],
        page: None,
        bbox: None,
        summary: None,
    }];
    let texts: Vec<String> = chunks.iter().map(|c| c.text.clone()).collect();
    let embs = ellisii.embedder().embed(&texts).await.unwrap();
    ellisii.store().upsert(nb, &chunks, &embs).await.unwrap();

    let hits = ellisii
        .search(
            "クエリ",
            SearchOptions {
                top_k: 4,
                semantic_weight: 0.5,
                caption_rerank: false,
                ce_rerank_top_n: 4,
                ..Default::default()
            },
        )
        .await
        .unwrap();
    assert_eq!(hits.len(), 1);
}

/// rewriter 有効 + `skip_ce_when_rewriting=true` (既定) で CE が skip されることを確認。
/// CE が動かないので BoostCompressor の score_passages は呼ばれず、boosted passage は
/// rank 1 に来ない (元の vec/RRF 順序のまま)。
#[tokio::test]
async fn ce_skipped_when_rewriter_active_by_default() {
    let nb = Uuid::new_v4();
    let ellisii = Ellisii::builder()
        .with_embedder(Arc::new(FixedEmbedder))
        .with_store_memory()
        .with_compressor(Arc::new(BoostCompressor))
        .with_query_rewriter(Arc::new(ScriptedRewriter {
            variant: "別の言い方".into(),
        }))
        .with_notebook_id(nb)
        .build()
        .unwrap();
    let chunks = vec![
        Chunk {
            id: Uuid::new_v4(), source_id: Uuid::nil(), ord: 0,
            text: "あ普通のチャンク 1。本文として十分な長さの説明をここに入れて noise filter の最低 25 文字を確実に超える。".into(),
            heading_path: vec![], page: None, bbox: None, summary: None,
        },
        Chunk {
            id: Uuid::new_v4(), source_id: Uuid::nil(), ord: 1,
            text: "い <BOOST> chunk と日本語ダミーをくっつけて noise filter を通す本文にしておく。".into(),
            heading_path: vec![], page: None, bbox: None, summary: None,
        },
        Chunk {
            id: Uuid::new_v4(), source_id: Uuid::nil(), ord: 2,
            text: "う 普通のチャンク 3。これも noise filter を通すためにダミー文を加えて 25 文字以上の本文にする。".into(),
            heading_path: vec![], page: None, bbox: None, summary: None,
        },
    ];
    let boosted_id = chunks[1].id;
    let texts: Vec<String> = chunks.iter().map(|c| c.text.clone()).collect();
    let embs = ellisii.embedder().embed(&texts).await.unwrap();
    ellisii.store().upsert(nb, &chunks, &embs).await.unwrap();

    let hits = ellisii
        .search(
            "通常のクエリです",
            SearchOptions {
                top_k: 3,
                semantic_weight: 0.5,
                caption_rerank: false,
                multi_query_max_variants: 1,
                ce_rerank_top_n: 3,
                ce_rerank_weight: 1.0,
                skip_ce_when_rewriting: true,
                ..Default::default()
            },
        )
        .await
        .unwrap();
    assert!(!hits.is_empty());
    // CE が skip されたので boosted passage は rank 1 に来ない (= CE は動いていない)。
    assert_ne!(
        hits[0].chunk.id, boosted_id,
        "CE should be skipped when rewriter is active; boosted passage must not be rank 1"
    );
}

/// 上と同じ条件で `skip_ce_when_rewriting=false` を明示すると、CE が走って boosted
/// passage が rank 1 に来る (= 既存挙動を opt-in で取り戻せる)。
#[tokio::test]
async fn ce_runs_when_rewriting_if_skip_disabled() {
    let nb = Uuid::new_v4();
    let ellisii = Ellisii::builder()
        .with_embedder(Arc::new(FixedEmbedder))
        .with_store_memory()
        .with_compressor(Arc::new(BoostCompressor))
        .with_query_rewriter(Arc::new(ScriptedRewriter {
            variant: "別の言い方".into(),
        }))
        .with_notebook_id(nb)
        .build()
        .unwrap();
    let chunks = vec![
        Chunk {
            id: Uuid::new_v4(), source_id: Uuid::nil(), ord: 0,
            text: "あ普通のチャンク 1。本文として十分な長さの説明をここに入れて noise filter の最低 25 文字を確実に超える。".into(),
            heading_path: vec![], page: None, bbox: None, summary: None,
        },
        Chunk {
            id: Uuid::new_v4(), source_id: Uuid::nil(), ord: 1,
            text: "い <BOOST> chunk と日本語ダミーをくっつけて noise filter を通す本文にしておく。".into(),
            heading_path: vec![], page: None, bbox: None, summary: None,
        },
        Chunk {
            id: Uuid::new_v4(), source_id: Uuid::nil(), ord: 2,
            text: "う 普通のチャンク 3。これも noise filter を通すためにダミー文を加えて 25 文字以上の本文にする。".into(),
            heading_path: vec![], page: None, bbox: None, summary: None,
        },
    ];
    let boosted_id = chunks[1].id;
    let texts: Vec<String> = chunks.iter().map(|c| c.text.clone()).collect();
    let embs = ellisii.embedder().embed(&texts).await.unwrap();
    ellisii.store().upsert(nb, &chunks, &embs).await.unwrap();

    let hits = ellisii
        .search(
            "通常のクエリです",
            SearchOptions {
                top_k: 3,
                semantic_weight: 0.5,
                caption_rerank: false,
                multi_query_max_variants: 1,
                ce_rerank_top_n: 3,
                ce_rerank_weight: 1.0,
                skip_ce_when_rewriting: false,
                ..Default::default()
            },
        )
        .await
        .unwrap();
    assert!(!hits.is_empty());
    assert_eq!(
        hits[0].chunk.id, boosted_id,
        "CE should run when skip_ce_when_rewriting=false; boosted passage must be rank 1"
    );
}

/// rewriter が specific クエリで skip された場合 (effective_max_variants=0) は CE が走る。
/// "第94条" は `is_specific_query` が true を返すので rewriter skip → CE 通常動作。
#[tokio::test]
async fn ce_runs_when_specific_query_skips_rewriter() {
    let nb = Uuid::new_v4();
    let ellisii = Ellisii::builder()
        .with_embedder(Arc::new(FixedEmbedder))
        .with_store_memory()
        .with_compressor(Arc::new(BoostCompressor))
        .with_query_rewriter(Arc::new(ScriptedRewriter {
            variant: "別の言い方".into(),
        }))
        .with_notebook_id(nb)
        .build()
        .unwrap();
    let chunks = vec![
        Chunk {
            id: Uuid::new_v4(), source_id: Uuid::nil(), ord: 0,
            text: "あ普通のチャンク 1。本文として十分な長さの説明をここに入れて noise filter の最低 25 文字を確実に超える。".into(),
            heading_path: vec![], page: None, bbox: None, summary: None,
        },
        Chunk {
            id: Uuid::new_v4(), source_id: Uuid::nil(), ord: 1,
            text: "い <BOOST> chunk と日本語ダミーをくっつけて noise filter を通す本文にしておく。".into(),
            heading_path: vec![], page: None, bbox: None, summary: None,
        },
    ];
    let boosted_id = chunks[1].id;
    let texts: Vec<String> = chunks.iter().map(|c| c.text.clone()).collect();
    let embs = ellisii.embedder().embed(&texts).await.unwrap();
    ellisii.store().upsert(nb, &chunks, &embs).await.unwrap();

    let hits = ellisii
        .search(
            "民法第94条の意思表示について教えて", // specific query → rewriter skip
            SearchOptions {
                top_k: 2,
                semantic_weight: 0.5,
                caption_rerank: false,
                multi_query_max_variants: 1,
                ce_rerank_top_n: 2,
                ce_rerank_weight: 1.0,
                skip_rewrite_on_specific: true,
                skip_ce_when_rewriting: true,
                ..Default::default()
            },
        )
        .await
        .unwrap();
    assert!(!hits.is_empty());
    // rewriter は specific でスキップされ、CE は通常動作 → boosted が rank 1。
    assert_eq!(
        hits[0].chunk.id, boosted_id,
        "rewriter skipped on specific query; CE should still run and promote BOOST"
    );
}
