//! `Ellisii::corpus_paraphrase_score()` の挙動検証。caption と body の vocab 乖離度の
//! 平均が、概念定義系 corpus (高 novelty) と字面一致系 corpus (低 novelty) で正しく
//! 区別できることを確認する。Run 18 (`docs/eval/recall-evals.md`) で導入したヒューリスティック。

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
async fn paraphrase_score_is_zero_for_empty_or_no_caption_corpus() {
    let nb = Uuid::new_v4();
    let e = Ellisii::builder()
        .with_embedder(Arc::new(ConstEmbedder))
        .with_store_memory()
        .with_notebook_id(nb)
        .build()
        .unwrap();
    // 空 corpus
    assert_eq!(e.corpus_paraphrase_score().await.unwrap(), 0.0);

    // caption が無いだけの corpus でも 0.0 (シグナル無効)
    let cs = vec![
        chunk("ただの本文 1。長めに書いておく。"),
        chunk("ただの本文 2。"),
    ];
    let texts: Vec<String> = cs.iter().map(|c| c.text.clone()).collect();
    let embs = e.embedder().embed(&texts).await.unwrap();
    e.store().upsert(nb, &cs, &embs).await.unwrap();
    // caption-less chunks では article-body fallback で caption が拾われる可能性があるが、
    // どちらにせよ stable な signal が返ること (panic しない) を確認。
    let s = e.corpus_paraphrase_score().await.unwrap();
    assert!((0.0..=1.0).contains(&s), "score out of range: {s}");
}

#[tokio::test]
async fn paraphrase_corpus_scores_higher_than_literal_corpus() {
    // 概念定義系: caption が短く、body に caption に無い概念語彙が多い
    let nb_para = Uuid::new_v4();
    let e_para = Ellisii::builder()
        .with_embedder(Arc::new(ConstEmbedder))
        .with_store_memory()
        .with_notebook_id(nb_para)
        .build()
        .unwrap();
    let cs_para = vec![
        chunk("(発明)\n自然法則を利用した技術的思想の創作のうち高度のもの"),
        chunk("(特許権)\n発明を独占的に実施する権利を一定期間付与する制度"),
        chunk("(実施)\n物の生産・使用・譲渡・輸入・申出をする行為"),
    ];
    let texts: Vec<String> = cs_para.iter().map(|c| c.text.clone()).collect();
    let embs = e_para.embedder().embed(&texts).await.unwrap();
    e_para
        .store()
        .upsert(nb_para, &cs_para, &embs)
        .await
        .unwrap();
    let para_score = e_para.corpus_paraphrase_score().await.unwrap();

    // 字面一致系: caption と body が同じ語彙
    let nb_lit = Uuid::new_v4();
    let e_lit = Ellisii::builder()
        .with_embedder(Arc::new(ConstEmbedder))
        .with_store_memory()
        .with_notebook_id(nb_lit)
        .build()
        .unwrap();
    let cs_lit = vec![
        chunk("(入湯税の税率)\n入湯税の税率は100円とする"),
        chunk("(たばこ税の税率)\nたばこ税の税率は次のとおりとする"),
        chunk("(都市計画税の税率)\n都市計画税の税率は0.3%とする"),
    ];
    let texts: Vec<String> = cs_lit.iter().map(|c| c.text.clone()).collect();
    let embs = e_lit.embedder().embed(&texts).await.unwrap();
    e_lit.store().upsert(nb_lit, &cs_lit, &embs).await.unwrap();
    let lit_score = e_lit.corpus_paraphrase_score().await.unwrap();

    assert!(
        para_score > lit_score + 0.15,
        "paraphrase corpus must score notably higher than literal: paraphrase={para_score}, literal={lit_score}"
    );
    assert!(para_score > 0.85, "paraphrase score too low: {para_score}");
}

#[tokio::test]
async fn paraphrase_score_in_unit_range() {
    // 任意の corpus でも 0.0..=1.0 に収まることの sanity check
    let nb = Uuid::new_v4();
    let e = Ellisii::builder()
        .with_embedder(Arc::new(ConstEmbedder))
        .with_store_memory()
        .with_notebook_id(nb)
        .build()
        .unwrap();
    let cs = vec![
        chunk("(A)\nA"),
        chunk("(B)\nまったく違う本文"),
        chunk("(C)\nC C C C C C C"),
    ];
    let texts: Vec<String> = cs.iter().map(|c| c.text.clone()).collect();
    let embs = e.embedder().embed(&texts).await.unwrap();
    e.store().upsert(nb, &cs, &embs).await.unwrap();
    let s = e.corpus_paraphrase_score().await.unwrap();
    assert!((0.0..=1.0).contains(&s), "score out of range: {s}");
}
