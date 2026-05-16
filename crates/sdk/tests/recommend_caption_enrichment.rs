//! `Ellisii::recommend_caption_enrichment()` の挙動検証。
//! Run 12n で 8 fixture × 同条件 A/B で頑健性を確認した q-cap match 閾値
//! (< 0.15 → On / >= 0.25 → Off / 中間 → Uncertain) が、SDK API 経由で
//! 正しく分岐することを単体で確かめる。

use async_trait::async_trait;
use ellisii_core::{Chunk, Result};
use ellisii_embed_core::Embedder;
use ellisii_sdk::{Ellisii, EnrichmentRecommendation};
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

fn chunk_with_caption(caption: &str, body: &str) -> Chunk {
    Chunk {
        id: Uuid::new_v4(),
        source_id: Uuid::nil(),
        ord: 0,
        text: format!("({caption})\n{body}"),
        heading_path: vec![],
        page: None,
        bbox: None,
        summary: None,
    }
}

async fn build() -> Ellisii {
    let nb = Uuid::new_v4();
    Ellisii::builder()
        .with_embedder(Arc::new(ConstEmbedder))
        .with_store_memory()
        .with_notebook_id(nb)
        .build()
        .unwrap()
}

#[tokio::test]
async fn returns_uncertain_for_empty_inputs() {
    let e = build().await;
    let qs: Vec<&str> = vec![];
    let r = e.recommend_caption_enrichment(&qs).await.unwrap();
    assert!(matches!(r, EnrichmentRecommendation::Uncertain { .. }));

    // queries はあるが captions 無し
    let qs2 = vec!["なにか"];
    let r2 = e.recommend_caption_enrichment(&qs2).await.unwrap();
    assert!(matches!(r2, EnrichmentRecommendation::Uncertain { .. }));
    assert_eq!(r2.q_cap_match(), 0.0);
}

#[tokio::test]
async fn recommends_on_for_paraphrase_heavy_query_vs_caption() {
    // civil-law-hard 風: caption は法律ターム、queries は日常シナリオ
    let e = build().await;
    let nb = e.notebook_id();
    let cs = vec![
        chunk_with_caption("虚偽表示", "通謀によって作った契約は無効とする説明本文。"),
        chunk_with_caption("公序良俗", "公の秩序に反する取引は無効である旨の本文。"),
        chunk_with_caption("即時取得", "盗品を善意で取得した場合の所有権に関する規定。"),
    ];
    let texts: Vec<String> = cs.iter().map(|c| c.text.clone()).collect();
    let embs = e.embedder().embed(&texts).await.unwrap();
    e.store().upsert(nb, &cs, &embs).await.unwrap();

    let qs = vec![
        "税逃れのために売買契約書だけ作った場合の効力",
        "違法薬物の売買を約束する取引は有効か",
        "中古品を盗品と知らずに購入した場合の所有権",
    ];
    let r = e.recommend_caption_enrichment(&qs).await.unwrap();
    assert!(
        matches!(r, EnrichmentRecommendation::On { .. }),
        "expected On for paraphrase-heavy queries, got {r:?}"
    );
    assert!(r.enrichment_on());
    assert!(!r.caption_rerank(), "On → cap rerank false 推奨");
}

#[tokio::test]
async fn recommends_off_for_literal_lookup_query_matching_caption() {
    // workplace-regs 風: caption に query キーワードが直接含まれる
    let e = build().await;
    let nb = e.notebook_id();
    let cs = vec![
        chunk_with_caption(
            "時間外労働の取り扱い",
            "残業時間が月60時間を超えた場合の割増賃金率の規定。",
        ),
        chunk_with_caption(
            "出生時育児休業",
            "出生時育児休業を取得できる期間と日数についての規定本文。",
        ),
        chunk_with_caption(
            "介護休業の申出",
            "介護休業は開始予定日の何週間前までに申し出る規定。",
        ),
    ];
    let texts: Vec<String> = cs.iter().map(|c| c.text.clone()).collect();
    let embs = e.embedder().embed(&texts).await.unwrap();
    e.store().upsert(nb, &cs, &embs).await.unwrap();

    let qs = vec![
        "1カ月の残業が60時間を超えた場合の取り扱い",
        "出生時育児休業の取得可能な期間と日数",
        "介護休業の申出は開始予定日の何週間前まで",
    ];
    let r = e.recommend_caption_enrichment(&qs).await.unwrap();
    assert!(
        matches!(r, EnrichmentRecommendation::Off { .. }),
        "expected Off for literal-lookup queries matching captions, got {r:?}"
    );
    assert!(!r.enrichment_on());
    assert!(r.caption_rerank(), "Off → cap rerank true (default) のまま");
}
