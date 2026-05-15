//! `EvalOptions.rewriter` を有効にしたときの multi-query 経路の wiring 検証。
//!
//! 真モデル無しでも「variant 注入で recall@k が上がる」最小ケースを確認できれば、
//! agentic search の前段としての効果を eval ハーネス上で測定可能にできている。
//!
//! 真モデル + 真 LlmRewriter の計測は `real_static_jp.rs` に同様の比較を別途追加する。

use async_trait::async_trait;
use ellisii_core::Result as CoreResult;
use ellisii_query_rewriter_core::{QueryRewriter, RewrittenQueries};
use ellisii_query_rewriter_passthrough::PassthroughRewriter;
use ellisii_rag::{eval::GoldenSet, MultiQueryOptions};
use ellisii_rag_eval_cli::{
    run_eval_with_options, Backend, CorpusEntry, EmbedderKind, EvalOptions,
};
use std::sync::Arc;

fn fixture() -> (Vec<CorpusEntry>, GoldenSet) {
    let corpus = vec![
        CorpusEntry {
            doc_id: "neko-proverbs".into(),
            text: "ねこに小判。猫は気まぐれだが、観察すると沢山の表情がある。".into(),
            title: "ねこの諺".into(),
            caption: String::new(),
        },
        CorpusEntry {
            doc_id: "inu-folk".into(),
            text: "犬の遠吠えと忠義に関する民話集。".into(),
            title: "犬の昔話".into(),
            caption: String::new(),
        },
        CorpusEntry {
            doc_id: "tori-myth".into(),
            text: "鳥の伝承。八咫烏や鳳凰など各地の神話に登場する鳥。".into(),
            title: "鳥の伝承".into(),
            caption: String::new(),
        },
    ];
    // クエリは漢字「猫」のみ。doc 本文は ひらがな「ねこ」中心 → variant 無しでは取りにくい。
    let golden: GoldenSet = serde_json::from_value(serde_json::json!({
        "name": "kana-variant-mini",
        "items": [
            { "query": "猫", "relevant": ["neko-proverbs"] }
        ]
    }))
    .unwrap();
    (corpus, golden)
}

/// PassthroughRewriter を渡したときは rewriter=None と同じ結果になる
/// (= multi-query 経路自体に回帰がない)。
#[tokio::test]
async fn passthrough_rewriter_matches_no_rewriter() {
    let (corpus, golden) = fixture();
    let embedder = EmbedderKind::Bigram { dim: 256 }.build().unwrap();

    let baseline = EvalOptions {
        backend: Backend::Sqlite,
        embedder: embedder.clone(),
        weights: vec![0.5, 0.75],
        k: 3,
        judge: None,
        rewriter: None,
        multi: MultiQueryOptions::default(),
    };
    let with_pass = EvalOptions {
        backend: Backend::Sqlite,
        embedder,
        weights: vec![0.5, 0.75],
        k: 3,
        judge: None,
        rewriter: Some(Arc::new(PassthroughRewriter)),
        multi: MultiQueryOptions {
            max_variants: 0,
            ..Default::default()
        },
    };

    let a = run_eval_with_options(&baseline, &corpus, &golden).await.unwrap();
    let b = run_eval_with_options(&with_pass, &corpus, &golden).await.unwrap();
    for (ra, rb) in a.iter().zip(b.iter()) {
        assert_eq!(ra.semantic, rb.semantic);
        assert!(
            (ra.summary.recall_at_k - rb.summary.recall_at_k).abs() < 1e-6,
            "passthrough must not regress recall (sem={}): {} vs {}",
            ra.semantic,
            ra.summary.recall_at_k,
            rb.summary.recall_at_k,
        );
    }
}

/// 漢字クエリ "猫" にひらがな variant "ねこ" を足すと、kana-only な doc を取りに行ける。
/// 文字バイグラム embedder + sqlite (FTS5+CharBigram) でも、variant 注入で
/// recall@k が同等以上に伸びることを検証する。
struct KanaRewriter;
#[async_trait]
impl QueryRewriter for KanaRewriter {
    async fn rewrite(&self, query: &str, _max: usize) -> CoreResult<RewrittenQueries> {
        let variants = if query == "猫" {
            vec!["ねこ".to_string()]
        } else {
            Vec::new()
        };
        Ok(RewrittenQueries {
            original: query.to_string(),
            variants,
        })
    }
}

#[tokio::test]
async fn kana_variant_does_not_regress_recall() {
    let (corpus, golden) = fixture();
    let embedder = EmbedderKind::Bigram { dim: 256 }.build().unwrap();

    let baseline = EvalOptions {
        backend: Backend::Sqlite,
        embedder: embedder.clone(),
        weights: vec![0.5, 0.75],
        k: 3,
        judge: None,
        rewriter: None,
        multi: MultiQueryOptions::default(),
    };
    let with_kana = EvalOptions {
        backend: Backend::Sqlite,
        embedder,
        weights: vec![0.5, 0.75],
        k: 3,
        judge: None,
        rewriter: Some(Arc::new(KanaRewriter)),
        multi: MultiQueryOptions {
            max_variants: 1,
            variant_weight: 0.7,
            ..Default::default()
        },
    };

    let base = run_eval_with_options(&baseline, &corpus, &golden).await.unwrap();
    let multi = run_eval_with_options(&with_kana, &corpus, &golden).await.unwrap();
    println!("=== kana-variant mini eval ===");
    for (b, m) in base.iter().zip(multi.iter()) {
        println!(
            "  semantic={:.2}  baseline recall={:.3}  multi recall={:.3}",
            b.semantic, b.summary.recall_at_k, m.summary.recall_at_k
        );
        assert!(
            m.summary.recall_at_k >= b.summary.recall_at_k - 1e-6,
            "multi-query recall must not regress vs baseline (sem={}): {} -> {}",
            b.semantic,
            b.summary.recall_at_k,
            m.summary.recall_at_k,
        );
    }
}
