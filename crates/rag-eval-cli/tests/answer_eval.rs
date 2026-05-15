//! retrieve → generate → judge を一気通貫した evaluation。
//!
//! `EvalOptions` に `judge: Option<Arc<dyn AnswerJudge>>` を渡すと、各 query について
//! retrieve した chunk を context に LLM で answer を生成し、その忠実度を採点する。
//! 採点結果は `EvalRow.faithfulness: Option<FaithfulnessSummary>` に乗る。
//!
//! 既存の retrieve のみ評価 (judge=None) の経路と振る舞いが衝突しないことも担保する。

use ellisii_rag::eval::{GoldenItem, GoldenSet};
use ellisii_rag_answer_eval::{heuristic::TokenOverlapJudge, AnswerJudge};
use ellisii_rag_eval_cli::{
    run_eval_with_options, Backend, CorpusEntry, EmbedderKind, EvalOptions, DEFAULT_DIM,
};
use std::sync::Arc;

fn corpus() -> Vec<CorpusEntry> {
    vec![
        CorpusEntry {
            doc_id: "minpou-94".into(),
            title: "第九十四条".into(),
            caption: "虚偽表示".into(),
            text: "相手方と通じてした虚偽の意思表示は、無効とする。\
                   前項の規定による意思表示の無効は、善意の第三者に対抗することができない。"
                .into(),
        },
        CorpusEntry {
            doc_id: "minpou-95".into(),
            title: "第九十五条".into(),
            caption: "錯誤".into(),
            text: "意思表示は、次に掲げる錯誤に基づくものであって、\
                   その錯誤が法律行為の目的及び取引上の社会通念に照らして重要なものであるときは、\
                   取り消すことができる。"
                .into(),
        },
    ]
}

fn golden() -> GoldenSet {
    GoldenSet {
        name: "ans-eval".into(),
        items: vec![
            GoldenItem {
                query: "通謀虚偽表示は無効か".into(),
                relevant: vec!["minpou-94".into()],
                tags: vec![],
            },
            GoldenItem {
                query: "錯誤による意思表示は取り消せるか".into(),
                relevant: vec!["minpou-95".into()],
                tags: vec![],
            },
        ],
    }
}

#[tokio::test]
async fn judge_none_keeps_existing_behavior() {
    let opts = EvalOptions {
        backend: Backend::Sqlite,
        embedder: EmbedderKind::Bigram { dim: DEFAULT_DIM }.build().unwrap(),
        weights: vec![0.75],
        k: 5,
        judge: None,
        rewriter: None,
        multi: ellisii_rag::MultiQueryOptions::default(),
    };
    let rows = run_eval_with_options(&opts, &corpus(), &golden())
        .await
        .unwrap();
    assert_eq!(rows.len(), 1);
    assert!(
        rows[0].faithfulness.is_none(),
        "judge=None should not produce faithfulness"
    );
}

#[tokio::test]
async fn judge_set_produces_faithfulness_summary() {
    let judge: Arc<dyn AnswerJudge> = Arc::new(TokenOverlapJudge::default());
    let opts = EvalOptions {
        backend: Backend::Sqlite,
        embedder: EmbedderKind::Bigram { dim: DEFAULT_DIM }.build().unwrap(),
        weights: vec![0.75],
        k: 5,
        judge: Some(judge),
        rewriter: None,
        multi: ellisii_rag::MultiQueryOptions::default(),
    };
    let rows = run_eval_with_options(&opts, &corpus(), &golden())
        .await
        .unwrap();
    let f = rows[0]
        .faithfulness
        .as_ref()
        .expect("judge=Some should produce faithfulness");
    assert_eq!(f.queries, 2);
    // EchoLlm は user prompt をエコーするので "answer" には context が含まれるが、
    // 同時に "質問:" "参考:" "<source id=...>" などの prompt 構造由来のトークンも
    // 含まれてしまう。これらは context に存在しないので score が下がる。
    // 統合配線が動いていることを担保するための緩い閾値 (≥ 0.5) のみチェックする。
    assert!(
        f.mean >= 0.5,
        "expected mean ≥ 0.5 with EchoLlm + TokenOverlapJudge, got {}",
        f.mean
    );
}
