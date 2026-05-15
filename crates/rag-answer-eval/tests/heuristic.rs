//! `TokenOverlapJudge` の挙動を rough にロックする。
//!
//! 設計意図:
//! - 100% answer のトークンが context に含まれていれば 1.0
//! - 全く無関係なら 0.0 付近
//! - 部分一致は中間
//!
//! あくまで決定的 baseline であって LLM judge の代替ではない。

use ellisii_rag_answer_eval::{heuristic::TokenOverlapJudge, AnswerJudge, JudgeInput};

#[tokio::test]
async fn fully_grounded_answer_scores_one() {
    let judge = TokenOverlapJudge::default();
    let contexts = vec![
        "民法第94条 通謀虚偽表示。相手方と通じてした虚偽の意思表示は、無効とする。".to_string(),
    ];
    let input = JudgeInput {
        question: "通謀虚偽表示は有効か",
        contexts: &contexts,
        answer: "通謀虚偽表示は無効とする。",
    };
    let s = judge.judge_faithfulness(&input).await.unwrap();
    assert!(s.score >= 0.95, "expected ≥0.95, got {}", s.score);
}

#[tokio::test]
async fn unrelated_answer_scores_zero() {
    let judge = TokenOverlapJudge::default();
    let contexts = vec!["民法第94条 通謀虚偽表示。".to_string()];
    let input = JudgeInput {
        question: "...",
        contexts: &contexts,
        answer: "ABCDEFG XYZ unrelated nonsense",
    };
    let s = judge.judge_faithfulness(&input).await.unwrap();
    assert!(s.score <= 0.1, "expected ≤0.1, got {}", s.score);
}

#[tokio::test]
async fn partial_overlap_in_between() {
    let judge = TokenOverlapJudge::default();
    let contexts = vec!["民法第94条 通謀虚偽表示は無効とする。".to_string()];
    // answer の半分は context に重なる、半分は無関係
    let input = JudgeInput {
        question: "...",
        contexts: &contexts,
        answer: "通謀虚偽表示は無効とする。 ABCDEFG nonsense",
    };
    let s = judge.judge_faithfulness(&input).await.unwrap();
    assert!(
        s.score > 0.3 && s.score < 0.95,
        "expected partial in (0.3, 0.95), got {}",
        s.score
    );
}

#[tokio::test]
async fn empty_answer_scores_one() {
    // 「何も答えていない」場合は逸脱しようがないので 1.0 とする (Ragas もこの定義)。
    let judge = TokenOverlapJudge::default();
    let contexts = vec!["context".to_string()];
    let input = JudgeInput {
        question: "...",
        contexts: &contexts,
        answer: "",
    };
    let s = judge.judge_faithfulness(&input).await.unwrap();
    assert!((s.score - 1.0).abs() < 1e-6);
}
