//! RAG の **answer 層** を定量評価するためのプリミティブ。
//!
//! `rag-eval-cli` が retrieve 層 (recall@k / nDCG@k / MRR) を見るのに対して、
//! こちらは **生成された answer が context に対してどれだけ忠実か** を測る。
//! Ragas でいう faithfulness に相当する層。
//!
//! 構成:
//! - [`AnswerJudge`] — judgement の trait。決定的な heuristic と LLM judge を切替可能。
//! - [`TokenOverlapJudge`] — 文字 bigram 重なりに基づく決定的 baseline (LLM 不要)。
//! - [`LlmJudge`] — `LlmBackend` を使って 0.0〜1.0 のスコアを取り出す Ragas-light 実装 (続編)。
//!
//! いずれも `(question, contexts, answer) → FaithfulnessScore` というシグネチャに揃える。

pub mod heuristic;
pub mod llm_judge;

use async_trait::async_trait;
use ellisii_core::Result;
use serde::{Deserialize, Serialize};

/// 1 件の judgement への入力。`contexts` は retrieve 層が返した上位 K 件の chunk テキスト。
#[derive(Debug, Clone)]
pub struct JudgeInput<'a> {
    pub question: &'a str,
    pub contexts: &'a [String],
    pub answer: &'a str,
}

/// 0.0 (= context から逸脱) 〜 1.0 (= 完全に grounded) のスコア。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FaithfulnessScore {
    pub score: f32,
    /// 任意の説明 (LLM judge は理由テキストを返せる)。決定的 judge は通常 None。
    #[serde(default)]
    pub explanation: Option<String>,
}

#[async_trait]
pub trait AnswerJudge: Send + Sync {
    async fn judge_faithfulness(&self, input: &JudgeInput<'_>) -> Result<FaithfulnessScore>;
}

/// 複数 query を集計したサマリ。
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct FaithfulnessSummary {
    pub queries: usize,
    pub mean: f32,
    pub min: f32,
    pub max: f32,
}

impl FaithfulnessSummary {
    pub fn from_scores(scores: &[FaithfulnessScore]) -> Self {
        if scores.is_empty() {
            return Self::default();
        }
        let n = scores.len() as f32;
        let mut sum = 0.0_f32;
        let mut mn = f32::INFINITY;
        let mut mx = f32::NEG_INFINITY;
        for s in scores {
            sum += s.score;
            if s.score < mn {
                mn = s.score;
            }
            if s.score > mx {
                mx = s.score;
            }
        }
        Self {
            queries: scores.len(),
            mean: sum / n,
            min: mn,
            max: mx,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn s(v: f32) -> FaithfulnessScore {
        FaithfulnessScore {
            score: v,
            explanation: None,
        }
    }

    #[test]
    fn summary_empty_is_zero() {
        let s = FaithfulnessSummary::from_scores(&[]);
        assert_eq!(s.queries, 0);
        assert_eq!(s.mean, 0.0);
    }

    #[test]
    fn summary_aggregates_basic_stats() {
        let scores = vec![s(1.0), s(0.5), s(0.0)];
        let summary = FaithfulnessSummary::from_scores(&scores);
        assert_eq!(summary.queries, 3);
        assert!((summary.mean - 0.5).abs() < 1e-6);
        assert_eq!(summary.min, 0.0);
        assert_eq!(summary.max, 1.0);
    }
}
