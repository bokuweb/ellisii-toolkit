//! 文字バイグラム重なりに基づく決定的 faithfulness 判定。LLM 不要。
//!
//! Ragas の faithfulness は「answer の各 claim が context に支持されるか」を
//! LLM に聞くが、それを真似る前にまず安価な baseline として bigram overlap で
//! 「answer の表面トークンが context にどれだけ含まれるか」を測る。
//!
//! 目的:
//! - LLM judge と相対比較するための reference 値
//! - LLM 無しでも CI で回せるリグレッション指標
//!
//! 限界:
//! - 言い換えに弱い (「無効である」と「効力が無い」を区別できない)
//! - 真の意味で grounded か否かは LLM judge に委ねる必要がある

use crate::{AnswerJudge, FaithfulnessScore, JudgeInput};
use async_trait::async_trait;
use ellisii_core::Result;
use std::collections::HashSet;

/// 文字 bigram 重なり率を返す決定的 judge。
pub struct TokenOverlapJudge {
    /// 1 文字以下のトークンを無視するか。日本語では false 推奨。
    pub min_chars: usize,
}

impl Default for TokenOverlapJudge {
    fn default() -> Self {
        Self { min_chars: 1 }
    }
}

/// LLM が「答えられない」を表明する典型フレーズ。これらが answer の大部分を
/// 占める場合、claim をしていないので faithfulness 違反は起こり得ない。
const REFUSAL_PHRASES: &[&str] = &[
    "参考資料に該当する情報は見つかりませんでした",
    "参考資料に該当する情報がありません",
    "参考資料には該当する情報がありません",
    "情報は見つかりませんでした",
    "情報なし",
    "回答できません",
    "答えられません",
];

/// answer が refusal とみなせるか。stock refusal フレーズを含み、かつ
/// answer 全体がそのフレーズの ~1.5x 以下に収まっているとき true。
/// これにより「短い refusal だけ」と「refusal を含む長文 (= 部分的に claim あり)」
/// を区別する。
fn looks_like_refusal(answer: &str) -> bool {
    let total = answer.chars().count();
    if total == 0 {
        return false;
    }
    for p in REFUSAL_PHRASES {
        if answer.contains(p) {
            let phrase_len = p.chars().count();
            // answer の主体が refusal phrase か (周辺は句点や軽い接続詞のみ想定)
            if total <= (phrase_len as f32 * 1.6) as usize + 8 {
                return true;
            }
        }
    }
    false
}

#[async_trait]
impl AnswerJudge for TokenOverlapJudge {
    async fn judge_faithfulness(&self, input: &JudgeInput<'_>) -> Result<FaithfulnessScore> {
        // 「答えられない」の表明は claim をしていないので faithfulness 違反は
        // 発生しえない。stock refusal 文字列を bigram で計ると context との
        // 重なりがほぼゼロで誤って 0.05 程度の低スコアになるため、特別扱い
        // で 1.0 (= 完全 grounded) として扱う。
        if looks_like_refusal(input.answer) {
            return Ok(FaithfulnessScore {
                score: 1.0,
                explanation: Some("refusal phrase: no claim made".into()),
            });
        }
        let answer_grams = bigrams(input.answer, self.min_chars);
        if answer_grams.is_empty() {
            // 何も答えていない → 逸脱しようがないので 1.0
            return Ok(FaithfulnessScore {
                score: 1.0,
                explanation: None,
            });
        }
        let mut ctx_grams: HashSet<(char, char)> = HashSet::new();
        for c in input.contexts {
            ctx_grams.extend(bigrams(c, self.min_chars));
        }
        let hit = answer_grams.iter().filter(|g| ctx_grams.contains(g)).count();
        let total = answer_grams.len();
        let score = hit as f32 / total as f32;
        Ok(FaithfulnessScore {
            score,
            explanation: None,
        })
    }
}

fn bigrams(s: &str, min_chars: usize) -> HashSet<(char, char)> {
    // 日本語の読点・句点や ASCII の punctuation を踏むと「は無」のような自然な
    // bigram が分断されて偽陰性になる。意味的 grounding を測るのが目的なので、
    // 表層の punctuation はノイズとして取り除いた上で bigram を作る。
    let chars: Vec<char> = s
        .chars()
        .filter(|c| !c.is_whitespace() && !is_skip_punct(*c))
        .collect();
    if chars.len() < min_chars.max(2) {
        return HashSet::new();
    }
    chars.windows(2).map(|w| (w[0], w[1])).collect()
}

fn is_skip_punct(c: char) -> bool {
    matches!(
        c,
        '、' | '。' | '・' | '「' | '」' | '『' | '』' | '（' | '）' | '【' | '】'
            | ',' | '.' | ';' | ':' | '!' | '?' | '(' | ')' | '[' | ']' | '"' | '\''
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn identical_strings_score_one() {
        let j = TokenOverlapJudge::default();
        let ctxs = vec!["abcde".to_string()];
        let input = JudgeInput {
            question: "",
            contexts: &ctxs,
            answer: "abcde",
        };
        let s = j.judge_faithfulness(&input).await.unwrap();
        assert!((s.score - 1.0).abs() < 1e-6);
    }

    #[tokio::test]
    async fn whitespace_is_ignored() {
        let j = TokenOverlapJudge::default();
        let ctxs = vec!["abcde".to_string()];
        let input = JudgeInput {
            question: "",
            contexts: &ctxs,
            answer: "  abcde  ",
        };
        let s = j.judge_faithfulness(&input).await.unwrap();
        assert!((s.score - 1.0).abs() < 1e-6);
    }

    #[tokio::test]
    async fn refusal_phrase_is_treated_as_grounded() {
        let j = TokenOverlapJudge::default();
        // 完全に context と無関係な refusal でも 1.0 (= claim していないので違反不能)
        let ctxs = vec!["まったく違う本文です。".to_string()];
        let input = JudgeInput {
            question: "Q",
            contexts: &ctxs,
            answer: "参考資料に該当する情報は見つかりませんでした。",
        };
        let s = j.judge_faithfulness(&input).await.unwrap();
        assert!((s.score - 1.0).abs() < 1e-6, "got {}", s.score);
    }

    #[tokio::test]
    async fn long_answer_with_refusal_substring_uses_normal_score() {
        let j = TokenOverlapJudge::default();
        // 長文の中で refusal フレーズを部分的に使う場合は通常 bigram 採点。
        // (context にない claim を主張している可能性があるため)
        let answer = "本問について、参考資料に該当する情報は見つかりませんでした。\
                      ただし、関連する条文として民法第94条が考えられ、\
                      虚偽表示は無効であるという原則が適用される可能性があります。\
                      また、第三者に対する効力は別途検討が必要となります。";
        let ctxs = vec!["まったく違う本文です。".to_string()];
        let input = JudgeInput {
            question: "Q",
            contexts: &ctxs,
            answer,
        };
        let s = j.judge_faithfulness(&input).await.unwrap();
        // refusal exemption は適用されず、通常 bigram で低スコア
        assert!(s.score < 0.5, "long answer should not get refusal exemption (got {})", s.score);
    }

    #[tokio::test]
    async fn alt_refusal_phrase_information_nashi() {
        let j = TokenOverlapJudge::default();
        let ctxs = vec!["ctx".to_string()];
        let input = JudgeInput {
            question: "Q",
            contexts: &ctxs,
            answer: "情報なし",
        };
        let s = j.judge_faithfulness(&input).await.unwrap();
        assert!((s.score - 1.0).abs() < 1e-6);
    }
}
