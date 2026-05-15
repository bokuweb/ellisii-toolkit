//! 任意の `LlmBackend` を judge として使う Ragas-light な faithfulness 実装。
//!
//! 流れ:
//! 1. (question, contexts, answer) を 1 つの prompt に組み立てる
//! 2. `LlmBackend::generate_stream` で応答を集める
//! 3. 応答テキストから最初に登場する 0.0〜1.0 の浮動小数点数を抜き出してスコアに採用
//! 4. 範囲外なら clamp、見つからなければ 0.0 + 説明にレスポンス本文を残す
//!
//! 局所 LLM (gemma / qwen) でも safety を取るため:
//! - 範囲外は無条件にクランプ
//! - 失敗時は 0.0 (= 信頼できない) に倒す。Ragas の faithfulness も同様に NaN を 0 扱い。

use crate::{AnswerJudge, FaithfulnessScore, JudgeInput};
use async_trait::async_trait;
use ellisii_core::Result;
use ellisii_llm_core::{LlmBackend, LlmRequest};
use std::sync::{Arc, Mutex};

pub struct LlmJudge<L: LlmBackend> {
    pub llm: L,
    pub max_tokens: u32,
    pub temperature: f32,
}

impl<L: LlmBackend> LlmJudge<L> {
    pub fn new(llm: L) -> Self {
        Self {
            llm,
            max_tokens: 64,
            temperature: 0.0,
        }
    }
}

#[async_trait]
impl<L: LlmBackend> AnswerJudge for LlmJudge<L> {
    async fn judge_faithfulness(&self, input: &JudgeInput<'_>) -> Result<FaithfulnessScore> {
        let context_block = input
            .contexts
            .iter()
            .enumerate()
            .map(|(i, c)| format!("[{}] {c}", i + 1))
            .collect::<Vec<_>>()
            .join("\n");
        let user = format!(
            "次の回答が、与えられた参考情報の範囲内でどれだけ忠実かを 0.0〜1.0 の数値 1 つで評価してください。\
             \n\n質問:\n{q}\n\n参考情報:\n{ctx}\n\n回答:\n{a}\n\nスコア (0.0〜1.0 の小数のみ):",
            q = input.question,
            ctx = context_block,
            a = input.answer,
        );
        let req = LlmRequest {
            system: "あなたは厳密な faithfulness 評価者です。回答が参考情報から逸脱していないかを 0.0〜1.0 の数値で評価し、数値以外の前置きは最小限に留めてください。".into(),
            history: Vec::new(),
            user,
            max_tokens: self.max_tokens,
            temperature: self.temperature,
        };

        let buf: Arc<Mutex<String>> = Arc::new(Mutex::new(String::new()));
        let buf2 = buf.clone();
        let cb: Box<dyn FnMut(String) + Send + 'static> = Box::new(move |t: String| {
            buf2.lock().unwrap().push_str(&t);
        });
        self.llm.generate_stream(req, cb).await?;
        let raw = buf.lock().unwrap().clone();
        let parsed = extract_first_unit_float(&raw);
        Ok(match parsed {
            Some(v) => FaithfulnessScore {
                score: v.clamp(0.0, 1.0),
                explanation: Some(raw),
            },
            None => FaithfulnessScore {
                score: 0.0,
                explanation: Some(format!("could not parse score from: {raw}")),
            },
        })
    }
}

/// テキスト中で最初に登場する浮動小数点数を取り出す。
/// "0.85", "0.42 と評価", "Score: 1.7" のいずれにもマッチ。
/// 整数のみ ("85") は採用しない (確率/比率として読めないため誤誘導を避ける)。
fn extract_first_unit_float(s: &str) -> Option<f32> {
    let bytes = s.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        // 数字または '.' の連続を取り出す
        if bytes[i].is_ascii_digit() || bytes[i] == b'.' {
            let start = i;
            let mut saw_digit = false;
            let mut saw_dot = false;
            while i < bytes.len() && (bytes[i].is_ascii_digit() || bytes[i] == b'.') {
                if bytes[i].is_ascii_digit() {
                    saw_digit = true;
                }
                if bytes[i] == b'.' {
                    if saw_dot {
                        break;
                    }
                    saw_dot = true;
                }
                i += 1;
            }
            if saw_digit && saw_dot {
                let token = std::str::from_utf8(&bytes[start..i]).ok()?;
                if let Ok(v) = token.parse::<f32>() {
                    return Some(v);
                }
            }
        } else {
            i += 1;
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extract_simple_float() {
        assert_eq!(extract_first_unit_float("0.85"), Some(0.85));
        assert_eq!(extract_first_unit_float("Score: 0.42 final"), Some(0.42));
        assert_eq!(extract_first_unit_float("評価不能"), None);
    }

    #[test]
    fn extract_skips_bare_integers() {
        // "85" だけだと確率なのか別の値か判別できないので拾わない。
        assert_eq!(extract_first_unit_float("85"), None);
        assert_eq!(extract_first_unit_float("85 then 0.5"), Some(0.5));
    }
}
