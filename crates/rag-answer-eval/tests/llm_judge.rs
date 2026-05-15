//! `LlmJudge` の挙動を、決定的な scripted LLM stub で固定する。
//!
//! 実 LLM (llama-cpp-2) は別経路で結合する想定。ここでは
//! - judge プロンプトに正しく context/answer を埋め込んでいるか
//! - LLM 応答 (テキスト) からスコアを正しくパースできるか
//! の 2 点だけをテストする。

use async_trait::async_trait;
use ellisii_core::Result;
use ellisii_llm_core::{LlmBackend, LlmRequest};
use ellisii_rag_answer_eval::{llm_judge::LlmJudge, AnswerJudge, JudgeInput};
use std::sync::{Arc, Mutex};

/// 事前に決めた一連のトークン列を順に流すだけの LLM stub。
/// `received_user` で最後に受け取った user prompt を取り出せる。
struct ScriptedLlm {
    response: String,
    received_user: Arc<Mutex<Option<String>>>,
}

impl ScriptedLlm {
    fn new(response: &str) -> (Self, Arc<Mutex<Option<String>>>) {
        let buf = Arc::new(Mutex::new(None));
        (
            Self {
                response: response.to_string(),
                received_user: buf.clone(),
            },
            buf,
        )
    }
}

#[async_trait]
impl LlmBackend for ScriptedLlm {
    async fn generate_stream(
        &self,
        req: LlmRequest,
        mut on_token: Box<dyn FnMut(String) + Send + 'static>,
    ) -> Result<()> {
        *self.received_user.lock().unwrap() = Some(req.user.clone());
        on_token(self.response.clone());
        Ok(())
    }
}

#[tokio::test]
async fn llm_judge_parses_score_from_response() {
    let (llm, _captured) = ScriptedLlm::new("0.85");
    let judge = LlmJudge::new(llm);
    let ctxs = vec!["context text".to_string()];
    let input = JudgeInput {
        question: "Q",
        contexts: &ctxs,
        answer: "A",
    };
    let s = judge.judge_faithfulness(&input).await.unwrap();
    assert!(
        (s.score - 0.85).abs() < 1e-3,
        "expected 0.85, got {}",
        s.score
    );
}

#[tokio::test]
async fn llm_judge_clamps_to_unit_interval() {
    // モデルが範囲外の値を返してきた場合は [0,1] にクランプする。
    let (llm, _) = ScriptedLlm::new("Score: 1.7");
    let judge = LlmJudge::new(llm);
    let ctxs = vec!["c".to_string()];
    let input = JudgeInput {
        question: "Q",
        contexts: &ctxs,
        answer: "A",
    };
    let s = judge.judge_faithfulness(&input).await.unwrap();
    assert!((s.score - 1.0).abs() < 1e-6, "got {}", s.score);
}

#[tokio::test]
async fn llm_judge_extracts_first_float_in_text() {
    // 思考の後に「最終的な答え: 0.42」のように出した場合も拾う。
    let (llm, _) =
        ScriptedLlm::new("ややズレている部分があるが概ね一致する。最終: 0.42 と評価する。");
    let judge = LlmJudge::new(llm);
    let ctxs = vec!["c".to_string()];
    let input = JudgeInput {
        question: "Q",
        contexts: &ctxs,
        answer: "A",
    };
    let s = judge.judge_faithfulness(&input).await.unwrap();
    assert!((s.score - 0.42).abs() < 1e-3, "got {}", s.score);
}

#[tokio::test]
async fn llm_judge_passes_question_context_answer_into_prompt() {
    let (llm, captured) = ScriptedLlm::new("0.5");
    let judge = LlmJudge::new(llm);
    let ctxs = vec!["民法第94条 通謀虚偽表示".to_string()];
    let input = JudgeInput {
        question: "通謀虚偽表示は無効か",
        contexts: &ctxs,
        answer: "通謀虚偽表示は無効である",
    };
    let _ = judge.judge_faithfulness(&input).await.unwrap();
    let prompt = captured
        .lock()
        .unwrap()
        .clone()
        .expect("user prompt captured");
    assert!(
        prompt.contains("通謀虚偽表示は無効か"),
        "prompt missing question: {prompt}"
    );
    assert!(
        prompt.contains("民法第94条"),
        "prompt missing context: {prompt}"
    );
    assert!(
        prompt.contains("通謀虚偽表示は無効である"),
        "prompt missing answer: {prompt}"
    );
}

#[tokio::test]
async fn llm_judge_returns_zero_when_no_number_in_response() {
    let (llm, _) = ScriptedLlm::new("評価できません");
    let judge = LlmJudge::new(llm);
    let ctxs = vec!["c".to_string()];
    let input = JudgeInput {
        question: "Q",
        contexts: &ctxs,
        answer: "A",
    };
    let s = judge.judge_faithfulness(&input).await.unwrap();
    // パースに失敗したら controvertial だが、安全側に倒して 0.0 ("不明 = 信頼できない")。
    assert!((s.score - 0.0).abs() < 1e-6, "got {}", s.score);
    assert!(
        s.explanation.is_some(),
        "explanation should record raw response"
    );
}
