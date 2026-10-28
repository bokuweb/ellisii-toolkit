//! 各 LLM family のチャットテンプレート整形。
//!
//! - **Gemma 4**: `<|turn>role\n…<turn|>`, with a dedicated system turn.
//! - **Qwen** (ChatML, Qwen 2.5 / 3 共通): `<|im_start|>system\n…<|im_end|>\n<|im_start|>user\n…<|im_end|>\n<|im_start|>assistant\n`
//! - **LFM3**: ChatML with an empty thinking block for non-thinking generation.

use ellisii_llm_core::{ChatRole, LlmRequest, ModelFamily};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FormattedPrompt {
    pub text: String,
    /// 推論側で stop に使うトークン文字列の候補。
    pub stop_sequences: Vec<String>,
}

pub fn format(family: ModelFamily, req: &LlmRequest) -> FormattedPrompt {
    match family {
        ModelFamily::Gemma4 => format_gemma4(req),
        ModelFamily::Qwen => format_qwen(req),
        ModelFamily::Lfm3 => format_lfm3(req),
    }
}

// Text-only, non-thinking E2B/E4B template from google/gemma-4-E2B-it.
// BOS is inserted by the inference backend.
fn format_gemma4(req: &LlmRequest) -> FormattedPrompt {
    let mut text = String::new();
    if !req.system.trim().is_empty() {
        text.push_str(&format!("<|turn>system\n{}<turn|>\n", req.system.trim()));
    }
    for turn in &req.history {
        let (role, content) = match turn.role {
            ChatRole::User => ("user", turn.content.trim().to_string()),
            ChatRole::Assistant => {
                // Match the canonical strip_thinking macro for past answers.
                let answer: String = turn
                    .content
                    .split("<channel|>")
                    .map(|part| part.split("<|channel>").next().unwrap_or_default())
                    .collect();
                ("model", answer.trim().to_string())
            }
        };
        text.push_str(&format!("<|turn>{role}\n{content}<turn|>\n"));
    }
    text.push_str(&format!(
        "<|turn>user\n{}<turn|>\n<|turn>model\n",
        req.user.trim()
    ));
    FormattedPrompt {
        text,
        stop_sequences: vec!["<turn|>".into(), "<|turn>".into()],
    }
}

fn format_qwen(req: &LlmRequest) -> FormattedPrompt {
    let mut text = String::new();
    if !req.system.trim().is_empty() {
        text.push_str("<|im_start|>system\n");
        text.push_str(req.system.trim());
        text.push_str("<|im_end|>\n");
    }
    for turn in &req.history {
        let role = match turn.role {
            ChatRole::User => "user",
            ChatRole::Assistant => "assistant",
        };
        text.push_str(&format!("<|im_start|>{role}\n"));
        text.push_str(&turn.content);
        text.push_str("<|im_end|>\n");
    }
    text.push_str("<|im_start|>user\n");
    text.push_str(&req.user);
    text.push_str("<|im_end|>\n<|im_start|>assistant\n");
    FormattedPrompt {
        text,
        stop_sequences: vec!["<|im_end|>".into(), "<|endoftext|>".into()],
    }
}

// Mirrors LiquidAI/LFM3-1.7B chat_template.jinja with enable_thinking=false.
// BOS is added by the inference backend, rather than embedded in this string.
fn format_lfm3(req: &LlmRequest) -> FormattedPrompt {
    format_lfm3_with_thinking(req, false)
}

/// LFM3 text template with an explicit reasoning mode.
/// Generated reasoning is included in the stream and counts toward max_tokens.
pub fn format_lfm3_with_thinking(req: &LlmRequest, enable_thinking: bool) -> FormattedPrompt {
    let mut history = req.history.clone();
    for turn in &mut history {
        if turn.role == ChatRole::Assistant {
            let answer = turn
                .content
                .rsplit_once("</think>")
                .map_or(turn.content.as_str(), |(_, answer)| answer.trim());
            turn.content = format!("<think></think>{answer}");
        }
    }
    let mut prompt = format_qwen(&LlmRequest {
        history,
        ..req.clone()
    });
    prompt.text.push_str(if enable_thinking {
        "<think>"
    } else {
        "<think></think>"
    });
    prompt
}

#[cfg(test)]
mod tests {
    use super::*;

    fn req(sys: &str, user: &str) -> LlmRequest {
        LlmRequest {
            system: sys.into(),
            history: vec![],
            user: user.into(),
            max_tokens: 16,
            temperature: 0.0,
        }
    }

    fn req_with_history(sys: &str, history: Vec<(ChatRole, &str)>, user: &str) -> LlmRequest {
        LlmRequest {
            system: sys.into(),
            history: history
                .into_iter()
                .map(|(r, c)| ellisii_llm_core::ChatTurn {
                    role: r,
                    content: c.into(),
                })
                .collect(),
            user: user.into(),
            max_tokens: 16,
            temperature: 0.0,
        }
    }

    #[test]
    fn gemma4_renders_canonical_system_and_history() {
        let p = format(
            ModelFamily::Gemma4,
            &req_with_history(
                " be terse ",
                vec![
                    (ChatRole::User, " 前の質問 "),
                    (
                        ChatRole::Assistant,
                        "<|channel>thought\nprivate<channel|> 前の回答 ",
                    ),
                ],
                " 新しい質問 ",
            ),
        );
        assert_eq!(
            p.text,
            concat!(
                "<|turn>system\nbe terse<turn|>\n",
                "<|turn>user\n前の質問<turn|>\n",
                "<|turn>model\n前の回答<turn|>\n",
                "<|turn>user\n新しい質問<turn|>\n<|turn>model\n"
            )
        );
    }

    #[test]
    fn qwen_renders_history_as_chatml() {
        let p = format(
            ModelFamily::Qwen,
            &req_with_history(
                "be terse",
                vec![(ChatRole::User, "Q1"), (ChatRole::Assistant, "A1")],
                "Q2",
            ),
        );
        assert!(p.text.contains("<|im_start|>system\nbe terse<|im_end|>"));
        assert!(p.text.contains("<|im_start|>user\nQ1<|im_end|>"));
        assert!(p.text.contains("<|im_start|>assistant\nA1<|im_end|>"));
        assert!(p
            .text
            .ends_with("<|im_start|>user\nQ2<|im_end|>\n<|im_start|>assistant\n"));
    }

    #[test]
    fn gemma4_wraps_user_with_canonical_turn_markers() {
        let p = format(ModelFamily::Gemma4, &req("   ", "hello"));
        assert_eq!(p.text, "<|turn>user\nhello<turn|>\n<|turn>model\n");
        assert_eq!(p.stop_sequences, vec!["<turn|>", "<|turn>"]);
    }

    #[test]
    fn qwen_uses_chatml() {
        let p = format(ModelFamily::Qwen, &req("be terse", "hi"));
        assert!(p.text.contains("<|im_start|>system\nbe terse<|im_end|>"));
        assert!(p.text.contains("<|im_start|>user\nhi<|im_end|>"));
        assert!(p.text.ends_with("<|im_start|>assistant\n"));
        assert!(p.stop_sequences.contains(&"<|im_end|>".to_string()));
    }

    #[test]
    fn qwen_omits_empty_system() {
        let p = format(ModelFamily::Qwen, &req("   ", "hi"));
        assert!(!p.text.contains("<|im_start|>system"));
    }
    #[test]
    fn lfm3_thinking_uses_canonical_open_reasoning_prefix() {
        let p = format_lfm3_with_thinking(&req("", "hello"), true);
        assert_eq!(
            p.text,
            "<|im_start|>user\nhello<|im_end|>\n<|im_start|>assistant\n<think>"
        );
        assert_eq!(p.stop_sequences, vec!["<|im_end|>", "<|endoftext|>"]);
    }

    #[test]
    fn lfm3_disables_thinking_and_strips_previous_reasoning() {
        let p = format(
            ModelFamily::Lfm3,
            &req_with_history(
                "日本語で回答",
                vec![
                    (ChatRole::User, "前の質問"),
                    (
                        ChatRole::Assistant,
                        "<think>private reasoning</think> 回答 ",
                    ),
                ],
                "次の質問",
            ),
        );
        assert_eq!(
            p.text,
            concat!(
                "<|im_start|>system\n日本語で回答<|im_end|>\n",
                "<|im_start|>user\n前の質問<|im_end|>\n",
                "<|im_start|>assistant\n<think></think>回答<|im_end|>\n",
                "<|im_start|>user\n次の質問<|im_end|>\n",
                "<|im_start|>assistant\n<think></think>"
            )
        );
        assert_eq!(p.stop_sequences, vec!["<|im_end|>", "<|endoftext|>"]);
    }

    #[test]
    fn lfm3_plain_assistant_history_gets_empty_thinking_prefix() {
        let p = format(
            ModelFamily::Lfm3,
            &req_with_history(
                "",
                vec![(ChatRole::User, "Q"), (ChatRole::Assistant, "A")],
                "Q2",
            ),
        );
        assert!(p.text.contains("assistant\n<think></think>A<|im_end|>"));
        assert!(!p.text.contains("<|im_start|>system"));
        assert!(p.text.ends_with("assistant\n<think></think>"));
    }
}
