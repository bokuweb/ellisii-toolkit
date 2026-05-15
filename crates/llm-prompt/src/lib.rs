//! 各 LLM family のチャットテンプレート整形。
//!
//! - **Gemma 4** (3 と互換テンプレ): `<start_of_turn>user\n…<end_of_turn>\n<start_of_turn>model\n`
//!   (Gemma は system role をサポートせず user に寄せるのが慣例)
//! - **Qwen** (ChatML, Qwen 2.5 / 3 共通): `<|im_start|>system\n…<|im_end|>\n<|im_start|>user\n…<|im_end|>\n<|im_start|>assistant\n`

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
    }
}

fn format_gemma4(req: &LlmRequest) -> FormattedPrompt {
    // Gemma は専用 system role が無く、最初の user ターンに system を埋め込む。
    let mut text = String::new();
    let mut first_user = true;
    for turn in &req.history {
        match turn.role {
            ChatRole::User => {
                let body = if first_user && !req.system.trim().is_empty() {
                    first_user = false;
                    format!("{}\n\n{}", req.system.trim(), turn.content)
                } else {
                    first_user = false;
                    turn.content.clone()
                };
                text.push_str(&format!("<start_of_turn>user\n{}<end_of_turn>\n", body));
            }
            ChatRole::Assistant => {
                text.push_str(&format!(
                    "<start_of_turn>model\n{}<end_of_turn>\n",
                    turn.content
                ));
            }
        }
    }
    let user_body = if first_user && !req.system.trim().is_empty() {
        format!("{}\n\n{}", req.system.trim(), req.user)
    } else {
        req.user.clone()
    };
    text.push_str(&format!(
        "<start_of_turn>user\n{}<end_of_turn>\n<start_of_turn>model\n",
        user_body
    ));
    FormattedPrompt {
        text,
        stop_sequences: vec!["<end_of_turn>".into(), "<start_of_turn>".into()],
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
    fn gemma4_renders_history_with_system_in_first_user() {
        let p = format(
            ModelFamily::Gemma4,
            &req_with_history(
                "be terse",
                vec![
                    (ChatRole::User, "前の質問"),
                    (ChatRole::Assistant, "前の回答"),
                ],
                "新しい質問",
            ),
        );
        // 1 ターン目に system が入る
        assert!(p.text.contains("be terse\n\n前の質問"));
        assert!(p
            .text
            .contains("<start_of_turn>model\n前の回答<end_of_turn>"));
        assert!(p
            .text
            .ends_with("<start_of_turn>user\n新しい質問<end_of_turn>\n<start_of_turn>model\n"));
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
    fn gemma4_wraps_user_with_turn_markers() {
        let p = format(ModelFamily::Gemma4, &req("", "hello"));
        assert!(p.text.starts_with("<start_of_turn>user\n"));
        assert!(p.text.contains("hello"));
        assert!(p.text.ends_with("<start_of_turn>model\n"));
        assert!(p.stop_sequences.contains(&"<end_of_turn>".to_string()));
    }

    #[test]
    fn gemma4_inlines_system_into_user() {
        let p = format(ModelFamily::Gemma4, &req("be terse", "hi"));
        assert!(p.text.contains("be terse"));
        assert!(p.text.contains("hi"));
        // Gemma は専用 system セグメントを持たない
        assert!(!p.text.contains("<|im_start|>"));
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
}
