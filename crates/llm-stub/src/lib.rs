use async_trait::async_trait;
use ellisii_core::Result;
use ellisii_llm_core::{LlmBackend, LlmRequest};

/// 入力をエコーするだけの配線確認用バックエンド。
pub struct EchoLlm;

#[async_trait]
impl LlmBackend for EchoLlm {
    async fn generate_stream(
        &self,
        req: LlmRequest,
        mut on_token: Box<dyn FnMut(String) + Send + 'static>,
    ) -> Result<()> {
        on_token("[stub] ".to_string());
        for word in req.user.split_whitespace() {
            on_token(word.to_string());
            on_token(" ".to_string());
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ellisii_llm_core::LlmRequest;
    use std::sync::{Arc, Mutex};

    fn collected() -> (
        Arc<Mutex<Vec<String>>>,
        Box<dyn FnMut(String) + Send + 'static>,
    ) {
        let buf: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
        let buf2 = buf.clone();
        let cb: Box<dyn FnMut(String) + Send + 'static> = Box::new(move |t: String| {
            buf2.lock().unwrap().push(t);
        });
        (buf, cb)
    }

    fn req(user: &str) -> LlmRequest {
        LlmRequest {
            system: "sys".into(),
            history: Vec::new(),
            user: user.into(),
            max_tokens: 16,
            temperature: 0.0,
        }
    }

    #[tokio::test]
    async fn echo_emits_prefix_then_words() {
        let (buf, cb) = collected();
        EchoLlm
            .generate_stream(req("hello world"), cb)
            .await
            .unwrap();
        let tokens = buf.lock().unwrap().clone();
        assert_eq!(tokens.first().map(String::as_str), Some("[stub] "));
        let body: String = tokens[1..].concat();
        assert_eq!(body.trim_end(), "hello world");
    }

    #[tokio::test]
    async fn echo_handles_empty_user() {
        let (buf, cb) = collected();
        EchoLlm.generate_stream(req(""), cb).await.unwrap();
        let tokens = buf.lock().unwrap().clone();
        assert_eq!(tokens, vec!["[stub] ".to_string()]);
    }

    #[tokio::test]
    async fn echo_collapses_whitespace_runs() {
        let (buf, cb) = collected();
        EchoLlm.generate_stream(req("a   b\tc"), cb).await.unwrap();
        let body: String = buf.lock().unwrap()[1..].concat();
        assert_eq!(body.trim_end(), "a b c");
    }
}
