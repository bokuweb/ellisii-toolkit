//! 実モデルを使った end-to-end 生成スモーク。
//!
//! 必要環境 (いずれも `--features litert` ビルド時):
//!   - `ELLISII_LITERT_MODEL` … `.litertlm` / `.task` モデルへの絶対パス
//!   - build 時に `LITERT_LM_LIB_DIR` … CLiteRTLM dylib のディレクトリ
//!
//! 実行: `ELLISII_LITERT_MODEL=/path/gemma-4-E2B-it.litertlm \
//!        cargo test -p ellisii-llm-litert --features litert -- --ignored`

use ellisii_llm_core::{LlmBackend, LlmRequest, ModelFamily};
use ellisii_llm_litert::{LiteRtBackend, LiteRtConfig};

#[tokio::test]
#[ignore = "requires a real LiteRT-LM model + --features litert"]
async fn generates_nonempty_streamed_text() {
    let model = std::env::var("ELLISII_LITERT_MODEL")
        .expect("set ELLISII_LITERT_MODEL to a .litertlm/.task path");

    let backend = LiteRtBackend::load(LiteRtConfig::new(model, ModelFamily::Gemma4))
        .expect("load LiteRT-LM model");

    let collected = std::sync::Arc::new(std::sync::Mutex::new(String::new()));
    let sink = collected.clone();

    let req = LlmRequest {
        system: "Answer in one short word.".into(),
        history: vec![],
        user: "What is the capital of Japan?".into(),
        max_tokens: 32,
        temperature: 0.0,
    };

    backend
        .generate_stream(
            req,
            Box::new(move |tok| sink.lock().unwrap().push_str(&tok)),
        )
        .await
        .expect("generation succeeds");

    let out = collected.lock().unwrap().clone();
    assert!(!out.trim().is_empty(), "expected non-empty output");
    assert!(
        out.to_lowercase().contains("tokyo"),
        "expected Tokyo in answer, got: {out:?}"
    );
}
