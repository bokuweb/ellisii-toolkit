//! `LlamaCppBackend::generate_stream` を **独立した 2 つのプロンプト**で連続呼び出ししても
//! KV cache が衝突しないことを確認する退行ガード。
//!
//! 修正前 (~~ee93f44 以前) は 2 回目の呼び出しで:
//!   ```text
//!   init: the tokens of sequence 0 in the input batch have inconsistent sequence positions
//!         X = 283, Y = 0
//!   decode: failed to initialize batch
//!   ```
//! を踏んで `Err` を返していた。SDK 経由で `LlmRewriter::rewrite` を回すと第 2 呼び出し以降
//! 全部この経路に落ち、rewriter が passthrough にフォールバックしていた
//! (`docs/eval/recall-evals.md` Run 7)。
//!
//! GGUF が無い環境では skip。
//!
//! ```sh
//! cargo test -p ellisii-llm-llamacpp --features gguf \
//!   --test kv_reset_regression -- --ignored --nocapture
//! ```

#![cfg(feature = "gguf")]

use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use ellisii_llm_core::{LlmBackend, LlmRequest, ModelFamily};
use ellisii_llm_llamacpp::{LlamaConfig, LlamaCppBackend};

fn locate_e4b() -> Option<PathBuf> {
    let p = std::env::var_os("HOME")
        .map(PathBuf::from)?
        .join("Library/Application Support/ellisii/models/gemma-4-E4B-it-IQ4_XS.gguf");
    if p.is_file() {
        Some(p)
    } else {
        None
    }
}

async fn collect(
    backend: &LlamaCppBackend,
    system: &str,
    user: &str,
) -> ellisii_core::Result<String> {
    let buf: Arc<Mutex<String>> = Arc::new(Mutex::new(String::new()));
    let buf2 = buf.clone();
    let cb: Box<dyn FnMut(String) + Send + 'static> =
        Box::new(move |t| buf2.lock().unwrap().push_str(&t));
    let req = LlmRequest {
        system: system.to_string(),
        history: Vec::new(),
        user: user.to_string(),
        max_tokens: 32,
        temperature: 0.0,
    };
    backend.generate_stream(req, cb).await?;
    let out = buf.lock().unwrap().clone();
    Ok(out)
}

#[tokio::test(flavor = "current_thread")]
#[ignore]
async fn independent_prompts_do_not_collide_in_kv_cache() {
    let Some(gguf) = locate_e4b() else {
        eprintln!("[skip] gemma-4-E4B-it-IQ4_XS.gguf not found");
        return;
    };
    let cfg = LlamaConfig::new(gguf, ModelFamily::Gemma4);
    let backend = LlamaCppBackend::load(cfg).expect("load gemma");

    // 同じ system だが user が違う 2 連続呼び出し (= LlmRewriter の典型的な使い方)
    let s = "あなたは検索クエリを書き換えるアシスタントです。";
    let r1 = collect(
        &backend,
        s,
        "「温泉に入ったときに課される税」を別表現で 1 つ書いてください。\n出力例:\n1. ...",
    )
    .await;
    assert!(r1.is_ok(), "1st call failed: {:?}", r1.err());
    assert!(
        !r1.as_ref().unwrap().is_empty(),
        "1st call produced no output"
    );

    let r2 = collect(
        &backend,
        s,
        "「市たばこ税の税率」を別表現で 1 つ書いてください。\n出力例:\n1. ...",
    )
    .await;
    assert!(
        r2.is_ok(),
        "2nd call failed (KV cache regression): {:?}",
        r2.err()
    );
    assert!(
        !r2.as_ref().unwrap().is_empty(),
        "2nd call produced no output"
    );

    // 完全に system も user も違う 3 番目の呼び出しでも生き残る (= prefix=0 でも安全)
    let r3 = collect(
        &backend,
        "あなたは丁寧な日本語の挨拶をしてください。",
        "おはよう",
    )
    .await;
    assert!(
        r3.is_ok(),
        "3rd call (totally different system) failed: {:?}",
        r3.err()
    );
}
