//! 残った 2 つの retrieve miss query について、rewriter (MultiExpand) が
//! どんな variant を生成しているかを直接観察する。
//!
//! 仮説: rewriter が target 概念 (通謀虚偽表示 / 天然果実) を出せていれば
//! retrieve 側で拾える可能性が高い。出せていなければ prompt の改良対象。
//!
//! 観察結果は PR #N で記録 — 旧 prompt は表層を言い換えていたが、
//! 「surface vs 条文用語」の対比例 + 2 例追加で target 用語を出すよう改善。
//!
//! `#[ignore]`。`cargo test -p ellisii-rag-eval-cli --test inspect_rewriter_misses -- --ignored --nocapture`

use ellisii_llm_core::{LlmBackend, ModelFamily};
use ellisii_llm_llamacpp::{LlamaConfig, LlamaCppBackend};
use ellisii_query_rewriter_core::QueryRewriter;
use ellisii_query_rewriter_llm::MultiExpandRewriter;
use std::path::PathBuf;
use std::sync::Arc;

fn locate_e4b() -> Option<PathBuf> {
    std::env::var_os("HOME")
        .map(|h| {
            PathBuf::from(&h)
                .join("Library/Application Support/ellisii/models/gemma-4-E4B-it-IQ4_XS.gguf")
        })
        .filter(|p| p.is_file())
}

struct SharedLlm(Arc<dyn LlmBackend>);
#[async_trait::async_trait]
impl LlmBackend for SharedLlm {
    async fn generate_stream(
        &self,
        req: ellisii_llm_core::LlmRequest,
        on_token: Box<dyn FnMut(String) + Send + 'static>,
    ) -> ellisii_core::Result<()> {
        self.0.generate_stream(req, on_token).await
    }
}

#[tokio::test]
#[ignore]
async fn inspect_rewriter_outputs_for_misses() {
    let e4b = locate_e4b().expect("Gemma 4 E4B not present");
    let cfg = LlamaConfig::new(e4b, ModelFamily::Gemma4);
    let llm: Arc<dyn LlmBackend> = Arc::new(LlamaCppBackend::load(cfg).expect("load gemma E4B"));
    let rewriter = MultiExpandRewriter::new(SharedLlm(Arc::clone(&llm)));

    let queries = &[
        (
            "知人と相談して、税逃れのために売買契約書だけ作った場合の効力",
            "通謀虚偽表示 (minpou-94)",
        ),
        ("畑で取れた野菜は誰のものか", "天然果実 (minpou-88)"),
        // sanity check: should already work (regression check)
        ("脅されて結ばされた契約はどう扱われるか", "強迫 (minpou-96)"),
    ];

    println!("\n=== rewriter output inspection ===\n");
    for (q, target) in queries {
        let r = rewriter.rewrite(q, 8).await.unwrap();
        println!("Q: {q}");
        println!("Target: {target}");
        println!("Variants:");
        for (i, v) in r.variants.iter().enumerate() {
            println!("  [{}] {}", i + 1, v);
        }
        println!();
    }
}
