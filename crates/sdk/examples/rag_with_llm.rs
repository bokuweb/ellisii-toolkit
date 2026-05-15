//! RAG: 検索 + LLM stream で回答を生成する例。
//!
//! `feature = "llamacpp"` を有効化し、GGUF モデル (Gemma 4 / Qwen 系) を
//! ローカルに置いた状態で実行します。
//!
//! 実行:
//! ```sh
//! cargo run -p ellisii-sdk --features llamacpp --example rag_with_llm -- \
//!   ./docs \
//!   "民法を要約して" \
//!   /path/to/gemma-4-E4B-it-IQ4_XS.gguf
//! ```

#[cfg(feature = "llamacpp")]
#[tokio::main(flavor = "current_thread")]
async fn main() -> anyhow::Result<()> {
    use ellisii_sdk::{AskOptions, Ellisii, IndexOptions, ModelFamily};

    let mut args = std::env::args().skip(1);
    let dir = args
        .next()
        .ok_or_else(|| anyhow::anyhow!("usage: rag_with_llm <dir> <query> <model_path>"))?;
    let query = args
        .next()
        .ok_or_else(|| anyhow::anyhow!("missing query"))?;
    let model_path = args
        .next()
        .ok_or_else(|| anyhow::anyhow!("missing model_path"))?;

    let dim = 64usize;
    let ellisii = Ellisii::builder()
        .with_embedder_dummy(dim)
        .with_store_memory()
        .with_llm_llamacpp(&model_path, ModelFamily::Gemma4)?
        .build()?;

    eprintln!("indexing {dir} ...");
    let _ = ellisii.index_dir(&dir, IndexOptions::default()).await?;

    eprintln!("\nQ: {query}\nA: ");
    use std::io::Write;
    let _hits = ellisii
        .ask(
            &query,
            AskOptions {
                top_k: 6,
                temperature: 0.2,
                max_tokens: 512,
                ..Default::default()
            },
            move |tok: String| {
                let mut out = std::io::stdout();
                let _ = out.write_all(tok.as_bytes());
                let _ = out.flush();
            },
        )
        .await?;
    println!();
    Ok(())
}

#[cfg(not(feature = "llamacpp"))]
fn main() {
    eprintln!(
        "this example requires the `llamacpp` feature.\n\
         re-run with: cargo run -p ellisii-sdk --features llamacpp --example rag_with_llm -- ..."
    );
    std::process::exit(2);
}
