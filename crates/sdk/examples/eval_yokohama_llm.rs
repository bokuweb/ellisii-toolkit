//! 横浜市市税条例の golden Q&A に対し、**実 LLM (gemma-4-E4B GGUF)** を使った
//! `LlmRewriter` 経由 multi-query retrieval を A/B で測る。
//!
//! `eval_yokohama.rs` は LLM 不要な決定的 synonym table で multi-query をシミュレートしたが、
//! ここは production 経路 (LlmRewriter + LlamaCppBackend) を実際に走らせて recall への
//! 影響を計測する。レイテンシも併記して latency / quality trade-off を見る。
//!
//! 実行 (gemma-4-E4B GGUF が必須):
//! ```sh
//! cargo run -p ellisii-sdk \
//!   --features static-jp,llamacpp \
//!   --example eval_yokohama_llm --release
//! ```
//!
//! 結果 (Run 7) は `docs/eval/recall-evals.md` に追記する。

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use ellisii_rag::eval::{summarize, GoldenSet};
use ellisii_sdk::{Ellisii, SearchOptions};
use uuid::Uuid;

const NOTEBOOK_ID: &str = "95339065-df88-4ee7-82c1-e11c587250e4";

fn home() -> PathBuf {
    PathBuf::from(std::env::var("HOME").expect("HOME"))
}
fn db_path() -> PathBuf {
    home().join("Library/Application Support/ellisii/ellisii.db")
}
fn embed_dir() -> PathBuf {
    home().join("Library/Application Support/ellisii/models/static-embedding-japanese")
}
fn gemma_e4b() -> Option<PathBuf> {
    let p = home().join("Library/Application Support/ellisii/models/gemma-4-E4B-it-IQ4_XS.gguf");
    if p.is_file() {
        Some(p)
    } else {
        None
    }
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> anyhow::Result<()> {
    #[cfg(not(all(feature = "static-jp", feature = "llamacpp")))]
    {
        anyhow::bail!("build with --features static-jp,llamacpp");
    }
    #[cfg(all(feature = "static-jp", feature = "llamacpp"))]
    return run().await;
}

#[cfg(all(feature = "static-jp", feature = "llamacpp"))]
async fn run() -> anyhow::Result<()> {
    use async_trait::async_trait;
    use ellisii_llm_core::{LlmBackend, ModelFamily};
    use ellisii_llm_llamacpp::{LlamaConfig, LlamaCppBackend};
    use ellisii_query_rewriter_llm::{LlmRewriter, MultiExpandRewriter};

    /// `Arc<dyn LlmBackend>` を generic な `L: LlmBackend` に橋渡しする薄い wrapper。
    /// `rag-eval-cli/tests/fusion_comparison.rs` と同じパターン。
    /// 既知の制限: 現状の `LlamaCppBackend::generate_stream` は独立プロンプト間で KV cache を
    /// 自動 reset しないため、`LlmRewriter` のような繰り返し呼び出し経路ではエラーで
    /// 落ちて `RewrittenQueries::just(query)` (passthrough) にフォールバックする。
    /// その場合の数値は cap-only と一致するはずなので、A/B が同じ値で並んだら KV 起因。
    /// 真の数値が欲しいときは src-tauri が `clear_kv_cache_seq` を呼ぶのと同等のリセットを
    /// LlamaCppBackend 側に入れるか、各 query で fresh backend を作る。
    struct SharedLlm(Arc<dyn LlmBackend>);
    #[async_trait]
    impl LlmBackend for SharedLlm {
        async fn generate_stream(
            &self,
            req: ellisii_llm_core::LlmRequest,
            on_token: Box<dyn FnMut(String) + Send + 'static>,
        ) -> ellisii_core::Result<()> {
            self.0.generate_stream(req, on_token).await
        }
    }

    let db = db_path();
    let embed = embed_dir();
    let Some(gguf) = gemma_e4b() else {
        anyhow::bail!(
            "gemma-4-E4B-it-IQ4_XS.gguf が見つかりません: {}",
            home()
                .join("Library/Application Support/ellisii/models/")
                .display()
        );
    };

    eprintln!("DB:    {}", db.display());
    eprintln!("Embed: {}", embed.display());
    eprintln!("LLM:   {}", gguf.display());

    // 1) golden 読み込み
    let golden_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/eval/yokohama/golden.json");
    let raw = std::fs::read_to_string(&golden_path)?;
    let gold: GoldenSet = GoldenSet::from_json_str(&raw)?;
    eprintln!("golden: {} ({} items)", gold.name, gold.items.len());

    // 2) LLM を 1 度だけロード、Ellisii と LlmRewriter で共有
    let cfg = LlamaConfig::new(gguf, ModelFamily::Gemma4);
    let llm: Arc<dyn LlmBackend> =
        Arc::new(LlamaCppBackend::load(cfg).map_err(|e| anyhow::anyhow!("load gemma: {e}"))?);

    let dim = 1024;
    let nb = Uuid::parse_str(NOTEBOOK_ID)?;

    // cap-only Ellisii から caption sample を取り出して caption-aware rewriter を作る。
    // (Run 37) ここで LLM rewriter に corpus の見出し語彙を教え込んで、Run 33 の
    // displacement 仮説を「prompt 側」で潰せるかを検証する。
    let ellisii_cap = Ellisii::builder()
        .with_embedder_static_jp(&embed)?
        .with_store_sqlite(&db, dim)?
        .with_notebook_id(nb)
        .build()?;
    let caption_hints = ellisii_cap.caption_samples(24).await?;
    eprintln!("caption hints: {} samples", caption_hints.len());

    let llm_rewriter = Arc::new(LlmRewriter::new(SharedLlm(Arc::clone(&llm))));
    let llm_rewriter_capaware = Arc::new(
        LlmRewriter::new(SharedLlm(Arc::clone(&llm))).with_caption_hints(caption_hints.clone()),
    );
    let multi_expand_rewriter = Arc::new(MultiExpandRewriter::new(SharedLlm(Arc::clone(&llm))));

    // 比較用 Ellisii: cap only / +LlmRewriter / +CaptionAwareLlm / +MultiExpand
    let ellisii_mq_llm = Ellisii::builder()
        .with_embedder_static_jp(&embed)?
        .with_store_sqlite(&db, dim)?
        .with_notebook_id(nb)
        .with_query_rewriter(llm_rewriter)
        .build()?;
    let ellisii_mq_capaware = Ellisii::builder()
        .with_embedder_static_jp(&embed)?
        .with_store_sqlite(&db, dim)?
        .with_notebook_id(nb)
        .with_query_rewriter(llm_rewriter_capaware)
        .build()?;
    let ellisii_mq_expand = Ellisii::builder()
        .with_embedder_static_jp(&embed)?
        .with_store_sqlite(&db, dim)?
        .with_notebook_id(nb)
        .with_query_rewriter(multi_expand_rewriter)
        .build()?;

    // 3) cap only と cap+mq(LLM) を順に実行。latency も計測。
    println!("\n=== Yokohama: caption rerank vs caption+multi-query (real LLM, gemma-4-E4B) ===");
    println!(
        "{:<32} {:<10} {:<10} {:<10} {:<10} {:<10}",
        "variant", "recall", "hit", "ndcg", "mrr", "elapsed"
    );

    async fn run_pairs(
        ellisii: &Ellisii,
        gold: &GoldenSet,
        k: usize,
        max_variants: usize,
        variant_caption_filter_threshold: f32,
    ) -> ellisii_core::Result<Vec<(Vec<String>, Vec<String>)>> {
        let mut pairs: Vec<(Vec<String>, Vec<String>)> = Vec::with_capacity(gold.items.len());
        for item in &gold.items {
            let hits = ellisii
                .search(
                    &item.query,
                    SearchOptions {
                        top_k: k,
                        semantic_weight: 0.5,
                        caption_rerank: true,
                        multi_query_max_variants: max_variants,
                        multi_query_variant_weight: 0.7,
                        variant_caption_filter_threshold,
                        ..Default::default()
                    },
                )
                .await?;
            let pred: Vec<String> = hits.iter().map(|h| h.chunk.id.to_string()).collect();
            pairs.push((pred, item.relevant.clone()));
        }
        Ok(pairs)
    }

    // Run 37: caption-aware prompt の効果検証。Run 35 までの結果と比較するため、
    // 既存の sweep variant は最小限に絞り、caption-aware variant を中心に並べる。
    // - cap only (baseline、no LLM)
    // - cap+LlmRewriter (no hints, no filter) — Run 33 baseline 再現
    // - cap+CapAwareLlm (with hints, no filter) — Run 37 主役
    // - cap+CapAwareLlm+filter@0.05 (with hints + Run 35 sweet spot)
    // - cap+MultiExpand (no hints) — 比較対照、Run 33 と同条件
    for &k in &[1usize, 5, 10] {
        let t0 = Instant::now();
        let cap_pairs = run_pairs(&ellisii_cap, &gold, k, 0, 0.0).await?;
        let elapsed_cap = t0.elapsed();
        let s_cap = summarize(&cap_pairs, k);
        println!(
            "{:<40} {:<10.3} {:<10.3} {:<10.3} {:<10.3} {:<10}",
            format!("cap only (k={})", k),
            s_cap.recall_at_k,
            s_cap.hit_at_k,
            s_cap.ndcg_at_k,
            s_cap.mrr,
            format!("{:.1}s", elapsed_cap.as_secs_f32())
        );

        let runs: &[(&str, &Ellisii, usize, f32)] = &[
            ("cap+LlmRewriter", &ellisii_mq_llm, 3, 0.0),
            ("cap+CapAwareLlm", &ellisii_mq_capaware, 3, 0.0),
            ("cap+CapAwareLlm+filter@0.05", &ellisii_mq_capaware, 3, 0.05),
            ("cap+MultiExpand", &ellisii_mq_expand, 6, 0.0),
        ];
        for &(name, e, mv, th) in runs {
            let t0 = Instant::now();
            let pairs = run_pairs(e, &gold, k, mv, th).await?;
            let elapsed = t0.elapsed();
            let s = summarize(&pairs, k);
            println!(
                "{:<40} {:<10.3} {:<10.3} {:<10.3} {:<10.3} {:<10}",
                format!("{name} (k={k})"),
                s.recall_at_k,
                s.hit_at_k,
                s.ndcg_at_k,
                s.mrr,
                format!("{:.1}s", elapsed.as_secs_f32())
            );
        }
    }
    Ok(())
}
