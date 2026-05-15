//! 特許法 (jp-tokkyo-hou) golden Q&A に対し、**実 LLM (gemma-4-E4B GGUF)** を使った
//! `LlmRewriter` / `MultiExpandRewriter` 経由 multi-query retrieval を A/B/C で測る。
//!
//! Run 9 (yokohama n=26) では caption rerank が top-1 を支配していて LLM rewriter の
//! 差が見えにくかった。jp-tokkyo-hou (n=64, paraphrase / hard 比 ~57%) で同じ実験を回し、
//! caption-rich + 抽象 paraphrase の混在環境で MultiExpand と LlmRewriter のどちらが
//! 有利かを実測する。
//!
//! 実行 (gemma-4-E4B GGUF + static-jp が必須):
//! ```sh
//! cargo run -p ellisii-sdk \
//!   --features static-jp,llamacpp \
//!   --example eval_tokkyo_llm --release
//! ```
//!
//! 結果は `docs/eval/recall-evals.md` Run 11 に追記する。

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use ellisii_core::Chunk;
use ellisii_rag::eval::{summarize, GoldenSet};
use ellisii_sdk::{Ellisii, SearchOptions};
use ellisii_store_core::VectorStore;
use serde::Deserialize;
use std::collections::HashMap;
use uuid::Uuid;

#[derive(Debug, Deserialize)]
struct CorpusEntry {
    doc_id: String,
    #[allow(dead_code)]
    title: String,
    caption: String,
    text: String,
}

fn home() -> PathBuf {
    PathBuf::from(std::env::var("HOME").expect("HOME"))
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

fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .join("rag/tests/fixtures/eval/jp-tokkyo-hou")
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
    use ellisii_embed_core::Embedder;
    use ellisii_llm_core::{LlmBackend, ModelFamily};
    use ellisii_llm_llamacpp::{LlamaConfig, LlamaCppBackend};
    use ellisii_query_rewriter_llm::{LlmRewriter, MultiExpandRewriter};

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

    // Setup: load corpus + golden
    let dir = fixture_dir();
    let corpus_json = std::fs::read_to_string(dir.join("corpus.json"))?;
    let corpus: Vec<CorpusEntry> = serde_json::from_str(&corpus_json)?;
    let golden_json = std::fs::read_to_string(dir.join("golden.json"))?;
    let gold: GoldenSet = GoldenSet::from_json_str(&golden_json)?;
    eprintln!("corpus: {} chunks, golden: {} ({} items)", corpus.len(), gold.name, gold.items.len());

    // Build chunks once + id_map
    let nb = Uuid::new_v4();
    let mut chunks: Vec<Chunk> = Vec::with_capacity(corpus.len());
    let mut texts: Vec<String> = Vec::with_capacity(corpus.len());
    let mut id_map: HashMap<Uuid, String> = HashMap::new();
    for (i, e) in corpus.iter().enumerate() {
        let cid = Uuid::new_v4();
        id_map.insert(cid, e.doc_id.clone());
        let txt = if e.caption.is_empty() {
            e.text.clone()
        } else {
            format!("({})\n{}", e.caption, e.text)
        };
        chunks.push(Chunk {
            id: cid,
            source_id: Uuid::new_v4(),
            ord: i as u32,
            text: txt.clone(),
            heading_path: vec![e.doc_id.clone()],
            page: None,
            bbox: None,
            summary: None,
        });
        texts.push(txt);
    }

    // LLM (1 度だけロード) + 3 つの Ellisii instance を共有 chunk で構築
    let Some(gguf) = gemma_e4b() else {
        anyhow::bail!("gemma-4-E4B-it-IQ4_XS.gguf が見つかりません");
    };
    let embed = embed_dir();
    eprintln!("embed: {}", embed.display());
    eprintln!("LLM:   {}", gguf.display());

    let cfg = LlamaConfig::new(gguf, ModelFamily::Gemma4);
    let llm: Arc<dyn LlmBackend> =
        Arc::new(LlamaCppBackend::load(cfg).map_err(|e| anyhow::anyhow!("load gemma: {e}"))?);
    let llm_rewriter = Arc::new(LlmRewriter::new(SharedLlm(Arc::clone(&llm))));
    let multi_expand_rewriter = Arc::new(MultiExpandRewriter::new(SharedLlm(Arc::clone(&llm))));

    async fn build(
        nb: Uuid,
        chunks: &[Chunk],
        texts: &[String],
        embed: &PathBuf,
        rewriter: Option<Arc<dyn ellisii_query_rewriter_core::QueryRewriter>>,
    ) -> anyhow::Result<Ellisii> {
        let mut b = Ellisii::builder()
            .with_embedder_static_jp(embed)?
            .with_store_memory()
            .with_notebook_id(nb);
        if let Some(r) = rewriter {
            b = b.with_query_rewriter(r);
        }
        let e = b.build()?;
        let embs = e.embedder().embed(texts).await?;
        e.store().upsert(nb, chunks, &embs).await?;
        Ok(e)
    }

    let ellisii_cap = build(nb, &chunks, &texts, &embed, None).await?;
    let ellisii_llm = build(nb, &chunks, &texts, &embed, Some(llm_rewriter)).await?;
    let ellisii_exp = build(nb, &chunks, &texts, &embed, Some(multi_expand_rewriter)).await?;

    println!("\n=== jp-tokkyo-hou: caption rerank vs caption+multi-query (gemma-4-E4B) ===");
    println!(
        "{:<32} {:<10} {:<10} {:<10} {:<10} {:<10}",
        "variant", "recall", "hit", "ndcg", "mrr", "elapsed"
    );

    async fn run_pairs(
        ellisii: &Ellisii,
        gold: &GoldenSet,
        id_map: &HashMap<Uuid, String>,
        k: usize,
        max_variants: usize,
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
                        ..Default::default()
                    },
                )
                .await?;
            let pred: Vec<String> = hits
                .iter()
                .filter_map(|h| id_map.get(&h.chunk.id).cloned())
                .collect();
            pairs.push((pred, item.relevant.clone()));
        }
        Ok(pairs)
    }

    for &k in &[5usize, 10] {
        let t0 = Instant::now();
        let cap = run_pairs(&ellisii_cap, &gold, &id_map, k, 0).await?;
        let dur_cap = t0.elapsed();
        let s_cap = summarize(&cap, k);

        let t0 = Instant::now();
        let llm_p = run_pairs(&ellisii_llm, &gold, &id_map, k, 3).await?;
        let dur_llm = t0.elapsed();
        let s_llm = summarize(&llm_p, k);

        let t0 = Instant::now();
        let exp = run_pairs(&ellisii_exp, &gold, &id_map, k, 6).await?;
        let dur_exp = t0.elapsed();
        let s_exp = summarize(&exp, k);

        println!(
            "{:<32} {:<10.3} {:<10.3} {:<10.3} {:<10.3} {:<10}",
            format!("cap only (k={})", k),
            s_cap.recall_at_k, s_cap.hit_at_k, s_cap.ndcg_at_k, s_cap.mrr,
            format!("{:.1}s", dur_cap.as_secs_f32())
        );
        println!(
            "{:<32} {:<10.3} {:<10.3} {:<10.3} {:<10.3} {:<10}",
            format!("cap+LlmRewriter (k={})", k),
            s_llm.recall_at_k, s_llm.hit_at_k, s_llm.ndcg_at_k, s_llm.mrr,
            format!("{:.1}s", dur_llm.as_secs_f32())
        );
        println!(
            "{:<32} {:<10.3} {:<10.3} {:<10.3} {:<10.3} {:<10}",
            format!("cap+MultiExpand (k={})", k),
            s_exp.recall_at_k, s_exp.hit_at_k, s_exp.ndcg_at_k, s_exp.mrr,
            format!("{:.1}s", dur_exp.as_secs_f32())
        );
    }
    Ok(())
}
