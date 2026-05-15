//! jp-cs-wiki-hard (CS Wikipedia、paraphrase + scenario) golden Q&A に対し、
//! **実 LLM (gemma-4-E4B GGUF)** + `LlmRewriter` / `MultiExpandRewriter` を回して
//! Run 34/35 (yokohama) / Run 38 (jp-civil-law-hard) の filter sweep を **第 3
//! corpus** で再計測する (Run 39)。
//!
//! 動機: Run 38 で「filter sweet spot は corpus 依存」が判明。yokohama は lookup
//! 寄り (rewriter 利得小)、civil-law-hard は paraphrase 100% (rewriter 利得大) と
//! 両極端だった。jp-cs-wiki-hard は **中庸 (Run 8: rewriter +9pp MRR)** で、
//! 3 点目の datapoint で「corpus 特性 → filter 効果」のマップを描けるかを検証する。
//!
//! 実行 (gemma-4-E4B GGUF + static-jp が必須):
//! ```sh
//! cargo run -p ellisii-sdk \
//!   --features static-jp,llamacpp \
//!   --example eval_cs_wiki_hard_llm --release
//! ```
//!
//! 結果は `docs/eval/recall-evals.md` Run 39 に追記する。

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
        .join("rag/tests/fixtures/eval/jp-cs-wiki-hard")
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

    // 1) corpus + golden 読み込み
    let dir = fixture_dir();
    let corpus_json = std::fs::read_to_string(dir.join("corpus.json"))?;
    let corpus: Vec<CorpusEntry> = serde_json::from_str(&corpus_json)?;
    let golden_json = std::fs::read_to_string(dir.join("golden.json"))?;
    let gold: GoldenSet = GoldenSet::from_json_str(&golden_json)?;
    eprintln!(
        "corpus: {} chunks, golden: {} ({} items)",
        corpus.len(),
        gold.name,
        gold.items.len()
    );

    // 2) Chunk + id_map を 1 度だけ構築
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

    println!(
        "\n=== jp-cs-wiki-hard: filter sweep (gemma-4-E4B, Run 39) ===\n\
         filter sweet spots from Run 34/35 (yokohama): LlmRewriter=0.05, MultiExpand=0.10"
    );
    println!(
        "{:<40} {:<10} {:<10} {:<10} {:<10} {:<10}",
        "variant", "recall", "hit", "ndcg", "mrr", "elapsed"
    );

    async fn run_pairs(
        ellisii: &Ellisii,
        gold: &GoldenSet,
        id_map: &HashMap<Uuid, String>,
        k: usize,
        max_variants: usize,
        filter_threshold: f32,
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
                        variant_caption_filter_threshold: filter_threshold,
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

    for &k in &[1usize, 5, 10] {
        let t0 = Instant::now();
        let cap = run_pairs(&ellisii_cap, &gold, &id_map, k, 0, 0.0).await?;
        let dur = t0.elapsed();
        let s = summarize(&cap, k);
        println!(
            "{:<40} {:<10.3} {:<10.3} {:<10.3} {:<10.3} {:<10}",
            format!("cap only (k={})", k),
            s.recall_at_k,
            s.hit_at_k,
            s.ndcg_at_k,
            s.mrr,
            format!("{:.1}s", dur.as_secs_f32())
        );

        let runs: &[(&str, &Ellisii, usize, f32)] = &[
            ("cap+LlmRewriter", &ellisii_llm, 3, 0.0),
            ("cap+LlmRew+filter@0.05", &ellisii_llm, 3, 0.05),
            ("cap+LlmRew+filter@0.10", &ellisii_llm, 3, 0.10),
            ("cap+MultiExpand", &ellisii_exp, 6, 0.0),
            ("cap+MultiExpand+filter@0.10", &ellisii_exp, 6, 0.10),
        ];
        for &(name, e, mv, th) in runs {
            let t0 = Instant::now();
            let pairs = run_pairs(e, &gold, &id_map, k, mv, th).await?;
            let dur = t0.elapsed();
            let s = summarize(&pairs, k);
            println!(
                "{:<40} {:<10.3} {:<10.3} {:<10.3} {:<10.3} {:<10}",
                format!("{name} (k={k})"),
                s.recall_at_k,
                s.hit_at_k,
                s.ndcg_at_k,
                s.mrr,
                format!("{:.1}s", dur.as_secs_f32())
            );
        }
    }
    Ok(())
}
