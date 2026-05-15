//! `eval_workplace_regs` の LLM rewriter 拡張版。caption rerank only (Run 1) を
//! baseline に、`LlmRewriter` と `MultiExpandRewriter` (gemma-4-E4B GGUF) で
//! 失敗 2 件 (法定休日 / 出張中の労働時間) を拾えるかを A/B 計測する。
//!
//! 実行 (gemma-4-E4B GGUF + static-jp が必須):
//! ```sh
//! cargo run -p ellisii-sdk \
//!   --features static-jp,llamacpp \
//!   --example eval_workplace_regs_llm --release
//! ```
//!
//! 結果は ellisii の `docs/eval/recall-evals.md` jp-workplace-regs セクションに
//! 「Run 2 (LLM rewriter)」として追記する。

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use ellisii_core::Chunk;
use ellisii_rag::eval::{summarize, GoldenSet};
use ellisii_sdk::{Ellisii, SearchOptions};
use serde::Deserialize;
use uuid::Uuid;

#[derive(Debug, Deserialize)]
struct CorpusEntry {
    doc_id: String,
    #[allow(dead_code)]
    parent_id: String,
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
        .join("rag/tests/fixtures/eval/jp-workplace-regs")
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

    let dir = fixture_dir();
    let corpus: Vec<CorpusEntry> =
        serde_json::from_str(&std::fs::read_to_string(dir.join("corpus.json"))?)?;
    let gold: GoldenSet =
        GoldenSet::from_json_str(&std::fs::read_to_string(dir.join("golden.json"))?)?;
    eprintln!(
        "corpus: {} chunks, golden: {} ({} items)",
        corpus.len(),
        gold.name,
        gold.items.len()
    );

    let nb = Uuid::new_v4();
    let src = Uuid::new_v4();
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
            source_id: src,
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
    let ellisii_llm_e = build(nb, &chunks, &texts, &embed, Some(llm_rewriter)).await?;
    let ellisii_exp = build(nb, &chunks, &texts, &embed, Some(multi_expand_rewriter)).await?;

    println!(
        "\n=== jp-workplace-regs: LLM rewriter A/B (gemma-4-E4B, Run 2) ===\n\
         baseline = caption rerank only (Run 1)"
    );
    println!(
        "{:<40} {:<10} {:<10} {:<10} {:<10} {:<10}",
        "variant", "recall", "hit", "ndcg", "mrr", "elapsed"
    );

    // 40 query × LLM rewrite で 1 sweep = 数百 LLM 呼出になるので k=5 のみで A/B 取る。
    // recall@5 は yokohama / 民法 eval と同じ主軸メトリクス。
    let k = 5usize;
    let t0 = Instant::now();
    eprintln!("[run] cap only ...");
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
        ("cap+LlmRewriter", &ellisii_llm_e, 3, 0.0),
        ("cap+LlmRew+filter@0.10", &ellisii_llm_e, 3, 0.10),
        ("cap+MultiExpand", &ellisii_exp, 6, 0.0),
        ("cap+MultiExpand+filter@0.10", &ellisii_exp, 6, 0.10),
    ];
    for &(name, e, mv, th) in runs {
        eprintln!("[run] {} ...", name);
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

    // Targeted failure inspection: 法定休日 / 出張中の労働時間 を Run 1 で外した
    // 2 query について、LlmRewriter / MultiExpand で top-5 がどう動いたか個別に出す。
    println!("\n=== Targeted: Run 1 failures at k=5 ===");
    let targets = ["法定休日は何曜日", "出張中の労働時間はどう扱われるか"];
    for q in targets {
        let item = gold
            .items
            .iter()
            .find(|i| i.query == q)
            .expect("query in golden");
        println!("query: {} (expected={:?})", q, item.relevant);
        for (label, eng, mv, th) in [
            ("cap only", &ellisii_cap, 0usize, 0.0_f32),
            ("LlmRew", &ellisii_llm_e, 3, 0.0),
            ("MultiExpand", &ellisii_exp, 6, 0.0),
        ] {
            let hits = eng
                .search(
                    q,
                    SearchOptions {
                        top_k: 5,
                        semantic_weight: 0.5,
                        caption_rerank: true,
                        multi_query_max_variants: mv,
                        multi_query_variant_weight: 0.7,
                        variant_caption_filter_threshold: th,
                        ..Default::default()
                    },
                )
                .await?;
            let pred: Vec<String> = hits
                .iter()
                .filter_map(|h| id_map.get(&h.chunk.id).cloned())
                .collect();
            println!("  {:<12} top5={:?}", label, pred);
        }
    }

    Ok(())
}

#[cfg(all(feature = "static-jp", feature = "llamacpp"))]
async fn run_pairs(
    ellisii: &Ellisii,
    gold: &GoldenSet,
    id_map: &HashMap<Uuid, String>,
    k: usize,
    max_variants: usize,
    filter_threshold: f32,
) -> ellisii_core::Result<Vec<(Vec<String>, Vec<String>)>> {
    let mut pairs = Vec::with_capacity(gold.items.len());
    for (i, item) in gold.items.iter().enumerate() {
        eprintln!("    [{}/{}] {}", i + 1, gold.items.len(), item.query);
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
