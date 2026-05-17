//! Run 12v: v6 で残る唯一の退行 (jp-tokkyo-hou MRR Δ -0.066) を
//! **検索時 LlmRewriter** で救済できるか実測する 2x2 ハーネス。
//!
//! CLAUDE.md「index 時 LLM 禁止 / 同等機能は search/ask 時 opt-in」 ルールの
//! 二段構え (index v6 LiteralOnly + search LLM) を初めて A/B で実証する。
//!
//! 4 variants (k=5, bigram, cap rerank=on, w=0.5):
//! 1. baseline   (no enrich, no rewriter)
//! 2. v6         (LawThesaurus::bundled v6 enrich, no rewriter)
//! 3. baseline+R (no enrich, LlmRewriter)
//! 4. v6+R       (v6 enrich, LlmRewriter)   ← hybrid
//!
//! 使い方 (gemma-4-E4B GGUF が必須):
//! ```sh
//! cargo run -p ellisii-sdk --features static-jp,llamacpp \
//!   --example eval_tokkyo_hybrid_v6 --release
//! ```

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use ellisii_core::Chunk;
use ellisii_jp_law_thesaurus::LawThesaurus;
use ellisii_rag::eval::{hit_at_k, reciprocal_rank, GoldenSet};
use ellisii_sdk::{Ellisii, SearchOptions};
use serde::Deserialize;
use uuid::Uuid;

#[derive(Debug, Deserialize, Clone)]
struct CorpusEntry {
    doc_id: String,
    #[serde(default)]
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
    use ellisii_llm_core::{LlmBackend, ModelFamily};
    use ellisii_llm_llamacpp::{LlamaConfig, LlamaCppBackend};
    use ellisii_query_rewriter_core::QueryRewriter;
    use ellisii_query_rewriter_llm::LlmRewriter;

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
        "fixture: jp-tokkyo-hou ({} chunks, {} queries)",
        corpus.len(),
        gold.items.len()
    );

    // Build raw + v6-enriched chunk sets sharing the same UUID set (same id_map).
    let nb = Uuid::new_v4();
    let src = Uuid::new_v4();
    let mut chunks_base: Vec<Chunk> = Vec::with_capacity(corpus.len());
    let mut id_map: HashMap<Uuid, String> = HashMap::new();
    for (i, e) in corpus.iter().enumerate() {
        let cid = Uuid::new_v4();
        id_map.insert(cid, e.doc_id.clone());
        let txt = if e.caption.is_empty() {
            e.text.clone()
        } else {
            format!("({})\n{}", e.caption, e.text)
        };
        chunks_base.push(Chunk {
            id: cid,
            source_id: src,
            ord: i as u32,
            text: txt,
            heading_path: vec![e.doc_id.clone()],
            page: None,
            bbox: None,
            summary: None,
        });
    }
    let thes = Arc::new(LawThesaurus::bundled());
    let mut chunks_enr = chunks_base.clone();
    let n_enriched = thes.enrich_chunks(&mut chunks_enr);
    eprintln!(
        "thesaurus: {} ({} entries), enriched {}/{}",
        thes.name(),
        thes.entry_count(),
        n_enriched,
        chunks_enr.len()
    );

    let Some(gguf) = gemma_e4b() else {
        anyhow::bail!("gemma-4-E4B-it-IQ4_XS.gguf が見つかりません");
    };
    eprintln!("LLM: {}", gguf.display());
    let cfg = LlamaConfig::new(gguf, ModelFamily::Gemma4);
    let llm: Arc<dyn LlmBackend> =
        Arc::new(LlamaCppBackend::load(cfg).map_err(|e| anyhow::anyhow!("load gemma: {e}"))?);
    let rewriter_factory =
        || Arc::new(LlmRewriter::new(SharedLlm(Arc::clone(&llm)))) as Arc<dyn QueryRewriter>;

    let embed = embed_dir();
    async fn build_engine(
        nb: Uuid,
        chunks: &[Chunk],
        embed: &PathBuf,
        rewriter: Option<Arc<dyn QueryRewriter>>,
    ) -> anyhow::Result<Ellisii> {
        let mut b = Ellisii::builder()
            .with_embedder_static_jp(embed)?
            .with_store_memory()
            .with_notebook_id(nb);
        if let Some(r) = rewriter {
            b = b.with_query_rewriter(r);
        }
        let e = b.build()?;
        let texts: Vec<String> = chunks.iter().map(|c| c.text.clone()).collect();
        let embs = e.embedder().embed(&texts).await?;
        e.store().upsert(nb, chunks, &embs).await?;
        Ok(e)
    }
    let e_baseline = build_engine(nb, &chunks_base, &embed, None).await?;
    let e_v6 = build_engine(nb, &chunks_enr, &embed, None).await?;
    let e_base_r = build_engine(nb, &chunks_base, &embed, Some(rewriter_factory())).await?;
    let e_v6_r = build_engine(nb, &chunks_enr, &embed, Some(rewriter_factory())).await?;

    async fn eval(
        ellisii: &Ellisii,
        gold: &GoldenSet,
        id_map: &HashMap<Uuid, String>,
        max_variants: usize,
    ) -> anyhow::Result<(f32, f32, f32)> {
        let mut pairs: Vec<(Vec<String>, Vec<String>)> = Vec::with_capacity(gold.items.len());
        let t0 = Instant::now();
        for item in &gold.items {
            let hits = ellisii
                .search(
                    &item.query,
                    SearchOptions {
                        top_k: 5,
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
        let elapsed = t0.elapsed().as_secs_f32();
        let n = pairs.len() as f32;
        let hit5 = pairs.iter().map(|(p, r)| hit_at_k(p, r, 5)).sum::<f32>() / n;
        let mrr = pairs
            .iter()
            .map(|(p, r)| reciprocal_rank(p, r))
            .sum::<f32>()
            / n;
        Ok((hit5, mrr, elapsed))
    }

    println!("\n=== jp-tokkyo-hou hybrid 2x2 (k=5, bigram, cap rerank=on) ===");
    println!("| variant            | hit@5 | MRR   | elapsed |");
    println!("|--------------------|------:|------:|--------:|");
    let (h, m, e) = eval(&e_baseline, &gold, &id_map, 0).await?;
    println!("| baseline           | {:.3} | {:.3} | {:.1}s |", h, m, e);
    let (h, m, e) = eval(&e_v6, &gold, &id_map, 0).await?;
    println!("| v6 enrich          | {:.3} | {:.3} | {:.1}s |", h, m, e);
    let (h, m, e) = eval(&e_base_r, &gold, &id_map, 3).await?;
    println!("| base + LlmRewriter | {:.3} | {:.3} | {:.1}s |", h, m, e);
    let (h, m, e) = eval(&e_v6_r, &gold, &id_map, 3).await?;
    println!(
        "| **v6 + LlmRewriter** (hybrid) | **{:.3}** | **{:.3}** | {:.1}s |",
        h, m, e
    );

    Ok(())
}
