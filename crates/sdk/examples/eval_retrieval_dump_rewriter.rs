//! `eval_retrieval_dump` の **LlmRewriter (gemma-4-E4B) 拡張版**。
//!
//! Run 12a で「civil-law-hard 残 4 件は paraphrase gap (シナリオ文 → 法律
//! ターム) で retrieval miss」と確定した。tokenizer 改善 (bigram → delarocha)
//! では橋渡しできないクラスなので、**LlmRewriter で query を法律ターム化
//! してから retrieve すれば top-5 に拾えるはず** という仮説を実測する。
//!
//! 比較対象:
//! - bigram         (no rewriter)
//! - bigram         + LlmRewriter (gemma-4-E4B, multi_query_max_variants=3)
//! - delarocha+NFKC (no rewriter)
//! - delarocha+NFKC + LlmRewriter
//!
//! 使い方 (gemma-4-E4B + delarocha 必須):
//! ```sh
//! ELLISII_EVAL_FIXTURE=jp-civil-law-hard \
//!   ELLISII_EVAL_QUERIES='q1|q2|...' \
//!   cargo run -p ellisii-sdk \
//!     --features static-jp,llamacpp,delarocha \
//!     --example eval_retrieval_dump_rewriter --release
//! ```

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use ellisii_core::Chunk;
use ellisii_jp_tokenizer_bigram::CharBigramTokenizer;
use ellisii_jp_tokenizer_core::JpTokenizer;
use ellisii_jp_tokenizer_nfkc::NfkcTokenizer;
use ellisii_rag::eval::GoldenSet;
use ellisii_sdk::{Ellisii, SearchOptions};
use ellisii_store_core::VectorStore;
use ellisii_store_sqlite::SqliteStore;
use serde::Deserialize;
use uuid::Uuid;

#[derive(Debug, Deserialize)]
struct CorpusEntry {
    doc_id: String,
    #[serde(default)]
    #[allow(dead_code)]
    parent_id: Option<String>,
    #[serde(default)]
    #[allow(dead_code)]
    title: String,
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
fn delarocha_dict() -> PathBuf {
    home().join("Library/Application Support/ellisii/models/delarocha/system.dic.zst")
}
fn gemma_e4b() -> PathBuf {
    home().join("Library/Application Support/ellisii/models/gemma-4-E4B-it-IQ4_XS.gguf")
}
fn fixture_dir() -> PathBuf {
    let name =
        std::env::var("ELLISII_EVAL_FIXTURE").unwrap_or_else(|_| "jp-civil-law-hard".to_string());
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .join("rag/tests/fixtures/eval")
        .join(name)
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> anyhow::Result<()> {
    #[cfg(not(all(feature = "static-jp", feature = "llamacpp", feature = "delarocha")))]
    {
        anyhow::bail!("build with --features static-jp,llamacpp,delarocha");
    }
    #[cfg(all(feature = "static-jp", feature = "llamacpp", feature = "delarocha"))]
    return run().await;
}

#[cfg(all(feature = "static-jp", feature = "llamacpp", feature = "delarocha"))]
async fn run() -> anyhow::Result<()> {
    use async_trait::async_trait;
    use ellisii_jp_tokenizer_delarocha::DelarochaTokenizer;
    use ellisii_llm_core::{LlmBackend, ModelFamily};
    use ellisii_llm_llamacpp::{LlamaConfig, LlamaCppBackend};
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
    eprintln!(
        "fixture: {}  corpus: {} chunks",
        dir.display(),
        corpus.len()
    );

    let queries: Vec<String> = if let Ok(qs) = std::env::var("ELLISII_EVAL_QUERIES") {
        qs.split('|').map(|s| s.trim().to_string()).collect()
    } else {
        let gold: GoldenSet =
            GoldenSet::from_json_str(&std::fs::read_to_string(dir.join("golden.json"))?)?;
        gold.items.iter().map(|i| i.query.clone()).collect()
    };
    eprintln!("queries: {}", queries.len());

    let nb = Uuid::new_v4();
    let src = Uuid::new_v4();
    let mut chunks: Vec<Chunk> = Vec::with_capacity(corpus.len());
    let mut texts: Vec<String> = Vec::with_capacity(corpus.len());
    let mut by_doc: HashMap<String, (String, String)> = HashMap::new();
    let mut id_map: HashMap<Uuid, String> = HashMap::new();
    for (i, e) in corpus.iter().enumerate() {
        let cid = Uuid::new_v4();
        id_map.insert(cid, e.doc_id.clone());
        by_doc.insert(e.doc_id.clone(), (e.caption.clone(), e.text.clone()));
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

    let embed = embed_dir();
    let dim = 1024;
    let bigram: Arc<dyn JpTokenizer> = Arc::new(CharBigramTokenizer::new());
    let dela_nfkc: Arc<dyn JpTokenizer> = Arc::new(NfkcTokenizer::new(Arc::new(
        DelarochaTokenizer::from_path(&delarocha_dict())
            .map_err(|e| anyhow::anyhow!("load delarocha: {e}"))?,
    )));
    let store_bigram: Arc<dyn VectorStore> =
        Arc::new(SqliteStore::open_in_memory_with_tokenizer(dim, bigram)?);
    let store_dela: Arc<dyn VectorStore> =
        Arc::new(SqliteStore::open_in_memory_with_tokenizer(dim, dela_nfkc)?);

    let gguf = gemma_e4b();
    if !gguf.is_file() {
        anyhow::bail!("gemma-4-E4B-it-IQ4_XS.gguf not found");
    }
    eprintln!("LLM: {}", gguf.display());
    let cfg = LlamaConfig::new(gguf, ModelFamily::Gemma4);
    let llm: Arc<dyn LlmBackend> =
        Arc::new(LlamaCppBackend::load(cfg).map_err(|e| anyhow::anyhow!("load gemma: {e}"))?);
    let rewriter = Arc::new(LlmRewriter::new(SharedLlm(Arc::clone(&llm))));

    // 4 variants of Ellisii. Same store, but rewriter ON/OFF.
    let bigram_engine = Ellisii::builder()
        .with_embedder_static_jp(&embed)?
        .with_store(store_bigram.clone())
        .with_notebook_id(nb)
        .build()?;
    let embs = bigram_engine.embedder().embed(&texts).await?;
    store_bigram.upsert(nb, &chunks, &embs).await?;
    store_dela.upsert(nb, &chunks, &embs).await?;

    let dela_engine = Ellisii::builder()
        .with_embedder_static_jp(&embed)?
        .with_store(store_dela.clone())
        .with_notebook_id(nb)
        .build()?;
    let bigram_rw = Ellisii::builder()
        .with_embedder_static_jp(&embed)?
        .with_store(store_bigram.clone())
        .with_query_rewriter(rewriter.clone())
        .with_notebook_id(nb)
        .build()?;
    let dela_rw = Ellisii::builder()
        .with_embedder_static_jp(&embed)?
        .with_store(store_dela.clone())
        .with_query_rewriter(rewriter.clone())
        .with_notebook_id(nb)
        .build()?;

    let opts_plain = SearchOptions {
        top_k: 5,
        semantic_weight: 0.5,
        caption_rerank: true,
        ..Default::default()
    };
    let opts_rw = SearchOptions {
        top_k: 5,
        semantic_weight: 0.5,
        caption_rerank: true,
        multi_query_max_variants: 3,
        multi_query_variant_weight: 0.7,
        skip_rewrite_on_specific: false,
        ..Default::default()
    };

    for q in &queries {
        println!("\n========================================");
        println!("query: {}", q);
        let runs: &[(&str, &Ellisii, &SearchOptions)] = &[
            ("bigram        ", &bigram_engine, &opts_plain),
            ("bigram   + RW ", &bigram_rw, &opts_rw),
            ("delarocha     ", &dela_engine, &opts_plain),
            ("delarocha+ RW ", &dela_rw, &opts_rw),
        ];
        for (label, eng, opts) in runs {
            let hits = eng.search(q, (*opts).clone()).await?;
            println!("[{}] top-5:", label);
            for (rank, h) in hits.iter().enumerate() {
                let did = id_map.get(&h.chunk.id).cloned().unwrap_or_default();
                let (cap, body) = by_doc.get(&did).cloned().unwrap_or_default();
                let preview: String = body.chars().take(60).collect();
                println!(
                    "  #{} {} | ({}) | {}{}",
                    rank + 1,
                    did,
                    cap,
                    preview,
                    if body.chars().count() > preview.chars().count() {
                        "..."
                    } else {
                        ""
                    }
                );
            }
        }
    }

    Ok(())
}
