//! Failing query を渡すと top-5 retrieval hits (doc_id + caption + body
//! preview) を dump する診断ハーネス。Run 11 で「残 4 件は retrieval は
//! 当たっているはず」と推定したが、推定ではなく実測でクラス分け (retrieval
//! が外している vs LLM 出力の質的問題) する。
//!
//! 既存 `eval_answer_tokenizer_facade.rs` と同じく bigram と delarocha+NFKC
//! の 2 store で top-5 を出すので、どちらが当たっているかも分かる。LLM は
//! 呼ばないので速い (1 query <1s)。
//!
//! 使い方:
//! ```sh
//! ELLISII_EVAL_FIXTURE=jp-civil-law-hard \
//!   ELLISII_EVAL_QUERIES='knowingly comma sep query list' \
//!   cargo run -p ellisii-sdk --features static-jp,delarocha \
//!     --example eval_retrieval_dump --release
//! ```
//!
//! `ELLISII_EVAL_QUERIES` 未指定なら fixture の `golden.json` 全件を dump
//! する。指定時は `|` (vertical bar) 区切り。

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
    #[cfg(not(all(feature = "static-jp", feature = "delarocha")))]
    {
        anyhow::bail!("build with --features static-jp,delarocha");
    }
    #[cfg(all(feature = "static-jp", feature = "delarocha"))]
    return run().await;
}

#[cfg(all(feature = "static-jp", feature = "delarocha"))]
async fn run() -> anyhow::Result<()> {
    use ellisii_jp_tokenizer_delarocha::DelarochaTokenizer;

    let dir = fixture_dir();
    let corpus: Vec<CorpusEntry> =
        serde_json::from_str(&std::fs::read_to_string(dir.join("corpus.json"))?)?;
    eprintln!(
        "fixture: {}\ncorpus: {} chunks",
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
    let mut by_doc: HashMap<String, (String, String)> = HashMap::new(); // doc_id -> (caption, body)
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

    for q in &queries {
        println!("\n========================================");
        println!("query: {}", q);
        for (label, eng) in [("bigram   ", &bigram_engine), ("delarocha", &dela_engine)] {
            let hits = eng
                .search(
                    q,
                    SearchOptions {
                        top_k: 5,
                        semantic_weight: 0.5,
                        caption_rerank: true,
                        ..Default::default()
                    },
                )
                .await?;
            println!("[{}] top-5:", label);
            for (rank, h) in hits.iter().enumerate() {
                let did = id_map.get(&h.chunk.id).cloned().unwrap_or_default();
                let (cap, body) = by_doc.get(&did).cloned().unwrap_or_default();
                let preview: String = body.chars().take(80).collect();
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
