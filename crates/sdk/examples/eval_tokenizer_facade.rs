//! Cross-corpus tokenizer facade verification.
//!
//! 新しく追加した `with_store_sqlite_with_tokenizer` / `with_store_sqlite_nfkc`
//! / `with_store_sqlite_delarocha` の 3 facade を、複数の golden corpus で
//! 同じ条件 (cap rerank=on, hybrid weights を sweep) で A/B 計測する。
//! jp-workplace-regs Run 6 で出した「sparse-only でしか効かない」傾向が他
//! コーパスでも成立するかを横展開する。
//!
//! 使い方:
//! ```sh
//! ELLISII_EVAL_FIXTURE=jp-civil-law-hard \
//!   cargo run -p ellisii-sdk --features static-jp,delarocha \
//!     --example eval_tokenizer_facade --release
//! ```
//!
//! `ELLISII_EVAL_FIXTURE` には `crates/rag/tests/fixtures/eval/<name>` の
//! `<name>` を渡す (例: `jp-civil-law-hard`, `jp-cs-wiki-hard`,
//! `jp-tokkyo-hou`, `jp-labor-law`, `jp-workplace-regs`).

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use ellisii_core::Chunk;
use ellisii_jp_tokenizer_bigram::CharBigramTokenizer;
use ellisii_jp_tokenizer_core::JpTokenizer;
use ellisii_jp_tokenizer_nfkc::NfkcTokenizer;
use ellisii_rag::eval::{summarize, GoldenSet};
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
        std::env::var("ELLISII_EVAL_FIXTURE").unwrap_or_else(|_| "jp-workplace-regs".to_string());
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
    let gold: GoldenSet =
        GoldenSet::from_json_str(&std::fs::read_to_string(dir.join("golden.json"))?)?;
    eprintln!(
        "fixture: {}\ncorpus:  {} chunks\ngolden:  {} ({} items)",
        dir.display(),
        corpus.len(),
        gold.name,
        gold.items.len(),
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

    let embed = embed_dir();
    let dela_path = delarocha_dict();
    let dim = 1024;
    eprintln!("embed:    {}", embed.display());
    eprintln!("delarocha:{}", dela_path.display());

    let bigram: Arc<dyn JpTokenizer> = Arc::new(CharBigramTokenizer::new());
    let nfkc: Arc<dyn JpTokenizer> =
        Arc::new(NfkcTokenizer::new(Arc::new(CharBigramTokenizer::new())));
    let dela: Arc<dyn JpTokenizer> = Arc::new(
        DelarochaTokenizer::from_path(&dela_path)
            .map_err(|e| anyhow::anyhow!("load delarocha: {e}"))?,
    );
    let nfkc_dela: Arc<dyn JpTokenizer> = Arc::new(NfkcTokenizer::new(Arc::new(
        DelarochaTokenizer::from_path(&dela_path)
            .map_err(|e| anyhow::anyhow!("load delarocha: {e}"))?,
    )));

    let stores: Vec<(&str, Arc<dyn VectorStore>)> = vec![
        (
            "bigram   (baseline)",
            Arc::new(SqliteStore::open_in_memory_with_tokenizer(dim, bigram)?),
        ),
        (
            "bigram+NFKC",
            Arc::new(SqliteStore::open_in_memory_with_tokenizer(dim, nfkc)?),
        ),
        (
            "delarocha",
            Arc::new(SqliteStore::open_in_memory_with_tokenizer(dim, dela)?),
        ),
        (
            "delarocha+NFKC",
            Arc::new(SqliteStore::open_in_memory_with_tokenizer(dim, nfkc_dela)?),
        ),
    ];

    let mut elliss: Vec<(&str, Ellisii)> = Vec::with_capacity(stores.len());
    let mut embs: Option<Vec<Vec<f32>>> = None;
    for (label, store) in &stores {
        let e = Ellisii::builder()
            .with_embedder_static_jp(&embed)?
            .with_store(store.clone())
            .with_notebook_id(nb)
            .build()?;
        let v = if let Some(v) = &embs {
            v.clone()
        } else {
            let v = e.embedder().embed(&texts).await?;
            embs = Some(v.clone());
            v
        };
        store.upsert(nb, &chunks, &v).await?;
        elliss.push((label, e));
    }

    println!(
        "\n=== Cross-corpus tokenizer A/B  fixture={}  (k=5, cap=on) ===",
        std::env::var("ELLISII_EVAL_FIXTURE").unwrap_or_else(|_| "jp-workplace-regs".into())
    );
    println!(
        "{:<22} {:>6} {:>8} {:>8} {:>8} {:>8}",
        "variant", "w", "recall", "hit", "ndcg", "mrr"
    );

    for (label, eng) in &elliss {
        for &w in &[0.0_f32, 0.5, 1.0] {
            let pairs = run_pairs(eng, &gold, &id_map, w, 5).await?;
            let s = summarize(&pairs, 5);
            println!(
                "{:<22} {:>6.2} {:>8.3} {:>8.3} {:>8.3} {:>8.3}",
                label, w, s.recall_at_k, s.hit_at_k, s.ndcg_at_k, s.mrr
            );
        }
    }

    Ok(())
}

#[cfg(all(feature = "static-jp", feature = "delarocha"))]
async fn run_pairs(
    ellisii: &Ellisii,
    gold: &GoldenSet,
    id_map: &HashMap<Uuid, String>,
    semantic_weight: f32,
    k: usize,
) -> ellisii_core::Result<Vec<(Vec<String>, Vec<String>)>> {
    let mut pairs = Vec::with_capacity(gold.items.len());
    for item in &gold.items {
        let hits = ellisii
            .search(
                &item.query,
                SearchOptions {
                    top_k: k,
                    semantic_weight,
                    caption_rerank: true,
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
