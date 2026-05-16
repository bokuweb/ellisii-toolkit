//! `eval_workplace_regs` の FTS5 トークナイザ A/B。
//!
//! `store-sqlite` の FTS5 トークナイザを bigram (デフォルト) と vaporetto
//! (形態素) で切り替え、jp-workplace-regs golden Q&A の recall がどう動くかを
//! 計測する。bigram は「法定休日」を `法定|定休|休日` に分解して「休日」系
//! 条文と過剰マッチしがちなので、形態素にすることで失敗 2 件 (法定休日 /
//! 出張中の労働時間) を救えるかを見たい。
//!
//! 実行 (vaporetto モデル必須):
//! ```sh
//! cargo run -p ellisii-sdk \
//!   --features static-jp \
//!   --example eval_workplace_regs_tok --release
//! ```
//!
//! 結果は `docs/eval/recall-evals.md` jp-workplace-regs セクションに
//! 「Run 4 (FTS5 tokenizer A/B)」として追記する。

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use ellisii_core::Chunk;
use ellisii_jp_tokenizer_bigram::CharBigramTokenizer;
use ellisii_jp_tokenizer_core::JpTokenizer;
use ellisii_rag::eval::{summarize, GoldenSet};
use ellisii_sdk::{Ellisii, SearchOptions};
use ellisii_store_core::VectorStore;
use ellisii_store_sqlite::SqliteStore;
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
fn vaporetto_model() -> PathBuf {
    home().join(
        "Library/Application Support/ellisii/models/vaporetto/bccwj-suw+unidic_pos+pron.model.zst",
    )
}
fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .join("rag/tests/fixtures/eval/jp-workplace-regs")
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> anyhow::Result<()> {
    #[cfg(not(feature = "static-jp"))]
    {
        anyhow::bail!("build with --features static-jp");
    }
    #[cfg(feature = "static-jp")]
    return run().await;
}

#[cfg(feature = "static-jp")]
async fn run() -> anyhow::Result<()> {
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

    let embed = embed_dir();
    eprintln!("embed: {}", embed.display());

    let dim = 1024;
    let bigram: Arc<dyn JpTokenizer> = Arc::new(CharBigramTokenizer::new());
    let store_bigram = Arc::new(SqliteStore::open_in_memory_with_tokenizer(dim, bigram)?);

    let vap_path = vaporetto_model();
    let store_vap: Option<Arc<dyn VectorStore>> = {
        #[cfg(feature = "vaporetto")]
        {
            if vap_path.is_file() {
                use ellisii_jp_tokenizer_vaporetto::VaporettoTokenizer;
                let v = VaporettoTokenizer::from_zst(&vap_path)
                    .map_err(|e| anyhow::anyhow!("load vaporetto: {e}"))?;
                let tok: Arc<dyn JpTokenizer> = Arc::new(v);
                Some(Arc::new(SqliteStore::open_in_memory_with_tokenizer(
                    dim, tok,
                )?))
            } else {
                eprintln!("[skip vaporetto] model not found at {}", vap_path.display());
                None
            }
        }
        #[cfg(not(feature = "vaporetto"))]
        {
            eprintln!(
                "[skip vaporetto] build with --features vaporetto to enable (model: {})",
                vap_path.display()
            );
            None
        }
    };

    let build = |store: Arc<dyn VectorStore>| -> anyhow::Result<Ellisii> {
        Ok(Ellisii::builder()
            .with_embedder_static_jp(&embed)?
            .with_store(store)
            .with_notebook_id(nb)
            .build()?)
    };

    let ellisii_bigram = build(store_bigram.clone())?;
    let embs = ellisii_bigram.embedder().embed(&texts).await?;
    store_bigram.upsert(nb, &chunks, &embs).await?;
    if let Some(s) = &store_vap {
        s.upsert(nb, &chunks, &embs).await?;
    }

    println!("\n=== jp-workplace-regs: FTS5 tokenizer A/B (Run 4, k=5) ===");
    println!(
        "{:<28} {:<10} {:<10} {:<10} {:<10}",
        "variant", "recall", "hit", "ndcg", "mrr"
    );

    let weights = [0.0_f32, 0.25, 0.5, 0.75, 1.0];
    for &w in &weights {
        let pairs = run_pairs(&ellisii_bigram, &gold, &id_map, w, true, 5).await?;
        let s = summarize(&pairs, 5);
        println!(
            "{:<28} {:<10.3} {:<10.3} {:<10.3} {:<10.3}",
            format!("bigram cap w={:.2}", w),
            s.recall_at_k,
            s.hit_at_k,
            s.ndcg_at_k,
            s.mrr
        );
        if let Some(s_vap) = &store_vap {
            let ellisii_vap = build(s_vap.clone())?;
            let pairs = run_pairs(&ellisii_vap, &gold, &id_map, w, true, 5).await?;
            let s = summarize(&pairs, 5);
            println!(
                "{:<28} {:<10.3} {:<10.3} {:<10.3} {:<10.3}",
                format!("vap    cap w={:.2}", w),
                s.recall_at_k,
                s.hit_at_k,
                s.ndcg_at_k,
                s.mrr
            );
        }
    }

    // Targeted failures
    if let Some(s_vap) = &store_vap {
        println!("\n=== Targeted failures @ k=5 (weight=0.5, cap=on) ===");
        let ellisii_vap = build(s_vap.clone())?;
        for q in ["法定休日は何曜日", "出張中の労働時間はどう扱われるか"] {
            let item = gold.items.iter().find(|i| i.query == q).unwrap();
            let hits_b = ellisii_bigram.search(q, opts(0.5, 5)).await?;
            let hits_v = ellisii_vap.search(q, opts(0.5, 5)).await?;
            let to_ids = |hs: &[ellisii_core::SearchHit]| -> Vec<String> {
                hs.iter()
                    .filter_map(|h| id_map.get(&h.chunk.id).cloned())
                    .collect()
            };
            println!("  {}  expected={:?}", q, item.relevant);
            println!("    bigram top5={:?}", to_ids(&hits_b));
            println!("    vap    top5={:?}", to_ids(&hits_v));
        }
    }

    Ok(())
}

#[cfg(feature = "static-jp")]
fn opts(semantic_weight: f32, top_k: usize) -> SearchOptions {
    SearchOptions {
        top_k,
        semantic_weight,
        caption_rerank: true,
        ..Default::default()
    }
}

#[cfg(feature = "static-jp")]
async fn run_pairs(
    ellisii: &Ellisii,
    gold: &GoldenSet,
    id_map: &HashMap<Uuid, String>,
    semantic_weight: f32,
    caption_rerank: bool,
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
                    caption_rerank,
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
