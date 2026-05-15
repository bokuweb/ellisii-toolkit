//! Markdown 経路の end-to-end eval (Run 28)。
//!
//! Run 24 で導入した `ChunkConfig::synthesize_caption_from_heading` は Run 25 で
//! eval_caption_synthesis.rs を使って **chunk テキストを直接 2 通りに作って**
//! 計測した。今回は **実際の chunker パスを通して** ingest 時の挙動が同じ結果を
//! 返すかを確認する e2e eval。
//!
//! 各 corpus entry を Markdown 1 block の `ParsedDocument` (heading_path =
//! `[normalize_title(title)]`) に組み立て、`chunker::chunk(doc, ..., cfg)` に
//! synthesize_caption_from_heading 真偽を切り替えて流す。
//!
//! 実行:
//! ```sh
//! cargo run -p ellisii-sdk --example eval_chunker_pipeline_md --release
//! ```
//!
//! `docs/eval/recall-evals.md` Run 28 を参照。

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use async_trait::async_trait;
use ellisii_chunker::{chunk as chunker_chunk, ChunkConfig};
use ellisii_core::{Chunk, Result, SourceKind};
use ellisii_embed_core::Embedder;
use ellisii_parsers_core::{ParsedBlock, ParsedDocument};
use ellisii_rag::eval::{summarize, GoldenSet};
use ellisii_sdk::{Ellisii, SearchOptions};
use serde::Deserialize;
use uuid::Uuid;

#[derive(Debug, Deserialize)]
struct CorpusEntry {
    doc_id: String,
    title: String,
    #[allow(dead_code)]
    caption: String,
    text: String,
}

struct BigramHashEmbedder {
    dim: usize,
}

#[async_trait]
impl Embedder for BigramHashEmbedder {
    fn dim(&self) -> usize {
        self.dim
    }
    async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        Ok(texts.iter().map(|t| bigram_vec(t, self.dim)).collect())
    }
}

fn bigram_vec(text: &str, dim: usize) -> Vec<f32> {
    let mut v = vec![0.0_f32; dim];
    let chars: Vec<char> = text.chars().filter(|c| !c.is_whitespace()).collect();
    if chars.len() < 2 {
        for c in &chars {
            let idx = (fnv1a(&c.to_string()) as usize) % dim;
            v[idx] += 1.0;
        }
        normalize(&mut v);
        return v;
    }
    for w in chars.windows(2) {
        let s: String = w.iter().collect();
        let idx = (fnv1a(&s) as usize) % dim;
        v[idx] += 1.0;
    }
    normalize(&mut v);
    v
}

fn normalize(v: &mut [f32]) {
    let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if n > 0.0 {
        for x in v.iter_mut() {
            *x /= n;
        }
    }
}

fn fnv1a(s: &str) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for b in s.as_bytes() {
        h ^= *b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

/// `(...)` 入りタイトルから括弧を剥がす。`maybe_inject_caption` の skip 条件回避。
fn normalize_title(t: &str) -> String {
    let mut out = String::new();
    let mut depth: i32 = 0;
    for c in t.chars() {
        if c == '(' || c == '(' {
            depth += 1;
        } else if (c == ')' || c == ')') && depth > 0 {
            depth -= 1;
        } else if depth == 0 {
            out.push(c);
        }
    }
    out.trim().to_string()
}

fn fixtures_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .join("rag/tests/fixtures/eval")
}

fn load_fixture(name: &str) -> (Vec<CorpusEntry>, GoldenSet) {
    let dir = fixtures_root().join(name);
    let corpus_json = std::fs::read_to_string(dir.join("corpus.json")).expect("read corpus");
    let corpus: Vec<CorpusEntry> = serde_json::from_str(&corpus_json).expect("parse corpus");
    let golden_json = std::fs::read_to_string(dir.join("golden.json")).expect("read golden");
    let golden = GoldenSet::from_json_str(&golden_json).expect("parse golden");
    (corpus, golden)
}

/// corpus を Markdown 1 block の ParsedDocument の集合に変換し、chunker を通して
/// Chunk リストを得る。`synthesize_caption_from_heading` は cfg で切り替える。
///
/// 戻り値は (chunks, doc_id_by_chunk_id)。doc_id は corpus entry の doc_id を
/// 全 chunk に伝播 (1 entry → 多 chunks の場合あり)。eval は doc_id 単位で
/// relevant 判定を行う設計なので、同じ entry から派生した chunks はどれが
/// hit しても同じ doc_id にマップされる。
fn run_chunker_for_corpus(
    corpus: &[CorpusEntry],
    synthesize: bool,
) -> (Vec<Chunk>, HashMap<Uuid, String>) {
    let cfg = ChunkConfig {
        synthesize_caption_from_heading: synthesize,
        ..Default::default()
    };
    let mut all_chunks: Vec<Chunk> = Vec::new();
    let mut id_map: HashMap<Uuid, String> = HashMap::new();
    for entry in corpus {
        let title = normalize_title(&entry.title);
        let heading_path = if title.is_empty() {
            vec![]
        } else {
            vec![title]
        };
        let doc = ParsedDocument {
            kind: SourceKind::Markdown,
            blocks: vec![ParsedBlock {
                text: entry.text.clone(),
                heading_path,
                page: None,
                bbox: None,
            }],
        };
        let source_id = Uuid::new_v4();
        let chunks = chunker_chunk(&doc, source_id, cfg);
        for c in chunks {
            id_map.insert(c.id, entry.doc_id.clone());
            all_chunks.push(c);
        }
    }
    (all_chunks, id_map)
}

async fn build_ellisii(chunks: &[Chunk]) -> Arc<Ellisii> {
    let dim = 256;
    let nb = Uuid::new_v4();
    let ellisii = Ellisii::builder()
        .with_embedder(Arc::new(BigramHashEmbedder { dim }))
        .with_store_memory()
        .with_notebook_id(nb)
        .build()
        .expect("build ellisii");
    let texts: Vec<String> = chunks.iter().map(|c| c.text.clone()).collect();
    let embedder = BigramHashEmbedder { dim };
    let embs = embedder.embed(&texts).await.expect("embed");
    use ellisii_store_core::VectorStore;
    ellisii
        .store()
        .upsert(nb, chunks, &embs)
        .await
        .expect("upsert");
    Arc::new(ellisii)
}

async fn measure(name: &str, synthesize: bool) -> anyhow::Result<(usize, f32, f32, f32)> {
    let (corpus, golden) = load_fixture(name);
    let (chunks, id_map) = run_chunker_for_corpus(&corpus, synthesize);
    let chunk_count = chunks.len();
    let ellisii = build_ellisii(&chunks).await;
    let density = ellisii.caption_density().await?;
    let mut pairs: Vec<(Vec<String>, Vec<String>)> = Vec::new();
    for item in &golden.items {
        let hits = ellisii
            .search(
                &item.query,
                SearchOptions {
                    top_k: 5,
                    semantic_weight: 0.5,
                    caption_rerank: true,
                    auto_adjust_weight: false,
                    ..Default::default()
                },
            )
            .await?;
        pairs.push((
            hits.iter()
                .filter_map(|h| id_map.get(&h.chunk.id).cloned())
                .collect(),
            item.relevant.clone(),
        ));
    }
    let s = summarize(&pairs, 5);
    let _ = density; // 出力で使う
    Ok((chunk_count, density, s.hit_at_k, s.mrr))
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> anyhow::Result<()> {
    println!("=== Markdown e2e: synthesize_caption_from_heading A/B (Run 28) ===");
    println!(
        "{:<20} {:<8} {:<8} {:<10} {:<10} {:<10} {:<8} {:<8}",
        "corpus", "mode", "chunks", "density", "hit@5", "mrr@5", "Δhit", "Δmrr"
    );
    for corpus in ["jp-cs-wiki", "jp-cs-wiki-hard"] {
        let (n_off, d_off, h_off, m_off) = measure(corpus, false).await?;
        let (n_on, d_on, h_on, m_on) = measure(corpus, true).await?;
        println!(
            "{:<20} {:<8} {:<8} {:<10.3} {:<10.3} {:<10.3}",
            corpus, "off", n_off, d_off, h_off, m_off
        );
        println!(
            "{:<20} {:<8} {:<8} {:<10.3} {:<10.3} {:<10.3} {:+<8.3} {:+<8.3}",
            corpus,
            "on",
            n_on,
            d_on,
            h_on,
            m_on,
            h_on - h_off,
            m_on - m_off,
        );
    }
    Ok(())
}
