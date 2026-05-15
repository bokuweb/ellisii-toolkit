//! `ChunkConfig::synthesize_caption_from_heading` (PR #112) の効果を計測する。
//!
//! Run 24 で導入した opt-in flag を **caption-less corpus** (jp-cs-wiki / -hard)
//! で A/B し、`heading_path[-1]` を疑似 caption として inject すると caption rerank が
//! 効くようになるか、すなわち hit@5 / mrr@5 が伸びるかを確認する。
//!
//! eval_fixtures.rs と違って **チャンクテキストを 2 通りに作り分けて直接 store に入れる**:
//!   - off: text = entry.text                            (caption rerank の対象外)
//!   - on:  text = format!("({title})\n{text}")          (caption rerank が効く)
//!
//! 実行:
//! ```sh
//! cargo run -p ellisii-sdk --example eval_caption_synthesis --release
//! ```
//!
//! `docs/eval/recall-evals.md` Run 25 を参照。

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use async_trait::async_trait;
use ellisii_core::{Chunk, Result};
use ellisii_embed_core::Embedder;
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

/// 文字バイグラムを次元 D にハッシュバケットへ落とす決定的 embedder。
/// `eval_fixtures.rs` と同じ実装。CI で外部モデル無しに動かせる。
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

/// title から括弧書きを取り除き、caption inject の skip 条件に引っかからないようにする
/// (`maybe_inject_caption` は `(` `)` を含む heading を skip する)。
/// 例: "ACID (コンピュータ科学)" → "ACID"
fn normalize_title(t: &str) -> String {
    // 半角 ( ) と全角 ( ) の両方を扱う。
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

/// `synthesize` が true なら本文先頭に `({normalize_title(title)})\n` を prepend する。
/// `false` なら entry.text のまま (= 既存挙動)。
async fn build_ellisii_for_mode(
    corpus: &[CorpusEntry],
    synthesize: bool,
) -> (Arc<Ellisii>, HashMap<Uuid, String>) {
    let dim = 256;
    let nb = Uuid::new_v4();
    let ellisii = Ellisii::builder()
        .with_embedder(Arc::new(BigramHashEmbedder { dim }))
        .with_store_memory()
        .with_notebook_id(nb)
        .build()
        .expect("build ellisii");
    let mut chunks = Vec::new();
    let mut texts = Vec::new();
    let mut id_map: HashMap<Uuid, String> = HashMap::new();
    for (i, e) in corpus.iter().enumerate() {
        let cid = Uuid::new_v4();
        id_map.insert(cid, e.doc_id.clone());
        let text = if synthesize {
            let title = normalize_title(&e.title);
            if title.is_empty() {
                e.text.clone()
            } else {
                format!("({})\n{}", title, e.text)
            }
        } else {
            e.text.clone()
        };
        chunks.push(Chunk {
            id: cid,
            source_id: Uuid::nil(),
            ord: i as u32,
            text: text.clone(),
            heading_path: vec![e.doc_id.clone()],
            page: None,
            bbox: None,
            summary: None,
        });
        texts.push(text);
    }
    let embedder = BigramHashEmbedder { dim };
    let embs = embedder.embed(&texts).await.expect("embed");
    ellisii.store().upsert(nb, &chunks, &embs).await.expect("upsert");
    (Arc::new(ellisii), id_map)
}

async fn measure(
    name: &str,
    synthesize: bool,
) -> anyhow::Result<(f32, f32, f32)> {
    let (corpus, golden) = load_fixture(name);
    let (ellisii, id_map) = build_ellisii_for_mode(&corpus, synthesize).await;
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
            hits.iter().filter_map(|h| id_map.get(&h.chunk.id).cloned()).collect(),
            item.relevant.clone(),
        ));
    }
    let _ = name; // avoid unused warning if println removed
    let s = summarize(&pairs, 5);
    Ok((density, s.hit_at_k, s.mrr))
}

/// クエリ集合の **タイトル直接マッチ度** (Run 26) を計算する。
/// caption synthesis (Run 24) を ON にしたときの利得を予測するシグナル。
fn measure_query_title_match(name: &str) -> f32 {
    let (corpus, golden) = load_fixture(name);
    let titles: Vec<String> = corpus
        .iter()
        .map(|e| normalize_title(&e.title))
        .filter(|t| !t.is_empty())
        .collect();
    let queries: Vec<&str> = golden.items.iter().map(|i| i.query.as_str()).collect();
    ellisii_rag::query_title_match_mean(&queries, &titles)
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> anyhow::Result<()> {
    println!("=== Query→title match signal (Run 26) — all 8 fixture corpora ===");
    println!("{:<20} {:<10}", "corpus", "q_title_match");
    for corpus in [
        "jp-civil-law",
        "jp-civil-law-hard",
        "jp-cs-wiki",
        "jp-cs-wiki-hard",
        "jp-patent-docs",
        "jp-patents",
        "jp-tokkyo-hou",
        "sql-antipatterns",
    ] {
        let s = measure_query_title_match(corpus);
        println!("{:<20} {:<10.3}", corpus, s);
    }

    println!("\n=== caption_synthesis_from_heading A/B (caption_rerank=true, k=5) ===");
    println!(
        "{:<20} {:<8} {:<10} {:<10} {:<10} {:<8} {:<8}",
        "corpus", "mode", "density", "hit@5", "mrr@5", "Δhit", "Δmrr"
    );
    for corpus in ["jp-cs-wiki", "jp-cs-wiki-hard"] {
        let (d_off, h_off, m_off) = measure(corpus, false).await?;
        let (d_on, h_on, m_on) = measure(corpus, true).await?;
        println!(
            "{:<20} {:<8} {:<10.3} {:<10.3} {:<10.3}",
            corpus, "off", d_off, h_off, m_off
        );
        println!(
            "{:<20} {:<8} {:<10.3} {:<10.3} {:<10.3} {:+<8.3} {:+<8.3}",
            corpus,
            "on",
            d_on,
            h_on,
            m_on,
            h_on - h_off,
            m_on - m_off
        );
    }
    Ok(())
}
