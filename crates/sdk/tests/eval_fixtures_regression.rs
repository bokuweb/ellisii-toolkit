//! Fixture-based RAG regression guard. CI-runnable (no external models).
//!
//! `crates/rag/tests/fixtures/eval/<corpus>/{corpus.json, golden.json}` を
//! 8 corpora ぶん回し、`caption_rerank=true` モードでの **hit@5 / MRR@5** が
//! 既知のフロアを下回ったら fail する。`eval_yokohama_regression` の fixture 版で、
//! あちらは外部 DB / static-jp モデルが要るため `#[ignore]` だが、こちらは
//! BigramHashEmbedder + InMemoryStore で完結するため CI に組み込める。
//!
//! 動機: PR #99 の caption IDF / PR #101 caption fallback / PR #105 skip CE /
//! PR #106 paraphrase score などで rerank 経路に手を入れているので、サイレント
//! 退行を防ぐ pre-merge ゲートが要る。`docs/eval/recall-evals.md` Run 23 参照。
//!
//! フロアは現行 main (PR #110 merge 後) で計測した値から **-3 pp 緩和**。
//! 真の retrieval 退行 (caption boost が壊れる、Jaccard floor が外れる、
//! IDF 計算が逆転する など) は確実に -5 pp 以上 動くので、3 pp 緩和は
//! flaky 退避に十分。

use async_trait::async_trait;
use ellisii_core::{Chunk, Result};
use ellisii_embed_core::Embedder;
use ellisii_rag::eval::{summarize, GoldenSet};
use ellisii_sdk::{Ellisii, SearchOptions};
use ellisii_store_core::VectorStore as _;
use serde::Deserialize;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use uuid::Uuid;

#[derive(Debug, Deserialize)]
struct CorpusEntry {
    doc_id: String,
    #[allow(dead_code)]
    title: String,
    caption: String,
    text: String,
}

/// 文字バイグラムを次元 D にハッシュバケットへ落とす決定的 embedder。
/// `eval_fixtures.rs` と同じ実装で、CI で外部モデル無しに動かせる。
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

fn fixtures_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .join("rag/tests/fixtures/eval")
}

fn load_fixture(name: &str) -> (Vec<CorpusEntry>, GoldenSet) {
    let dir = fixtures_root().join(name);
    let corpus_json = std::fs::read_to_string(dir.join("corpus.json"))
        .unwrap_or_else(|_| panic!("read {name}/corpus.json"));
    let corpus: Vec<CorpusEntry> = serde_json::from_str(&corpus_json)
        .unwrap_or_else(|e| panic!("parse {name}/corpus.json: {e}"));
    let golden_json = std::fs::read_to_string(dir.join("golden.json"))
        .unwrap_or_else(|_| panic!("read {name}/golden.json"));
    let golden = GoldenSet::from_json_str(&golden_json)
        .unwrap_or_else(|e| panic!("parse {name}/golden.json: {e}"));
    (corpus, golden)
}

async fn build_ellisii(corpus: &[CorpusEntry]) -> (Arc<Ellisii>, HashMap<Uuid, String>) {
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
        let txt = if e.caption.is_empty() {
            e.text.clone()
        } else {
            format!("({})\n{}", e.caption, e.text)
        };
        chunks.push(Chunk {
            id: cid,
            source_id: Uuid::nil(),
            ord: i as u32,
            text: txt.clone(),
            heading_path: vec![e.doc_id.clone()],
            page: None,
            bbox: None,
            summary: None,
        });
        texts.push(txt);
    }
    let embedder = BigramHashEmbedder { dim };
    let embs = embedder.embed(&texts).await.unwrap();
    ellisii
        .store()
        .upsert(nb, &chunks, &embs)
        .await
        .expect("upsert chunks");
    (Arc::new(ellisii), id_map)
}

async fn measure_hit_mrr_at_5(
    name: &str,
) -> (f32, f32) {
    let (corpus, golden) = load_fixture(name);
    let (ellisii, id_map) = build_ellisii(&corpus).await;
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
            .await
            .expect("search");
        pairs.push((
            hits.iter().filter_map(|h| id_map.get(&h.chunk.id).cloned()).collect(),
            item.relevant.clone(),
        ));
    }
    let s = summarize(&pairs, 5);
    (s.hit_at_k, s.mrr)
}

/// 各 corpus の (hit@5 floor, mrr@5 floor)。現行 main で計測した値 -3pp 緩和。
/// この値を 1 つでも下回ったら本テストは fail する。
const FLOORS: &[(&str, f32, f32)] = &[
    ("jp-civil-law", 0.85, 0.72),
    ("jp-civil-law-hard", 0.36, 0.24),
    ("jp-cs-wiki", 0.87, 0.74),
    ("jp-cs-wiki-hard", 0.62, 0.45),
    ("jp-labor-law", 0.87, 0.71),
    // jp-multihop v2 (Run 46) は規則側 doc から定義側語彙を抜き、等級コード参照に
    // した hard 版。BigramHash + caption rerank で hit@5=0.810, mrr@5=0.786 を計測。
    // floor は -3pp。
    ("jp-multihop", 0.78, 0.75),
    // jp-faq (Run 63 で追加): customer support FAQ 形式の Q-A pair corpus。
    // BigramHash + caption rerank で hit@5=0.682, mrr@5=0.442 → -3pp 余裕。
    ("jp-faq", 0.65, 0.41),
    ("jp-patent-docs", 0.86, 0.53),
    ("jp-patents", 0.69, 0.50),
    ("jp-tokkyo-hou", 0.68, 0.58),
    ("sql-antipatterns", 0.72, 0.58),
];

#[tokio::test]
async fn fixture_recall_does_not_regress() {
    // 全 9 corpus を回し、フロアを下回るものをまとめて報告する。
    // 1 つだけ落ちた場合でも、他の corpus への影響を見たいので最後にまとめて assert。
    let mut failures: Vec<String> = Vec::new();
    for (name, hit_floor, mrr_floor) in FLOORS {
        let (hit, mrr) = measure_hit_mrr_at_5(name).await;
        eprintln!(
            "{:<28} hit@5={:.3} (floor={:.2})  mrr@5={:.3} (floor={:.2})",
            name, hit, hit_floor, mrr, mrr_floor
        );
        if hit + 1e-3 < *hit_floor {
            failures.push(format!(
                "{name}: hit@5={hit:.3} below floor {hit_floor:.2}"
            ));
        }
        if mrr + 1e-3 < *mrr_floor {
            failures.push(format!("{name}: mrr@5={mrr:.3} below floor {mrr_floor:.2}"));
        }
    }
    assert!(
        failures.is_empty(),
        "fixture eval regressions detected:\n  - {}",
        failures.join("\n  - ")
    );
}
