//! Hard golden で baseline がどの query で失敗してるかを覗く診断テスト。
//!
//! `#[ignore]`。
//! `cargo test -p ellisii-rag-eval-cli --test diagnose_civil_law_hard -- --ignored --nocapture`

use ellisii_core::{Chunk, SearchHit};
use ellisii_embed_core::Embedder;
use ellisii_llm_stub::EchoLlm;
use ellisii_rag::{eval::GoldenSet, HybridWeights, RagEngine};
use ellisii_rag_eval_cli::{CorpusEntry, EmbedderKind};
use ellisii_store_core::VectorStore;
use ellisii_store_sqlite::SqliteStore;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use uuid::Uuid;

fn locate_static_jp() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("ELLISII_STATIC_JP_DIR") {
        let pb = PathBuf::from(p);
        if pb.is_dir() {
            return Some(pb);
        }
    }
    if let Some(home) = std::env::var_os("HOME") {
        let mac = PathBuf::from(&home)
            .join("Library/Application Support/ellisii/models/static-embedding-japanese");
        if mac.is_dir() {
            return Some(mac);
        }
    }
    None
}

fn load_fixture(domain: &str) -> (Vec<CorpusEntry>, GoldenSet) {
    let base = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .join("rag/tests/fixtures/eval")
        .join(domain);
    let corpus: Vec<CorpusEntry> =
        serde_json::from_str(&std::fs::read_to_string(base.join("corpus.json")).unwrap()).unwrap();
    let golden: GoldenSet =
        serde_json::from_str(&std::fs::read_to_string(base.join("golden.json")).unwrap()).unwrap();
    (corpus, golden)
}

struct DynEmb(Arc<dyn Embedder>);
#[async_trait::async_trait]
impl Embedder for DynEmb {
    fn dim(&self) -> usize {
        self.0.dim()
    }
    async fn embed(&self, texts: &[String]) -> ellisii_core::Result<Vec<Vec<f32>>> {
        self.0.embed(texts).await
    }
}

#[tokio::test]
#[ignore]
async fn diagnose_failures() {
    let static_jp = locate_static_jp().expect("static-jp model not present");
    let (corpus, golden) = load_fixture("jp-civil-law-hard");
    let embedder = EmbedderKind::StaticJp { model_dir: static_jp }.build().unwrap();
    let dim = embedder.dim();

    // Build store
    let store = SqliteStore::open_in_memory(dim).unwrap();
    let nb = Uuid::new_v4();
    let mut chunks = Vec::new();
    let mut texts = Vec::new();
    let mut id_map: HashMap<Uuid, String> = HashMap::new();
    for (ord, e) in corpus.iter().enumerate() {
        let cid = Uuid::new_v4();
        id_map.insert(cid, e.doc_id.clone());
        let body = if e.caption.is_empty() && e.title.is_empty() {
            e.text.clone()
        } else {
            format!("{} {} {}", e.title, e.caption, e.text)
        };
        chunks.push(Chunk {
            id: cid,
            source_id: Uuid::new_v4(),
            ord: ord as u32,
            text: body.clone(),
            heading_path: vec![e.doc_id.clone()],
            page: None,
            bbox: None,
            summary: None,
        });
        texts.push(body);
    }
    let dyn_emb = DynEmb(embedder);
    let embs = dyn_emb.embed(&texts).await.unwrap();
    store.upsert(nb, &chunks, &embs).await.unwrap();
    let engine = RagEngine {
        embedder: dyn_emb,
        store,
        llm: EchoLlm,
    };

    println!("\n=== diagnostic: jp-civil-law-hard, semantic=0.75, k=10 ===");
    let mut failures = 0;
    let mut rank_distribution: Vec<usize> = Vec::new();
    for item in &golden.items {
        let hits: Vec<SearchHit> = engine
            .retrieve_weighted(Some(nb), &item.query, 10, HybridWeights { semantic: 0.75 })
            .await
            .unwrap();
        let predicted: Vec<String> = hits
            .iter()
            .filter_map(|h| id_map.get(&h.chunk.id).cloned())
            .collect();
        let want: &str = &item.relevant[0];
        let pos = predicted.iter().position(|p| p == want);
        match pos {
            Some(r) => {
                rank_distribution.push(r + 1);
                if r >= 3 {
                    println!("  [rank {}] {} (want {}) → {:?}", r + 1, item.query, want, &predicted[..5.min(predicted.len())]);
                }
            }
            None => {
                failures += 1;
                println!("  [MISS]  {} (want {}) → {:?}", item.query, want, &predicted[..5.min(predicted.len())]);
            }
        }
    }
    println!("\n  hit ranks: {:?}", rank_distribution);
    println!("  total queries: {}", golden.items.len());
    println!("  recall@10 misses: {}", failures);
    let mut counts = [0usize; 11];
    for r in &rank_distribution {
        if *r <= 10 {
            counts[*r] += 1;
        }
    }
    println!("  rank histogram:");
    for r in 1..=10 {
        if counts[r] > 0 {
            println!("    rank {:2}: {} {}", r, "█".repeat(counts[r]), counts[r]);
        }
    }
}
