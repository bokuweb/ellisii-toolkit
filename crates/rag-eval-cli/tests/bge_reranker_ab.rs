//! bge-reranker-v2-m3 vs open-provence-xsmall の A/B 計測。
//!
//! `crates/provence-onnx` は HF cross-encoder の汎用 ONNX runner なので、
//! `tokenizer.json + model.onnx (input_ids/attention_mask → logits)` の
//! 仕様を満たすモデルなら open-provence と bge-reranker 両方を同じ infra で
//! ロードできる。差し替えは「モデルディレクトリ」を切り替えるだけ。
//!
//! 既定で `#[ignore]`、両モデル + static-jp embedder が揃ったローカル環境で
//! 手動実行する:
//!
//! ```sh
//! cargo test -p ellisii-rag-eval-cli --test bge_reranker_ab \
//!     -- --ignored --nocapture
//! ```
//!
//! モデル配置 (既定パス):
//! - `~/Library/Application Support/ellisii/models/open-provence/{tokenizer.json,model.onnx}`
//! - `~/Library/Application Support/ellisii/models/bge-reranker-v2-m3/{tokenizer.json,model.onnx}`
//! - `~/Library/Application Support/ellisii/models/static-embedding-japanese/`
//!
//! bge-reranker-v2-m3 の fetch は [`scripts/fetch-bge-reranker.sh`](../../../scripts/fetch-bge-reranker.sh)
//! を使う (HuggingFace の `BAAI/bge-reranker-v2-m3` から ONNX を取得)。

use ellisii_core::{Chunk, SearchHit};
use ellisii_embed_core::Embedder;
use ellisii_llm_stub::EchoLlm;
use ellisii_provence_core::ContextCompressor;
use ellisii_provence_onnx::{ProvenceConfig, ProvenceOnnx};
use ellisii_rag::{
    eval::{summarize, GoldenSet},
    HybridWeights, RagEngine,
};
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

fn locate_open_provence() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("ELLISII_OPEN_PROVENCE_DIR") {
        let pb = PathBuf::from(p);
        if pb.is_dir() {
            return Some(pb);
        }
    }
    if let Some(home) = std::env::var_os("HOME") {
        let mac =
            PathBuf::from(&home).join("Library/Application Support/ellisii/models/open-provence");
        if mac.is_dir() {
            return Some(mac);
        }
    }
    None
}

fn locate_bge_reranker() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("ELLISII_BGE_RERANKER_DIR") {
        let pb = PathBuf::from(p);
        if pb.is_dir() {
            return Some(pb);
        }
    }
    if let Some(home) = std::env::var_os("HOME") {
        let mac = PathBuf::from(&home)
            .join("Library/Application Support/ellisii/models/bge-reranker-v2-m3");
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

/// CE rerank を適用して hits を並び替え、score をブレンドして返す。
/// `ce_weight` は 0.0..=1.0、CE 比率 (1.0 で pure CE)。`ce_rerank.rs` と同じ実装。
async fn apply_ce_rerank(
    compressor: &dyn ContextCompressor,
    query: &str,
    mut hits: Vec<SearchHit>,
    ce_weight: f32,
) -> Vec<SearchHit> {
    if hits.len() <= 1 {
        return hits;
    }
    let texts: Vec<String> = hits.iter().map(|h| h.chunk.text.clone()).collect();
    let scores = match compressor.score_passages(query, &texts).await {
        Ok(s) if s.len() == hits.len() => s,
        _ => return hits,
    };
    let max_orig = hits
        .iter()
        .map(|h| h.score.abs())
        .fold(0.0_f32, f32::max)
        .max(1e-6);
    for (h, ce) in hits.iter_mut().zip(scores.iter()) {
        let norm_orig = (h.score / max_orig).clamp(0.0, 1.0);
        h.score = ce_weight * ce + (1.0 - ce_weight) * norm_orig;
    }
    hits.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    hits
}

/// 1 corpus / 1 reranker の組合せで上位 top_k の (recall, hit, nDCG, MRR) を返す。
async fn measure_one(
    engine: &RagEngine<DynEmb, SqliteStore, EchoLlm>,
    nb: Uuid,
    id_map: &HashMap<Uuid, String>,
    golden: &GoldenSet,
    reranker: Option<&dyn ContextCompressor>,
    label: &str,
    ce_pool: usize,
    top_k: usize,
    ce_weight: f32,
) -> (f32, f32, f32, f32, f32) {
    let t0 = std::time::Instant::now();
    let mut pairs = Vec::with_capacity(golden.items.len());
    for item in &golden.items {
        let mut hits = engine
            .retrieve_weighted(
                Some(nb),
                &item.query,
                ce_pool,
                HybridWeights { semantic: 0.75 },
            )
            .await
            .unwrap();
        if let Some(r) = reranker {
            hits = apply_ce_rerank(r, &item.query, hits, ce_weight).await;
        }
        hits.truncate(top_k);
        let predicted: Vec<String> = hits
            .iter()
            .filter_map(|h| id_map.get(&h.chunk.id).cloned())
            .collect();
        pairs.push((predicted, item.relevant.clone()));
    }
    let s = summarize(&pairs, top_k);
    let dt = t0.elapsed().as_secs_f32();
    println!(
        "  {:<24} recall={:.3}  hit={:.3}  nDCG={:.3}  MRR={:.3}  ({:.1}s)",
        label, s.recall_at_k, s.hit_at_k, s.ndcg_at_k, s.mrr, dt,
    );
    (s.recall_at_k, s.hit_at_k, s.ndcg_at_k, s.mrr, dt)
}

async fn measure_domain(domain: &str) {
    let static_jp = locate_static_jp().expect("static-jp model not present");
    let provence_dir = locate_open_provence().expect("open-provence model not present");
    let bge_dir = locate_bge_reranker()
        .expect("bge-reranker-v2-m3 model not present (see scripts/fetch-bge-reranker.sh)");

    let (corpus, golden) = load_fixture(domain);
    let queries = golden.items.len();
    let embedder = EmbedderKind::StaticJp {
        model_dir: static_jp,
    }
    .build()
    .unwrap();
    let dim = embedder.dim();
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

    // CE config は ce_rerank.rs と同じ既定で揃える: pool=2x, weight=0.7
    let top_k = 10usize;
    let ce_pool = top_k * 2;
    let ce_weight = 0.7;

    let provence =
        ProvenceOnnx::load(&provence_dir, ProvenceConfig::default()).expect("load open-provence");
    let bge =
        ProvenceOnnx::load(&bge_dir, ProvenceConfig::default()).expect("load bge-reranker-v2-m3");

    println!(
        "\n=== bge-reranker-v2-m3 vs open-provence on {domain} (k={top_k}, ce_pool={ce_pool}, w={ce_weight}, queries={queries}) ==="
    );
    println!("  config                   recall  hit    nDCG   MRR     time");

    let (b_rec, _b_hit, b_ndcg, b_mrr, _) = measure_one(
        &engine,
        nb,
        &id_map,
        &golden,
        None,
        "baseline (no CE)",
        ce_pool,
        top_k,
        0.0,
    )
    .await;
    let (p_rec, _p_hit, p_ndcg, p_mrr, _) = measure_one(
        &engine,
        nb,
        &id_map,
        &golden,
        Some(&provence),
        "open-provence-xsmall",
        ce_pool,
        top_k,
        ce_weight,
    )
    .await;
    let (g_rec, _g_hit, g_ndcg, g_mrr, _) = measure_one(
        &engine,
        nb,
        &id_map,
        &golden,
        Some(&bge),
        "bge-reranker-v2-m3",
        ce_pool,
        top_k,
        ce_weight,
    )
    .await;

    println!(
        "  Δ(bge - provence)        recall={:+.3} nDCG={:+.3} MRR={:+.3}",
        g_rec - p_rec,
        g_ndcg - p_ndcg,
        g_mrr - p_mrr,
    );
    println!(
        "  Δ(bge - baseline)        recall={:+.3} nDCG={:+.3} MRR={:+.3}",
        g_rec - b_rec,
        g_ndcg - b_ndcg,
        g_mrr - b_mrr,
    );
}

#[tokio::test]
#[ignore]
async fn measure_civil_law_hard() {
    measure_domain("jp-civil-law-hard").await;
}

#[tokio::test]
#[ignore]
async fn measure_cs_wiki_hard() {
    measure_domain("jp-cs-wiki-hard").await;
}

#[tokio::test]
#[ignore]
async fn measure_tokkyo_hou() {
    measure_domain("jp-tokkyo-hou").await;
}
