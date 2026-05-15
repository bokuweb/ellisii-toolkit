//! 新 prompt 環境 (PR #56) で「production config (score-max + MultiExpand) に
//! CE rerank を載せる効果」を計測する。PR #54 の調査では旧 prompt + RRF +
//! LlmRewriter で「CE は rewriter とスタックしない」と結論したが、prompt 改善
//! 後の score-max + MultiExpand 環境でも同じ結論になるか確認する。
//!
//! `#[ignore]`。`cargo test -p ellisii-rag-eval-cli --test ce_stack_with_new_prompt -- --ignored --nocapture`

use ellisii_core::{Chunk, SearchHit};
use ellisii_embed_core::Embedder;
use ellisii_llm_core::{LlmBackend, ModelFamily};
use ellisii_llm_llamacpp::{LlamaConfig, LlamaCppBackend};
use ellisii_llm_stub::EchoLlm;
use ellisii_provence_core::ContextCompressor;
use ellisii_provence_onnx::{ProvenceConfig, ProvenceOnnx};
use ellisii_query_rewriter_core::QueryRewriter;
use ellisii_query_rewriter_llm::MultiExpandRewriter;
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

fn locate_provence() -> Option<PathBuf> {
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

fn locate_e4b() -> Option<PathBuf> {
    std::env::var_os("HOME")
        .map(|h| {
            PathBuf::from(&h)
                .join("Library/Application Support/ellisii/models/gemma-4-E4B-it-IQ4_XS.gguf")
        })
        .filter(|p| p.is_file())
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

struct SharedLlm(Arc<dyn LlmBackend>);
#[async_trait::async_trait]
impl LlmBackend for SharedLlm {
    async fn generate_stream(
        &self,
        req: ellisii_llm_core::LlmRequest,
        on_token: Box<dyn FnMut(String) + Send + 'static>,
    ) -> ellisii_core::Result<()> {
        self.0.generate_stream(req, on_token).await
    }
}

async fn score_max_retrieve<S: VectorStore + Send + Sync>(
    engine: &RagEngine<DynEmb, S, EchoLlm>,
    nb: Uuid,
    rewriter: &dyn QueryRewriter,
    query: &str,
    top_k: usize,
    weights: HybridWeights,
) -> Vec<SearchHit> {
    let rewritten = rewriter.rewrite(query, 8).await.unwrap();
    let queries = rewritten.all();
    let per_query_k = top_k.max(6);
    let mut merged: HashMap<Uuid, SearchHit> = HashMap::new();
    for q in &queries {
        let hits = match engine
            .retrieve_weighted(Some(nb), q, per_query_k, weights)
            .await
        {
            Ok(h) => h,
            Err(_) => continue,
        };
        for h in hits {
            merged
                .entry(h.chunk.id)
                .and_modify(|e| {
                    if h.score > e.score {
                        *e = h.clone();
                    }
                })
                .or_insert(h);
        }
    }
    let mut hits: Vec<SearchHit> = merged.into_values().collect();
    hits.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    hits.truncate(top_k);
    hits
}

async fn apply_ce(
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

async fn measure_domain(
    domain: &str,
    static_jp: &PathBuf,
    provence_dir: &PathBuf,
    llm: Arc<dyn LlmBackend>,
) {
    let (corpus, golden) = load_fixture(domain);
    let queries_n = golden.items.len();

    let embedder = EmbedderKind::StaticJp {
        model_dir: static_jp.clone(),
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

    let provence = ProvenceOnnx::load(provence_dir, ProvenceConfig::default()).expect("load CE");
    let rewriter = MultiExpandRewriter::new(SharedLlm(Arc::clone(&llm)));
    let top_k = 10usize;
    let ce_pool = top_k * 2;
    let weights = HybridWeights { semantic: 0.75 };

    println!("\n=== {domain} (k={top_k}, ce_pool={ce_pool}, queries={queries_n}) ===");
    println!("  config                              recall  hit    nDCG   MRR     time");

    // (a) score-max + MultiExpand (= production の本体検索段)
    let t0 = std::time::Instant::now();
    let mut pairs = Vec::with_capacity(golden.items.len());
    for item in &golden.items {
        let hits = score_max_retrieve(&engine, nb, &rewriter, &item.query, top_k, weights).await;
        let predicted: Vec<String> = hits
            .iter()
            .filter_map(|h| id_map.get(&h.chunk.id).cloned())
            .collect();
        pairs.push((predicted, item.relevant.clone()));
    }
    let s_a = summarize(&pairs, top_k);
    println!(
        "  score-max + MultiExpand             {:.3}   {:.3}  {:.3}  {:.3}   {:.1}s",
        s_a.recall_at_k,
        s_a.hit_at_k,
        s_a.ndcg_at_k,
        s_a.mrr,
        t0.elapsed().as_secs_f32()
    );

    // (b) score-max + MultiExpand + CE rerank
    let t0 = std::time::Instant::now();
    let mut pairs = Vec::with_capacity(golden.items.len());
    for item in &golden.items {
        let hits = score_max_retrieve(&engine, nb, &rewriter, &item.query, ce_pool, weights).await;
        let mut hits = apply_ce(&provence, &item.query, hits, 0.7).await;
        hits.truncate(top_k);
        let predicted: Vec<String> = hits
            .iter()
            .filter_map(|h| id_map.get(&h.chunk.id).cloned())
            .collect();
        pairs.push((predicted, item.relevant.clone()));
    }
    let s_b = summarize(&pairs, top_k);
    println!(
        "  score-max + MultiExpand + CE w=0.7  {:.3}   {:.3}  {:.3}  {:.3}   {:.1}s",
        s_b.recall_at_k,
        s_b.hit_at_k,
        s_b.ndcg_at_k,
        s_b.mrr,
        t0.elapsed().as_secs_f32()
    );

    // (c) score-max + MultiExpand + CE w=0.5 (PR #54 で best recall を出した w)
    let t0 = std::time::Instant::now();
    let mut pairs = Vec::with_capacity(golden.items.len());
    for item in &golden.items {
        let hits = score_max_retrieve(&engine, nb, &rewriter, &item.query, ce_pool, weights).await;
        let mut hits = apply_ce(&provence, &item.query, hits, 0.5).await;
        hits.truncate(top_k);
        let predicted: Vec<String> = hits
            .iter()
            .filter_map(|h| id_map.get(&h.chunk.id).cloned())
            .collect();
        pairs.push((predicted, item.relevant.clone()));
    }
    let s_c = summarize(&pairs, top_k);
    println!(
        "  score-max + MultiExpand + CE w=0.5  {:.3}   {:.3}  {:.3}  {:.3}   {:.1}s",
        s_c.recall_at_k,
        s_c.hit_at_k,
        s_c.ndcg_at_k,
        s_c.mrr,
        t0.elapsed().as_secs_f32()
    );

    println!(
        "  Δ (b - a) +CE w=0.7                  recall={:+.3}  nDCG={:+.3}  MRR={:+.3}",
        s_b.recall_at_k - s_a.recall_at_k,
        s_b.ndcg_at_k - s_a.ndcg_at_k,
        s_b.mrr - s_a.mrr,
    );
    println!(
        "  Δ (c - a) +CE w=0.5                  recall={:+.3}  nDCG={:+.3}  MRR={:+.3}",
        s_c.recall_at_k - s_a.recall_at_k,
        s_c.ndcg_at_k - s_a.ndcg_at_k,
        s_c.mrr - s_a.mrr,
    );
}

#[tokio::test]
#[ignore]
async fn measure_civil_law_hard_only() {
    let static_jp = locate_static_jp().expect("static-jp not present");
    let provence_dir = locate_provence().expect("open-provence not present");
    let e4b = locate_e4b().expect("Gemma 4 E4B not present");

    let cfg = LlamaConfig::new(e4b, ModelFamily::Gemma4);
    let llm: Arc<dyn LlmBackend> = Arc::new(LlamaCppBackend::load(cfg).expect("load gemma"));

    measure_domain(
        "jp-civil-law-hard",
        &static_jp,
        &provence_dir,
        Arc::clone(&llm),
    )
    .await;
}

#[tokio::test]
#[ignore]
async fn measure_cs_wiki_only() {
    let static_jp = locate_static_jp().expect("static-jp not present");
    let provence_dir = locate_provence().expect("open-provence not present");
    let e4b = locate_e4b().expect("Gemma 4 E4B not present");

    let cfg = LlamaConfig::new(e4b, ModelFamily::Gemma4);
    let llm: Arc<dyn LlmBackend> = Arc::new(LlamaCppBackend::load(cfg).expect("load gemma"));

    measure_domain(
        "jp-cs-wiki-hard",
        &static_jp,
        &provence_dir,
        Arc::clone(&llm),
    )
    .await;
}
