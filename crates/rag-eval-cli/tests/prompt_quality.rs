//! 本命 config (score-max merge + MultiExpand rewriter) を両 hard fixture で
//! 計測する。prompt 改善の効果をクロスドメインで検証するため、civil-law-hard
//! (法律, 概念絡みあり) と cs-wiki-hard (技術, 概念独立) の両方で確認する。
//!
//! `#[ignore]`。`cargo test -p ellisii-rag-eval-cli --test prompt_quality -- --ignored --nocapture`

use ellisii_core::{Chunk, SearchHit};
use ellisii_embed_core::Embedder;
use ellisii_llm_core::{LlmBackend, ModelFamily};
use ellisii_llm_llamacpp::{LlamaConfig, LlamaCppBackend};
use ellisii_llm_stub::EchoLlm;
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

fn locate_e4b() -> Option<PathBuf> {
    std::env::var_os("HOME")
        .map(|h| {
            PathBuf::from(&h).join("Library/Application Support/ellisii/models/gemma-4-E4B-it-IQ4_XS.gguf")
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
        let hits = match engine.retrieve_weighted(Some(nb), q, per_query_k, weights).await {
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
    hits.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
    hits.truncate(top_k);
    hits
}

async fn measure(domain: &str, static_jp: &PathBuf, llm: Arc<dyn LlmBackend>) {
    let (corpus, golden) = load_fixture(domain);
    let queries_n = golden.items.len();

    let embedder = EmbedderKind::StaticJp { model_dir: static_jp.clone() }.build().unwrap();
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

    let rewriter = MultiExpandRewriter::new(SharedLlm(Arc::clone(&llm)));
    let top_k = 10usize;
    let weights = HybridWeights { semantic: 0.75 };

    // baseline (no rewriter)
    let t0 = std::time::Instant::now();
    let mut pairs = Vec::with_capacity(golden.items.len());
    for item in &golden.items {
        let hits = engine
            .retrieve_weighted(Some(nb), &item.query, top_k, weights)
            .await
            .unwrap();
        let predicted: Vec<String> = hits.iter().filter_map(|h| id_map.get(&h.chunk.id).cloned()).collect();
        pairs.push((predicted, item.relevant.clone()));
    }
    let s_base = summarize(&pairs, top_k);
    let dt_base = t0.elapsed();

    // score-max + MultiExpand
    let t0 = std::time::Instant::now();
    let mut pairs = Vec::with_capacity(golden.items.len());
    for item in &golden.items {
        let hits = score_max_retrieve(&engine, nb, &rewriter, &item.query, top_k, weights).await;
        let predicted: Vec<String> = hits.iter().filter_map(|h| id_map.get(&h.chunk.id).cloned()).collect();
        pairs.push((predicted, item.relevant.clone()));
    }
    let s_multi = summarize(&pairs, top_k);
    let dt_multi = t0.elapsed();

    println!("\n=== {domain} (k={top_k}, queries={queries_n}) ===");
    println!(
        "  baseline (no rewriter)         recall={:.3}  hit={:.3}  nDCG={:.3}  MRR={:.3}  ({:.1}s)",
        s_base.recall_at_k, s_base.hit_at_k, s_base.ndcg_at_k, s_base.mrr, dt_base.as_secs_f32()
    );
    println!(
        "  score-max + MultiExpand (本命) recall={:.3}  hit={:.3}  nDCG={:.3}  MRR={:.3}  ({:.1}s)",
        s_multi.recall_at_k, s_multi.hit_at_k, s_multi.ndcg_at_k, s_multi.mrr, dt_multi.as_secs_f32()
    );
    println!(
        "  Δ (multi - baseline)            recall={:+.3}  hit={:+.3}  nDCG={:+.3}  MRR={:+.3}",
        s_multi.recall_at_k - s_base.recall_at_k,
        s_multi.hit_at_k - s_base.hit_at_k,
        s_multi.ndcg_at_k - s_base.ndcg_at_k,
        s_multi.mrr - s_base.mrr,
    );
}

#[tokio::test]
#[ignore]
async fn measure_both_domains() {
    let static_jp = locate_static_jp().expect("static-jp model not present");
    let e4b = locate_e4b().expect("Gemma 4 E4B GGUF not present");

    // LLM を 1 度だけロードして両 domain で共有
    let cfg = LlamaConfig::new(e4b, ModelFamily::Gemma4);
    let llm: Arc<dyn LlmBackend> = Arc::new(LlamaCppBackend::load(cfg).expect("load gemma E4B"));

    measure("jp-civil-law-hard", &static_jp, Arc::clone(&llm)).await;
    measure("jp-cs-wiki-hard", &static_jp, Arc::clone(&llm)).await;
    measure("sql-antipatterns", &static_jp, Arc::clone(&llm)).await;
    measure("jp-patents", &static_jp, Arc::clone(&llm)).await;
}
