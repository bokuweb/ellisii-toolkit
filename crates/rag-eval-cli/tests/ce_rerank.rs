//! Cross-encoder 再ランクの単独効果を hard golden で計測する。
//!
//! 既存の src-tauri は retrieve → CE rerank (compressor.score_passages) →
//! top_k と pipeline するが、rag-eval-cli の eval ハーネスは CE rerank を
//! 通っていなかった。CE rerank が hard golden で実際に nDCG/MRR/recall を
//! どれだけ伸ばすかを単独で計測する。
//!
//! 計測内容:
//!   1. baseline   : retrieve_weighted のみ
//!   2. CE 0.7/0.3 : src-tauri と同じ (CE 0.7 + 元 score 0.3)
//!   3. CE pure    : CE スコアだけで並べ替え (元 score 無視)
//!
//! `#[ignore]`。`cargo test -p ellisii-rag-eval-cli --test ce_rerank -- --ignored --nocapture`

use ellisii_core::{Chunk, SearchHit};
use ellisii_embed_core::Embedder;
use ellisii_llm_stub::EchoLlm;
use ellisii_provence_core::ContextCompressor;
use ellisii_provence_onnx::{ProvenceConfig, ProvenceOnnx};
use ellisii_rag::{
    eval::{summarize, GoldenSet},
    HybridWeights, MultiQueryOptions, RagEngine,
};
use ellisii_query_rewriter_core::QueryRewriter;
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
        let mac = PathBuf::from(&home).join("Library/Application Support/ellisii/models/open-provence");
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
/// `ce_weight` は 0.0..=1.0、CE 比率 (1.0 で pure CE)。
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
        _ => return hits, // 失敗時は元順序を尊重
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
    hits.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
    hits
}

#[tokio::test]
#[ignore]
async fn measure_ce_rerank_civil_law_hard() {
    let static_jp = locate_static_jp().expect("static-jp model not present");
    let provence_dir = locate_provence().expect("open-provence model not present");
    let (corpus, golden) = load_fixture("jp-civil-law-hard");
    let queries = golden.items.len();

    // Embedder + store の構築
    let embedder = EmbedderKind::StaticJp { model_dir: static_jp }.build().unwrap();
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

    // open-provence モデルをロード
    let provence = ProvenceOnnx::load(&provence_dir, ProvenceConfig::default())
        .expect("load open-provence");

    // top_k=10, ce_pool=20 (src-tauri と同じ「2x 候補→CE で並べ替え→top_k」)
    let top_k = 10usize;
    let ce_pool = top_k * 2;

    // (1) CE weight sweep
    let weights: &[(&str, Option<f32>)] = &[
        ("baseline (no CE)", None),
        ("CE w=0.3", Some(0.3)),
        ("CE w=0.5", Some(0.5)),
        ("CE w=0.6", Some(0.6)),
        ("CE w=0.7 (default)", Some(0.7)),
        ("CE w=0.8", Some(0.8)),
        ("CE w=0.9", Some(0.9)),
        ("CE w=1.0 (pure)", Some(1.0)),
    ];

    println!("\n=== (1) CE weight sweep on jp-civil-law-hard (k={top_k}, ce_pool={ce_pool}, queries={queries}) ===");
    println!("  config                recall  hit    nDCG   MRR     time");
    for (label, ce_weight) in weights {
        let t0 = std::time::Instant::now();
        let mut pairs = Vec::with_capacity(golden.items.len());
        for item in &golden.items {
            let mut hits = engine
                .retrieve_weighted(Some(nb), &item.query, ce_pool, HybridWeights { semantic: 0.75 })
                .await
                .unwrap();
            if let Some(w) = ce_weight {
                hits = apply_ce_rerank(&provence, &item.query, hits, *w).await;
            }
            hits.truncate(top_k);
            let predicted: Vec<String> = hits
                .iter()
                .filter_map(|h| id_map.get(&h.chunk.id).cloned())
                .collect();
            pairs.push((predicted, item.relevant.clone()));
        }
        let s = summarize(&pairs, top_k);
        let dt = t0.elapsed();
        println!(
            "  {:<21} {:.3}   {:.3}  {:.3}  {:.3}   {:.1}s",
            label, s.recall_at_k, s.hit_at_k, s.ndcg_at_k, s.mrr, dt.as_secs_f32()
        );
    }

    // (2) ce_pool sweep at w=0.7
    println!("\n=== (2) ce_pool sweep at w=0.7 (k={top_k}) ===");
    println!("  ce_pool  recall  hit    nDCG   MRR     time");
    for &pool in &[10usize, 20, 30, 50] {
        let t0 = std::time::Instant::now();
        let mut pairs = Vec::with_capacity(golden.items.len());
        for item in &golden.items {
            let mut hits = engine
                .retrieve_weighted(Some(nb), &item.query, pool, HybridWeights { semantic: 0.75 })
                .await
                .unwrap();
            hits = apply_ce_rerank(&provence, &item.query, hits, 0.7).await;
            hits.truncate(top_k);
            let predicted: Vec<String> = hits
                .iter()
                .filter_map(|h| id_map.get(&h.chunk.id).cloned())
                .collect();
            pairs.push((predicted, item.relevant.clone()));
        }
        let s = summarize(&pairs, top_k);
        let dt = t0.elapsed();
        println!(
            "  {:<7} {:.3}   {:.3}  {:.3}  {:.3}   {:.1}s",
            pool, s.recall_at_k, s.hit_at_k, s.ndcg_at_k, s.mrr, dt.as_secs_f32()
        );
    }
}

/// LlmRewriter (paraphrase) と CE rerank を重ねた効果を計測。
/// 仮説: rewriter で recall を伸ばし、CE で順位を整える → 加法的に効くはず。
#[tokio::test]
#[ignore]
async fn measure_rewriter_plus_ce_civil_law_hard() {
    use ellisii_llm_core::{LlmBackend, ModelFamily};
    use ellisii_llm_llamacpp::{LlamaConfig, LlamaCppBackend};
    use ellisii_query_rewriter_llm::LlmRewriter;

    let static_jp = locate_static_jp().expect("static-jp model not present");
    let provence_dir = locate_provence().expect("open-provence model not present");
    let e4b = std::env::var_os("HOME")
        .map(|h| {
            PathBuf::from(&h).join("Library/Application Support/ellisii/models/gemma-4-E4B-it-IQ4_XS.gguf")
        })
        .filter(|p| p.is_file())
        .expect("gemma-4-E4B GGUF not present");
    let (corpus, golden) = load_fixture("jp-civil-law-hard");
    let queries = golden.items.len();

    // build engine
    let embedder = EmbedderKind::StaticJp { model_dir: static_jp }.build().unwrap();
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

    // CE
    let provence = ProvenceOnnx::load(&provence_dir, ProvenceConfig::default()).expect("load CE");

    // LLM (rewriter)
    let cfg = LlamaConfig::new(e4b, ModelFamily::Gemma4);
    let llm: Arc<dyn LlmBackend> = Arc::new(LlamaCppBackend::load(cfg).expect("load gemma E4B"));
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
    let rewriter = LlmRewriter::new(SharedLlm(llm.clone()));

    let top_k = 10usize;
    let ce_pool = 20usize;
    let weights = HybridWeights { semantic: 0.75 };
    let multi_opts = MultiQueryOptions {
        weights,
        max_variants: 3,
        variant_weight: 0.7,
    };

    // 4 configs: baseline / +CE / +Rewriter / +Rewriter +CE
    println!("\n=== Rewriter × CE stack on jp-civil-law-hard (k={top_k}, ce_pool={ce_pool}, queries={queries}) ===");
    println!("  config                  recall  hit    nDCG   MRR     time");

    // (a) baseline
    let t0 = std::time::Instant::now();
    let mut pairs = Vec::with_capacity(golden.items.len());
    for item in &golden.items {
        let mut hits = engine
            .retrieve_weighted(Some(nb), &item.query, top_k, weights)
            .await
            .unwrap();
        hits.truncate(top_k);
        let predicted: Vec<String> = hits.iter().filter_map(|h| id_map.get(&h.chunk.id).cloned()).collect();
        pairs.push((predicted, item.relevant.clone()));
    }
    let s = summarize(&pairs, top_k);
    println!(
        "  {:<23} {:.3}   {:.3}  {:.3}  {:.3}   {:.1}s",
        "baseline", s.recall_at_k, s.hit_at_k, s.ndcg_at_k, s.mrr, t0.elapsed().as_secs_f32()
    );

    // (b) +CE
    let t0 = std::time::Instant::now();
    let mut pairs = Vec::with_capacity(golden.items.len());
    for item in &golden.items {
        let mut hits = engine
            .retrieve_weighted(Some(nb), &item.query, ce_pool, weights)
            .await
            .unwrap();
        hits = apply_ce_rerank(&provence, &item.query, hits, 0.7).await;
        hits.truncate(top_k);
        let predicted: Vec<String> = hits.iter().filter_map(|h| id_map.get(&h.chunk.id).cloned()).collect();
        pairs.push((predicted, item.relevant.clone()));
    }
    let s = summarize(&pairs, top_k);
    println!(
        "  {:<23} {:.3}   {:.3}  {:.3}  {:.3}   {:.1}s",
        "+CE (0.7/0.3)", s.recall_at_k, s.hit_at_k, s.ndcg_at_k, s.mrr, t0.elapsed().as_secs_f32()
    );

    // (c) +Rewriter
    let t0 = std::time::Instant::now();
    let mut pairs = Vec::with_capacity(golden.items.len());
    for item in &golden.items {
        let mut hits = engine
            .retrieve_multi(Some(nb), &item.query, top_k, &rewriter, multi_opts)
            .await
            .unwrap();
        hits.truncate(top_k);
        let predicted: Vec<String> = hits.iter().filter_map(|h| id_map.get(&h.chunk.id).cloned()).collect();
        pairs.push((predicted, item.relevant.clone()));
    }
    let s = summarize(&pairs, top_k);
    println!(
        "  {:<23} {:.3}   {:.3}  {:.3}  {:.3}   {:.1}s",
        "+Rewriter", s.recall_at_k, s.hit_at_k, s.ndcg_at_k, s.mrr, t0.elapsed().as_secs_f32()
    );

    // (d) +Rewriter +CE
    let t0 = std::time::Instant::now();
    let mut pairs = Vec::with_capacity(golden.items.len());
    for item in &golden.items {
        let mut hits = engine
            .retrieve_multi(Some(nb), &item.query, ce_pool, &rewriter, multi_opts)
            .await
            .unwrap();
        hits = apply_ce_rerank(&provence, &item.query, hits, 0.7).await;
        hits.truncate(top_k);
        let predicted: Vec<String> = hits.iter().filter_map(|h| id_map.get(&h.chunk.id).cloned()).collect();
        pairs.push((predicted, item.relevant.clone()));
    }
    let s = summarize(&pairs, top_k);
    println!(
        "  {:<23} {:.3}   {:.3}  {:.3}  {:.3}   {:.1}s",
        "+Rewriter +CE", s.recall_at_k, s.hit_at_k, s.ndcg_at_k, s.mrr, t0.elapsed().as_secs_f32()
    );

    let _ = QueryRewriter::rewrite(&rewriter, "_", 0).await; // silence unused warning
}
