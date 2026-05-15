//! 「score-max merge vs RRF 融合」× 「LlmRewriter (paraphrase) vs MultiExpand
//! (paraphrase+sub+HyDE)」 の 2×2 比較。
//!
//! 目的: src-tauri 本体は score-max merge + MultiExpand 系のロジックで動いて
//! いるが、PR #53 で測ったのは RRF + 各種 rewriter だった。本体経路に近い
//! score-max でも同じ結論 (LlmRewriter > MultiExpand) になるかを確認し、
//! src-tauri の retrieve_multi 移行を data-driven に判断する。
//!
//! `#[ignore]`。`cargo test -p ellisii-rag-eval-cli --test fusion_comparison -- --ignored --nocapture`

use ellisii_core::{Chunk, SearchHit};
use ellisii_embed_core::Embedder;
use ellisii_llm_core::{LlmBackend, ModelFamily};
use ellisii_llm_llamacpp::{LlamaConfig, LlamaCppBackend};
use ellisii_llm_stub::EchoLlm;
use ellisii_query_rewriter_core::QueryRewriter;
use ellisii_query_rewriter_llm::{LlmRewriter, MultiExpandRewriter};
use ellisii_rag::{
    eval::{summarize, GoldenSet},
    HybridWeights, MultiQueryOptions, RagEngine,
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

/// score-max merge: 各 variant query で個別に retrieve_weighted し、
/// chunk.id で重複排除しつつ最大スコアを保持して降順で top_k を返す。
/// src-tauri の `run_stream` 内ループと同じロジック。
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

#[tokio::test]
#[ignore]
async fn fusion_x_rewriter_2x2() {
    let static_jp = locate_static_jp().expect("static-jp model not present");
    let e4b = locate_e4b().expect("Gemma 4 E4B GGUF not present");
    let (corpus, golden) = load_fixture("jp-civil-law-hard");
    let queries_n = golden.items.len();

    // Build engine
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

    // LLM (1 度だけロード、両 rewriter で共有)
    let cfg = LlamaConfig::new(e4b, ModelFamily::Gemma4);
    let llm: Arc<dyn LlmBackend> = Arc::new(LlamaCppBackend::load(cfg).expect("load gemma E4B"));
    let llm_rewriter = LlmRewriter::new(SharedLlm(Arc::clone(&llm)));
    let multi_expand = MultiExpandRewriter::new(SharedLlm(Arc::clone(&llm)));

    let top_k = 10usize;
    let weights = HybridWeights { semantic: 0.75 };

    let multi_opts_paraphrase = MultiQueryOptions {
        weights,
        max_variants: 3,
        variant_weight: 0.7,
    };
    let multi_opts_full = MultiQueryOptions {
        weights,
        max_variants: 6,
        variant_weight: 1.0,
    };

    println!(
        "\n=== Fusion × Rewriter 2x2 on jp-civil-law-hard (k={top_k}, queries={queries_n}) ===\n"
    );

    // (1) score-max + LlmRewriter (paraphrase only)
    {
        let t0 = std::time::Instant::now();
        let mut pairs = Vec::with_capacity(golden.items.len());
        for item in &golden.items {
            let hits =
                score_max_retrieve(&engine, nb, &llm_rewriter, &item.query, top_k, weights).await;
            let predicted: Vec<String> = hits
                .iter()
                .filter_map(|h| id_map.get(&h.chunk.id).cloned())
                .collect();
            pairs.push((predicted, item.relevant.clone()));
        }
        let s = summarize(&pairs, top_k);
        println!(
            "  {:<35} recall={:.3}  hit={:.3}  nDCG={:.3}  MRR={:.3}  ({:.1}s)",
            "score-max + LlmRewriter",
            s.recall_at_k,
            s.hit_at_k,
            s.ndcg_at_k,
            s.mrr,
            t0.elapsed().as_secs_f32()
        );
    }

    // (2) score-max + MultiExpand (paraphrase + sub + HyDE) — = src-tauri 本体に近い
    {
        let t0 = std::time::Instant::now();
        let mut pairs = Vec::with_capacity(golden.items.len());
        for item in &golden.items {
            let hits =
                score_max_retrieve(&engine, nb, &multi_expand, &item.query, top_k, weights).await;
            let predicted: Vec<String> = hits
                .iter()
                .filter_map(|h| id_map.get(&h.chunk.id).cloned())
                .collect();
            pairs.push((predicted, item.relevant.clone()));
        }
        let s = summarize(&pairs, top_k);
        println!(
            "  {:<35} recall={:.3}  hit={:.3}  nDCG={:.3}  MRR={:.3}  ({:.1}s)",
            "score-max + MultiExpand (本体)",
            s.recall_at_k,
            s.hit_at_k,
            s.ndcg_at_k,
            s.mrr,
            t0.elapsed().as_secs_f32()
        );
    }

    // (3) RRF + LlmRewriter
    {
        let t0 = std::time::Instant::now();
        let mut pairs = Vec::with_capacity(golden.items.len());
        for item in &golden.items {
            let hits = engine
                .retrieve_multi(
                    Some(nb),
                    &item.query,
                    top_k,
                    &llm_rewriter,
                    multi_opts_paraphrase,
                )
                .await
                .unwrap_or_default();
            let predicted: Vec<String> = hits
                .iter()
                .filter_map(|h| id_map.get(&h.chunk.id).cloned())
                .collect();
            pairs.push((predicted, item.relevant.clone()));
        }
        let s = summarize(&pairs, top_k);
        println!(
            "  {:<35} recall={:.3}  hit={:.3}  nDCG={:.3}  MRR={:.3}  ({:.1}s)",
            "RRF + LlmRewriter",
            s.recall_at_k,
            s.hit_at_k,
            s.ndcg_at_k,
            s.mrr,
            t0.elapsed().as_secs_f32()
        );
    }

    // (4) RRF + MultiExpand
    {
        let t0 = std::time::Instant::now();
        let mut pairs = Vec::with_capacity(golden.items.len());
        for item in &golden.items {
            let hits = engine
                .retrieve_multi(Some(nb), &item.query, top_k, &multi_expand, multi_opts_full)
                .await
                .unwrap_or_default();
            let predicted: Vec<String> = hits
                .iter()
                .filter_map(|h| id_map.get(&h.chunk.id).cloned())
                .collect();
            pairs.push((predicted, item.relevant.clone()));
        }
        let s = summarize(&pairs, top_k);
        println!(
            "  {:<35} recall={:.3}  hit={:.3}  nDCG={:.3}  MRR={:.3}  ({:.1}s)",
            "RRF + MultiExpand",
            s.recall_at_k,
            s.hit_at_k,
            s.ndcg_at_k,
            s.mrr,
            t0.elapsed().as_secs_f32()
        );
    }
}
