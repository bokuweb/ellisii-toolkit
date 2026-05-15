//! ellisii-rag-eval — RAG 検索品質を headless で定量評価する。
//!
//! `main.rs` は薄い shim。実装はすべてここ (lib) に寄せて、
//! 統合テストから直接 `run_eval` を叩けるようにする。
//!
//! 評価のパイプライン:
//! 1. `CorpusEntry` 列を [`build_engine_*`] で `RagEngine` + `id_map` に変換
//! 2. [`run_eval`] が weights ごとに `retrieve_weighted` を回し
//!    [`ellisii_rag::eval::summarize`] でメトリクスへ落とす
//!
//! バックエンドは `store-memory` (default、外部依存無し) と `store-sqlite`
//! (FTS5 + BM25 + 日本語トークナイザ) を切替可能。`--backend` フラグの実体。

use anyhow::{anyhow, Result};
use async_trait::async_trait;
use ellisii_core::{Chunk, Result as CoreResult, SearchHit};
use ellisii_embed_core::Embedder;
use ellisii_embed_static_jp::StaticJpEmbedder;
use ellisii_llm_core::{LlmBackend, LlmRequest};
use ellisii_llm_stub::EchoLlm;
use ellisii_query_rewriter_core::QueryRewriter;
use ellisii_rag::{
    eval::{summarize, EvalSummary, GoldenSet},
    HybridWeights, MultiQueryOptions, RagEngine,
};
use ellisii_rag_answer_eval::{AnswerJudge, FaithfulnessScore, FaithfulnessSummary, JudgeInput};
use ellisii_store_core::VectorStore;
use ellisii_store_memory::InMemoryStore;
use ellisii_store_sqlite::SqliteStore;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use uuid::Uuid;

pub const DEFAULT_DIM: usize = 256;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Backend {
    /// `InMemoryStore` — keyword はクエリ全体の部分文字列マッチ (naive)。
    Memory,
    /// `SqliteStore` (in-memory) — FTS5 + BM25 + CharBigram トークナイザ。
    Sqlite,
}

impl Backend {
    pub fn parse(s: &str) -> Result<Backend> {
        match s {
            "memory" => Ok(Backend::Memory),
            "sqlite" => Ok(Backend::Sqlite),
            other => Err(anyhow!(
                "unknown backend {other:?} (expected memory | sqlite)"
            )),
        }
    }
    pub fn as_str(&self) -> &'static str {
        match self {
            Backend::Memory => "memory",
            Backend::Sqlite => "sqlite",
        }
    }
}

#[derive(Debug, Deserialize, Clone)]
pub struct CorpusEntry {
    pub doc_id: String,
    pub text: String,
    #[serde(default)]
    pub title: String,
    #[serde(default)]
    pub caption: String,
}

#[derive(Debug, Serialize, Clone)]
pub struct EvalRow {
    pub semantic: f32,
    pub summary: EvalSummary,
    /// `EvalOptions.judge` が `Some` のときだけ埋まる。answer 層 (Ragas faithfulness) のサマリ。
    #[serde(skip_serializing_if = "Option::is_none")]
    pub faithfulness: Option<FaithfulnessSummary>,
}

#[derive(Debug, Serialize)]
pub struct EvalReport {
    pub corpus_path: String,
    pub golden_path: String,
    pub golden_name: String,
    pub backend: String,
    pub corpus_size: usize,
    pub queries: usize,
    pub k: usize,
    pub rows: Vec<EvalRow>,
}

/// 文字バイグラムを `dim` 次元にハッシュバケットへ落とす決定的 embedder。
/// 実 embedder ではないが、共通バイグラム数で cosine 類似度が単調になるので
/// 「同義クエリ → 関連 doc が高スコア」という最低限の semantic 性質を満たす。
pub struct CharBigramEmbedder {
    pub dim: usize,
}

#[async_trait]
impl Embedder for CharBigramEmbedder {
    fn dim(&self) -> usize {
        self.dim
    }
    async fn embed(&self, texts: &[String]) -> CoreResult<Vec<Vec<f32>>> {
        Ok(texts.iter().map(|t| bigram_vec(t, self.dim)).collect())
    }
}

fn bigram_vec(text: &str, dim: usize) -> Vec<f32> {
    let mut v = vec![0.0_f32; dim];
    let chars: Vec<char> = text.chars().collect();
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

/// `Arc<dyn Embedder>` を `Embedder` として扱うためのラッパ。
/// `RagEngine` が `E: Embedder` を要求するので、CLI から実装を差し替えるには
/// 一旦コンクリート型に閉じる必要がある。
pub struct DynEmbedder {
    inner: Arc<dyn Embedder>,
}

impl DynEmbedder {
    pub fn new(inner: Arc<dyn Embedder>) -> Self {
        Self { inner }
    }
}

#[async_trait]
impl Embedder for DynEmbedder {
    fn dim(&self) -> usize {
        self.inner.dim()
    }
    async fn embed(&self, texts: &[String]) -> CoreResult<Vec<Vec<f32>>> {
        self.inner.embed(texts).await
    }
}

pub struct EngineCtx<S: VectorStore + Send + Sync> {
    pub engine: RagEngine<DynEmbedder, S, EchoLlm>,
    pub nb: Uuid,
    pub id_map: HashMap<Uuid, String>,
}

/// CLI / テストから注入する eval パラメータ一式。
pub struct EvalOptions {
    pub backend: Backend,
    pub embedder: Arc<dyn Embedder>,
    pub weights: Vec<f32>,
    pub k: usize,
    /// `Some` の場合、各 query について retrieve→generate→judge を一気通貫し
    /// `EvalRow.faithfulness` に answer 層のサマリを埋める。`None` (既定) なら retrieve のみ。
    pub judge: Option<Arc<dyn AnswerJudge>>,
    /// `Some` で multi-query 経路に切り替える。`weights` の各 semantic に対し
    /// `retrieve_multi` を使い、rewriter が生成した variants でも検索する。
    /// `None` (既定) なら従来の `retrieve_weighted` を使う。
    pub rewriter: Option<Arc<dyn QueryRewriter>>,
    /// `rewriter` が `Some` のときに使う multi-query 設定。`weights` は
    /// 各 row で semantic を上書きするので意味なし。`max_variants` と
    /// `variant_weight` のみが効く。
    pub multi: MultiQueryOptions,
}

/// CLI から指定可能な embedder 種別。
#[derive(Debug, Clone)]
pub enum EmbedderKind {
    /// 内蔵の文字バイグラム hash 埋め込み (256dim、決定的、外部依存無し)。
    Bigram { dim: usize },
    /// `embed-static-jp` を `<model_dir>` から読み込む (tokenizer.json + model.safetensors)。
    StaticJp { model_dir: std::path::PathBuf },
}

impl EmbedderKind {
    pub fn build(&self) -> Result<Arc<dyn Embedder>> {
        match self {
            EmbedderKind::Bigram { dim } => Ok(Arc::new(CharBigramEmbedder { dim: *dim })),
            EmbedderKind::StaticJp { model_dir } => {
                let e = StaticJpEmbedder::from_dir(model_dir)
                    .map_err(|e| anyhow!("load static-jp from {model_dir:?}: {e}"))?;
                Ok(Arc::new(e))
            }
        }
    }
}

fn build_chunks(corpus: &[CorpusEntry]) -> (Vec<Chunk>, Vec<String>, HashMap<Uuid, String>) {
    let mut chunks = Vec::with_capacity(corpus.len());
    let mut texts = Vec::with_capacity(corpus.len());
    let mut id_map = HashMap::new();
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
    (chunks, texts, id_map)
}

pub async fn build_engine_memory(
    corpus: &[CorpusEntry],
    embedder: Arc<dyn Embedder>,
) -> Result<EngineCtx<InMemoryStore>> {
    let dyn_embedder = DynEmbedder::new(embedder);
    let store = InMemoryStore::new();
    let nb = Uuid::new_v4();
    let (chunks, texts, id_map) = build_chunks(corpus);
    let embs = dyn_embedder
        .embed(&texts)
        .await
        .map_err(|e| anyhow!("{e}"))?;
    store
        .upsert(nb, &chunks, &embs)
        .await
        .map_err(|e| anyhow!("{e}"))?;
    Ok(EngineCtx {
        engine: RagEngine {
            embedder: dyn_embedder,
            store,
            llm: EchoLlm,
        },
        nb,
        id_map,
    })
}

pub async fn build_engine_sqlite(
    corpus: &[CorpusEntry],
    embedder: Arc<dyn Embedder>,
) -> Result<EngineCtx<SqliteStore>> {
    let dim = embedder.dim();
    let dyn_embedder = DynEmbedder::new(embedder);
    let store = SqliteStore::open_in_memory(dim).map_err(|e| anyhow!("{e}"))?;
    let nb = Uuid::new_v4();
    let (chunks, texts, id_map) = build_chunks(corpus);
    let embs = dyn_embedder
        .embed(&texts)
        .await
        .map_err(|e| anyhow!("{e}"))?;
    store
        .upsert(nb, &chunks, &embs)
        .await
        .map_err(|e| anyhow!("{e}"))?;
    Ok(EngineCtx {
        engine: RagEngine {
            embedder: dyn_embedder,
            store,
            llm: EchoLlm,
        },
        nb,
        id_map,
    })
}

fn hits_to_doc_ids(hits: &[SearchHit], id_map: &HashMap<Uuid, String>) -> Vec<String> {
    hits.iter()
        .filter_map(|h| id_map.get(&h.chunk.id).cloned())
        .collect()
}

pub async fn run_eval<S: VectorStore + Send + Sync>(
    ctx: &EngineCtx<S>,
    golden: &GoldenSet,
    weights: &[f32],
    k: usize,
) -> Result<Vec<EvalRow>> {
    run_eval_inner(
        ctx,
        golden,
        weights,
        k,
        None,
        None,
        MultiQueryOptions::default(),
    )
    .await
}

async fn run_eval_inner<S: VectorStore + Send + Sync>(
    ctx: &EngineCtx<S>,
    golden: &GoldenSet,
    weights: &[f32],
    k: usize,
    judge: Option<&dyn AnswerJudge>,
    rewriter: Option<&dyn QueryRewriter>,
    multi: MultiQueryOptions,
) -> Result<Vec<EvalRow>> {
    let mut rows = Vec::with_capacity(weights.len());
    for &w in weights {
        let mut pairs = Vec::with_capacity(golden.items.len());
        let mut scores: Vec<FaithfulnessScore> = Vec::new();
        for item in &golden.items {
            let hits = if let Some(rw) = rewriter {
                let opts = MultiQueryOptions {
                    weights: HybridWeights { semantic: w },
                    ..multi
                };
                ctx.engine
                    .retrieve_multi(Some(ctx.nb), &item.query, k, rw, opts)
                    .await
                    .map_err(|e| anyhow!("retrieve_multi {:?}: {e}", item.query))?
            } else {
                ctx.engine
                    .retrieve_weighted(Some(ctx.nb), &item.query, k, HybridWeights { semantic: w })
                    .await
                    .map_err(|e| anyhow!("retrieve {:?}: {e}", item.query))?
            };
            pairs.push((hits_to_doc_ids(&hits, &ctx.id_map), item.relevant.clone()));

            if let Some(j) = judge {
                let contexts: Vec<String> = hits.iter().map(|h| h.chunk.text.clone()).collect();
                let answer = generate_answer(&ctx.engine, &item.query, &contexts).await?;
                let s = j
                    .judge_faithfulness(&JudgeInput {
                        question: &item.query,
                        contexts: &contexts,
                        answer: &answer,
                    })
                    .await
                    .map_err(|e| anyhow!("judge {:?}: {e}", item.query))?;
                scores.push(s);
            }
        }
        rows.push(EvalRow {
            semantic: w,
            summary: summarize(&pairs, k),
            faithfulness: if judge.is_some() {
                Some(FaithfulnessSummary::from_scores(&scores))
            } else {
                None
            },
        });
    }
    Ok(rows)
}

/// retrieve した contexts と query を組み立てて、ctx の LLM に投げて
/// 出てきたトークン列を 1 つの answer 文字列にする。
async fn generate_answer<S: VectorStore + Send + Sync>(
    engine: &RagEngine<DynEmbedder, S, EchoLlm>,
    query: &str,
    contexts: &[String],
) -> Result<String> {
    let context_block = contexts
        .iter()
        .enumerate()
        .map(|(i, c)| format!("<source id={}>{c}</source>", i + 1))
        .collect::<Vec<_>>()
        .join("\n");
    let req = LlmRequest {
        system: "あなたは厳密な参考文献付きアシスタントです。<source>の範囲で答えてください。"
            .into(),
        history: Vec::new(),
        user: format!("質問: {query}\n\n参考:\n{context_block}"),
        max_tokens: 512,
        temperature: 0.0,
    };
    let buf: Arc<std::sync::Mutex<String>> = Arc::new(std::sync::Mutex::new(String::new()));
    let buf2 = buf.clone();
    let cb: Box<dyn FnMut(String) + Send + 'static> = Box::new(move |t: String| {
        buf2.lock().unwrap().push_str(&t);
    });
    engine
        .llm
        .generate_stream(req, cb)
        .await
        .map_err(|e| anyhow!("{e}"))?;
    let out = buf.lock().unwrap().clone();
    Ok(out)
}

/// 後方互換: 既存 default 設定 (CharBigramEmbedder + DEFAULT_DIM) で eval を実行する。
pub async fn run_eval_with_backend(
    backend: Backend,
    corpus: &[CorpusEntry],
    golden: &GoldenSet,
    weights: &[f32],
    k: usize,
) -> Result<Vec<EvalRow>> {
    let opts = EvalOptions {
        backend,
        embedder: Arc::new(CharBigramEmbedder { dim: DEFAULT_DIM }),
        weights: weights.to_vec(),
        k,
        judge: None,
        rewriter: None,
        multi: MultiQueryOptions::default(),
    };
    run_eval_with_options(&opts, corpus, golden).await
}

/// 注入可能な embedder / judge を受け取って eval を実行する。
pub async fn run_eval_with_options(
    opts: &EvalOptions,
    corpus: &[CorpusEntry],
    golden: &GoldenSet,
) -> Result<Vec<EvalRow>> {
    let judge_ref: Option<&dyn AnswerJudge> = opts.judge.as_deref();
    let rewriter_ref: Option<&dyn QueryRewriter> = opts.rewriter.as_deref();
    match opts.backend {
        Backend::Memory => {
            let ctx = build_engine_memory(corpus, opts.embedder.clone()).await?;
            run_eval_inner(
                &ctx,
                golden,
                &opts.weights,
                opts.k,
                judge_ref,
                rewriter_ref,
                opts.multi,
            )
            .await
        }
        Backend::Sqlite => {
            let ctx = build_engine_sqlite(corpus, opts.embedder.clone()).await?;
            run_eval_inner(
                &ctx,
                golden,
                &opts.weights,
                opts.k,
                judge_ref,
                rewriter_ref,
                opts.multi,
            )
            .await
        }
    }
}

pub fn validate_golden_against_corpus(
    corpus: &[CorpusEntry],
    golden: &GoldenSet,
) -> Vec<(String, String)> {
    let ids: std::collections::HashSet<&str> = corpus.iter().map(|c| c.doc_id.as_str()).collect();
    let mut missing = Vec::new();
    for it in &golden.items {
        for r in &it.relevant {
            if !ids.contains(r.as_str()) {
                missing.push((it.query.clone(), r.clone()));
            }
        }
    }
    missing
}
