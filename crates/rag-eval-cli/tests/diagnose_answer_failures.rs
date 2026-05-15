//! Answer 層の失敗パターンを特定するための diagnostic。
//! production retrieve config (score-max + MultiExpand) で answer を生成し、
//! faith.score の低い順に answer 内容と context を dump する。
//!
//! `#[ignore]`。`cargo test -p ellisii-rag-eval-cli --test diagnose_answer_failures -- --ignored --nocapture`

use ellisii_core::{Chunk, SearchHit};
use ellisii_embed_core::Embedder;
use ellisii_llm_core::{LlmBackend, LlmRequest, ModelFamily};
use ellisii_llm_llamacpp::{LlamaConfig, LlamaCppBackend};
use ellisii_llm_stub::EchoLlm;
use ellisii_query_rewriter_core::QueryRewriter;
use ellisii_query_rewriter_llm::MultiExpandRewriter;
use ellisii_rag::{eval::GoldenSet, HybridWeights, RagEngine};
use ellisii_rag_answer_eval::{heuristic::TokenOverlapJudge, AnswerJudge, JudgeInput};
use ellisii_rag_eval_cli::{CorpusEntry, EmbedderKind};
use ellisii_store_core::VectorStore;
use ellisii_store_sqlite::SqliteStore;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
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

fn format_user(q: &str, contexts: &[String]) -> String {
    let ctx = contexts
        .iter()
        .enumerate()
        .map(|(i, c)| format!("[{}] {c}", i + 1))
        .collect::<Vec<_>>()
        .join("\n");
    format!(
        "## 参考資料 (これらだけを根拠にしてください)\n{ctx}\n\n## 質問\n{q}\n\n## 回答指示\n\
         - 上記の参考資料のみを根拠に、**日本語**で簡潔に答える。\n\
         - 主張ごとに `[1]` `[2]` の形式で本文中に番号を挿入する (まとめてではなく主張の直後)。\n\
         - 該当情報が無い場合は『参考資料に該当する情報は見つかりませんでした。』と 1 行で返す。\n\
         - 参考資料の見出しや条文番号を長く丸写しせず、要点をまとめて回答する。",
        q = q
    )
}

const SYSTEM_PROMPT: &str = "あなたは厳密な参考文献付きアシスタントです。<source>に無い情報は答えず、引用を [1][2] の形式で付けてください。";

async fn generate(llm: &dyn LlmBackend, q: &str, contexts: &[String]) -> String {
    let req = LlmRequest {
        system: SYSTEM_PROMPT.into(),
        history: Vec::new(),
        user: format_user(q, contexts),
        max_tokens: 512,
        temperature: 0.2,
    };
    let buf: Arc<Mutex<String>> = Arc::new(Mutex::new(String::new()));
    let buf2 = buf.clone();
    let cb: Box<dyn FnMut(String) + Send + 'static> =
        Box::new(move |t: String| buf2.lock().unwrap().push_str(&t));
    if llm.generate_stream(req, cb).await.is_err() {
        return String::new();
    }
    let out = buf.lock().unwrap().clone();
    out
}

#[tokio::test]
#[ignore]
async fn diagnose() {
    let static_jp = locate_static_jp().expect("static-jp not present");
    let e4b = locate_e4b().expect("Gemma 4 E4B not present");
    let (corpus, golden) = load_fixture("jp-civil-law-hard");

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

    let cfg = LlamaConfig::new(e4b, ModelFamily::Gemma4);
    let llm: Arc<dyn LlmBackend> = Arc::new(LlamaCppBackend::load(cfg).expect("load gemma E4B"));
    let rewriter = MultiExpandRewriter::new(SharedLlm(Arc::clone(&llm)));
    let judge = TokenOverlapJudge::default();
    let top_k = 6usize;
    let weights = HybridWeights { semantic: 0.75 };

    let mut rows: Vec<(f32, String, String, Vec<String>, String)> = Vec::new();
    for item in &golden.items {
        let hits = score_max_retrieve(&engine, nb, &rewriter, &item.query, top_k, weights).await;
        let predicted_ids: Vec<String> = hits
            .iter()
            .filter_map(|h| id_map.get(&h.chunk.id).cloned())
            .collect();
        let contexts: Vec<String> = hits.iter().map(|h| h.chunk.text.clone()).collect();
        let answer = generate(&*llm, &item.query, &contexts).await;
        let s = judge
            .judge_faithfulness(&JudgeInput {
                question: &item.query,
                contexts: &contexts,
                answer: &answer,
            })
            .await
            .unwrap();
        let want = item.relevant.first().cloned().unwrap_or_default();
        rows.push((s.score, item.query.clone(), want, predicted_ids, answer));
    }

    rows.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    println!("\n=== answer faithfulness diagnostic (worst → best) ===\n");
    for (i, (score, q, want, pred, answer)) in rows.iter().enumerate() {
        let answer_short: String = answer.chars().take(180).collect();
        let preview = if answer.chars().count() > 180 { format!("{answer_short}...") } else { answer_short };
        let in_top = pred.iter().any(|p| p == want);
        let mark = if in_top { "✓ retrieve hit" } else { "✗ retrieve miss" };
        println!(
            "[{i:2}] faith={score:.3}  {mark}  want={want}",
        );
        println!("    Q: {q}");
        println!("    pred: {:?}", &pred[..3.min(pred.len())]);
        println!("    A: {preview}");
        println!();
    }
}
