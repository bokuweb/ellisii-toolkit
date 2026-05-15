//! 実モデル (Gemma 4 E4B + static-embedding-japanese) で multi-query 効果を測る。
//!
//! 既定で `#[ignore]`。ローカルにモデルが配置されている場合のみ
//! `cargo test -p ellisii-rag-eval-cli --test real_multi_query_e4b -- --ignored --nocapture`
//! で実行する。
//!
//! 期待する出力: 民法 / CS Wiki golden で、baseline (rewriter=None) と
//! LlmRewriter(Gemma-4-E4B) で recall@10 / nDCG / MRR を比較するレポート。
//!
//! semantic weight は既定 (0.75) のみで実行。weight sweep までやると runtime が
//! ~10 倍に膨らむため別タスクに分ける。

use ellisii_llm_core::{LlmBackend, ModelFamily};
use ellisii_llm_llamacpp::{LlamaConfig, LlamaCppBackend};
use ellisii_query_rewriter_llm::{LlmRewriter, MultiExpandRewriter};
use ellisii_rag::{eval::GoldenSet, MultiQueryOptions};
use ellisii_rag_eval_cli::{
    run_eval_with_options, Backend, CorpusEntry, EmbedderKind, EvalOptions,
};
use std::path::PathBuf;
use std::sync::Arc;

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
    if let Ok(p) = std::env::var("ELLISII_GEMMA_E4B_PATH") {
        let pb = PathBuf::from(p);
        if pb.is_file() {
            return Some(pb);
        }
    }
    if let Some(home) = std::env::var_os("HOME") {
        let mac = PathBuf::from(&home)
            .join("Library/Application Support/ellisii/models/gemma-4-E4B-it-IQ4_XS.gguf");
        if mac.is_file() {
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

/// 1 つの domain について baseline と multi-query の両方を回し、表形式で印字する。
async fn measure_domain(domain: &str, embedder_dir: &PathBuf, e4b_path: &PathBuf) {
    let (corpus, golden) = load_fixture(domain);
    let queries = golden.items.len();
    let embedder = EmbedderKind::StaticJp {
        model_dir: embedder_dir.clone(),
    }
    .build()
    .expect("load static-jp");

    // Gemma 4 E4B (IQ4_XS) を 1 度だけロード。LlmRewriter で共有する。
    let cfg = LlamaConfig::new(e4b_path.clone(), ModelFamily::Gemma4);
    let llm = LlamaCppBackend::load(cfg).expect("load gemma-4-E4B");
    let llm_arc: Arc<dyn LlmBackend> = Arc::new(llm);

    // 各 evaluation で同じ rewriter を共有するため、Arc<dyn LlmBackend> を持つ
    // ラッパ型を作って LlmRewriter に渡す。
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

    let baseline = EvalOptions {
        backend: Backend::Sqlite,
        embedder: embedder.clone(),
        weights: vec![0.75],
        k: 10,
        judge: None,
        rewriter: None,
        multi: MultiQueryOptions::default(),
    };
    let multi = EvalOptions {
        backend: Backend::Sqlite,
        embedder,
        weights: vec![0.75],
        k: 10,
        judge: None,
        rewriter: Some(Arc::new(LlmRewriter::new(SharedLlm(llm_arc.clone())))),
        multi: MultiQueryOptions {
            max_variants: 3,
            variant_weight: 0.7,
            ..Default::default()
        },
    };

    let t0 = std::time::Instant::now();
    let base_rows = run_eval_with_options(&baseline, &corpus, &golden).await.unwrap();
    let dt_base = t0.elapsed();

    let t1 = std::time::Instant::now();
    let multi_rows = run_eval_with_options(&multi, &corpus, &golden).await.unwrap();
    let dt_multi = t1.elapsed();

    let b = &base_rows[0].summary;
    let m = &multi_rows[0].summary;

    println!("\n=== {domain} (k=10, semantic=0.75, queries={queries}) ===");
    println!(
        "  baseline       recall={:.3}  hit={:.3}  nDCG={:.3}  MRR={:.3}  ({:.1}s)",
        b.recall_at_k, b.hit_at_k, b.ndcg_at_k, b.mrr, dt_base.as_secs_f32()
    );
    println!(
        "  multi-query    recall={:.3}  hit={:.3}  nDCG={:.3}  MRR={:.3}  ({:.1}s)",
        m.recall_at_k, m.hit_at_k, m.ndcg_at_k, m.mrr, dt_multi.as_secs_f32()
    );
    println!(
        "  Δ (multi - base) recall={:+.3}  hit={:+.3}  nDCG={:+.3}  MRR={:+.3}",
        m.recall_at_k - b.recall_at_k,
        m.hit_at_k - b.hit_at_k,
        m.ndcg_at_k - b.ndcg_at_k,
        m.mrr - b.mrr,
    );
}

#[tokio::test]
#[ignore]
async fn measure_civil_law() {
    let static_jp = locate_static_jp().expect("static-jp model not present");
    let e4b = locate_e4b().expect("Gemma 4 E4B GGUF not present");
    measure_domain("jp-civil-law", &static_jp, &e4b).await;
}

#[tokio::test]
#[ignore]
async fn measure_cs_wiki() {
    let static_jp = locate_static_jp().expect("static-jp model not present");
    let e4b = locate_e4b().expect("Gemma 4 E4B GGUF not present");
    measure_domain("jp-cs-wiki", &static_jp, &e4b).await;
}

/// Hard golden (シナリオ・間接参照ベース) で baseline vs LlmRewriter を比較。
/// 既存 golden は天井 (recall@10=1.0) に達していて差が出なかったため、
/// 余地のある golden で再測する。
#[tokio::test]
#[ignore]
async fn measure_civil_law_hard() {
    let static_jp = locate_static_jp().expect("static-jp model not present");
    let e4b = locate_e4b().expect("Gemma 4 E4B GGUF not present");
    measure_domain("jp-civil-law-hard", &static_jp, &e4b).await;
}

/// Hard golden で MultiExpandRewriter (paraphrase + decompose + HyDE) を計測。
/// LlmRewriter (paraphrase のみ) より「答え側の語彙」も検索対象に乗るため、
/// 同義語ギャップ ("脅されて → 強迫") を埋められる仮説。
#[tokio::test]
#[ignore]
async fn measure_civil_law_hard_with_multi_expand() {
    let static_jp = locate_static_jp().expect("static-jp model not present");
    let e4b_path = locate_e4b().expect("Gemma 4 E4B GGUF not present");

    let (corpus, golden) = load_fixture("jp-civil-law-hard");
    let queries = golden.items.len();
    let embedder = EmbedderKind::StaticJp { model_dir: static_jp }.build().unwrap();

    let cfg = LlamaConfig::new(e4b_path, ModelFamily::Gemma4);
    let llm = LlamaCppBackend::load(cfg).expect("load gemma-4-E4B");
    let llm_arc: Arc<dyn LlmBackend> = Arc::new(llm);

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

    let baseline = EvalOptions {
        backend: Backend::Sqlite,
        embedder: embedder.clone(),
        weights: vec![0.75],
        k: 10,
        judge: None,
        rewriter: None,
        multi: MultiQueryOptions::default(),
    };
    let mut multi_opts = MultiQueryOptions::default();
    multi_opts.max_variants = 6; // paraphrase 3 + sub 3 + hyde 1 を取りこぼさない
    multi_opts.variant_weight = 1.0; // expand_all_in_one は等価扱いを想定
    let multi = EvalOptions {
        backend: Backend::Sqlite,
        embedder,
        weights: vec![0.75],
        k: 10,
        judge: None,
        rewriter: Some(Arc::new(MultiExpandRewriter::new(SharedLlm(llm_arc.clone())))),
        multi: multi_opts,
    };

    let t0 = std::time::Instant::now();
    let base_rows = run_eval_with_options(&baseline, &corpus, &golden).await.unwrap();
    let dt_base = t0.elapsed();

    let t1 = std::time::Instant::now();
    let multi_rows = run_eval_with_options(&multi, &corpus, &golden).await.unwrap();
    let dt_multi = t1.elapsed();

    let b = &base_rows[0].summary;
    let m = &multi_rows[0].summary;
    println!("\n=== jp-civil-law-hard MultiExpand (k=10, semantic=0.75, queries={queries}) ===");
    println!(
        "  baseline       recall={:.3}  hit={:.3}  nDCG={:.3}  MRR={:.3}  ({:.1}s)",
        b.recall_at_k, b.hit_at_k, b.ndcg_at_k, b.mrr, dt_base.as_secs_f32()
    );
    println!(
        "  multi-expand   recall={:.3}  hit={:.3}  nDCG={:.3}  MRR={:.3}  ({:.1}s)",
        m.recall_at_k, m.hit_at_k, m.ndcg_at_k, m.mrr, dt_multi.as_secs_f32()
    );
    println!(
        "  Δ (multi - base) recall={:+.3}  hit={:+.3}  nDCG={:+.3}  MRR={:+.3}",
        m.recall_at_k - b.recall_at_k,
        m.hit_at_k - b.hit_at_k,
        m.ndcg_at_k - b.ndcg_at_k,
        m.mrr - b.mrr,
    );
}

/// jp-multihop (社内規程 / 多段参照) で MultiExpandRewriter を計測。
/// 多段クエリは「定義側ドキュメント」と「規則側ドキュメント」の両方を
/// recall に乗せる必要があり、HyDE (答え側語彙) と decompose (定義側を切り出す
/// サブ質問) の寄与が大きいと予想される。
#[tokio::test]
#[ignore]
async fn measure_multihop_with_multi_expand() {
    let static_jp = locate_static_jp().expect("static-jp model not present");
    let e4b_path = locate_e4b().expect("Gemma 4 E4B GGUF not present");

    let (corpus, golden) = load_fixture("jp-multihop");
    let queries = golden.items.len();
    let embedder = EmbedderKind::StaticJp { model_dir: static_jp }.build().unwrap();

    let cfg = LlamaConfig::new(e4b_path, ModelFamily::Gemma4);
    let llm = LlamaCppBackend::load(cfg).expect("load gemma-4-E4B");
    let llm_arc: Arc<dyn LlmBackend> = Arc::new(llm);

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

    // corpus が 22 docs しか無く top-10 だと baseline が天井 (recall=1.0) になるため
    // multi-hop の本領 (定義側ドキュメントを上位に持ち上げられるか) は k=2 / k=5 で測る。
    let mk_baseline = |k: usize| EvalOptions {
        backend: Backend::Sqlite,
        embedder: embedder.clone(),
        weights: vec![0.75],
        k,
        judge: None,
        rewriter: None,
        multi: MultiQueryOptions::default(),
    };
    let mut multi_opts = MultiQueryOptions::default();
    multi_opts.max_variants = 6;
    multi_opts.variant_weight = 1.0;
    let multi_opts_c = multi_opts;
    let mk_multi = |k: usize| EvalOptions {
        backend: Backend::Sqlite,
        embedder: embedder.clone(),
        weights: vec![0.75],
        k,
        judge: None,
        rewriter: Some(Arc::new(MultiExpandRewriter::new(SharedLlm(llm_arc.clone())))),
        multi: multi_opts_c,
    };

    println!("\n=== jp-multihop MultiExpand (semantic=0.75, queries={queries}) ===");
    for k in [2usize, 5, 10] {
        let t0 = std::time::Instant::now();
        let base_rows = run_eval_with_options(&mk_baseline(k), &corpus, &golden).await.unwrap();
        let dt_base = t0.elapsed();
        let t1 = std::time::Instant::now();
        let multi_rows = run_eval_with_options(&mk_multi(k), &corpus, &golden).await.unwrap();
        let dt_multi = t1.elapsed();
        let b = &base_rows[0].summary;
        let m = &multi_rows[0].summary;
        println!(
            "  k={:<2} baseline    recall={:.3}  hit={:.3}  nDCG={:.3}  MRR={:.3}  ({:.1}s)",
            k, b.recall_at_k, b.hit_at_k, b.ndcg_at_k, b.mrr, dt_base.as_secs_f32()
        );
        println!(
            "  k={:<2} multi-exp   recall={:.3}  hit={:.3}  nDCG={:.3}  MRR={:.3}  ({:.1}s)",
            k, m.recall_at_k, m.hit_at_k, m.ndcg_at_k, m.mrr, dt_multi.as_secs_f32()
        );
        println!(
            "  k={:<2} Δ           recall={:+.3}  hit={:+.3}  nDCG={:+.3}  MRR={:+.3}",
            k,
            m.recall_at_k - b.recall_at_k,
            m.hit_at_k - b.hit_at_k,
            m.ndcg_at_k - b.ndcg_at_k,
            m.mrr - b.mrr,
        );
    }
}
