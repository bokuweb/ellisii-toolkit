//! Ellisii SDK — 「ディレクトリを index して、検索 / RAG する」を 1 つの API
//! にまとめた facade。
//!
//! - [`Ellisii::index_dir`] でディレクトリ配下を再帰 index
//! - [`Ellisii::search`] で類似検索 + キーワード検索の hybrid retrieval
//! - LLM を組み込んだ場合のみ [`Ellisii::ask`] で RAG (token streaming)
//!
//! 詳細は `docs/sdk.md` と `crates/sdk/examples/` を参照。
//!
//! # Quick start (in-memory, モデル不要)
//!
//! ```no_run
//! use ellisii_sdk::{Ellisii, IndexOptions, SearchOptions};
//!
//! # async fn run() -> anyhow::Result<()> {
//! let ellisii = Ellisii::builder()
//!     .with_embedder_dummy(64)
//!     .with_store_memory()
//!     .build()?;
//! ellisii.index_dir("./docs", IndexOptions::default()).await?;
//! let hits = ellisii.search("query", SearchOptions::default()).await?;
//! for h in hits {
//!     println!("{:.3} {}", h.score, h.chunk.text);
//! }
//! # Ok(()) }
//! ```

use std::path::{Path, PathBuf};
use std::sync::Arc;

use ellisii_core::{Chunk, Error, Result, SearchHit};
use ellisii_embed_core::Embedder;
use ellisii_ingest::Ingestor;
use ellisii_jp_tokenizer_bigram::CharBigramTokenizer;
use ellisii_jp_tokenizer_core::JpTokenizer;
use ellisii_llm_core::{LlmBackend, LlmRequest};
use ellisii_parsers_core::detect_kind;
use ellisii_provence_core::ContextCompressor;
use ellisii_query_rewriter_core::{QueryRewriter, RewrittenQueries};
use ellisii_rag::intent_classifier::{
    CachingClassifier, Intent, IntentClassifier, LlmIntentClassifier,
};
use ellisii_rag::HybridWeights;
use ellisii_store_core::{Scope, VectorStore};
use ellisii_store_memory::InMemoryStore;
use ellisii_store_sqlite::SqliteStore;
use uuid::Uuid;

pub mod index_cache;
pub use index_cache::{fingerprint, IndexCache, IndexEntry, JsonIndexCache, MemoryIndexCache};

/// 公開する基本型 (再エクスポート)。`use ellisii_sdk::prelude::*` で一括取得。
pub mod prelude {
    pub use ellisii_core::{Chunk, Error, HitSource, Result, SearchHit, SourceKind};
    pub use ellisii_ingest::{IngestReport, Progress};
    pub use ellisii_llm_core::{ModelFamily, ModelSpec};
    pub use ellisii_store_core::Scope;
}

pub use ellisii_core::{HitSource, SourceKind};
pub use ellisii_ingest::{IngestReport, Progress};
pub use ellisii_llm_core::{ModelFamily, ModelSpec};

// ─── Builder ─────────────────────────────────────────────────────────────

/// [`Ellisii`] のセットアップを段階的に行うビルダー。
///
/// 必須:
/// - embedder (e.g. `with_embedder_dummy(dim)` / `with_embedder_static_jp(...)`)
/// - store    (e.g. `with_store_memory()` / `with_store_sqlite(path, dim)`)
///
/// オプション:
/// - llm    (`with_llm_*` を呼んだ場合のみ [`Ellisii::ask`] が使える)
/// - notebook_id (1 アプリ = 1 notebook で十分なら省略可、`Uuid::nil()` が既定)
pub struct EllisiiBuilder {
    embedder: Option<Arc<dyn Embedder>>,
    store: Option<Arc<dyn VectorStore>>,
    llm: Option<Arc<dyn LlmBackend>>,
    intent_classifier: Option<Arc<dyn IntentClassifier>>,
    index_cache: Option<Arc<dyn IndexCache>>,
    query_rewriter: Option<Arc<dyn QueryRewriter>>,
    compressor: Option<Arc<dyn ContextCompressor>>,
    chunker: Option<Arc<dyn ellisii_chunker::Chunker>>,
    notebook_id: Option<Uuid>,
}

impl Default for EllisiiBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl EllisiiBuilder {
    pub fn new() -> Self {
        Self {
            embedder: None,
            store: None,
            llm: None,
            intent_classifier: None,
            index_cache: None,
            query_rewriter: None,
            compressor: None,
            chunker: None,
            notebook_id: None,
        }
    }

    /// 任意の [`Embedder`] 実装を渡す。
    pub fn with_embedder(mut self, e: Arc<dyn Embedder>) -> Self {
        self.embedder = Some(e);
        self
    }

    /// テスト・配線確認用のダミー embedder。指定 dim のランダム vector を返す。
    /// 検索精度は出ないが、deps なく動く。
    pub fn with_embedder_dummy(mut self, dim: usize) -> Self {
        self.embedder = Some(Arc::new(ellisii_embed_dummy::DummyEmbedder::new(dim)));
        self
    }

    /// 日本語向け静的埋め込み (static-embedding-japanese)。
    /// `feature = "static-jp"` 必須。
    #[cfg(feature = "static-jp")]
    pub fn with_embedder_static_jp<P: AsRef<Path>>(mut self, model_dir: P) -> Result<Self> {
        let m = ellisii_embed_static_jp::StaticJpEmbedder::from_dir(model_dir.as_ref())
            .map_err(|e| Error::Embed(format!("load static-jp: {e}")))?;
        self.embedder = Some(Arc::new(m));
        Ok(self)
    }

    /// 任意の [`VectorStore`] 実装を渡す。
    pub fn with_store(mut self, s: Arc<dyn VectorStore>) -> Self {
        self.store = Some(s);
        self
    }

    /// in-memory ストア (テスト / 短命プロセス向け)。永続化なし。
    pub fn with_store_memory(mut self) -> Self {
        self.store = Some(Arc::new(InMemoryStore::new()));
        self
    }

    /// sqlite + sqlite-vec + FTS5 ストア。永続化あり。
    /// `dim` は埋め込み次元 (embedder と一致させる)。FTS5 tokenizer は char-bigram
    /// (依存ゼロ・デフォルト)。形態素を使いたいときは
    /// [`with_store_sqlite_with_tokenizer`] を使う。
    pub fn with_store_sqlite<P: AsRef<Path>>(mut self, db_path: P, dim: usize) -> Result<Self> {
        let tokenizer: Arc<dyn JpTokenizer> = Arc::new(CharBigramTokenizer::new());
        let store = SqliteStore::open_with_tokenizer(db_path.as_ref(), dim, tokenizer)?;
        self.store = Some(Arc::new(store));
        Ok(self)
    }

    /// sqlite + sqlite-vec + FTS5 ストアを **任意の [`JpTokenizer`]** で開く。
    /// vaporetto / delarocha などの形態素 tokenizer を本番 index に流す入口。
    pub fn with_store_sqlite_with_tokenizer<P: AsRef<Path>>(
        mut self,
        db_path: P,
        dim: usize,
        tokenizer: Arc<dyn JpTokenizer>,
    ) -> Result<Self> {
        let store = SqliteStore::open_with_tokenizer(db_path.as_ref(), dim, tokenizer)?;
        self.store = Some(Arc::new(store));
        Ok(self)
    }

    /// sqlite ストアを **NFKC 正規化を被せた bigram tokenizer** で開く。
    /// 半角/全角数字や半角カナの揺れを FTS5 indexer 側で吸収できる。
    /// query 側は [`Ellisii::search`] 入口で同じ正規化を行うので index/query が
    /// 揃う。
    pub fn with_store_sqlite_nfkc<P: AsRef<Path>>(self, db_path: P, dim: usize) -> Result<Self> {
        let inner: Arc<dyn JpTokenizer> = Arc::new(CharBigramTokenizer::new());
        let tok: Arc<dyn JpTokenizer> =
            Arc::new(ellisii_jp_tokenizer_nfkc::NfkcTokenizer::new(inner));
        self.with_store_sqlite_with_tokenizer(db_path, dim, tok)
    }

    /// vaporetto モデルをロードして sqlite ストアの FTS5 tokenizer に流す
    /// convenience。`feature = "vaporetto"` 必須。モデルは `.model.zst`。
    #[cfg(feature = "vaporetto")]
    pub fn with_store_sqlite_vaporetto<P: AsRef<Path>, Q: AsRef<Path>>(
        self,
        db_path: P,
        dim: usize,
        model_path: Q,
    ) -> Result<Self> {
        use ellisii_jp_tokenizer_vaporetto::VaporettoTokenizer;
        let tok = VaporettoTokenizer::from_zst(model_path.as_ref())
            .map_err(|e| Error::Store(format!("load vaporetto: {e}")))?;
        self.with_store_sqlite_with_tokenizer(db_path, dim, Arc::new(tok))
    }

    /// delarocha (Vibrato-system 互換) をロードして sqlite ストアの FTS5
    /// tokenizer に流す convenience。`feature = "delarocha"` 必須。
    /// 辞書は `system.dic` または `system.dic.zst`。
    #[cfg(feature = "delarocha")]
    pub fn with_store_sqlite_delarocha<P: AsRef<Path>, Q: AsRef<Path>>(
        self,
        db_path: P,
        dim: usize,
        dict_path: Q,
    ) -> Result<Self> {
        use ellisii_jp_tokenizer_delarocha::DelarochaTokenizer;
        let tok = DelarochaTokenizer::from_path(dict_path.as_ref())
            .map_err(|e| Error::Store(format!("load delarocha: {e}")))?;
        self.with_store_sqlite_with_tokenizer(db_path, dim, Arc::new(tok))
    }

    /// **コーパスに合った tokenizer を自動選択** して sqlite ストアを開く
    /// (Run 8 / 6 corpus 横展開を根拠とした defensible default)。
    ///
    /// 動作:
    /// - `delarocha_dict` が `Some(path)` で `feature = "delarocha"` の両方が
    ///   揃っていれば **delarocha + NFKC** で開く (6 corpus で常に bigram
    ///   以上、悪化させたケースは無いという経験則)。
    /// - そうでなければ **bigram + NFKC** で開く (依存ゼロ・ranker は揃う)。
    ///
    /// `sample_texts` は判断の診断 signals (英字比率 / zenkaku digit /
    /// kanji digit) を [`ellisii_jp_tokenizer_core::recommend_tokenizer`] に
    /// 流して `tracing::info!` で出力する。`None` を渡すと signals は出力されない
    /// (= tokenizer 選択ロジック自体には影響しない)。
    pub fn with_store_sqlite_auto<P: AsRef<Path>>(
        self,
        db_path: P,
        dim: usize,
        delarocha_dict: Option<&Path>,
        sample_texts: Option<&[&str]>,
    ) -> Result<Self> {
        let dict_available = delarocha_dict.is_some() && cfg!(feature = "delarocha");
        if let Some(samples) = sample_texts {
            let (pick, sig) = ellisii_jp_tokenizer_core::recommend_tokenizer(
                samples.iter().copied(),
                dict_available,
            );
            tracing::info!(
                "tokenizer_auto: pick={:?} chars={} en_ratio={:.3} zen_digit={} kanji_digit={}",
                pick,
                sig.total_chars,
                sig.en_ratio,
                sig.has_zenkaku_digit,
                sig.has_kanji_digit,
            );
        }
        #[cfg(feature = "delarocha")]
        {
            if let Some(dict) = delarocha_dict {
                use ellisii_jp_tokenizer_delarocha::DelarochaTokenizer;
                let inner = DelarochaTokenizer::from_path(dict)
                    .map_err(|e| Error::Store(format!("load delarocha: {e}")))?;
                let nfkc: Arc<dyn JpTokenizer> = Arc::new(
                    ellisii_jp_tokenizer_nfkc::NfkcTokenizer::new(Arc::new(inner)),
                );
                return self.with_store_sqlite_with_tokenizer(db_path, dim, nfkc);
            }
        }
        #[cfg(not(feature = "delarocha"))]
        let _ = delarocha_dict;
        self.with_store_sqlite_nfkc(db_path, dim)
    }

    /// 任意の [`LlmBackend`] 実装を渡す。指定すると [`Ellisii::ask`] が使えるようになる。
    pub fn with_llm(mut self, llm: Arc<dyn LlmBackend>) -> Self {
        self.llm = Some(llm);
        self
    }

    /// llama.cpp (GGUF) バックエンド。`feature = "llamacpp"` 必須。
    #[cfg(feature = "llamacpp")]
    pub fn with_llm_llamacpp<P: AsRef<Path>>(
        mut self,
        model_path: P,
        family: ModelFamily,
    ) -> Result<Self> {
        let cfg = ellisii_llm_llamacpp::LlamaConfig::new(model_path.as_ref().to_path_buf(), family);
        let backend = ellisii_llm_llamacpp::LlamaCppBackend::load(cfg)
            .map_err(|e| Error::Llm(format!("load llama: {e}")))?;
        self.llm = Some(Arc::new(backend));
        Ok(self)
    }

    /// 冪等 ingest 用キャッシュ。省略時は無効 (毎回新しい source_id で再登録される)。
    pub fn with_index_cache(mut self, c: Arc<dyn IndexCache>) -> Self {
        self.index_cache = Some(c);
        self
    }

    /// JSON ファイル 1 本を `IndexCache` として使う簡易版。
    pub fn with_index_cache_json<P: AsRef<Path>>(mut self, path: P) -> Result<Self> {
        let c = JsonIndexCache::open(path.as_ref())?;
        self.index_cache = Some(Arc::new(c));
        Ok(self)
    }

    /// 意図分類器を明示的に渡す。省略時は LLM を組み込んだ場合に
    /// [`LlmIntentClassifier`] + [`CachingClassifier`] (cap=256) を自動構築する。
    /// LLM 無しなら classifier も未設定のまま (= [`AskOptions::route_by_intent`]
    /// は無効化される)。
    pub fn with_intent_classifier(mut self, c: Arc<dyn IntentClassifier>) -> Self {
        self.intent_classifier = Some(c);
        self
    }

    /// クエリ書き換え器を設定する。設定すると `SearchOptions::multi_query_max_variants > 0`
    /// のとき multi-query retrieval (元クエリ + variants の RRF 統合) が有効になる。
    /// 省略時は Passthrough (= 元クエリのみ) と等価。LLM が組み込み済みなら、
    /// `ellisii_query_rewriter_llm::LlmRewriter::new(llm)` を渡すと言い換え経路で
    /// recall を補強できる。
    pub fn with_query_rewriter(mut self, r: Arc<dyn QueryRewriter>) -> Self {
        self.query_rewriter = Some(r);
        self
    }

    /// Cross-encoder rerank / context compressor (Provence など) を設定する。
    /// `SearchOptions::ce_rerank_top_n > 0` のとき、検索の最終 pool の上位 N 件を
    /// `ContextCompressor::score_passages` で再スコアリングして並べ直す。
    /// 省略時は CE rerank が完全に skip される (= passthrough)。
    pub fn with_compressor(mut self, c: Arc<dyn ContextCompressor>) -> Self {
        self.compressor = Some(c);
        self
    }

    /// Provence ONNX cross-encoder をロードして compressor として登録する。
    /// `feature = "provence-onnx"` 必須。`model_dir` は `model.onnx` と
    /// `tokenizer.json` を含むディレクトリ (例:
    /// `~/Library/Application Support/ellisii/models/open-provence/`)。
    /// `keep_threshold` は文単位 compress 用 (rerank だけなら無関係)、0.1 が無難。
    #[cfg(feature = "provence-onnx")]
    pub fn with_compressor_provence_onnx<P: AsRef<Path>>(
        mut self,
        model_dir: P,
        keep_threshold: f32,
    ) -> Result<Self> {
        let cfg = ellisii_provence_onnx::ProvenceConfig {
            keep_threshold,
            ..Default::default()
        };
        let p = ellisii_provence_onnx::ProvenceOnnx::load(model_dir.as_ref(), cfg)
            .map_err(|e| Error::Other(anyhow::anyhow!("load provence-onnx: {e}")))?;
        self.compressor = Some(Arc::new(p));
        Ok(self)
    }

    /// notebook_id を明示する (省略時は `Uuid::nil()`)。複数 namespace を持ちたい
    /// アプリで使う。
    pub fn with_notebook_id(mut self, id: Uuid) -> Self {
        self.notebook_id = Some(id);
        self
    }

    /// 任意の [`ellisii_chunker::Chunker`] 実装を渡す (HANDOFF B4 / 2026-05-11)。
    ///
    /// - 未設定なら `DefaultChunker::default()` (= 既存挙動) が走る
    /// - 設定すると `index_file` / `index_dir` の chunking 経路がこの実装を使う
    /// - 既に chunk 済のデータを食わせたい場合は [`Ellisii::index_chunks`] を使い、
    ///   このメソッドは経由しない
    pub fn with_chunker(mut self, c: Arc<dyn ellisii_chunker::Chunker>) -> Self {
        self.chunker = Some(c);
        self
    }

    pub fn build(self) -> Result<Ellisii> {
        let embedder = self
            .embedder
            .ok_or_else(|| Error::Other(anyhow::anyhow!("embedder is required")))?;
        let store = self
            .store
            .ok_or_else(|| Error::Other(anyhow::anyhow!("store is required")))?;
        let notebook_id = self.notebook_id.unwrap_or_else(Uuid::nil);

        // Ingestor は generic 型なので、Arc<dyn> を内側で保持できる薄い wrapper
        // を作って渡す (src-tauri の DynEmbedder / DynStore と同じパターン)。
        let mut ingestor = Ingestor::new(
            Arc::new(DynEmbedder(embedder.clone())),
            Arc::new(DynStore(store.clone())),
        );
        if let Some(c) = self.chunker.clone() {
            ingestor = ingestor.with_chunker(c);
        }

        // intent_classifier の自動構築:
        //   - 明示的に with_intent_classifier(_) されていればそれを使う
        //   - そうでなくて LLM があるなら LlmIntentClassifier(+Caching) を自動構築
        //   - LLM も無ければ None
        let intent_classifier = match (self.intent_classifier, self.llm.clone()) {
            (Some(c), _) => Some(c),
            (None, Some(llm)) => {
                let inner = LlmIntentClassifier::new(llm);
                Some(Arc::new(CachingClassifier::new(inner, 256)) as Arc<dyn IntentClassifier>)
            }
            (None, None) => None,
        };

        Ok(Ellisii {
            embedder,
            store,
            llm: self.llm,
            intent_classifier,
            index_cache: self.index_cache,
            query_rewriter: self.query_rewriter,
            compressor: self.compressor,
            ingestor,
            notebook_id,
            caption_cache: std::sync::Mutex::new(None),
            defined_terms_cache: std::sync::Mutex::new(None),
            caption_idf_cache: std::sync::Mutex::new(None),
            heading_cache: std::sync::Mutex::new(None),
            heading_density_cache: std::sync::Mutex::new(None),
            source_count_cache: std::sync::Mutex::new(None),
        })
    }
}

// ─── Wrappers (Arc<dyn> を Ingestor の generic に渡すため) ───────────────

struct DynEmbedder(Arc<dyn Embedder>);
struct DynStore(Arc<dyn VectorStore>);

#[async_trait::async_trait]
impl Embedder for DynEmbedder {
    fn dim(&self) -> usize {
        self.0.dim()
    }
    async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        self.0.embed(texts).await
    }
}

#[async_trait::async_trait]
impl VectorStore for DynStore {
    async fn upsert(
        &self,
        notebook_id: Uuid,
        chunks: &[Chunk],
        embeddings: &[Vec<f32>],
    ) -> Result<()> {
        self.0.upsert(notebook_id, chunks, embeddings).await
    }
    async fn search(&self, scope: Scope, query: &[f32], top_k: usize) -> Result<Vec<SearchHit>> {
        self.0.search(scope, query, top_k).await
    }
    async fn keyword_search(
        &self,
        scope: Scope,
        query: &str,
        top_k: usize,
    ) -> Result<Vec<SearchHit>> {
        self.0.keyword_search(scope, query, top_k).await
    }
    async fn delete_by_source(&self, source_id: Uuid) -> Result<usize> {
        self.0.delete_by_source(source_id).await
    }
    async fn delete_by_notebook(&self, notebook_id: Uuid) -> Result<usize> {
        self.0.delete_by_notebook(notebook_id).await
    }
    async fn count_chunks(&self, source_id: Uuid) -> Result<usize> {
        self.0.count_chunks(source_id).await
    }
    async fn texts_by_source(&self, source_id: Uuid) -> Result<Vec<String>> {
        self.0.texts_by_source(source_id).await
    }
    async fn neighbor_chunks(
        &self,
        source_id: Uuid,
        ord_center: u32,
        window: u32,
    ) -> Result<Vec<(u32, String)>> {
        self.0.neighbor_chunks(source_id, ord_center, window).await
    }
    async fn representative_chunks(&self, scope: Scope, per_source: usize) -> Result<Vec<Chunk>> {
        self.0.representative_chunks(scope, per_source).await
    }
    async fn representative_chunks_for_topic(
        &self,
        scope: Scope,
        per_source: usize,
        topic: &str,
    ) -> Result<Vec<Chunk>> {
        self.0
            .representative_chunks_for_topic(scope, per_source, topic)
            .await
    }
}

// ─── Options ─────────────────────────────────────────────────────────────

/// [`Ellisii::index_dir`] のオプション。
#[derive(Default)]
pub struct IndexOptions {
    /// 取り込む拡張子の許可リスト (lowercase, ドット無し)。`None` なら自動判定
    /// ([`detect_kind`] が認識する種別すべて) を取り込む。
    pub include_extensions: Option<Vec<String>>,
    /// 隠しファイル (`.` 始まり) を再帰対象に含めるか。既定 `false`。
    pub follow_hidden: bool,
    /// シンボリックリンクを辿るか。既定 `false`。
    pub follow_symlinks: bool,
    /// ファイルごとの進捗コールバック。`None` で無効。
    pub on_progress: Option<Box<dyn Fn(IndexEvent) + Send + Sync>>,
    /// 同時に走らせる ingest の上限。1 なら順次。`None` も 1 と同じ扱い。
    /// I/O bound (parser + embed) なので 4〜8 程度まで上げると効果的。
    /// **進捗コールバックは順序を保証しない** (並列起動順)。
    pub concurrency: Option<usize>,
}

/// `index_dir` 中の各ファイル状態通知。
#[derive(Debug, Clone)]
pub enum IndexEvent {
    Started {
        path: PathBuf,
    },
    Ingested {
        path: PathBuf,
        chunks: usize,
    },
    /// IndexCache が有効で、指紋一致 = ファイル未変更で再 ingest を skip した。
    Unchanged {
        path: PathBuf,
    },
    Skipped {
        path: PathBuf,
        reason: String,
    },
    Failed {
        path: PathBuf,
        error: String,
    },
}

/// [`Ellisii::index_file`] の結果。冪等キャッシュ有効時は `Unchanged` も返り得る。
#[derive(Debug, Clone)]
pub enum IngestPathOutcome {
    Ingested(IngestReport),
    Unchanged,
}

impl IngestPathOutcome {
    pub fn is_unchanged(&self) -> bool {
        matches!(self, IngestPathOutcome::Unchanged)
    }
    pub fn report(&self) -> Option<&IngestReport> {
        match self {
            IngestPathOutcome::Ingested(r) => Some(r),
            IngestPathOutcome::Unchanged => None,
        }
    }
}

/// [`Ellisii::index_dir`] の結果サマリ。
#[derive(Debug, Clone, Default)]
pub struct IndexReport {
    pub total_files: usize,
    pub ingested: usize,
    /// IndexCache hit で再 ingest を省略したファイル数。
    pub unchanged: usize,
    pub skipped: usize,
    pub failed: usize,
    pub total_chunks: usize,
}

/// [`Ellisii::search`] のオプション。
#[derive(Debug, Clone)]
pub struct SearchOptions {
    pub top_k: usize,
    /// 0.0 = キーワード検索のみ / 1.0 = ベクトル検索のみ / 0.5 = 等重み hybrid。
    pub semantic_weight: f32,
    /// chunk 先頭の `(...)` 見出しを使った rerank を適用するか。法令や条文系の
    /// 文書で recall@K を底上げする。既定 `true`。
    pub caption_rerank: bool,
    /// `with_query_rewriter` 経由でクエリ書き換え器が設定されているとき、追加で
    /// 生成する variant 数の上限。0 で multi-query を無効化 (= passthrough)。
    /// rewriter 未設定 / LLM 失敗時は安全に passthrough にフォールバック。既定 0。
    pub multi_query_max_variants: usize,
    /// multi-query で variant ranking に掛ける重み (0.0..=1.0)。元クエリは常に 1.0。
    /// 既定 0.7。
    pub multi_query_variant_weight: f32,
    /// `with_compressor` 経由で cross-encoder (Provence など) を設定済みのとき、
    /// 検索の最終 pool の上位 N 件を `score_passages` で再スコアリングして並べ直す。
    /// 0 で CE rerank を無効化 (= passthrough)。compressor 未設定 / 呼び出し失敗時は
    /// 安全に passthrough にフォールバック。既定 0。
    pub ce_rerank_top_n: usize,
    /// CE rerank で「CE スコア : 元 RRF score」の混合比率。1.0 で pure CE、0.0 で
    /// 元順序を尊重 (= rerank 無効と等価)。`src-tauri` の既定 0.7 に揃えてある。
    pub ce_rerank_weight: f32,
    /// クエリの文字種比率に応じて [`SearchOptions::semantic_weight`] を **±0.2** ほど
    /// 自動調整する (`ellisii_rag::adjust_hybrid_weight_for_query`)。漢字 / 数字 / 英数字
    /// が多いクエリ (条文番号 / ID 系) は lexical 寄りに、ひらがな・カタカナ主体の
    /// 抽象クエリは semantic 寄りに振る。`src-tauri::run_stream` と同じヒューリスティック。
    /// 既定 `true`。明示的に重みを固定したいときは `false` にする。
    pub auto_adjust_weight: bool,
    /// クエリが既に十分 specific (条文番号 / 引用 / URL / コードスニペット / 50 文字以上の長文)
    /// な場合に、`multi_query_max_variants > 0` でも rewriter 呼び出しを **skip** する。
    /// LLM 呼び出しコストの節約 + specific クエリで rewriter が誤爆 (variant がノイズになり
    /// recall を下げる) するのを防ぐ。判定ロジックは `ellisii_rag::is_specific_query`。
    /// `src-tauri::run_stream` と同じヒューリスティック (PR #65 で validate)。既定 `true`。
    pub skip_rewrite_on_specific: bool,
    /// rewriter が **実際に variant を生成した** クエリでは CE rerank を **自動 skip** する。
    /// 動機: jp-civil-law-hard / docs/eval/recall-evals.md Run 16 の計測で
    /// 「Rewriter alone (hit 0.893 / nDCG 0.731)」 vs 「Rewriter + CE (hit 0.893 / nDCG 0.715)」
    /// と CE 併用で nDCG -0.016 / MRR -0.020 の退行が観測された。rewriter で promote した
    /// 多様な hit を CE が「単純な query-passage 類似度」で再評価するため、paraphrase 経由の
    /// 正解チャンクが沈んでしまうのが原因。
    /// `multi_query_max_variants > 0` かつ rewriter が有効に動いた場合のみ CE を skip する
    /// (specific クエリで rewriter が skip されたケースでは CE は通る)。既定 `true`。
    pub skip_ce_when_rewriting: bool,
    /// rewriter が生成した variant を corpus caption との overlap で post-filter する閾値。
    /// `0.0` (既定) で無効 = 従来通り全 variant を search に流す。`> 0.0` のとき、
    /// `ellisii_rag::rerank::max_caption_overlap(variant, corpus_captions) < threshold`
    /// な variant を drop する (元クエリは常に保持)。
    /// Run 33 (yokohama LLM rewriter -2.4pt 退行) のフォロー: LLM が caption と無関係な
    /// 流れ弾 variant を出して正解 chunk を displace する副作用を抑える。
    /// `caption_rerank` が動く前提と同じ caption cache を再利用するためコストはほぼゼロ。
    pub variant_caption_filter_threshold: f32,
    /// 同一 `source_id` (= 1 つの取り込み元 doc) から最終 top-K に残す chunk の上限。
    /// `0` (既定) で無効 (= passthrough)。`> 0` のとき、rerank 後 / truncate 前に
    /// [`ellisii_rag::rerank::dedup_by_source_in_place`] で「同一 source の chunk は
    /// 最大 N 件まで」と上から打ち切る。
    ///
    /// 動機: 長い PDF を chunker が複数 chunk に分割した場合、上位 K が同一 source の
    /// 連続 chunk で埋まり、別 source の重要 chunk が押し出される top-K 偏りを抑える。
    /// fixture corpora (1 doc = 1 source) では no-op だが、production の chunker
    /// 出力 (1 source → 複数 chunks) で diversity guard として効く。
    /// score / order は変えず、上限を超えた hit を**削るだけ** (MMR の極端 simplification)。
    pub max_chunks_per_source: usize,
    /// `chunk.heading_path` の文字列を query と bigram 比較し、最大一致率を
    /// `score` に加算する rerank ([`ellisii_rag::rerank::heading_boost_in_place`])。
    /// `caption_rerank` の後段で動く想定で、caption が空 / 短い / 曖昧な chunk でも
    /// 「章タイトル / 見出し」が query と一致すれば top-K に押し上げられる。
    /// Markdown / 法令 / 技術マニュアルなど heading_path が安定して充実している
    /// 取り込みパスで効きやすい。既定 `false` (= 無効) で behavior 完全温存。
    pub heading_rerank: bool,
    /// `heading_rerank` を `heading_density()` ベースで自動推奨する (Run 54)。
    /// `true` のとき、`heading_rerank == false` でも notebook の
    /// `heading_density >= Ellisii::HEADING_RERANK_AUTO_THRESHOLD` (=0.8) なら
    /// 内部で `heading_rerank` を on にする。`heading_rerank == true` (= 明示 ON)
    /// は常に優先される。
    ///
    /// 動機: Markdown / マニュアル / 法令系の取り込みで「heading_path が rich」を
    /// 自動検出して、ユーザが手動で flag を切り替えなくても recall ゲインが得られる
    /// ようにする。density 計算は内部キャッシュされ、ingest 後に invalidate される。
    /// 既定 `false` (= 無効) で behavior 完全温存。
    pub auto_heading_rerank: bool,
    /// `source_count` ベースで `max_chunks_per_source` を自動推奨する (Run 56)。
    /// `true` のとき、`max_chunks_per_source == 0` でも notebook の
    /// `source_count >= Ellisii::SOURCE_DEDUP_AUTO_THRESHOLD` (= 3) なら、
    /// 内部で `max_chunks_per_source = 1` を適用する。
    /// `max_chunks_per_source > 0` (= 明示) は常に優先される。
    ///
    /// 動機: multi-source notebook で「同一 source の連続 chunk が top-K を独占」
    /// するのを自動回避。Run 48-50 で実証した dedup1 ゲイン (jp-manual k=3 で
    /// +0.117) を、ユーザが source 数を知らなくても享受できるようにする。
    /// source 数は ingest 後の最初の検索で 1 度だけ計算し cache される。
    /// 既定 `false` (= 無効) で behavior 完全温存。
    pub auto_max_chunks_per_source: bool,
}

impl Default for SearchOptions {
    fn default() -> Self {
        Self {
            top_k: 6,
            semantic_weight: 0.5,
            caption_rerank: true,
            multi_query_max_variants: 0,
            multi_query_variant_weight: 0.7,
            ce_rerank_top_n: 0,
            ce_rerank_weight: 0.7,
            auto_adjust_weight: true,
            skip_rewrite_on_specific: true,
            skip_ce_when_rewriting: true,
            variant_caption_filter_threshold: 0.0,
            max_chunks_per_source: 0,
            heading_rerank: false,
            auto_heading_rerank: false,
            auto_max_chunks_per_source: false,
        }
    }
}

impl SearchOptions {
    /// Run 47-56 までで実証された **production-ready auto-tuning preset** (Run 57)。
    ///
    /// 全ての self-observation シグナル (`heading_density`, `source_count`) を
    /// 観測して動的に rerank/dedup を切り替える。corpus 特性に依存しない安全な
    /// 既定として推奨される構成:
    ///
    /// - `caption_rerank`: `true` (既定、Run 9 で実証)
    /// - `auto_heading_rerank`: `true` (Run 54、`heading_density ≥ 0.8` で発動)
    /// - `auto_max_chunks_per_source`: `true` (Run 56、`source_count ≥ 3` で発動)
    ///
    /// jp-manual fixture で Run 52 で実証した combo preset 効果 (k=2 Δrec=+0.150,
    /// k=5 Δrec=+0.100) が、ユーザが corpus 特性を知らなくても自動適用される。
    /// signal が閾値未満の corpus では完全 no-op (= `Default::default()` と同じ
    /// 結果になる) なので、production で「とりあえず付けておく」のが安全。
    ///
    /// `top_k` などのフィールドは [`SearchOptions::default()`] の値を継承する。
    /// 上書きしたい場合は `..SearchOptions::auto_tuning()` を spread する:
    ///
    /// ```ignore
    /// SearchOptions { top_k: 10, ..SearchOptions::auto_tuning() }
    /// ```
    pub fn auto_tuning() -> Self {
        Self {
            auto_heading_rerank: true,
            auto_max_chunks_per_source: true,
            ..Self::default()
        }
    }
}

/// [`Ellisii::ask`] のオプション。
#[derive(Debug, Clone)]
pub struct AskOptions {
    pub top_k: usize,
    pub semantic_weight: f32,
    pub max_tokens: u32,
    pub temperature: f32,
    /// 上書き system prompt。`None` なら SDK 既定の「資料に無い情報は答えない」系を使う。
    pub system: Option<String>,
    /// 意図分類でルーティングを切り替えるかどうか。`true` かつ classifier が
    /// 構築済み (= `with_llm_*` 経由 or `with_intent_classifier` 明示) のときのみ有効:
    /// - Summary{None}     → store の `representative_chunks` (TOC ライク取り出し)
    /// - Summary{Some(t)}  → store の `representative_chunks_for_topic`
    /// - Compare           → 全 source 横断 representative_chunks (per_source 少なめ)
    /// - Lookup / Smalltalk → 通常 hybrid retrieve
    ///
    /// 既定 `true`。
    pub route_by_intent: bool,
    /// chunk 先頭の `(...)` 見出しを使った rerank を適用するか。Lookup 系で recall を
    /// 底上げするのに有効。Summary/Compare ルートでは効かない (代表 chunk を
    /// 順序維持のまま使うため)。既定 `true`。
    pub caption_rerank: bool,
    /// multi-query 上限 (元クエリ + variant 数)。詳細は [`SearchOptions`] と同じ。既定 0。
    pub multi_query_max_variants: usize,
    /// multi-query variant の重み。詳細は [`SearchOptions`] と同じ。既定 0.7。
    pub multi_query_variant_weight: f32,
    /// CE rerank で再スコアリングする top-N。詳細は [`SearchOptions`] と同じ。既定 0。
    pub ce_rerank_top_n: usize,
    /// CE rerank の混合重み。詳細は [`SearchOptions`] と同じ。既定 0.7。
    pub ce_rerank_weight: f32,
    /// クエリの文字種比率に応じて重みを自動調整する。詳細は [`SearchOptions`] と同じ。
    /// 既定 `true`。
    pub auto_adjust_weight: bool,
    /// specific クエリで rewriter を skip するか。詳細は [`SearchOptions`] と同じ。既定 `true`。
    pub skip_rewrite_on_specific: bool,
    /// rewriter が実際に動いたクエリで CE rerank を自動 skip する。詳細は [`SearchOptions`]
    /// と同じ (Run 16 の non-composition 退行を防ぐ)。既定 `true`。
    pub skip_ce_when_rewriting: bool,
    /// rewriter variant を corpus caption overlap で post-filter する閾値。詳細は
    /// [`SearchOptions::variant_caption_filter_threshold`] と同じ。既定 `0.0` (無効)。
    pub variant_caption_filter_threshold: f32,
    /// 同一 source からの chunk 上限。詳細は
    /// [`SearchOptions::max_chunks_per_source`] と同じ。既定 `0` (無効)。
    pub max_chunks_per_source: usize,
    /// heading_path rerank を有効化するか。詳細は
    /// [`SearchOptions::heading_rerank`] と同じ。既定 `false` (無効)。
    pub heading_rerank: bool,
    /// `heading_density` ベースで heading_rerank を自動推奨。詳細は
    /// [`SearchOptions::auto_heading_rerank`] と同じ。既定 `false` (無効)。
    pub auto_heading_rerank: bool,
    /// `source_count` ベースで `max_chunks_per_source` を自動推奨 (Run 56)。詳細は
    /// [`SearchOptions::auto_max_chunks_per_source`] と同じ。既定 `false` (無効)。
    pub auto_max_chunks_per_source: bool,
    /// LLM が `[1]` 形式の citation marker を **1 つも出さなかった** 場合、
    /// 厳格化した system prompt で **1 度だけ** 再生成する (Run 64)。
    ///
    /// 動作:
    /// 1. 通常通り `generate_stream` で初回応答をストリーミング
    /// 2. 終了後、応答バッファに `[N]` marker が無く、かつ hits が空でない場合、
    ///    on_token に区切り (`\n\n---\n[出典付きで再生成]\n\n`) を出してから
    ///    厳格 prompt で再ストリーミング
    /// 3. 2 度目も citation 無しなら諦める (再々試行はしない)
    ///
    /// 既定 `false`。production で確実に citation を要求したいときに opt-in。
    /// retry には LLM 呼び出しコスト + latency がかかるため、安易な ON は非推奨。
    ///
    /// `hits` が空 (general mode / smalltalk) のときは元から citation 不要なので
    /// 本フラグは無効化される。
    pub no_citation_retry: bool,
}

impl Default for AskOptions {
    fn default() -> Self {
        Self {
            top_k: 6,
            semantic_weight: 0.5,
            max_tokens: 512,
            temperature: 0.2,
            system: None,
            route_by_intent: true,
            caption_rerank: true,
            multi_query_max_variants: 0,
            multi_query_variant_weight: 0.7,
            ce_rerank_top_n: 0,
            ce_rerank_weight: 0.7,
            auto_adjust_weight: true,
            skip_rewrite_on_specific: true,
            skip_ce_when_rewriting: true,
            variant_caption_filter_threshold: 0.0,
            max_chunks_per_source: 0,
            heading_rerank: false,
            auto_heading_rerank: false,
            auto_max_chunks_per_source: false,
            no_citation_retry: false,
        }
    }
}

impl AskOptions {
    /// [`SearchOptions::auto_tuning()`] と同じ思想の **AskOptions 用 preset** (Run 57)。
    /// retrieval 側に auto_heading_rerank / auto_max_chunks_per_source を ON にする
    /// ほかは [`AskOptions::default()`] の値を継承する。LLM パラメータ (max_tokens 等)
    /// を上書きする場合は spread を使う:
    ///
    /// ```ignore
    /// AskOptions { max_tokens: 1024, ..AskOptions::auto_tuning() }
    /// ```
    pub fn auto_tuning() -> Self {
        Self {
            auto_heading_rerank: true,
            auto_max_chunks_per_source: true,
            ..Self::default()
        }
    }
}

const DEFAULT_RAG_SYSTEM: &str = "あなたは厳密な参考文献付きアシスタントです。<source>に無い情報は答えず、引用を [1][2] の形式で付けてください。";

// ─── Ellisii (facade) ────────────────────────────────────────────────────

/// Lazily-built `(chunk_id, label)` list used by the caption / heading /
/// defined-term rerankers. Wrapped in `Arc` so each search call can share
/// the snapshot without copying.
type CaptionListCache = std::sync::Mutex<Option<Arc<Vec<(Uuid, String)>>>>;

pub struct Ellisii {
    embedder: Arc<dyn Embedder>,
    store: Arc<dyn VectorStore>,
    llm: Option<Arc<dyn LlmBackend>>,
    intent_classifier: Option<Arc<dyn IntentClassifier>>,
    index_cache: Option<Arc<dyn IndexCache>>,
    query_rewriter: Option<Arc<dyn QueryRewriter>>,
    compressor: Option<Arc<dyn ContextCompressor>>,
    ingestor: Ingestor<DynEmbedder, DynStore>,
    notebook_id: Uuid,
    /// caption rerank 用の lazy cache。ingest 後に invalidate される。
    caption_cache: CaptionListCache,
    /// caption の TF-IDF 重み (`caption -> [0,1]` の hashmap)。caption_cache と一緒に
    /// 構築 / invalidate される。Run 12 で観測された「同種 caption が複数文書に出ると
    /// 正解外の chunk を引き上げる」問題の対策。
    caption_idf_cache: std::sync::Mutex<Option<Arc<std::collections::HashMap<String, f32>>>>,
    /// heading rerank 用の lazy cache。caption と同じく ingest で invalidate される。
    heading_cache: CaptionListCache,
    /// Run 42: 本文中の定義語 (`「X」という。`) を `(chunk_id, term)` で 1 row/term。
    /// caption / heading と同じく lazy build。`invalidate_caption_cache` で同時破棄される。
    defined_terms_cache: CaptionListCache,
    /// Run 53/54: `heading_density()` の lazy cache。`auto_heading_rerank` の判定で
    /// 毎 search 毎に store を叩かないようにする。`invalidate_caption_cache` で同時破棄。
    heading_density_cache: std::sync::Mutex<Option<f32>>,
    /// Run 56: `source_count()` の lazy cache。`auto_max_chunks_per_source` の判定で
    /// 毎 search 毎に DISTINCT source_id クエリを発行しないようにする。
    /// `invalidate_caption_cache` で同時破棄。
    source_count_cache: std::sync::Mutex<Option<usize>>,
}

impl Ellisii {
    pub fn builder() -> EllisiiBuilder {
        EllisiiBuilder::new()
    }

    /// 既に chunk 済の `Vec<Chunk>` を直接 ingest する経路 (HANDOFF B4 / 2026-05-11)。
    ///
    /// parser / chunker / OCR を一切経由せず、与えられた chunk を embed → store する。
    /// 想定ユースケース:
    /// - 外部ツールで chunking 済のコーパスを取り込みたい
    /// - chunk-aware な前処理 (例: 章タイトルを heading_path に明示的に積む) を
    ///   SDK 利用者側で行う
    ///
    /// chunk 側で `source_id` / `ord` / `heading_path` を埋める責務は呼び出し側に
    /// ある。返り値は store に書き込めた chunk 数 (0 入力なら 0)。
    /// `index_cache` は使われない (= 冪等管理は呼び出し側のスコープ)。
    pub async fn index_chunks(&self, chunks: Vec<Chunk>) -> Result<usize> {
        if chunks.is_empty() {
            return Ok(0);
        }
        // embed → store。Ingestor::embed_and_store と同じ「heading_path を text 前に
        // prepend してから embed する」regime を踏襲する (rerank との一貫性を保つ)。
        let texts: Vec<String> = chunks
            .iter()
            .map(|c| {
                if c.heading_path.is_empty() {
                    c.text.clone()
                } else {
                    format!("{}\n{}", c.heading_path.join(" / "), c.text)
                }
            })
            .collect();
        let mut total = 0usize;
        let batch_size = 16usize;
        for batch_start in (0..chunks.len()).step_by(batch_size) {
            let batch_end = (batch_start + batch_size).min(chunks.len());
            let batch_chunks = &chunks[batch_start..batch_end];
            let batch_texts: Vec<String> = texts[batch_start..batch_end].to_vec();
            let embs = self.embedder.embed(&batch_texts).await?;
            self.store
                .upsert(self.notebook_id, batch_chunks, &embs)
                .await?;
            total += batch_chunks.len();
        }
        // caption / heading / defined-terms cache を invalidate (新 chunk が増えた)
        self.invalidate_caption_cache();
        Ok(total)
    }

    /// 単一ファイルを ingest。`with_index_cache` を設定済みなら指紋一致で skip
    /// する冪等動作。
    pub async fn index_file<P: AsRef<Path>>(&self, path: P) -> Result<IngestPathOutcome> {
        let path = path.as_ref();
        self.ingest_with_cache(path).await
    }

    /// index_file の内部実装。冪等処理を担当。
    async fn ingest_with_cache(&self, path: &Path) -> Result<IngestPathOutcome> {
        let path_str = path
            .to_str()
            .ok_or_else(|| Error::Other(anyhow::anyhow!("non-utf8 path: {}", path.display())))?;
        let cache_key = path_str.to_string();
        let fp = fingerprint(path);

        // cache hit && 指紋一致 → skip
        if let Some(cache) = self.index_cache.as_ref() {
            if let Some(prev) = cache.get(&cache_key).await? {
                if prev.fingerprint == fp {
                    return Ok(IngestPathOutcome::Unchanged);
                }
                // 内容が変わっている → 古い source_id の chunk を消してから再 ingest
                let _ = self.store.delete_by_source(prev.source_id).await;
                let _ = cache.forget(&cache_key).await;
            }
        }

        let source_id = Uuid::new_v4();
        let report = self
            .ingestor
            .ingest_file(path_str, self.notebook_id, source_id, None)
            .await?;
        self.invalidate_caption_cache();
        if let Some(cache) = self.index_cache.as_ref() {
            cache
                .put(
                    &cache_key,
                    IndexEntry {
                        source_id,
                        fingerprint: fp,
                    },
                )
                .await?;
        }
        Ok(IngestPathOutcome::Ingested(report))
    }

    /// ディレクトリ配下を再帰的に walk して、parser が認識できるファイルを順に
    /// ingest する。並列化はしない (大量ファイル時は呼び出し側で chunk して並列化推奨)。
    pub async fn index_dir<P: AsRef<Path>>(
        &self,
        dir: P,
        opts: IndexOptions,
    ) -> Result<IndexReport> {
        let dir = dir.as_ref();
        if !dir.is_dir() {
            return Err(Error::Other(anyhow::anyhow!(
                "not a directory: {}",
                dir.display()
            )));
        }
        let extensions_lower: Option<Vec<String>> = opts
            .include_extensions
            .as_ref()
            .map(|v| v.iter().map(|s| s.to_ascii_lowercase()).collect());
        let walker = walkdir::WalkDir::new(dir).follow_links(opts.follow_symlinks);
        let follow_hidden = opts.follow_hidden;
        let entries: Box<dyn Iterator<Item = walkdir::Result<walkdir::DirEntry>>> = if follow_hidden
        {
            Box::new(walker.into_iter())
        } else {
            Box::new(walker.into_iter().filter_entry(is_not_hidden))
        };
        let mut report = IndexReport::default();
        // 進捗コールバックを Arc にして全 task で共有 (buffer_unordered 内で
        // 複数 future に move したいため)。
        let on_progress: Option<Arc<dyn Fn(IndexEvent) + Send + Sync>> =
            opts.on_progress.map(Arc::from);
        let emit = |ev: IndexEvent| {
            if let Some(cb) = &on_progress {
                cb(ev);
            }
        };
        // 1) walk しつつフィルタリングして対象 path を集める。
        let mut targets: Vec<PathBuf> = Vec::new();
        for entry in entries.filter_map(|e| e.ok()) {
            if !entry.file_type().is_file() {
                continue;
            }
            let path = entry.path();
            if let Some(ref allow) = extensions_lower {
                let ext = path
                    .extension()
                    .and_then(|x| x.to_str())
                    .map(|s| s.to_ascii_lowercase())
                    .unwrap_or_default();
                if !allow.contains(&ext) {
                    continue;
                }
            }
            let path_str = match path.to_str() {
                Some(s) => s,
                None => {
                    report.skipped += 1;
                    emit(IndexEvent::Skipped {
                        path: path.to_path_buf(),
                        reason: "non-utf8 path".into(),
                    });
                    continue;
                }
            };
            if detect_kind(path_str).is_none() {
                report.skipped += 1;
                emit(IndexEvent::Skipped {
                    path: path.to_path_buf(),
                    reason: "unsupported file type".into(),
                });
                continue;
            }
            report.total_files += 1;
            targets.push(path.to_path_buf());
        }

        // 2) ingest を並列実行 (buffer_unordered)。
        //    concurrency=1 なら逐次、>1 なら同時並行。tokio::spawn は使わず
        //    現在 task の中で多重ポーリングするので Send/'static 制約は不要。
        let concurrency = opts.concurrency.unwrap_or(1).max(1);
        use futures_util::StreamExt;
        let stream = futures_util::stream::iter(targets.into_iter().map(|path| {
            let this = self;
            let cb = on_progress.clone();
            async move {
                this.ingest_with_progress(&path, |ev| {
                    if let Some(cb) = &cb {
                        cb(ev);
                    }
                })
                .await
            }
        }));
        let mut buffered = stream.buffer_unordered(concurrency);
        while let Some(outcome) = buffered.next().await {
            match outcome {
                FileOutcome::Ingested { chunks } => {
                    report.ingested += 1;
                    report.total_chunks += chunks;
                }
                FileOutcome::Unchanged => report.unchanged += 1,
                FileOutcome::Failed => report.failed += 1,
            }
        }
        Ok(report)
    }

    /// 1 ファイルを ingest しつつ Started / Ingested / Unchanged / Failed の
    /// IndexEvent を進捗コールバックに流す。`index_dir` が同時並行で呼ぶ。
    async fn ingest_with_progress<F: Fn(IndexEvent)>(&self, path: &Path, emit: F) -> FileOutcome {
        emit(IndexEvent::Started {
            path: path.to_path_buf(),
        });
        match self.ingest_with_cache(path).await {
            Ok(IngestPathOutcome::Ingested(r)) => {
                emit(IndexEvent::Ingested {
                    path: path.to_path_buf(),
                    chunks: r.chunks_stored,
                });
                FileOutcome::Ingested {
                    chunks: r.chunks_stored,
                }
            }
            Ok(IngestPathOutcome::Unchanged) => {
                emit(IndexEvent::Unchanged {
                    path: path.to_path_buf(),
                });
                FileOutcome::Unchanged
            }
            Err(e) => {
                emit(IndexEvent::Failed {
                    path: path.to_path_buf(),
                    error: e.to_string(),
                });
                FileOutcome::Failed
            }
        }
    }

    /// 類似検索 + キーワード検索を統合した hybrid retrieval。LLM は呼ばない。
    ///
    /// クエリは入口で NFKC 正規化される (半角/全角数字・半角カナ等の揺れを吸収)。
    /// 同じ正規化を index 側でも掛けたい場合は [`Self::builder()`] の
    /// `with_store_sqlite_nfkc` を使う。
    pub async fn search(&self, query: &str, opts: SearchOptions) -> Result<Vec<SearchHit>> {
        let query_owned = ellisii_jp_tokenizer_nfkc::nfkc(query);
        let query = query_owned.as_str();
        let base = opts.semantic_weight.clamp(0.0, 1.0);
        let semantic = if opts.auto_adjust_weight {
            ellisii_rag::adjust_hybrid_weight_for_query(base, query)
        } else {
            base
        };
        let weights = HybridWeights { semantic };
        let pool_top = (opts.top_k * 5).max(opts.top_k);
        // specific クエリ (条文番号 / 引用 / URL / コードスニペット / 50 文字以上) では
        // rewriter を skip する。LLM 呼び出しコストの節約 + variant ノイズ防止。
        let effective_max_variants = if opts.skip_rewrite_on_specific
            && opts.multi_query_max_variants > 0
            && ellisii_rag::is_specific_query(query)
        {
            tracing::debug!(
                "skipping rewriter: query is specific (rule-based) — `{}`",
                query
            );
            0
        } else {
            opts.multi_query_max_variants
        };
        let mut fused = self
            .hybrid_pool(
                query,
                opts.top_k * 5,
                weights,
                effective_max_variants,
                opts.multi_query_variant_weight,
                pool_top,
                opts.variant_caption_filter_threshold,
            )
            .await?;
        if opts.caption_rerank {
            self.apply_caption_rerank(query, &mut fused).await?;
        }
        // auto_heading_rerank: 明示 ON でなくても density 高ければ自動 on (Run 54)。
        let heading_on = opts.heading_rerank
            || (opts.auto_heading_rerank
                && self.heading_density().await.unwrap_or(0.0)
                    >= Self::HEADING_RERANK_AUTO_THRESHOLD);
        if heading_on {
            // caption の後段で heading_path[-1] (= 章タイトル / 見出し) の bigram 一致で
            // 追加 boost。caption boost と同じ alpha=1.0 を採用。
            ellisii_rag::rerank::heading_boost_in_place(query, &mut fused, 1.0);
        }
        // Run 16 で観測した non-composition 退行 (rewriter + CE で nDCG -0.016) を防ぐため、
        // rewriter が実際に variant を生成したクエリでは CE を自動 skip する。
        let ce_should_run = opts.ce_rerank_top_n > 0
            && !(opts.skip_ce_when_rewriting && effective_max_variants > 0);
        if ce_should_run {
            self.apply_ce_rerank(
                query,
                &mut fused,
                opts.ce_rerank_top_n,
                opts.ce_rerank_weight,
            )
            .await;
        } else if opts.ce_rerank_top_n > 0
            && opts.skip_ce_when_rewriting
            && effective_max_variants > 0
        {
            tracing::debug!(
                "skipping CE rerank: rewriter active (variants={}) — see recall-evals Run 16",
                effective_max_variants
            );
        }
        // MMR-lite: 同一 source からの chunk を `max_chunks_per_source` 件までに制限。
        // 0 で passthrough。truncate の直前に呼んで、上位 K を多様な source で
        // 埋められるようにする (production の chunker 出力で意味あり)。
        // auto_max_chunks_per_source: 明示指定が無くても multi-source notebook なら
        // 自動で dedup=1 を適用 (Run 56)。
        let effective_max_per_source = if opts.max_chunks_per_source > 0 {
            opts.max_chunks_per_source
        } else if opts.auto_max_chunks_per_source
            && self.source_count().await.unwrap_or(0) >= Self::SOURCE_DEDUP_AUTO_THRESHOLD
        {
            1
        } else {
            0
        };
        if effective_max_per_source > 0 {
            ellisii_rag::rerank::dedup_by_source_in_place(&mut fused, effective_max_per_source);
        }
        fused.truncate(opts.top_k);
        Ok(fused)
    }

    /// 元クエリ (+ optional variants) ごとに vec/kw 検索を走らせて RRF で 1 つに融合する。
    /// variants は `query_rewriter` が設定されているときだけ生成。LLM 失敗 / 未設定時は
    /// passthrough にフォールバックして検索が止まらないようにする。
    ///
    /// `variant_caption_filter_threshold > 0.0` で、生成された variants のうち corpus caption
    /// との max overlap が threshold 未満のものを drop する (元クエリは常に保持)。Run 33 followup。
    #[allow(clippy::too_many_arguments)]
    async fn hybrid_pool(
        &self,
        query: &str,
        per_ranking_top_k: usize,
        weights: HybridWeights,
        max_variants: usize,
        variant_weight: f32,
        pool_top: usize,
        variant_caption_filter_threshold: f32,
    ) -> Result<Vec<SearchHit>> {
        let mut queries = if max_variants > 0 {
            if let Some(rewriter) = self.query_rewriter.as_ref() {
                match rewriter.rewrite(query, max_variants).await {
                    Ok(r) => r.all(),
                    Err(e) => {
                        tracing::warn!("query rewrite failed: {e}; falling back to passthrough");
                        RewrittenQueries::just(query).all()
                    }
                }
            } else {
                RewrittenQueries::just(query).all()
            }
        } else {
            RewrittenQueries::just(query).all()
        };

        if variant_caption_filter_threshold > 0.0 && queries.len() > 1 {
            let captions = self
                .captions()
                .await
                .unwrap_or_else(|_| Arc::new(Vec::new()));
            if !captions.is_empty() {
                let before = queries.len();
                let original = queries[0].clone();
                queries.retain(|q| {
                    q == &original
                        || ellisii_rag::rerank::max_caption_overlap(q, &captions)
                            >= variant_caption_filter_threshold
                });
                tracing::debug!(
                    "variant caption filter: kept {}/{} (threshold={})",
                    queries.len(),
                    before,
                    variant_caption_filter_threshold
                );
            }
        }

        let w_vec = weights.vector();
        let w_kw = weights.keyword();
        let mut rankings: Vec<(Vec<SearchHit>, f32)> = Vec::with_capacity(queries.len() * 2);
        for (i, q) in queries.iter().enumerate() {
            let q_w = if i == 0 { 1.0 } else { variant_weight };
            let q_emb = self.embedder.embed(std::slice::from_ref(q)).await?;
            let vec_hits = self
                .store
                .search(Some(self.notebook_id), &q_emb[0], per_ranking_top_k)
                .await?;
            let kw_hits = self
                .store
                .keyword_search(Some(self.notebook_id), q, per_ranking_top_k)
                .await?;
            rankings.push((vec_hits, w_vec * q_w));
            rankings.push((kw_hits, w_kw * q_w));
        }
        Ok(ellisii_rag::rrf_weighted(&rankings, pool_top))
    }

    /// caption + heading rerank: pool 内 hit にローカル boost を入れたあと、
    /// notebook 全 caption を全走査して pool 外の chunk も引き上げる。heading は
    /// caption が無いケースのフォールバックとして弱めの in-pool boost でだけ使う
    /// (heading は複数 chunk で共有されることが多く、injection に使うと有効
    /// チャンク以外も等しく持ち上がってしまうため)。
    /// store の `all_captions` / `all_headings` が空 (default 実装) の場合は
    /// pool 内 boost だけが効く。
    async fn apply_caption_rerank(&self, query: &str, pool: &mut Vec<SearchHit>) -> Result<()> {
        // caption の IDF 表 (= 文書間で頻出する caption を減衰させる重み) を
        // captions と一緒に確保する。corpus に caption が無いか単一しか無い場合は
        // 空 hashmap で fall through (caption_*_with_idf は idf 未登録 caption に
        // weight=1.0 を返すので既存挙動と同じ)。
        let captions = self.captions().await?;
        let idf = self.caption_idf().await?;

        // 1) pool 内 boost (caption only) — IDF 重み付き。
        ellisii_rag::rerank::caption_boost_in_place_with_idf(query, pool, 1.0, &idf);

        // 2) caption-index 全走査
        if captions.is_empty() {
            // caption が無い文書では、heading-index 側で pool 外引き上げを試す。
            let headings = self.headings().await?;
            if !headings.is_empty() {
                let empty = std::collections::HashMap::new();
                self.inject_from_index(query, pool, &headings, 8, 0.6, &empty)
                    .await?;
            }
            pool.sort_by(|a, b| {
                b.score
                    .partial_cmp(&a.score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            return Ok(());
        }
        self.inject_from_index(query, pool, &captions, 8, 1.5, &idf)
            .await?;

        // 3) defined-terms index (Run 42): body 中の `「X」という。` 系定義語を
        //    使って pool 外から chunk を inject。caption が短く query 中心語と
        //    乖離するケース (例: yokohama [39]) を救う経路。IDF は持たないので
        //    空 hashmap で fall through。alpha は caption (1.5) より控えめに。
        let defined_terms = self.defined_terms().await?;
        if !defined_terms.is_empty() {
            let empty = std::collections::HashMap::new();
            self.inject_from_index(query, pool, &defined_terms, 4, 0.8, &empty)
                .await?;
        }

        pool.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        Ok(())
    }

    /// `index` (例: caption / heading) を query と全走査して、上位 N 件を pool に注入する。
    /// pool に居る id は score 加算、未収の id は store から chunk を引き直して push。
    /// `idf` 空の場合は IDF 重み付けをしない (= 既存挙動)。
    async fn inject_from_index(
        &self,
        query: &str,
        pool: &mut Vec<SearchHit>,
        index: &[(Uuid, String)],
        top_n: usize,
        bonus_alpha: f32,
        idf: &std::collections::HashMap<String, f32>,
    ) -> Result<()> {
        let scored = ellisii_rag::rerank::apply_caption_index_with_idf(
            query,
            pool,
            index,
            top_n,
            bonus_alpha,
            idf,
        );
        let in_pool: std::collections::HashSet<Uuid> = pool.iter().map(|h| h.chunk.id).collect();
        let missing_ids: Vec<Uuid> = scored
            .keys()
            .filter(|id| !in_pool.contains(id))
            .copied()
            .collect();
        if !missing_ids.is_empty() {
            let extra = self.store.get_chunks_by_ids(&missing_ids).await?;
            for chunk in extra {
                let score = *scored.get(&chunk.id).unwrap_or(&0.0);
                pool.push(SearchHit {
                    chunk,
                    score,
                    source: ellisii_core::HitSource::Hybrid,
                });
            }
        }
        for h in pool.iter_mut() {
            if let Some(s) = scored.get(&h.chunk.id) {
                h.score = *s;
            }
        }
        Ok(())
    }

    /// pool 上位 `top_n` を `ContextCompressor::score_passages` (cross-encoder) で再スコアリングし、
    /// 元 RRF score とブレンドして並べ直す。アルゴリズムは `src-tauri` および
    /// `crates/rag-eval-cli/tests/ce_rerank.rs` と同じ:
    ///   blended = ce_weight * ce + (1 - ce_weight) * (orig_score / max_orig)
    /// compressor 未設定 / 失敗 / pool が空のときは何もしない (= passthrough)。
    async fn apply_ce_rerank(
        &self,
        query: &str,
        pool: &mut [SearchHit],
        top_n: usize,
        ce_weight: f32,
    ) {
        let compressor = match self.compressor.as_ref() {
            Some(c) => c,
            None => return,
        };
        let n = top_n.min(pool.len());
        if n <= 1 {
            return;
        }
        let texts: Vec<String> = pool[..n].iter().map(|h| h.chunk.text.clone()).collect();
        let scores = match compressor.score_passages(query, &texts).await {
            Ok(s) if s.len() == n => s,
            Ok(_) => return,
            Err(e) => {
                tracing::warn!("ce rerank score_passages failed: {e}; falling back to passthrough");
                return;
            }
        };
        let max_orig = pool[..n]
            .iter()
            .map(|h| h.score.abs())
            .fold(0.0_f32, f32::max)
            .max(1e-6);
        for (h, ce) in pool[..n].iter_mut().zip(scores.iter()) {
            let norm_orig = (h.score / max_orig).clamp(0.0, 1.0);
            h.score = ce_weight * ce + (1.0 - ce_weight) * norm_orig;
        }
        // top_n の中だけソート (top_n 以下の順序は触らない)。
        let head = &mut pool[..n];
        head.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
    }

    /// corpus の caption (`(...)` 見出し) を最大 `max` 件サンプルして返す。
    /// caption-aware な rewriter (Run 37) を構築する際に使う想定:
    ///
    /// ```ignore
    /// let hints = ellisii.caption_samples(24).await?;
    /// let rewriter = LlmRewriter::new(llm).with_caption_hints(hints);
    /// ```
    ///
    /// 内部キャッシュ ([`Ellisii::invalidate_caption_cache`]) を再利用するため、複数回
    /// 呼んでも追加の DB アクセスは発生しない。
    pub async fn caption_samples(&self, max: usize) -> Result<Vec<String>> {
        let captions = self.captions().await?;
        Ok(captions.iter().take(max).map(|(_, c)| c.clone()).collect())
    }

    /// caption index を lazy に取得する。1 回 build したら notebook_id 配下で固定。
    /// 再 ingest 時に [`Ellisii::invalidate_caption_cache`] を呼べば次回 search で
    /// 再構築される。
    async fn captions(&self) -> Result<Arc<Vec<(Uuid, String)>>> {
        {
            let lock = self.caption_cache.lock().expect("poisoned");
            if let Some(c) = lock.as_ref() {
                return Ok(c.clone());
            }
        }
        let v = self.store.all_captions(Some(self.notebook_id)).await?;
        let arc = Arc::new(v);
        let mut lock = self.caption_cache.lock().expect("poisoned");
        if lock.is_none() {
            *lock = Some(arc.clone());
        }
        Ok(lock.as_ref().unwrap().clone())
    }

    /// caption の TF-IDF 重み表を lazy に作る。caption_cache と一緒に用意され、
    /// `invalidate_caption_cache` で同時に破棄される。
    async fn caption_idf(&self) -> Result<Arc<std::collections::HashMap<String, f32>>> {
        {
            let lock = self.caption_idf_cache.lock().expect("poisoned");
            if let Some(c) = lock.as_ref() {
                return Ok(c.clone());
            }
        }
        let captions = self.captions().await?;
        let map = ellisii_rag::rerank::compute_caption_idf(&captions);
        let arc = Arc::new(map);
        let mut lock = self.caption_idf_cache.lock().expect("poisoned");
        if lock.is_none() {
            *lock = Some(arc.clone());
        }
        Ok(lock.as_ref().unwrap().clone())
    }

    /// defined-terms index を lazy に取得する (Run 42)。caption / heading と同じく
    /// ingest で invalidate される。
    async fn defined_terms(&self) -> Result<Arc<Vec<(Uuid, String)>>> {
        {
            let lock = self.defined_terms_cache.lock().expect("poisoned");
            if let Some(c) = lock.as_ref() {
                return Ok(c.clone());
            }
        }
        let v = self.store.all_defined_terms(Some(self.notebook_id)).await?;
        let arc = Arc::new(v);
        let mut lock = self.defined_terms_cache.lock().expect("poisoned");
        if lock.is_none() {
            *lock = Some(arc.clone());
        }
        Ok(lock.as_ref().unwrap().clone())
    }

    /// heading index を lazy に取得する。caption と同じく ingest で invalidate される。
    async fn headings(&self) -> Result<Arc<Vec<(Uuid, String)>>> {
        {
            let lock = self.heading_cache.lock().expect("poisoned");
            if let Some(c) = lock.as_ref() {
                return Ok(c.clone());
            }
        }
        let v = self.store.all_headings(Some(self.notebook_id)).await?;
        let arc = Arc::new(v);
        let mut lock = self.heading_cache.lock().expect("poisoned");
        if lock.is_none() {
            *lock = Some(arc.clone());
        }
        Ok(lock.as_ref().unwrap().clone())
    }

    /// caption / heading rerank 用 cache を破棄する。再 ingest や手動で chunk を
    /// 更新した直後に呼ぶ。
    pub fn invalidate_caption_cache(&self) {
        if let Ok(mut lock) = self.caption_cache.lock() {
            *lock = None;
        }
        if let Ok(mut lock) = self.caption_idf_cache.lock() {
            *lock = None;
        }
        if let Ok(mut lock) = self.heading_cache.lock() {
            *lock = None;
        }
        if let Ok(mut lock) = self.defined_terms_cache.lock() {
            *lock = None;
        }
        if let Ok(mut lock) = self.heading_density_cache.lock() {
            *lock = None;
        }
        if let Ok(mut lock) = self.source_count_cache.lock() {
            *lock = None;
        }
    }

    /// 設定済み notebook の **caption 密度** (= caption を持つ chunk / 全 chunk)。
    ///
    /// `caption_rerank` の効きを事前に見積もるためのヘルパ。値域は `0.0..=1.0`、chunk が
    /// 1 件も無いときは `0.0`。法令系 (条文タイトル付きのドキュメント) はこの値が
    /// 0.5 以上になりやすく、原稿テキストや FAQ 系は低い。
    ///
    /// **Rule of thumb** (`docs/eval/recall-evals.md` Run 9):
    /// - `>= 0.5`: caption rerank が支配的に効く → rewriter は **`LlmRewriter`**
    ///   (paraphrase-only) で latency / 品質のバランスが取れる。
    /// - `< 0.5`: caption が薄いので **`MultiExpandRewriter`** (HyDE + decompose +
    ///   paraphrase) で本文側の語彙を補う recall ゲインが見込める。
    /// - `0.0` or 近い値: そもそも caption rerank が無効化されている。`SearchOptions::caption_rerank`
    ///   を `false` にしてレイテンシを節約してよい。
    pub async fn caption_density(&self) -> Result<f32> {
        let total = self
            .store
            .count_chunks_in_scope(Some(self.notebook_id))
            .await?;
        if total == 0 {
            return Ok(0.0);
        }
        let captions = self.captions().await?;
        Ok(captions.len() as f32 / total as f32)
    }

    /// **Heading density** — chunk の `heading_path` がどれだけ「真の見出し」を
    /// 含んでいるかの heuristic。値域は `0.0..=1.0`、chunk が 1 件も無いときは `0.0`。
    ///
    /// 判定: heading_path 連結文字列が **8 文字以上** かつ **非 ASCII 文字を含む**
    /// chunk の割合 (Run 62 で 4 → 8 に refine)。Markdown / マニュアル系の
    /// **記述的タイトル** (e.g. "1.4 ロードバランサ") では高く、doc-id 様の
    /// 英数字 ID や短い topic 名 (e.g. "ACID", "TLS", "第十三条") では低い。
    ///
    /// **Rule of thumb** (`docs/eval/recall-evals.md` Run 51 / 53 / 62):
    /// - `>= 0.8`: heading_path がリッチ (e.g. jp-manual avg 11.8 chars) →
    ///   `SearchOptions::heading_rerank = true` で小 K (≤5) で +0.03〜+0.10 の recall
    ///   ゲインが期待できる
    /// - `0.4..0.8`: 中間 → opt-in 可、ただし絶対 gain は限定的
    /// - `< 0.4`: heading が doc-id 様 / 短い topic 名 / 空 → `heading_rerank` は
    ///   no-op か小退行。既定の `false` のままが安全
    ///
    /// 閾値 8 は Run 62 の corpus 横断調査から決定: title 平均長 ≥8 chars だった
    /// jp-manual (11.8) / jp-labor-law (9.7) は Run 51 で heading_rerank +0.10 /
    /// +0.03、平均 6.9 chars の jp-cs-wiki-hard (e.g. "B木"=2, "TLS"=3, "ACID
    /// (コンピュータ科学)"=18) は -0.05 という分離が出た。旧閾値 4 では
    /// jp-cs-wiki-hard / jp-civil-law-hard も density >= 0.8 を返してしまい、
    /// 「ON 推奨」と誤検知していた (Run 59 で観測)。
    ///
    /// 結果は内部キャッシュしないので、ingest 直後に 1 度呼んで保存することを推奨。
    pub async fn heading_density(&self) -> Result<f32> {
        if let Some(v) = self
            .heading_density_cache
            .lock()
            .expect("poisoned")
            .as_ref()
        {
            return Ok(*v);
        }
        let total = self
            .store
            .count_chunks_in_scope(Some(self.notebook_id))
            .await?;
        let value = if total == 0 {
            0.0
        } else {
            let headings = self.store.all_headings(Some(self.notebook_id)).await?;
            let rich = headings
                .iter()
                .filter(|(_, h)| h.chars().count() >= 8 && !h.is_ascii())
                .count();
            rich as f32 / total as f32
        };
        *self.heading_density_cache.lock().expect("poisoned") = Some(value);
        Ok(value)
    }

    /// `heading_density` の **rule of thumb 閾値** (Run 53)。`heading_rerank` 自動推奨に使う。
    pub const HEADING_RERANK_AUTO_THRESHOLD: f32 = 0.8;

    /// 設定済み notebook の **distinct source 数**。multi-source 判定に使う。
    /// 内部 cache されているので、ingest 後の最初の呼び出し以降は O(1)。
    /// 再 ingest 時は [`Ellisii::invalidate_caption_cache`] で破棄される。
    ///
    /// **Rule of thumb** (`docs/eval/recall-evals.md` Run 56):
    /// - `>= 3` (= [`Self::SOURCE_DEDUP_AUTO_THRESHOLD`]): multi-source notebook
    ///   → `auto_max_chunks_per_source` で dedup1 を自動 ON する候補
    /// - `< 3`: 単独 source / 2 source 程度 → dedup は無効でよい (Lookup でも
    ///   parent 多様性の必要性が小さい)
    pub async fn source_count(&self) -> Result<usize> {
        if let Some(v) = self.source_count_cache.lock().expect("poisoned").as_ref() {
            return Ok(*v);
        }
        let n = self
            .store
            .count_sources_in_scope(Some(self.notebook_id))
            .await?;
        *self.source_count_cache.lock().expect("poisoned") = Some(n);
        Ok(n)
    }

    /// `source_count` の **rule of thumb 閾値** (Run 56)。`auto_max_chunks_per_source`
    /// で dedup1 を自動 ON するかの判定。
    pub const SOURCE_DEDUP_AUTO_THRESHOLD: usize = 3;

    /// corpus が **paraphrase-friendly** かどうかの指標 (0.0..=1.0)。caption に無い
    /// 語彙を body がどれだけ新規導入しているかの平均で、`ellisii_core::caption::body_vocab_novelty`
    /// を `all_captions` の上限 256 件サンプルに対して計算する。
    ///
    /// **Rule of thumb** (`docs/eval/recall-evals.md` Run 11 / 18):
    /// - `>= 0.85`: 概念定義系 (特許法・百科事典) → `multi_query_max_variants >= 2`
    ///   で **LLM rewriter が +5pt 以上 MRR を伸ばす** ことが多い。
    /// - `0.70..0.85`: 中間。rewriter は試す価値があるが、improvements は小さい。
    /// - `< 0.70`: 字面一致主体 (税率・固有名詞中心) → rewriter の利得は小さく、
    ///   variant ノイズで逆に退行することも。`multi_query_max_variants = 0` のままが安全。
    ///
    /// caption 無しの corpus / chunk 取得失敗時は `0.0` を返す (シグナル無効)。
    /// 結果は内部キャッシュしないので、ingest 直後に 1 度呼んで保存することを推奨。
    pub async fn corpus_paraphrase_score(&self) -> Result<f32> {
        let captions = self.captions().await?;
        if captions.is_empty() {
            return Ok(0.0);
        }
        // 大規模 corpus でも O(N) を抑えるため上限 256 件まで。
        const SAMPLE_LIMIT: usize = 256;
        let take = captions.len().min(SAMPLE_LIMIT);
        let ids: Vec<Uuid> = captions.iter().take(take).map(|(id, _)| *id).collect();
        let chunks = self.store.get_chunks_by_ids(&ids).await?;
        if chunks.is_empty() {
            return Ok(0.0);
        }
        let cap_by_id: std::collections::HashMap<Uuid, &str> = captions
            .iter()
            .take(take)
            .map(|(id, c)| (*id, c.as_str()))
            .collect();
        let mut sum = 0.0f32;
        let mut n = 0usize;
        for c in &chunks {
            let Some(cap) = cap_by_id.get(&c.id) else {
                continue;
            };
            // body = chunk.text から caption の文字列を 1 度だけ取り除いた残り。
            // 無ければ chunk.text 全体を body とみなす。
            let body = c.text.replacen(cap, "", 1);
            sum += ellisii_core::caption::body_vocab_novelty(cap, &body);
            n += 1;
        }
        if n == 0 {
            Ok(0.0)
        } else {
            Ok(sum / n as f32)
        }
    }

    /// 各クエリの char-bigram のうち、corpus body のいずれかに literal に出現する
    /// 割合 (= query-side recall) の最大値を取り、クエリ集合全体で平均する。
    ///
    /// **解釈** (`docs/eval/recall-evals.md` Run 22):
    /// - `>= 0.7`: クエリ語彙が既に corpus body に literal に出現 → paraphrase
    ///   rewrite で recall が伸びる余地は小さい (yokohama-style literal lookup)
    /// - `<= 0.4`: クエリと body の lexical gap が大きい → paraphrase rewrite で
    ///   bridge する価値あり
    ///
    /// 動機: Run 21 の `specific_query_ratio` だけでは捕捉できない false positive
    /// (yokohama「税率はいくら」のような自然文 literal lookup) を query-vs-corpus
    /// 軸で補足する。`specific_query_ratio` と組み合わせると rewriter ON/OFF の
    /// 強い OFF 推奨ガードになる。
    ///
    /// 実装: `all_captions` の対応 chunk 上限 256 件をサンプルして bodies として渡す。
    /// caption 無し corpus では 0.0 を返す (シグナル無効)。
    pub async fn query_body_literal_match<S: AsRef<str>>(&self, queries: &[S]) -> Result<f32> {
        if queries.is_empty() {
            return Ok(0.0);
        }
        let captions = self.captions().await?;
        if captions.is_empty() {
            return Ok(0.0);
        }
        const SAMPLE_LIMIT: usize = 256;
        let take = captions.len().min(SAMPLE_LIMIT);
        let ids: Vec<Uuid> = captions.iter().take(take).map(|(id, _)| *id).collect();
        let chunks = self.store.get_chunks_by_ids(&ids).await?;
        if chunks.is_empty() {
            return Ok(0.0);
        }
        let bodies: Vec<&str> = chunks.iter().map(|c| c.text.as_str()).collect();
        Ok(ellisii_rag::query_body_recall_mean(queries, &bodies))
    }

    /// query 集合の **タイトル直接マッチ度** (0.0..=1.0、Run 26 の signal)。
    /// `ChunkConfig::synthesize_caption_from_heading` (Run 24) を有効化すべきかの予測指標。
    ///
    /// **解釈** (`docs/eval/recall-evals.md` Run 26):
    /// - `>= 0.3`: query がタイトル直接マッチ寄り (FAQ / 概念定義 lookup)
    ///   → caption synthesis ON で MRR が伸びる (Run 25 jp-cs-wiki = 0.452 → +16.7pt)
    /// - `< 0.3`: query が paraphrase / 概念ジャンプ寄り
    ///   → caption synthesis ON で MRR が下がるリスク (Run 25 jp-cs-wiki-hard = 0.051 → −4.5pt)
    ///
    /// **使い方** (起動時 1 度計算 → `ChunkConfig` に反映する想定):
    /// ```ignore
    /// let signal = ellisii.query_title_match(&golden_queries).await?;
    /// let cfg = ellisii_chunker::ChunkConfig {
    ///     synthesize_caption_from_heading: signal >= 0.3,
    ///     ..Default::default()
    /// };
    /// ```
    ///
    /// 実装: `captions` または `headings` の対応 chunk 上限 256 件をサンプルし、
    /// 各 chunk の `heading_path.last()` を title として `query_title_match_mean` に渡す。
    /// title が抽出できない (空 corpus / heading_path 全部空) なら `0.0`。
    pub async fn query_title_match<S: AsRef<str>>(&self, queries: &[S]) -> Result<f32> {
        if queries.is_empty() {
            return Ok(0.0);
        }
        const SAMPLE_LIMIT: usize = 256;
        // chunk id を集める。captions が空でも headings は埋まっていることがあるので両方見る。
        let ids: Vec<Uuid> = {
            let captions = self.captions().await?;
            if !captions.is_empty() {
                captions
                    .iter()
                    .take(SAMPLE_LIMIT)
                    .map(|(id, _)| *id)
                    .collect()
            } else {
                let headings = self.headings().await?;
                if headings.is_empty() {
                    return Ok(0.0);
                }
                headings
                    .iter()
                    .take(SAMPLE_LIMIT)
                    .map(|(id, _)| *id)
                    .collect()
            }
        };
        let chunks = self.store.get_chunks_by_ids(&ids).await?;
        let titles: Vec<String> = chunks
            .iter()
            .filter_map(|c| c.heading_path.last().cloned())
            .filter(|t| !t.trim().is_empty())
            .collect();
        if titles.is_empty() {
            return Ok(0.0);
        }
        Ok(ellisii_rag::query_title_match_mean(queries, &titles))
    }

    /// 検索 → LLM stream で RAG 回答。`with_llm_*` で LLM を組み込んでいない場合は
    /// エラー。`on_token` は生成トークン到着のたびに呼ばれる。
    /// 戻り値は引用に使ったヒット (citation 表示用)。
    ///
    /// `opts.route_by_intent = true` (既定) かつ classifier が構築済みのときは、
    /// クエリの意図に応じて retrieval 戦略を切り替える ([`AskOptions`] 参照)。
    ///
    /// クエリは入口で NFKC 正規化される ([`Self::search`] と同じ)。
    pub async fn ask<F>(&self, query: &str, opts: AskOptions, on_token: F) -> Result<Vec<SearchHit>>
    where
        F: FnMut(String) + Send + 'static,
    {
        let query_owned = ellisii_jp_tokenizer_nfkc::nfkc(query);
        let query = query_owned.as_str();
        let llm = self
            .llm
            .as_ref()
            .ok_or_else(|| Error::Llm("no LLM configured (call with_llm_*)".into()))?;

        let intent: Option<Intent> = if opts.route_by_intent {
            if let Some(c) = self.intent_classifier.as_ref() {
                match c.classify(query).await {
                    Ok(i) => Some(i),
                    Err(e) => {
                        tracing::warn!("intent classify failed: {e}; falling back to hybrid");
                        None
                    }
                }
            } else {
                None
            }
        } else {
            None
        };

        let hits = match &intent {
            Some(Intent::Summary { topic }) => {
                let reps = match topic.as_deref().filter(|t| !t.is_empty()) {
                    Some(t) => {
                        self.store
                            .representative_chunks_for_topic(Some(self.notebook_id), opts.top_k, t)
                            .await?
                    }
                    None => {
                        self.store
                            .representative_chunks(Some(self.notebook_id), opts.top_k)
                            .await?
                    }
                };
                reps.into_iter()
                    .map(|c| SearchHit {
                        chunk: c,
                        score: 1.0,
                        source: ellisii_core::HitSource::Vector,
                    })
                    .collect()
            }
            Some(Intent::Compare { .. }) => {
                // 複数 source を覆面的に拾うため per_source は小さめ
                let reps = self
                    .store
                    .representative_chunks(Some(self.notebook_id), (opts.top_k / 2).max(2))
                    .await?;
                reps.into_iter()
                    .map(|c| SearchHit {
                        chunk: c,
                        score: 1.0,
                        source: ellisii_core::HitSource::Vector,
                    })
                    .collect()
            }
            Some(Intent::Smalltalk) | Some(Intent::Lookup) | None => {
                let base = opts.semantic_weight.clamp(0.0, 1.0);
                let semantic = if opts.auto_adjust_weight {
                    ellisii_rag::adjust_hybrid_weight_for_query(base, query)
                } else {
                    base
                };
                let weights = HybridWeights { semantic };
                let pool_top = (opts.top_k * 5).max(opts.top_k);
                let effective_max_variants = if opts.skip_rewrite_on_specific
                    && opts.multi_query_max_variants > 0
                    && ellisii_rag::is_specific_query(query)
                {
                    tracing::debug!(
                        "skipping rewriter: query is specific (rule-based) — `{}`",
                        query
                    );
                    0
                } else {
                    opts.multi_query_max_variants
                };
                let mut fused = self
                    .hybrid_pool(
                        query,
                        opts.top_k * 5,
                        weights,
                        effective_max_variants,
                        opts.multi_query_variant_weight,
                        pool_top,
                        opts.variant_caption_filter_threshold,
                    )
                    .await?;
                if opts.caption_rerank {
                    self.apply_caption_rerank(query, &mut fused).await?;
                }
                let heading_on = opts.heading_rerank
                    || (opts.auto_heading_rerank
                        && self.heading_density().await.unwrap_or(0.0)
                            >= Self::HEADING_RERANK_AUTO_THRESHOLD);
                if heading_on {
                    ellisii_rag::rerank::heading_boost_in_place(query, &mut fused, 1.0);
                }
                // Run 16 で観測した non-composition 退行 (rewriter + CE で nDCG -0.016) を防ぐ。
                let ce_should_run = opts.ce_rerank_top_n > 0
                    && !(opts.skip_ce_when_rewriting && effective_max_variants > 0);
                if ce_should_run {
                    self.apply_ce_rerank(
                        query,
                        &mut fused,
                        opts.ce_rerank_top_n,
                        opts.ce_rerank_weight,
                    )
                    .await;
                } else if opts.ce_rerank_top_n > 0
                    && opts.skip_ce_when_rewriting
                    && effective_max_variants > 0
                {
                    tracing::debug!(
                        "skipping CE rerank: rewriter active (variants={}) — see recall-evals Run 16",
                        effective_max_variants
                    );
                }
                let effective_max_per_source = if opts.max_chunks_per_source > 0 {
                    opts.max_chunks_per_source
                } else if opts.auto_max_chunks_per_source
                    && self.source_count().await.unwrap_or(0) >= Self::SOURCE_DEDUP_AUTO_THRESHOLD
                {
                    1
                } else {
                    0
                };
                if effective_max_per_source > 0 {
                    ellisii_rag::rerank::dedup_by_source_in_place(
                        &mut fused,
                        effective_max_per_source,
                    );
                }
                fused.truncate(opts.top_k);
                fused
            }
        };
        if let Some(i) = &intent {
            tracing::debug!("ask: intent={:?}, hits={}", i, hits.len());
        }

        let context = hits
            .iter()
            .enumerate()
            .map(|(i, h)| format!("<source id={}>{}</source>", i + 1, h.chunk.text))
            .collect::<Vec<_>>()
            .join("\n");
        let base_system = opts
            .system
            .clone()
            .unwrap_or_else(|| DEFAULT_RAG_SYSTEM.to_string());
        let user_text = format!("質問: {query}\n\n参考:\n{context}");
        let req = LlmRequest {
            system: base_system.clone(),
            history: Vec::new(),
            user: user_text.clone(),
            max_tokens: opts.max_tokens,
            temperature: opts.temperature,
        };

        // Run 64: no_citation_retry が有効なら、on_token を 2 周渡せるよう
        // Arc<Mutex<F>> に詰めて token をバッファしつつ pass-through する。
        // hits が空のとき (general mode / smalltalk) は retry 不要。
        let retry_enabled = opts.no_citation_retry && !hits.is_empty();
        if !retry_enabled {
            llm.generate_stream(req, Box::new(on_token)).await?;
            return Ok(hits);
        }

        let answer_buf: Arc<std::sync::Mutex<String>> =
            Arc::new(std::sync::Mutex::new(String::new()));
        let on_token_arc: Arc<std::sync::Mutex<F>> = Arc::new(std::sync::Mutex::new(on_token));

        let make_cb = |buf: Arc<std::sync::Mutex<String>>,
                       f: Arc<std::sync::Mutex<F>>|
         -> Box<dyn FnMut(String) + Send + 'static> {
            Box::new(move |tok: String| {
                if let Ok(mut b) = buf.lock() {
                    b.push_str(&tok);
                }
                if let Ok(mut g) = f.lock() {
                    (*g)(tok);
                }
            })
        };

        llm.generate_stream(req, make_cb(answer_buf.clone(), on_token_arc.clone()))
            .await?;

        let answer1 = answer_buf.lock().expect("poisoned").clone();
        let stats = ellisii_rag::citation::verify_citations(&answer1, &hits);
        if stats.total > 0 {
            return Ok(hits);
        }
        tracing::info!(
            "no_citation_retry: 初回応答に [N] marker 無し (hits={}, len={}); 厳格 prompt で再生成",
            hits.len(),
            answer1.chars().count()
        );

        // Divider をユーザ側 on_token に投げる (UI で「retry に入った」のマーカー)。
        const RETRY_DIVIDER: &str = "\n\n---\n[出典付きで再生成]\n\n";
        if let Ok(mut g) = on_token_arc.lock() {
            (*g)(RETRY_DIVIDER.into());
        }
        if let Ok(mut b) = answer_buf.lock() {
            b.push_str(RETRY_DIVIDER);
        }

        // 厳格 prompt: 既存 system に「必ず [N] 形式で引用」を追加。
        let strict_system = format!(
            "{}\n\n重要: 回答には必ず参考資料の番号 ([1] [2] ...) を [N] 形式で引用すること。\
             引用無しの回答は許可されない。資料に該当が無ければその旨を述べた上で、最低 1 つの [N] で根拠を示す。",
            base_system
        );
        let retry_req = LlmRequest {
            system: strict_system,
            history: Vec::new(),
            user: user_text,
            max_tokens: opts.max_tokens,
            temperature: opts.temperature,
        };
        llm.generate_stream(retry_req, make_cb(answer_buf.clone(), on_token_arc))
            .await?;

        // Run 66: faithfulness ratio gate。retry 後の最終応答で unsupported_ratio が
        // 閾値超なら tracing::warn を残す。production 側ではこの warn を拾って UI に
        // 警告フラグを立てるなどの対応に使う。
        let final_answer = answer_buf.lock().expect("poisoned").clone();
        let final_stats = ellisii_rag::citation::verify_citations(&final_answer, &hits);
        if final_stats.is_unsupported_high() {
            tracing::warn!(
                "citation faithfulness gate: unsupported_ratio={:.2} (total={}, unsupported={}); answer may contain hallucinated citations",
                final_stats.unsupported_ratio(),
                final_stats.total,
                final_stats.unsupported,
            );
        }
        Ok(hits)
    }

    /// 設定済みの意図分類器を取り出す escape hatch。
    pub fn intent_classifier(&self) -> Option<Arc<dyn IntentClassifier>> {
        self.intent_classifier.clone()
    }

    /// 内部の VectorStore を直接触りたいときの escape hatch。
    pub fn store(&self) -> Arc<dyn VectorStore> {
        self.store.clone()
    }

    /// 内部の Embedder を直接触りたいときの escape hatch。
    pub fn embedder(&self) -> Arc<dyn Embedder> {
        self.embedder.clone()
    }

    /// 設定済みの notebook_id (= namespace)。
    pub fn notebook_id(&self) -> Uuid {
        self.notebook_id
    }
}

/// `index_dir` のファイル単位結果 (集計用、外部公開しない)。
enum FileOutcome {
    Ingested { chunks: usize },
    Unchanged,
    Failed,
}

fn is_not_hidden(entry: &walkdir::DirEntry) -> bool {
    // walkdir の `filter_entry` は root も含めて全エントリで呼ばれる。
    // ユーザが渡した root 自体が `.` 始まりのこと (例: macOS の tempfile) も
    // あるので、depth=0 (root) は無条件に通す。
    if entry.depth() == 0 {
        return true;
    }
    !entry
        .file_name()
        .to_str()
        .map(|s| s.starts_with('.') && s != "." && s != "..")
        .unwrap_or(false)
}
