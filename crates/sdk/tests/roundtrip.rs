//! ディレクトリ index → search の最小限の round-trip テスト。
//! 外部モデル不要 (DummyEmbedder + InMemoryStore)。

use ellisii_sdk::{
    Ellisii, IndexEvent, IndexOptions, IngestPathOutcome, MemoryIndexCache, SearchOptions,
};
use std::sync::{Arc, Mutex};

#[tokio::test]
async fn index_dir_then_search_finds_indexed_text() {
    let tmp = tempfile::tempdir().unwrap();
    let docs = tmp.path().join("docs");
    std::fs::create_dir_all(&docs).unwrap();
    std::fs::write(
        docs.join("a.txt"),
        // is_retrieval_noise の content_chars < 25 フィルタを抜けるため、
        // 各ファイルを 25 文字以上の内容語にする。
        "民法は私法の基本ルールを定めた法律であり、契約や所有権、相続といった日常生活に関わる規律を広く扱います。",
    )
    .unwrap();
    std::fs::write(
        docs.join("b.md"),
        "# 物権\n\n所有権と用益物権の章。占有権や地上権、地役権、留置権なども含めて体系的に整理されています。",
    )
    .unwrap();
    // .ds_store のような認識外ファイルは skip されるはず
    std::fs::write(docs.join(".DS_Store"), b"junk").unwrap();
    // サブディレクトリも再帰される
    let sub = docs.join("sub");
    std::fs::create_dir_all(&sub).unwrap();
    std::fs::write(sub.join("c.txt"), "債権について。契約に基づく給付請求権から不法行為に基づく損害賠償請求権までを扱う民法の重要分野です。").unwrap();

    let ellisii = Ellisii::builder()
        .with_embedder_dummy(64)
        .with_store_memory()
        .build()
        .unwrap();

    let events: Arc<Mutex<Vec<IndexEvent>>> = Arc::new(Mutex::new(Vec::new()));
    let events_cb = events.clone();
    let report = ellisii
        .index_dir(
            &docs,
            IndexOptions {
                on_progress: Some(Box::new(move |ev| {
                    events_cb.lock().unwrap().push(ev);
                })),
                ..Default::default()
            },
        )
        .await
        .unwrap();

    assert_eq!(report.ingested, 3, "3 ファイルが取り込まれているはず");
    assert!(report.total_chunks >= 3);
    assert!(events
        .lock()
        .unwrap()
        .iter()
        .any(|e| matches!(e, IndexEvent::Ingested { .. })));

    let hits = ellisii
        .search(
            "民法",
            SearchOptions {
                top_k: 5,
                ..Default::default()
            },
        )
        .await
        .unwrap();
    assert!(!hits.is_empty(), "「民法」で何かしらヒットするはず");
    assert!(
        hits.iter().any(|h| h.chunk.text.contains("民法")),
        "ヒットの中に「民法」を含むチャンクがあるはず"
    );
}

#[tokio::test]
async fn second_index_with_cache_marks_files_unchanged() {
    let tmp = tempfile::tempdir().unwrap();
    std::fs::write(tmp.path().join("a.txt"), "hello").unwrap();
    std::fs::write(tmp.path().join("b.md"), "# h\n\nworld").unwrap();
    let cache = Arc::new(MemoryIndexCache::new());
    let ellisii = Ellisii::builder()
        .with_embedder_dummy(32)
        .with_store_memory()
        .with_index_cache(cache.clone())
        .build()
        .unwrap();

    let r1 = ellisii
        .index_dir(tmp.path(), IndexOptions::default())
        .await
        .unwrap();
    assert_eq!(r1.ingested, 2);
    assert_eq!(r1.unchanged, 0);

    // 同じ内容で 2 回目: 全部 Unchanged になるはず
    let r2 = ellisii
        .index_dir(tmp.path(), IndexOptions::default())
        .await
        .unwrap();
    assert_eq!(r2.ingested, 0);
    assert_eq!(r2.unchanged, 2);
}

#[tokio::test]
async fn modified_file_is_reingested_with_replacement() {
    let tmp = tempfile::tempdir().unwrap();
    let f = tmp.path().join("a.txt");
    // is_retrieval_noise の content_chars < 25 を抜けるため英字 30 文字以上に。
    std::fs::write(
        &f,
        "old content describing a long paragraph about the law and contracts for testing only",
    )
    .unwrap();
    let cache = Arc::new(MemoryIndexCache::new());
    let ellisii = Ellisii::builder()
        .with_embedder_dummy(32)
        .with_store_memory()
        .with_index_cache(cache.clone())
        .build()
        .unwrap();

    let _ = ellisii
        .index_dir(tmp.path(), IndexOptions::default())
        .await
        .unwrap();
    // 古い内容のヒット数を控える
    let hits_old = ellisii
        .search("old", SearchOptions::default())
        .await
        .unwrap();
    assert!(hits_old.iter().any(|h| h.chunk.text.contains("old")));

    // 内容を変える + mtime も変わるよう sleep
    std::thread::sleep(std::time::Duration::from_millis(20));
    std::fs::write(
        &f,
        "new shiny content with a long body about a freshly modified document file for testing",
    )
    .unwrap();

    let r2 = ellisii
        .index_dir(tmp.path(), IndexOptions::default())
        .await
        .unwrap();
    assert_eq!(r2.ingested, 1, "modified file は再 ingest される");
    assert_eq!(r2.unchanged, 0);

    // 古い内容のヒットは消えていて、新しいヒットが出るはず
    let hits_new = ellisii
        .search("shiny", SearchOptions::default())
        .await
        .unwrap();
    assert!(hits_new.iter().any(|h| h.chunk.text.contains("shiny")));
    let hits_old2 = ellisii
        .search("old", SearchOptions::default())
        .await
        .unwrap();
    assert!(
        !hits_old2.iter().any(|h| h.chunk.text.contains("old")),
        "古い chunk は delete_by_source で消えているはず"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn parallel_index_dir_ingests_all_files() {
    let tmp = tempfile::tempdir().unwrap();
    for i in 0..10 {
        std::fs::write(
            tmp.path().join(format!("doc{i}.txt")),
            format!("file number {i} contents"),
        )
        .unwrap();
    }
    let ellisii = Ellisii::builder()
        .with_embedder_dummy(32)
        .with_store_memory()
        .build()
        .unwrap();
    let report = ellisii
        .index_dir(
            tmp.path(),
            IndexOptions {
                concurrency: Some(4),
                ..Default::default()
            },
        )
        .await
        .unwrap();
    assert_eq!(report.ingested, 10);
    assert!(report.total_chunks >= 10);
}

#[tokio::test]
async fn index_file_returns_unchanged_outcome() {
    let tmp = tempfile::tempdir().unwrap();
    let f = tmp.path().join("a.txt");
    std::fs::write(&f, "content").unwrap();
    let cache = Arc::new(MemoryIndexCache::new());
    let ellisii = Ellisii::builder()
        .with_embedder_dummy(32)
        .with_store_memory()
        .with_index_cache(cache)
        .build()
        .unwrap();

    let o1 = ellisii.index_file(&f).await.unwrap();
    assert!(matches!(o1, IngestPathOutcome::Ingested(_)));

    let o2 = ellisii.index_file(&f).await.unwrap();
    assert!(o2.is_unchanged());
}

#[tokio::test]
async fn search_returns_empty_when_index_is_empty() {
    let ellisii = Ellisii::builder()
        .with_embedder_dummy(32)
        .with_store_memory()
        .build()
        .unwrap();
    let hits = ellisii
        .search("anything", SearchOptions::default())
        .await
        .unwrap();
    assert!(hits.is_empty());
}

#[tokio::test]
async fn ask_without_llm_returns_error() {
    let ellisii = Ellisii::builder()
        .with_embedder_dummy(32)
        .with_store_memory()
        .build()
        .unwrap();
    let res = ellisii
        .ask(
            "anything",
            ellisii_sdk::AskOptions::default(),
            |_t: String| {},
        )
        .await;
    assert!(res.is_err(), "LLM 未設定時はエラーを返すはず");
}

#[tokio::test]
async fn ask_with_intent_classifier_routes_summary_to_representative() {
    use async_trait::async_trait;
    use ellisii_core::Result;
    use ellisii_llm_core::{LlmBackend, LlmRequest};
    use ellisii_rag::intent_classifier::{Intent, IntentClassifier};
    use std::sync::Arc;

    // Summary{None} を必ず返す classifier
    struct AlwaysSummary;
    #[async_trait]
    impl IntentClassifier for AlwaysSummary {
        async fn classify(&self, _q: &str) -> Result<Intent> {
            Ok(Intent::summary_all())
        }
    }
    // 受け取った user prompt を記録するだけの LLM
    struct CaptureLlm(Arc<Mutex<String>>);
    #[async_trait]
    impl LlmBackend for CaptureLlm {
        async fn generate_stream(
            &self,
            req: LlmRequest,
            mut on_token: Box<dyn FnMut(String) + Send + 'static>,
        ) -> Result<()> {
            *self.0.lock().unwrap() = req.user.clone();
            on_token("ok".into());
            Ok(())
        }
    }

    let captured = Arc::new(Mutex::new(String::new()));
    let llm = Arc::new(CaptureLlm(captured.clone()));
    let ellisii = Ellisii::builder()
        .with_embedder_dummy(32)
        .with_store_memory()
        .with_llm(llm)
        .with_intent_classifier(Arc::new(AlwaysSummary))
        .build()
        .unwrap();

    // 取り込み: heading_path の異なる 2 source を作って representative_chunks が
    // 拾えるようにする
    let tmp = tempfile::tempdir().unwrap();
    std::fs::write(tmp.path().join("a.md"), "# 第一編 総則\n\n前文").unwrap();
    std::fs::write(tmp.path().join("b.md"), "# 第二編 物権\n\n物権の冒頭").unwrap();
    let _ = ellisii
        .index_dir(tmp.path(), ellisii_sdk::IndexOptions::default())
        .await
        .unwrap();

    let hits = ellisii
        .ask("全体を要約して", ellisii_sdk::AskOptions::default(), |_| {})
        .await
        .unwrap();
    assert!(!hits.is_empty());
    let user_prompt = captured.lock().unwrap().clone();
    assert!(
        user_prompt.contains("<source"),
        "context が組み立てられているはず"
    );
    // Summary route の chunk は heading top-level の先頭が含まれる
    assert!(user_prompt.contains("前文") || user_prompt.contains("物権"));
}

#[tokio::test]
async fn ask_with_stub_llm_streams_tokens() {
    use async_trait::async_trait;
    use ellisii_core::Result;
    use ellisii_llm_core::{LlmBackend, LlmRequest};
    use std::sync::Arc;

    struct CannedLlm;
    #[async_trait]
    impl LlmBackend for CannedLlm {
        async fn generate_stream(
            &self,
            _req: LlmRequest,
            mut on_token: Box<dyn FnMut(String) + Send + 'static>,
        ) -> Result<()> {
            on_token("hello ".to_string());
            on_token("world".to_string());
            Ok(())
        }
    }

    let ellisii = Ellisii::builder()
        .with_embedder_dummy(32)
        .with_store_memory()
        .with_llm(Arc::new(CannedLlm))
        .build()
        .unwrap();
    let collected: Arc<Mutex<String>> = Arc::new(Mutex::new(String::new()));
    let cb_buf = collected.clone();
    let _hits = ellisii
        .ask(
            "test",
            ellisii_sdk::AskOptions::default(),
            move |t: String| cb_buf.lock().unwrap().push_str(&t),
        )
        .await
        .unwrap();
    assert_eq!(collected.lock().unwrap().as_str(), "hello world");
}

#[tokio::test]
async fn index_chunks_indexes_pre_chunked_data_without_parser() {
    use ellisii_core::Chunk;
    use uuid::Uuid;

    let ellisii = Ellisii::builder()
        .with_embedder_dummy(32)
        .with_store_memory()
        .build()
        .unwrap();

    // 外部で chunk 済の体裁。25 文字以上の content を持つ 2 chunk。
    let source_id = Uuid::new_v4();
    let chunks = vec![
        Chunk {
            id: Uuid::new_v4(),
            source_id,
            ord: 0,
            text: "事前に分割済みの 1 つ目のチャンクで、検索対象として十分な長さの本文を含みます。"
                .into(),
            heading_path: vec!["プリチャンク".into(), "セクションA".into()],
            page: None,
            bbox: None,
            summary: None,
        },
        Chunk {
            id: Uuid::new_v4(),
            source_id,
            ord: 1,
            text: "事前に分割済みの 2 つ目のチャンク。別のテーマを 25 文字以上で記述しています。"
                .into(),
            heading_path: vec!["プリチャンク".into(), "セクションB".into()],
            page: None,
            bbox: None,
            summary: None,
        },
    ];

    let written = ellisii.index_chunks(chunks).await.unwrap();
    assert_eq!(written, 2);

    // 検索で hit すること (DummyEmbedder は精度低いので kw 経路でも探せる term を使う)
    let hits = ellisii
        .search(
            "プリチャンク",
            SearchOptions {
                top_k: 5,
                ..Default::default()
            },
        )
        .await
        .unwrap();
    assert!(
        !hits.is_empty(),
        "「プリチャンク」で何らかの hit が返るはず"
    );
}

#[tokio::test]
async fn builder_with_chunker_uses_custom_chunker() {
    use ellisii_chunker::{ChunkConfig, Chunker, DefaultChunker};
    use ellisii_core::Chunk;
    use ellisii_parsers_core::ParsedDocument;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use uuid::Uuid;

    // 呼び出し回数をカウントするだけの薄い custom chunker
    struct CountingChunker {
        inner: DefaultChunker,
        calls: Arc<AtomicUsize>,
    }
    impl Chunker for CountingChunker {
        fn chunk(&self, doc: &ParsedDocument, source_id: Uuid) -> Vec<Chunk> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            self.inner.chunk(doc, source_id)
        }
    }

    let calls = Arc::new(AtomicUsize::new(0));
    let chunker = Arc::new(CountingChunker {
        inner: DefaultChunker::new(ChunkConfig::default()),
        calls: calls.clone(),
    });

    let tmp = tempfile::tempdir().unwrap();
    std::fs::write(
        tmp.path().join("a.txt"),
        "民法は私法の基本ルールを定めた法律であり、契約や所有権、相続といった日常生活に関わる規律を広く扱います。",
    )
    .unwrap();

    let ellisii = Ellisii::builder()
        .with_embedder_dummy(32)
        .with_store_memory()
        .with_chunker(chunker)
        .build()
        .unwrap();
    let _ = ellisii.index_file(tmp.path().join("a.txt")).await.unwrap();

    assert!(
        calls.load(Ordering::SeqCst) >= 1,
        "custom chunker が呼ばれた形跡が無い"
    );
}
