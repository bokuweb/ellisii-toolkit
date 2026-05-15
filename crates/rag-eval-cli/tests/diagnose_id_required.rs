//! 「IDリクワイアドとキーレスエントリについて解説して」が retrieve で
//! 本文チャンクを top_k に乗せられるかを **本番 DB に対して** 直接検証する。
//!
//! 動機: 手動で Tauri を起動して再生する代わりに、~/Library/Application
//! Support/ellisii/ellisii.db を直接開いて hybrid retrieve のみを走らせ、
//! どのチャンクが何位で来るかをログに吐く。Provence や LLM の影響を排除し、
//! 純粋に「retrieve 段階で本文が拾えているか」を切り分ける。
//!
//! 実行: `cargo test -p ellisii-rag-eval-cli --test diagnose_id_required \
//!        -- --ignored --nocapture`
//!
//! 必要環境:
//!   - ~/Library/Application Support/ellisii/ellisii.db (1024dim static-jp で
//!     既に ingest 済み)
//!   - ~/Library/Application Support/ellisii/models/static-embedding-japanese
//!   - ~/Library/Application Support/ellisii/models/vaporetto/...model.zst

use ellisii_embed_core::Embedder;
use ellisii_jp_tokenizer_core::JpTokenizer;
use ellisii_jp_tokenizer_vaporetto::VaporettoTokenizer;
use ellisii_rag::HybridWeights;
use ellisii_rag_eval_cli::EmbedderKind;
use ellisii_store_core::{Scope, VectorStore};
use ellisii_store_sqlite::SqliteStore;
use std::path::PathBuf;
use std::sync::Arc;
use uuid::Uuid;

const NOTEBOOK_ID: &str = "ad657643-c080-495c-8121-7d2d6d097127";
const QUERY: &str = "IDリクワイアドとキーレスエントリについて解説して";

fn data_dir() -> PathBuf {
    let home = std::env::var_os("HOME").expect("HOME");
    PathBuf::from(home).join("Library/Application Support/ellisii")
}

#[tokio::test]
#[ignore]
async fn diagnose_id_required_retrieval() {
    let dd = data_dir();
    let db = dd.join("ellisii.db");
    assert!(db.is_file(), "ellisii.db missing at {db:?}");

    let static_jp = dd.join("models/static-embedding-japanese");
    assert!(static_jp.is_dir(), "static-jp model dir missing");
    let vaporetto_model = dd.join("models/vaporetto/bccwj-suw+unidic_pos+pron.model.zst");
    assert!(vaporetto_model.is_file(), "vaporetto model missing");

    let embedder = EmbedderKind::StaticJp {
        model_dir: static_jp,
    }
    .build()
    .expect("static-jp build");
    let dim = embedder.dim();

    let tokenizer: Arc<dyn JpTokenizer> =
        Arc::new(VaporettoTokenizer::from_zst(&vaporetto_model).expect("vaporetto load"));
    let store = SqliteStore::open_with_tokenizer(&db, dim, tokenizer).expect("open ellisii.db");

    let scope: Scope = Some(Uuid::parse_str(NOTEBOOK_ID).unwrap());

    // 元クエリ + 別表記 variants で検索する。retrieve_multi と同様の
    // 「クエリ集合を vec/kw 両方で叩いて RRF 融合」を、テスト側で直接実装する。
    // 別表記は MultiExpand が出すであろう想定パターンを事前に与えて、
    // rewriter の影響を切り離して retrieve 単体の地力を見る。
    let query_variants: Vec<&str> = vec![
        QUERY,
        "IDリクワイアド",
        "キーレスエントリ",
        "ID Required アンチパターン",
        "Keyless Entry アンチパターン",
        "とりあえずID",
        "外部キー嫌い",
    ];

    let top_k = 20usize;
    let weights = HybridWeights::default();

    // 元クエリだけの結果と、variants を加えた結果の 2 通りを比較する。
    let single_hits = retrieve_one(&store, &embedder, scope, QUERY, top_k, weights).await;
    println!("\n=== single-query top {top_k} ===");
    print_hits(&single_hits);

    let mut all_rankings: Vec<(Vec<ellisii_core::SearchHit>, f32)> = Vec::new();
    for (i, q) in query_variants.iter().enumerate() {
        let q_emb = embedder.embed(&[q.to_string()]).await.unwrap();
        let vec_hits = store.search(scope, &q_emb[0], top_k * 5).await.unwrap();
        let kw_hits = store.keyword_search(scope, q, top_k * 5).await.unwrap();
        let qw = if i == 0 { 1.0 } else { 0.7 };
        all_rankings.push((vec_hits, weights.vector() * qw));
        all_rankings.push((kw_hits, weights.keyword() * qw));
    }
    let multi_hits = ellisii_rag::rrf_weighted(&all_rankings, top_k);
    println!("\n=== multi-query (元 + 6 variants) top {top_k} ===");
    print_hits(&multi_hits);

    // アサーション: 本文チャンクが top に来ているか。
    //
    // ground truth (ellisii.db を SELECT で確認したもの):
    //   IDリクワイアド本文: ord 732〜747 (heading_path "3章 IDリクワイアド..."、
    //                       Page 68〜69 の解説文)
    //   キーレスエントリ本文: ord 891 などに「次のような言葉を耳にしたら、
    //                         それはおそらく『キーレスエントリ(外部キー嫌い)
    //                         アンチパターンの兆候があることを示しています。」
    //                         を含む文。
    //
    // 「成功」の最低ライン: multi-query 経路で top_k=20 の中に
    //   - heading_path に "3章 IDリクワイアド" を含む本文 chunk が **2 件以上**
    //   - text に "キーレスエントリ" または "外部キー嫌い" を含む本文 chunk が
    //     **1 件以上**
    // 両方含まれること。これが満たされれば LLM 段で本物の解説に到達できる。
    let id_body_count = multi_hits
        .iter()
        .filter(|h| {
            h.chunk
                .heading_path
                .iter()
                .any(|s| s.contains("IDリクワイアド"))
                && h.chunk.text.chars().count() > 50
        })
        .count();
    let keyless_body_count = multi_hits
        .iter()
        .filter(|h| {
            (h.chunk.text.contains("キーレスエントリ") || h.chunk.text.contains("外部キー嫌い"))
                && h.chunk.text.chars().count() > 50
        })
        .count();
    println!(
        "\n[summary] id_body_in_top{top_k}={id_body_count}  keyless_body_in_top{top_k}={keyless_body_count}"
    );

    assert!(
        id_body_count >= 2,
        "expected >=2 body chunks under '3章 IDリクワイアド' in top {top_k}, got {id_body_count}"
    );
    assert!(
        keyless_body_count >= 1,
        "expected >=1 chunk containing 'キーレスエントリ' / '外部キー嫌い' in top {top_k}, got {keyless_body_count}"
    );
}

async fn retrieve_one(
    store: &SqliteStore,
    embedder: &Arc<dyn Embedder>,
    scope: Scope,
    query: &str,
    top_k: usize,
    weights: HybridWeights,
) -> Vec<ellisii_core::SearchHit> {
    let q_emb = embedder.embed(&[query.to_string()]).await.unwrap();
    let vec_hits = store.search(scope, &q_emb[0], top_k * 5).await.unwrap();
    let kw_hits = store.keyword_search(scope, query, top_k * 5).await.unwrap();
    ellisii_rag::rrf_weighted(
        &[(vec_hits, weights.vector()), (kw_hits, weights.keyword())],
        top_k,
    )
}

fn print_hits(hits: &[ellisii_core::SearchHit]) {
    for (i, h) in hits.iter().enumerate() {
        let snippet: String = h.chunk.text.chars().take(60).collect();
        let hp = h.chunk.heading_path.join(" / ");
        println!(
            "  [{rank:>2}] score={s:.4} ord={ord:>4} src={src:?} hp=[{hp}] text={snippet}",
            rank = i + 1,
            s = h.score,
            ord = h.chunk.ord,
            src = h.source,
        );
    }
}
