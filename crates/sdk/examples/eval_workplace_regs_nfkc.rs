//! `eval_workplace_regs` の NFKC 正規化 A/B。
//!
//! 現状 ellisii は **chunker / query 側どちらでも NFKC 正規化していない**
//! ことが Run 6 後の調査で判明した。jp-workplace-regs corpus には半角 763 個 /
//! 全角 1061 個の数字が同じ文書内で混在しており、production で user が全角で
//! 投げると FTS5 sparse 経路 (および char-level embedder) が取り逃すリスクが
//! ある。本ハーネスは「synthesised zenkaku query」を作って NFKC ON/OFF を
//! A/B で計測し、影響度を定量化する。
//!
//! 実行:
//! ```sh
//! cargo run -p ellisii-sdk --features static-jp \
//!   --example eval_workplace_regs_nfkc --release
//! ```
//!
//! 結果は `docs/eval/recall-evals.md` jp-workplace-regs セクション
//! 「Run 7 (NFKC normalization A/B)」として追記する。

use std::collections::HashMap;
use std::path::PathBuf;

use std::sync::Arc;

use ellisii_core::Chunk;
use ellisii_jp_tokenizer_bigram::CharBigramTokenizer;
use ellisii_jp_tokenizer_core::JpTokenizer;
use ellisii_rag::eval::{summarize, GoldenSet};
use ellisii_sdk::{Ellisii, SearchOptions};
use ellisii_store_core::VectorStore;
use ellisii_store_sqlite::SqliteStore;
use serde::Deserialize;
use unicode_normalization::UnicodeNormalization;
use uuid::Uuid;

#[derive(Debug, Deserialize)]
struct CorpusEntry {
    doc_id: String,
    #[allow(dead_code)]
    parent_id: String,
    #[allow(dead_code)]
    title: String,
    caption: String,
    text: String,
}

fn home() -> PathBuf {
    PathBuf::from(std::env::var("HOME").expect("HOME"))
}
fn embed_dir() -> PathBuf {
    home().join("Library/Application Support/ellisii/models/static-embedding-japanese")
}
fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .join("rag/tests/fixtures/eval/jp-workplace-regs")
}

fn nfkc(s: &str) -> String {
    s.nfkc().collect()
}

/// 半角数字 / 半角記号 / 半角英字 を全角に倒す (synthetic stress test 用)。
fn to_zenkaku(s: &str) -> String {
    s.chars()
        .map(|c| match c {
            '0'..='9' => char::from_u32(c as u32 - '0' as u32 + '０' as u32).unwrap(),
            'A'..='Z' => char::from_u32(c as u32 - 'A' as u32 + 'Ａ' as u32).unwrap(),
            'a'..='z' => char::from_u32(c as u32 - 'a' as u32 + 'ａ' as u32).unwrap(),
            '(' => '（',
            ')' => '）',
            '!' => '！',
            '?' => '？',
            ',' => '，',
            '.' => '．',
            _ => c,
        })
        .collect()
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> anyhow::Result<()> {
    #[cfg(not(feature = "static-jp"))]
    {
        anyhow::bail!("build with --features static-jp");
    }
    #[cfg(feature = "static-jp")]
    return run().await;
}

#[cfg(feature = "static-jp")]
async fn run() -> anyhow::Result<()> {
    let dir = fixture_dir();
    let corpus: Vec<CorpusEntry> =
        serde_json::from_str(&std::fs::read_to_string(dir.join("corpus.json"))?)?;
    let gold: GoldenSet =
        GoldenSet::from_json_str(&std::fs::read_to_string(dir.join("golden.json"))?)?;
    eprintln!(
        "corpus: {} chunks, golden: {} ({} items)",
        corpus.len(),
        gold.name,
        gold.items.len()
    );

    let nb = Uuid::new_v4();
    let src = Uuid::new_v4();

    // 2 つのコーパスを用意:
    //   raw  = corpus.json そのまま (混在表記が残る)
    //   norm = chunk text を NFKC 正規化 (zen 数字 / 記号 → han)
    let build_chunks = |normalize: bool| -> (Vec<Chunk>, Vec<String>, HashMap<Uuid, String>) {
        let mut chunks = Vec::with_capacity(corpus.len());
        let mut texts = Vec::with_capacity(corpus.len());
        let mut id_map = HashMap::new();
        for (i, e) in corpus.iter().enumerate() {
            let cid = Uuid::new_v4();
            id_map.insert(cid, e.doc_id.clone());
            let raw_txt = if e.caption.is_empty() {
                e.text.clone()
            } else {
                format!("({})\n{}", e.caption, e.text)
            };
            let txt = if normalize { nfkc(&raw_txt) } else { raw_txt };
            chunks.push(Chunk {
                id: cid,
                source_id: src,
                ord: i as u32,
                text: txt.clone(),
                heading_path: vec![e.doc_id.clone()],
                page: None,
                bbox: None,
                summary: None,
            });
            texts.push(txt);
        }
        (chunks, texts, id_map)
    };

    let embed = embed_dir();
    eprintln!("embed: {}", embed.display());

    // index_raw (NFKC off) / index_norm (NFKC on)
    let (chunks_raw, texts_raw, id_map_raw) = build_chunks(false);
    let (chunks_norm, texts_norm, id_map_norm) = build_chunks(true);

    // store-memory は substring match なので FTS5 経路の zen/han 不一致を再現できない。
    // sqlite + bigram tokenizer を使って本物の FTS5 pipeline で計測する。
    let dim = 1024;
    let mk_store = || -> anyhow::Result<Arc<dyn VectorStore>> {
        let tok: Arc<dyn JpTokenizer> = Arc::new(CharBigramTokenizer::new());
        Ok(Arc::new(SqliteStore::open_in_memory_with_tokenizer(
            dim, tok,
        )?))
    };
    let store_raw = mk_store()?;
    let store_norm = mk_store()?;
    let ellisii_raw = Ellisii::builder()
        .with_embedder_static_jp(&embed)?
        .with_store(store_raw.clone())
        .with_notebook_id(nb)
        .build()?;
    let embs = ellisii_raw.embedder().embed(&texts_raw).await?;
    store_raw.upsert(nb, &chunks_raw, &embs).await?;

    let ellisii_norm = Ellisii::builder()
        .with_embedder_static_jp(&embed)?
        .with_store(store_norm.clone())
        .with_notebook_id(nb)
        .build()?;
    let embs2 = ellisii_norm.embedder().embed(&texts_norm).await?;
    store_norm.upsert(nb, &chunks_norm, &embs2).await?;

    // クエリ 4 系列:
    //   q_han      = golden の query (半角主体)
    //   q_zen      = 半角→全角 / 記号→全角に倒した擬似クエリ
    //   そのいずれかを normalize=on (NFKC) で送るかどうかでさらに分岐。
    let queries_han: Vec<&str> = gold.items.iter().map(|i| i.query.as_str()).collect();
    let queries_zen: Vec<String> = queries_han.iter().map(|q| to_zenkaku(q)).collect();

    println!("\n=== jp-workplace-regs: NFKC A/B (Run 7, k=5, cap=on, w=0.5) ===");
    println!(
        "{:<40} {:<10} {:<10} {:<10} {:<10}",
        "variant", "recall", "hit", "ndcg", "mrr"
    );

    // 4 セル: (index, query) を {raw, norm} × {han-q, zen-q-raw, zen-q-norm}
    // index=raw, q=han (= 既存 Run 1 と同等の baseline)
    let s = eval_variant(
        &ellisii_raw,
        &gold,
        &id_map_raw,
        &queries_han
            .iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>(),
        5,
    )
    .await?;
    println!(
        "{:<40} {:<10.3} {:<10.3} {:<10.3} {:<10.3}",
        "index=raw  q=han  (baseline)", s.recall_at_k, s.hit_at_k, s.ndcg_at_k, s.mrr
    );

    // index=raw, q=zen raw (= user が全角で投げて、何もしない悪い状態)
    let s = eval_variant(&ellisii_raw, &gold, &id_map_raw, &queries_zen, 5).await?;
    println!(
        "{:<40} {:<10.3} {:<10.3} {:<10.3} {:<10.3}",
        "index=raw  q=zen   (no normalize)", s.recall_at_k, s.hit_at_k, s.ndcg_at_k, s.mrr
    );

    // index=raw, q=zen-NFKC (= query 側だけ NFKC、index は混在のまま)
    let queries_zen_nfkc: Vec<String> = queries_zen.iter().map(|q| nfkc(q)).collect();
    let s = eval_variant(&ellisii_raw, &gold, &id_map_raw, &queries_zen_nfkc, 5).await?;
    println!(
        "{:<40} {:<10.3} {:<10.3} {:<10.3} {:<10.3}",
        "index=raw  q=zen→NFKC (query-side only)", s.recall_at_k, s.hit_at_k, s.ndcg_at_k, s.mrr
    );

    // index=norm, q=zen-NFKC (= 推奨; both sides NFKC)
    let s = eval_variant(&ellisii_norm, &gold, &id_map_norm, &queries_zen_nfkc, 5).await?;
    println!(
        "{:<40} {:<10.3} {:<10.3} {:<10.3} {:<10.3}",
        "index=norm q=zen→NFKC (both sides)", s.recall_at_k, s.hit_at_k, s.ndcg_at_k, s.mrr
    );

    // index=norm, q=han (han はそもそも NFKC 後と同じだから、 normalize による 副作用が無いかの sanity)
    let s = eval_variant(
        &ellisii_norm,
        &gold,
        &id_map_norm,
        &queries_han
            .iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>(),
        5,
    )
    .await?;
    println!(
        "{:<40} {:<10.3} {:<10.3} {:<10.3} {:<10.3}",
        "index=norm q=han   (sanity)", s.recall_at_k, s.hit_at_k, s.ndcg_at_k, s.mrr
    );

    println!("\nzenkaku sample queries (q=zen):");
    for q in queries_zen.iter().take(3) {
        println!("  - {}", q);
    }

    // ---- Stress test: 数字を含む 6 query で zen / kanji-digit / NFKC を見る ----
    // golden 40 件のうち数字を含むのは 4 件しかなく、A/B が差が出にくい。
    // 「production で user が打つ可能性のある表記」を 6 件用意して、index と
    // query 双方の正規化が無いと壊れることを示す。
    println!("\n=== Stress test (digit-bearing queries, k=5) ===");
    let stress: &[(&str, &str, &str, &str)] = &[
        // (query_han, query_zen, query_kanji, expected_doc_id)
        (
            "週40時間労働",
            "週４０時間労働",
            "週四十時間労働",
            "shuugyou-8",
        ),
        (
            "勤続20年で何日",
            "勤続２０年で何日",
            "勤続二十年で何日",
            "refresh-4",
        ),
        (
            "通算93日間の介護休業",
            "通算９３日間の介護休業",
            "通算九十三日間の介護休業",
            "ikukai-20",
        ),
        (
            "試用期間2カ月",
            "試用期間２カ月",
            "試用期間二カ月",
            "shuugyou-6",
        ),
        (
            "自己都合退職は30日前",
            "自己都合退職は３０日前",
            "自己都合退職は三十日前",
            "shuugyou-35",
        ),
        (
            "パートが14日前に退職届",
            "パートが１４日前に退職届",
            "パートが十四日前に退職届",
            "part-17",
        ),
    ];

    // hybrid (w=0.5) と sparse-only (w=1.0) の両方で見る。NFKC が最も効くのは
    // FTS5 だけが ranker のとき。
    for weight in [0.5_f32, 1.0_f32] {
        println!("\n  weight = {:.1} (semantic_weight)", weight);
        println!(
            "  {:<18} {:<6} {:<6} {:<6}",
            "variant", "han", "zen", "kanji"
        );
        run_stress(
            stress,
            weight,
            &ellisii_raw,
            &ellisii_norm,
            &id_map_raw,
            &id_map_norm,
        )
        .await?;
    }

    Ok(())
}

#[cfg(feature = "static-jp")]
async fn run_stress(
    stress: &[(&str, &str, &str, &str)],
    weight: f32,
    ellisii_raw: &Ellisii,
    ellisii_norm: &Ellisii,
    id_map_raw: &HashMap<Uuid, String>,
    id_map_norm: &HashMap<Uuid, String>,
) -> anyhow::Result<()> {
    for (label, idx, q_norm, use_norm_id_map) in [
        ("index=raw ", ellisii_raw, false, false),
        ("index=raw +qNFKC", ellisii_raw, true, false),
        ("index=norm+qNFKC", ellisii_norm, true, true),
    ] {
        let id_map = if use_norm_id_map {
            &id_map_norm
        } else {
            &id_map_raw
        };
        let mut han_hits = 0;
        let mut zen_hits = 0;
        let mut kan_hits = 0;
        for (q_han, q_zen, q_kan, exp) in stress {
            for (q, counter) in [
                (*q_han, &mut han_hits),
                (*q_zen, &mut zen_hits),
                (*q_kan, &mut kan_hits),
            ] {
                let q_eff = if q_norm { nfkc(q) } else { q.to_string() };
                let hits = idx.search(&q_eff, opts(weight, 5)).await?;
                let pred: Vec<String> = hits
                    .iter()
                    .filter_map(|h| id_map.get(&h.chunk.id).cloned())
                    .collect();
                if pred.iter().any(|p| p == exp) {
                    *counter += 1;
                }
            }
        }
        let n = stress.len();
        println!(
            "  {:<18} {}/{}   {}/{}   {}/{}",
            label, han_hits, n, zen_hits, n, kan_hits, n
        );
    }

    Ok(())
}

#[cfg(feature = "static-jp")]
fn opts(semantic_weight: f32, top_k: usize) -> SearchOptions {
    SearchOptions {
        top_k,
        semantic_weight,
        caption_rerank: true,
        ..Default::default()
    }
}

#[cfg(feature = "static-jp")]
async fn eval_variant(
    ellisii: &Ellisii,
    gold: &GoldenSet,
    id_map: &HashMap<Uuid, String>,
    queries: &[String],
    k: usize,
) -> ellisii_core::Result<ellisii_rag::eval::EvalSummary> {
    let mut pairs = Vec::with_capacity(gold.items.len());
    for (item, q) in gold.items.iter().zip(queries.iter()) {
        let hits = ellisii
            .search(
                q,
                SearchOptions {
                    top_k: k,
                    semantic_weight: 0.5,
                    caption_rerank: true,
                    ..Default::default()
                },
            )
            .await?;
        let pred: Vec<String> = hits
            .iter()
            .filter_map(|h| id_map.get(&h.chunk.id).cloned())
            .collect();
        pairs.push((pred, item.relevant.clone()));
    }
    Ok(summarize(&pairs, k))
}
