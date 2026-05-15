//! End-to-end RAG retrieval eval.
//!
//! `eval_fixture.rs` は metric が動くことを合成データで保証するだけだが、
//! こちらは実際に `RagEngine::retrieve_weighted` を `InMemoryStore` と
//! 決定的な char-bigram embedder で回し、複数の hybrid 設定で
//! recall@k / nDCG@k / MRR を計測する。
//!
//! 目的:
//! - eval ハーネスが「埋め込み → 検索 → 採点」までシームレスに繋がることを
//!   配線として担保する (将来 sqlite-vec / 実 embedder に差し替えても通る形)。
//! - hybrid (semantic=0.5) が vector-only より劣化していない、
//!   という最低限のリグレッション保証を入れる。
//! - `cargo test -p ellisii-rag end_to_end_eval -- --nocapture` で
//!   人間が読める比較レポートを出す。
//!
//! 注意: `InMemoryStore::keyword_search` は「クエリ全体の部分文字列マッチ」と
//! 単純化されているため、複数トークンの日本語クエリではほぼ 0 ヒットになる。
//! これは store-memory 側の制約であって RAG の品質ではない。
//! 実 keyword 経路 (FTS5 + BM25) を計測したいときは store-sqlite に
//! 切り替えて同じ golden で再実行する想定。

use async_trait::async_trait;
use ellisii_core::{Chunk, Result, SearchHit};
use ellisii_embed_core::Embedder;
use ellisii_llm_stub::EchoLlm;
use ellisii_rag::{
    eval::{summarize, EvalSummary, GoldenItem, GoldenSet},
    HybridWeights, RagEngine,
};
use ellisii_store_core::VectorStore;
use ellisii_store_memory::InMemoryStore;
use std::collections::HashMap;
use uuid::Uuid;

/// 文字バイグラムを 64 次元にハッシュバケットへ落とす決定的 embedder。
/// 実 embedder ではないが、共通バイグラム数に応じて cos 類似度が単調になるので
/// 「同義語クエリ → 関連 doc が高スコア」という最低限の semantic 性質を満たす。
struct CharBigramEmbedder {
    dim: usize,
}

#[async_trait]
impl Embedder for CharBigramEmbedder {
    fn dim(&self) -> usize {
        self.dim
    }
    async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        Ok(texts.iter().map(|t| bigram_vec(t, self.dim)).collect())
    }
}

fn bigram_vec(text: &str, dim: usize) -> Vec<f32> {
    let mut v = vec![0.0_f32; dim];
    let chars: Vec<char> = text.chars().collect();
    if chars.len() < 2 {
        // 単独文字も unigram として 1 つは入れておく
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

/// (doc-id 文字列, テキスト) のペア。doc-id は golden の `relevant` と一致。
const CORPUS: &[(&str, &str)] = &[
    (
        "minpou-94",
        "民法第94条 通謀虚偽表示。相手方と通じてした虚偽の意思表示は無効とする。\
         ただし、その無効は善意の第三者に対抗することができない。",
    ),
    (
        "minpou-93",
        "民法第93条 心裡留保。意思表示は、表意者がその真意ではないことを\
         知ってしたときであっても、その効力を妨げられない。",
    ),
    (
        "minpou-95",
        "民法第95条 錯誤。意思表示は、法律行為の目的及び取引上の社会通念に照らして\
         重要な錯誤に基づくものであるときは取り消すことができる。",
    ),
    (
        "minpou-15",
        "民法第15条 補助開始の審判。精神上の障害により事理を弁識する能力が\
         不十分である者については、家庭裁判所は補助開始の審判をすることができる。",
    ),
    (
        "minpou-16",
        "民法第16条 被補助人及び補助人。補助開始の審判を受けた者は被補助人とし、\
         これに補助人を付する。",
    ),
    (
        "minpou-7",
        "民法第7条 後見開始の審判。精神上の障害により事理を弁識する能力を\
         欠く常況にある者については、家庭裁判所は後見開始の審判をすることができる。",
    ),
    (
        "noise-1",
        "商法第501条 絶対的商行為。次に掲げる行為は商行為とする。",
    ),
    (
        "noise-2",
        "刑法第199条 殺人。人を殺した者は、死刑又は無期若しくは5年以上の懲役に処する。",
    ),
];

async fn build_engine() -> (
    RagEngine<CharBigramEmbedder, InMemoryStore, EchoLlm>,
    Uuid,
    HashMap<Uuid, String>,
) {
    let embedder = CharBigramEmbedder { dim: 128 };
    let store = InMemoryStore::new();
    let nb = Uuid::new_v4();

    let mut chunks = Vec::new();
    let mut texts = Vec::new();
    let mut id_map: HashMap<Uuid, String> = HashMap::new();
    for (ord, (doc_id, text)) in CORPUS.iter().enumerate() {
        let cid = Uuid::new_v4();
        id_map.insert(cid, (*doc_id).to_string());
        chunks.push(Chunk {
            id: cid,
            source_id: Uuid::new_v4(),
            ord: ord as u32,
            text: (*text).to_string(),
            heading_path: vec![(*doc_id).to_string()],
            page: None,
            bbox: None,
            summary: None,
        });
        texts.push((*text).to_string());
    }
    let embs = embedder.embed(&texts).await.unwrap();
    store.upsert(nb, &chunks, &embs).await.unwrap();

    (
        RagEngine {
            embedder,
            store,
            llm: EchoLlm,
        },
        nb,
        id_map,
    )
}

fn golden() -> GoldenSet {
    GoldenSet {
        name: "e2e-jp-law".into(),
        items: vec![
            GoldenItem {
                query: "民法第94条の意思表示について教えて".into(),
                relevant: vec!["minpou-94".into()],
                tags: vec!["article-id".into()],
            },
            GoldenItem {
                query: "通謀虚偽表示は無効か".into(),
                relevant: vec!["minpou-94".into()],
                tags: vec!["concept".into()],
            },
            GoldenItem {
                query: "補助開始の審判の効果".into(),
                relevant: vec!["minpou-15".into(), "minpou-16".into()],
                tags: vec!["multi-relevant".into()],
            },
            GoldenItem {
                query: "錯誤による意思表示は取り消せるか".into(),
                relevant: vec!["minpou-95".into()],
                tags: vec!["concept".into()],
            },
        ],
    }
}

fn hits_to_doc_ids(hits: &[SearchHit], id_map: &HashMap<Uuid, String>) -> Vec<String> {
    hits.iter()
        .filter_map(|h| id_map.get(&h.chunk.id).cloned())
        .collect()
}

async fn eval_with_weights(
    engine: &RagEngine<CharBigramEmbedder, InMemoryStore, EchoLlm>,
    nb: Uuid,
    id_map: &HashMap<Uuid, String>,
    golden: &GoldenSet,
    weights: HybridWeights,
    k: usize,
) -> EvalSummary {
    let mut pairs: Vec<(Vec<String>, Vec<String>)> = Vec::new();
    for item in &golden.items {
        let hits = engine
            .retrieve_weighted(Some(nb), &item.query, k, weights)
            .await
            .unwrap();
        let pred = hits_to_doc_ids(&hits, id_map);
        pairs.push((pred, item.relevant.clone()));
    }
    summarize(&pairs, k)
}

#[tokio::test]
async fn end_to_end_retrieve_eval_reports_metrics() {
    let (engine, nb, id_map) = build_engine().await;
    let g = golden();
    let k = 5;

    let kw_only =
        eval_with_weights(&engine, nb, &id_map, &g, HybridWeights { semantic: 0.0 }, k).await;
    let hybrid =
        eval_with_weights(&engine, nb, &id_map, &g, HybridWeights { semantic: 0.5 }, k).await;
    let vec_only =
        eval_with_weights(&engine, nb, &id_map, &g, HybridWeights { semantic: 1.0 }, k).await;

    println!(
        "\n=== RAG retrieval eval (k={k}, n_queries={}) ===\n\
         setting       recall@k   hit@k   nDCG@k    MRR\n\
         keyword-only  {:>7.3}   {:>5.3}   {:>6.3}   {:>5.3}\n\
         hybrid 0.5    {:>7.3}   {:>5.3}   {:>6.3}   {:>5.3}\n\
         vector-only   {:>7.3}   {:>5.3}   {:>6.3}   {:>5.3}\n",
        g.items.len(),
        kw_only.recall_at_k,
        kw_only.hit_at_k,
        kw_only.ndcg_at_k,
        kw_only.mrr,
        hybrid.recall_at_k,
        hybrid.hit_at_k,
        hybrid.ndcg_at_k,
        hybrid.mrr,
        vec_only.recall_at_k,
        vec_only.hit_at_k,
        vec_only.ndcg_at_k,
        vec_only.mrr,
    );

    // Sanity: vector / hybrid 経路は corpus 内に正解 chunk があるので必ずヒットする。
    // keyword-only は store-memory が naive なため 0 でも許容 (上のモジュールコメント参照)。
    assert!(
        vec_only.hit_at_k > 0.0,
        "vector-only should hit at least one query"
    );
    assert!(
        hybrid.hit_at_k > 0.0,
        "hybrid should hit at least one query"
    );

    // リグレッション保証: hybrid は vector-only より recall/MRR が劣化しない。
    // (keyword 側が naive で 0 になるので、最低でも vector に揃うことを期待)
    assert!(
        hybrid.recall_at_k + 1e-6 >= vec_only.recall_at_k,
        "hybrid recall@k {} regressed below vector-only {}",
        hybrid.recall_at_k,
        vec_only.recall_at_k,
    );
    assert!(
        hybrid.mrr + 1e-6 >= vec_only.mrr,
        "hybrid MRR {} regressed below vector-only {}",
        hybrid.mrr,
        vec_only.mrr,
    );
}

#[tokio::test]
async fn keyword_search_finds_exact_article_id() {
    // 「民法第94条」のような明示クエリでは keyword 経路が確実にヒットする、
    // という前提が崩れていないことを別建てで確認する。
    let (engine, nb, id_map) = build_engine().await;
    let hits = engine
        .retrieve_weighted(Some(nb), "民法第94条", 5, HybridWeights { semantic: 0.0 })
        .await
        .unwrap();
    let ids = hits_to_doc_ids(&hits, &id_map);
    assert!(ids.contains(&"minpou-94".to_string()), "got: {ids:?}");
}
