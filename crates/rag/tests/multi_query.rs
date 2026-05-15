//! `RagEngine::retrieve_multi` の配線テスト。
//!
//! - PassthroughRewriter を渡したとき、結果セットが従来の `retrieve_weighted` と
//!   同じ chunk 集合を含むこと (回帰しない)。
//! - 言い換えクエリを 1 つ追加するスクリプト rewriter で、
//!   通常 retrieve では拾えなかったチャンクが top-k に入りうること。

use async_trait::async_trait;
use ellisii_core::{Chunk, Result};
use ellisii_embed_core::Embedder;
use ellisii_llm_stub::EchoLlm;
use ellisii_query_rewriter_core::{QueryRewriter, RewrittenQueries};
use ellisii_query_rewriter_passthrough::PassthroughRewriter;
use ellisii_rag::{HybridWeights, MultiQueryOptions, RagEngine};
use ellisii_store_core::VectorStore;
use ellisii_store_memory::InMemoryStore;
use uuid::Uuid;

/// 文字を SUM したベクトルを返す決定的 embedder。
struct CharSumEmbedder {
    dim: usize,
}

#[async_trait]
impl Embedder for CharSumEmbedder {
    fn dim(&self) -> usize {
        self.dim
    }
    async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        Ok(texts
            .iter()
            .map(|t| {
                let mut v = vec![0.0_f32; self.dim];
                for c in t.chars() {
                    v[(c as usize) % self.dim] += 1.0;
                }
                v
            })
            .collect())
    }
}

async fn build_engine() -> RagEngine<CharSumEmbedder, InMemoryStore, EchoLlm> {
    let embedder = CharSumEmbedder { dim: 64 };
    let store = InMemoryStore::new();
    let nb = Uuid::new_v4();
    // 注意: `ellisii_core::is_retrieval_noise` は content_chars < 25 のチャンクを
    // RRF 段階で除外するため、test fixture は **25 文字以上の内容語** を持つ必要がある。
    // これより短いと rrf_weighted の結果が空になる。
    let chunks: Vec<Chunk> = [
        "猫の慣用句について、日本語ではよく猫の手も借りたい等の表現が使われる。",
        "ねこの諺としては、ねこに小判やねこをかぶる、ねこの額のような表現がある。",
        "犬の昔話には、桃太郎の犬や花咲か爺さんに登場する忠犬など色々ある。",
        "鳥の伝承では鶴の恩返しや、鳳凰や雀の御宿といった話がよく語られる。",
    ]
    .into_iter()
    .enumerate()
    .map(|(i, t)| Chunk {
        id: Uuid::new_v4(),
        source_id: Uuid::nil(),
        ord: i as u32,
        text: t.into(),
        heading_path: vec![],
        page: None,
        bbox: None,
        summary: None,
    })
    .collect();
    let embs = embedder
        .embed(&chunks.iter().map(|c| c.text.clone()).collect::<Vec<_>>())
        .await
        .unwrap();
    store.upsert(nb, &chunks, &embs).await.unwrap();
    RagEngine {
        embedder,
        store,
        llm: EchoLlm,
    }
}

#[tokio::test]
async fn passthrough_multi_query_does_not_regress_single_query() {
    let eng = build_engine().await;
    let single = eng
        .retrieve_weighted(
            None,
            "猫の慣用句について教えて",
            4,
            HybridWeights::default(),
        )
        .await
        .unwrap();
    let multi = eng
        .retrieve_multi(
            None,
            "猫の慣用句について教えて",
            4,
            &PassthroughRewriter,
            MultiQueryOptions {
                max_variants: 0,
                ..Default::default()
            },
        )
        .await
        .unwrap();
    let single_ids: Vec<Uuid> = single.iter().map(|h| h.chunk.id).collect();
    let multi_ids: Vec<Uuid> = multi.iter().map(|h| h.chunk.id).collect();
    assert_eq!(
        single_ids, multi_ids,
        "passthrough must equal single-query order"
    );
}

/// 元クエリだけだと拾いにくい variant を追加すると、その variant に該当する
/// chunk が top-k に入る (= 言い換えで recall が伸びることの最小ケース)。
struct ScriptedRewriter {
    variant: String,
}

#[async_trait]
impl QueryRewriter for ScriptedRewriter {
    async fn rewrite(&self, query: &str, _max: usize) -> Result<RewrittenQueries> {
        Ok(RewrittenQueries {
            original: query.to_string(),
            variants: vec![self.variant.clone()],
        })
    }
}

#[tokio::test]
async fn variant_query_can_surface_additional_hit() {
    let eng = build_engine().await;
    // 元クエリ "猫" 単体だと "ねこの諺" は kana のみで一致しにくい。
    // variant "ねこ" を追加すると、"ねこの諺" のスコアが押し上がる想定。
    let multi = eng
        .retrieve_multi(
            None,
            "猫について教えてほしいです",
            4,
            &ScriptedRewriter {
                variant: "ねこの諺について教えてほしいです".into(),
            },
            MultiQueryOptions {
                max_variants: 1,
                ..Default::default()
            },
        )
        .await
        .unwrap();
    let texts: Vec<&str> = multi.iter().map(|h| h.chunk.text.as_str()).collect();
    assert!(
        texts.iter().any(|t| t.starts_with("ねこの諺")),
        "variant query containing 'ねこの諺' should surface that chunk in top-4 (got {:?})",
        texts
    );
}
