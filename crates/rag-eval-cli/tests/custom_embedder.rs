//! 任意の `Embedder` 実装を eval ハーネスに注入できることを検証する。
//!
//! 内蔵の `CharBigramEmbedder` は決定的だが擬似 semantic でしかない。
//! 実モデル (例: `embed-static-jp`) や別のテスト用 embedder を CLI / lib から
//! 差し替えられるよう、`run_eval_with_options(EvalOptions { embedder, backend, ... })`
//! 経路を追加する。本テストはその注入 API の存在をロックする。
//!
//! 実モデルを要するテストは別ファイル (`real_static_jp.rs`) で `#[ignore]` 付きで提供。

use async_trait::async_trait;
use ellisii_core::Result as CoreResult;
use ellisii_embed_core::Embedder;
use ellisii_rag::eval::{GoldenItem, GoldenSet};
use ellisii_rag_eval_cli::{run_eval_with_options, Backend, CorpusEntry, EvalOptions};
use std::sync::Arc;

/// 「クエリと最も同じ文字を共有する doc を高スコアにする」決定的 embedder。
/// 配線テスト専用 — 各 doc に固有のキーワードを 1 文字含めると、その文字を持つ
/// クエリが対応 doc に高スコアを付ける。
struct OneHotCharEmbedder {
    dim: usize,
}

#[async_trait]
impl Embedder for OneHotCharEmbedder {
    fn dim(&self) -> usize {
        self.dim
    }
    async fn embed(&self, texts: &[String]) -> CoreResult<Vec<Vec<f32>>> {
        Ok(texts
            .iter()
            .map(|t| {
                let mut v = vec![0.0_f32; self.dim];
                for c in t.chars() {
                    let idx = (c as usize) % self.dim;
                    v[idx] += 1.0;
                }
                let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
                if n > 0.0 {
                    for x in v.iter_mut() {
                        *x /= n;
                    }
                }
                v
            })
            .collect())
    }
}

#[tokio::test]
async fn run_eval_accepts_custom_embedder() {
    // `ellisii_core::is_retrieval_noise` は content_chars < 25 のチャンクを
    // RRF 段階で除外するため、test fixture を 25 文字以上にする。
    let corpus = vec![
        CorpusEntry {
            doc_id: "alpha".into(),
            title: "".into(),
            caption: "".into(),
            text: "アルファ・テストの説明文章。アルファは最初の文字を意味するギリシャ文字です。"
                .into(),
        },
        CorpusEntry {
            doc_id: "beta".into(),
            title: "".into(),
            caption: "".into(),
            text: "ベータ・テストの説明文章。ベータは二番目の文字を意味するギリシャ文字です。"
                .into(),
        },
        CorpusEntry {
            doc_id: "gamma".into(),
            title: "".into(),
            caption: "".into(),
            text: "ガンマ・テストの説明文章。ガンマは三番目の文字を意味するギリシャ文字です。"
                .into(),
        },
    ];
    let golden = GoldenSet {
        name: "custom-embed".into(),
        items: vec![
            GoldenItem {
                query: "アルファ".into(),
                relevant: vec!["alpha".into()],
                tags: vec![],
            },
            GoldenItem {
                query: "ベータ".into(),
                relevant: vec!["beta".into()],
                tags: vec![],
            },
            GoldenItem {
                query: "ガンマ".into(),
                relevant: vec!["gamma".into()],
                tags: vec![],
            },
        ],
    };

    let opts = EvalOptions {
        backend: Backend::Memory,
        embedder: Arc::new(OneHotCharEmbedder { dim: 256 }),
        weights: vec![1.0], // 純 vector
        k: 3,
        judge: None,
        rewriter: None,
        multi: ellisii_rag::MultiQueryOptions::default(),
    };
    let rows = run_eval_with_options(&opts, &corpus, &golden)
        .await
        .expect("eval succeeds");
    assert_eq!(rows.len(), 1);
    let s = &rows[0].summary;
    assert_eq!(s.queries, 3);
    // 文字共有による単純な vector でも 3/3 ヒットするはず。
    assert!(
        s.hit_at_k >= 1.0,
        "expected hit@3=1.0 with one-hot char embedder, got {}",
        s.hit_at_k
    );
}
