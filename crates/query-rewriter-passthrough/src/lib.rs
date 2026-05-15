//! 入力クエリをそのまま返す `PassthroughRewriter`。
//!
//! agentic / multi-query を有効化していない経路のデフォルトとして利用する。
//! retrieve のフォールバックや、書き換えが効くかの A/B 比較の baseline 用。

use async_trait::async_trait;
use ellisii_core::Result;
use ellisii_query_rewriter_core::{QueryRewriter, RewrittenQueries};

pub struct PassthroughRewriter;

#[async_trait]
impl QueryRewriter for PassthroughRewriter {
    async fn rewrite(&self, query: &str, _max_variants: usize) -> Result<RewrittenQueries> {
        Ok(RewrittenQueries::just(query))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn returns_only_the_original_query() {
        let r = PassthroughRewriter.rewrite("猫の慣用句", 5).await.unwrap();
        assert_eq!(r.original, "猫の慣用句");
        assert!(r.variants.is_empty());
        assert_eq!(r.all(), vec!["猫の慣用句".to_string()]);
    }

    #[tokio::test]
    async fn ignores_max_variants() {
        // max_variants が 0 でも 100 でも結果は同じ
        let a = PassthroughRewriter.rewrite("q", 0).await.unwrap();
        let b = PassthroughRewriter.rewrite("q", 100).await.unwrap();
        assert_eq!(a, b);
    }
}
