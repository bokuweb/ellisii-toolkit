//! `QueryRewriter` trait と関連型。
//!
//! retrieve 前にクエリを N 個に展開するための抽象。
//! - 元クエリは常に結果に含める (recall を落とさない安全網)。
//! - 実装は `query-rewriter-passthrough` (no-op) と
//!   `query-rewriter-llm` (LLM で言い換え生成) を別 crate に分離。

use async_trait::async_trait;
use ellisii_core::Result;

/// 書き換え結果。`original` は必ず元クエリ。`variants` は追加の言い換え。
///
/// `all()` で「元クエリを先頭にした全クエリ列」を返す。RRF にそのまま渡す想定。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RewrittenQueries {
    pub original: String,
    pub variants: Vec<String>,
}

impl RewrittenQueries {
    pub fn just(original: impl Into<String>) -> Self {
        Self {
            original: original.into(),
            variants: Vec::new(),
        }
    }

    /// 元クエリを先頭に、続けて variants を返す。空文字 / 元と完全一致 / 重複は除く。
    pub fn all(&self) -> Vec<String> {
        let mut out = Vec::with_capacity(1 + self.variants.len());
        out.push(self.original.clone());
        for v in &self.variants {
            let v = v.trim();
            if v.is_empty() || v == self.original {
                continue;
            }
            if out.iter().any(|x| x == v) {
                continue;
            }
            out.push(v.to_string());
        }
        out
    }
}

#[async_trait]
pub trait QueryRewriter: Send + Sync {
    /// 元クエリを受け取り、書き換え結果を返す。
    ///
    /// `max_variants` は元を除いた追加クエリ数の上限。実装は超えてはならない。
    async fn rewrite(&self, query: &str, max_variants: usize) -> Result<RewrittenQueries>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn just_has_only_original() {
        let r = RewrittenQueries::just("民法 第709条");
        assert_eq!(r.all(), vec!["民法 第709条".to_string()]);
    }

    #[test]
    fn all_dedups_and_drops_empty_and_equal_to_original() {
        let r = RewrittenQueries {
            original: "猫".into(),
            variants: vec![
                "".into(),
                "  ".into(),
                "猫".into(),
                "ねこ".into(),
                "ねこ".into(),
                "ネコ".into(),
            ],
        };
        assert_eq!(r.all(), vec!["猫", "ねこ", "ネコ"]);
    }

    #[test]
    fn all_preserves_variant_order() {
        let r = RewrittenQueries {
            original: "a".into(),
            variants: vec!["c".into(), "b".into(), "d".into()],
        };
        assert_eq!(r.all(), vec!["a", "c", "b", "d"]);
    }
}
