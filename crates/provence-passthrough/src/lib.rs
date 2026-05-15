//! 圧縮を行わない既定実装。Provence モデルが未配置の環境でも動くようにする。

use async_trait::async_trait;
use ellisii_core::Result;
use ellisii_provence_core::{
    split_sentences, CompressedContext, ContextCompressor, ScoredSentence,
};

#[derive(Default, Clone, Copy)]
pub struct PassthroughCompressor;

impl PassthroughCompressor {
    pub fn new() -> Self {
        Self
    }
}

#[async_trait]
impl ContextCompressor for PassthroughCompressor {
    fn is_active(&self) -> bool {
        false
    }

    async fn compress(&self, _query: &str, text: &str) -> Result<CompressedContext> {
        let original_chars = text.chars().count();
        let sentences = split_sentences(text)
            .into_iter()
            .map(|s| ScoredSentence {
                text: s,
                score: 1.0,
                kept: true,
            })
            .collect();
        Ok(CompressedContext {
            kept_text: text.to_string(),
            original_chars,
            kept_chars: original_chars,
            sentences,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn passthrough_keeps_everything() {
        let c = PassthroughCompressor::new();
        let r = c.compress("query", "東京の天気は？快晴。").await.unwrap();
        assert_eq!(r.kept_text, "東京の天気は？快晴。");
        assert_eq!(r.ratio(), 1.0);
        assert!(r.sentences.iter().all(|s| s.kept));
    }

    #[test]
    fn passthrough_is_inactive() {
        assert!(!PassthroughCompressor::new().is_active());
    }

    #[tokio::test]
    async fn passthrough_handles_empty_text() {
        let c = PassthroughCompressor::new();
        let r = c.compress("q", "").await.unwrap();
        assert_eq!(r.kept_text, "");
        assert_eq!(r.original_chars, 0);
        assert_eq!(r.kept_chars, 0);
        assert!(r.sentences.is_empty());
        assert_eq!(r.ratio(), 1.0);
    }

    #[tokio::test]
    async fn passthrough_emits_one_sentence_per_input_sentence() {
        let c = PassthroughCompressor::new();
        let r = c.compress("q", "A。B。C。").await.unwrap();
        assert_eq!(r.sentences.len(), 3);
        assert!(r.sentences.iter().all(|s| (s.score - 1.0).abs() < 1e-6));
    }

    #[tokio::test]
    async fn passthrough_score_passages_returns_ones() {
        let c = PassthroughCompressor::new();
        let scores = c
            .score_passages("q", &["a".into(), "b".into()])
            .await
            .unwrap();
        assert_eq!(scores, vec![1.0, 1.0]);
    }

    #[tokio::test]
    async fn passthrough_char_counts_use_chars_not_bytes() {
        let c = PassthroughCompressor::new();
        let r = c.compress("q", "あいう").await.unwrap();
        assert_eq!(r.original_chars, 3);
        assert_eq!(r.kept_chars, 3);
    }
}
