//! NFKC 正規化を裏 tokenizer に被せる薄いラッパー。
//!
//! 用途: FTS5 indexer に渡る前に文字列を NFKC 化することで、半角 ⇄ 全角 数字
//! / 記号 / 英字 を統一する。半角カタカナ → 全角カタカナの同化も同時に起きる。
//!
//! 同じ tokenizer をユーザー側 (index) と検索側 (query) の両方に流せば、
//! `週４０時間` と `週40時間` が同じ tokenset になり、FTS5 sparse 経路の
//! 一致漏れを防げる。
//!
//! ```
//! use std::sync::Arc;
//! use ellisii_jp_tokenizer_core::JpTokenizer;
//! use ellisii_jp_tokenizer_nfkc::NfkcTokenizer;
//!
//! struct Identity;
//! impl JpTokenizer for Identity {
//!     fn tokenize(&self, text: &str) -> Vec<String> {
//!         text.chars().map(|c| c.to_string()).collect()
//!     }
//! }
//!
//! let inner: Arc<dyn JpTokenizer> = Arc::new(Identity);
//! let tok = NfkcTokenizer::new(inner);
//! assert_eq!(tok.tokenize("１２"), tok.tokenize("12"));
//! ```

use ellisii_jp_tokenizer_core::JpTokenizer;
use std::sync::Arc;
use unicode_normalization::UnicodeNormalization;

/// `JpTokenizer` を NFKC 正規化で前処理してから呼び出すラッパー。
pub struct NfkcTokenizer {
    inner: Arc<dyn JpTokenizer>,
}

impl NfkcTokenizer {
    pub fn new(inner: Arc<dyn JpTokenizer>) -> Self {
        Self { inner }
    }
}

impl JpTokenizer for NfkcTokenizer {
    fn tokenize(&self, text: &str) -> Vec<String> {
        let normalized: String = text.nfkc().collect();
        self.inner.tokenize(&normalized)
    }
}

/// 純粋関数の NFKC ヘルパー。SDK の query 入口など、tokenizer を経由しない
/// パスでも同じ正規化を呼べるよう公開する。
pub fn nfkc(s: &str) -> String {
    s.nfkc().collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Bigram;
    impl JpTokenizer for Bigram {
        fn tokenize(&self, text: &str) -> Vec<String> {
            text.chars()
                .collect::<Vec<_>>()
                .windows(2)
                .map(|w| w.iter().collect::<String>())
                .collect()
        }
    }

    #[test]
    fn zenkaku_digits_normalize_to_hankaku() {
        let inner: Arc<dyn JpTokenizer> = Arc::new(Bigram);
        let tok = NfkcTokenizer::new(inner);
        assert_eq!(tok.tokenize("週４０時間"), tok.tokenize("週40時間"));
    }

    #[test]
    fn nfkc_helper_strips_zenkaku() {
        assert_eq!(nfkc("ＡＢＣ１２３"), "ABC123");
        assert_eq!(nfkc("（テスト）"), "(テスト)");
    }

    #[test]
    fn halfwidth_katakana_normalize_to_fullwidth() {
        // NFKC は ｱ → ア に倒す
        let inner: Arc<dyn JpTokenizer> = Arc::new(Bigram);
        let tok = NfkcTokenizer::new(inner);
        assert_eq!(tok.tokenize("ｶﾀｶﾅ"), tok.tokenize("カタカナ"));
    }
}
