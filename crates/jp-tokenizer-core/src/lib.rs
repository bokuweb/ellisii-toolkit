//! 日本語向けトークナイザの抽象。
//!
//! 既定の `unicode61` FTS5 トークナイザは日本語の語境界を扱えないため、
//! 取り込み時 / クエリ時に **同じトークナイザ** で前段分割し、空白区切り
//! の文字列を FTS5 に流す方針で使う。
//!
//! 実装は別 crate に分離する:
//! - `ellisii-jp-tokenizer-bigram` … 文字 bigram (依存ゼロ・既定)
//! - `ellisii-jp-tokenizer-vaporetto` … Vaporetto (高品質、モデルファイル要)

pub trait JpTokenizer: Send + Sync {
    fn tokenize(&self, text: &str) -> Vec<String>;

    /// FTS5 に投入できる「空白区切りの一連のトークン」を返す。
    fn tokenize_for_fts(&self, text: &str) -> String {
        self.tokenize(text).join(" ")
    }
}

/// CJK (ひらがな/カタカナ/漢字/CJK 拡張) かどうか
pub fn is_cjk(c: char) -> bool {
    matches!(
        c,
        '\u{3040}'..='\u{309f}' // ひらがな
            | '\u{30a0}'..='\u{30ff}' // カタカナ
            | '\u{31f0}'..='\u{31ff}' // カタカナ拡張
            | '\u{3400}'..='\u{4dbf}' // CJK 拡張 A
            | '\u{4e00}'..='\u{9fff}' // CJK 統合漢字
            | '\u{f900}'..='\u{faff}' // CJK 互換漢字
            | '\u{ff66}'..='\u{ff9d}' // 半角カタカナ
    )
}

pub fn is_word_char(c: char) -> bool {
    c.is_alphanumeric() || is_cjk(c)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cjk_detects_hiragana_katakana_kanji() {
        assert!(is_cjk('あ'));
        assert!(is_cjk('カ'));
        assert!(is_cjk('漢'));
        assert!(is_cjk('ｶ'));
    }

    #[test]
    fn cjk_rejects_latin_and_punct() {
        assert!(!is_cjk('a'));
        assert!(!is_cjk('1'));
        assert!(!is_cjk(' '));
        assert!(!is_cjk('。'));
    }

    #[test]
    fn word_char_includes_alphanumeric_and_cjk() {
        assert!(is_word_char('A'));
        assert!(is_word_char('9'));
        assert!(is_word_char('字'));
        assert!(!is_word_char(' '));
        assert!(!is_word_char(','));
    }

    struct WhitespaceTok;
    impl JpTokenizer for WhitespaceTok {
        fn tokenize(&self, text: &str) -> Vec<String> {
            text.split_whitespace().map(|s| s.to_string()).collect()
        }
    }

    #[test]
    fn default_tokenize_for_fts_joins_with_single_space() {
        let t = WhitespaceTok;
        assert_eq!(t.tokenize_for_fts("a  b\tc"), "a b c");
        assert_eq!(t.tokenize_for_fts(""), "");
    }
}
