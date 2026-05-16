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

/// `recommend_tokenizer` が返す軽量レポート。SDK の `with_store_sqlite_auto`
/// から呼ばれるほか、ユーザが手動で「このコーパスにはどの tokenizer が
/// 向くか」を確認する用途でも使える。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TokenizerRecommendation {
    /// `CharBigramTokenizer` + NFKC ラッパー (依存ゼロ)。
    BigramNfkc,
    /// 形態素 tokenizer (Vibrato 互換 = delarocha) + NFKC ラッパー。
    /// 辞書ファイルが利用可能なときのみ推奨される。
    MorphemeNfkc,
}

/// シンプルな corpus signal を出す診断レポート。`recommend_tokenizer` の
/// 判断材料を可視化するためのもので、しきい値そのものは持たない。
#[derive(Debug, Clone, Copy)]
pub struct CorpusSignals {
    /// サンプリングした全文字数 (CJK + ASCII)。
    pub total_chars: usize,
    /// ASCII alphabet `[A-Za-z]` の割合 (0.0 - 1.0)。`>= 0.10` なら英字混在
    /// (例: jp-cs-wiki-hard) と判断し、形態素 tokenizer の単語境界が効きやすい。
    pub en_ratio: f32,
    /// 半角/全角/漢数字を含むかの indicator (Run 7 で NFKC 適用根拠に使う)。
    pub has_zenkaku_digit: bool,
    pub has_kanji_digit: bool,
}

/// 与えられたテキスト列をサンプリングして tokenizer を推奨する。
///
/// 学習方針 (Run 8 / 6 corpus 横展開の結果):
/// - 形態素 tokenizer (delarocha 等) は全 6 corpus で bigram 以上の recall/nDCG
///   を示し、悪化させたケースは無い。**形態素辞書が手元にあるなら常にそちらを
///   選ぶ** のが defensible。
/// - 辞書が無い場合は bigram + NFKC を返す (NFKC は recall 中立だが query
///   正規化の決定性のため常に on)。
///
/// 返り値とともに [`CorpusSignals`] も返すので、ユーザが判断材料を確認できる。
pub fn recommend_tokenizer<'a, I>(
    samples: I,
    morpheme_dict_available: bool,
) -> (TokenizerRecommendation, CorpusSignals)
where
    I: IntoIterator<Item = &'a str>,
{
    let mut total = 0usize;
    let mut en = 0usize;
    let mut has_zen_digit = false;
    let mut has_kanji_digit = false;
    for s in samples {
        for c in s.chars() {
            total += 1;
            if c.is_ascii_alphabetic() {
                en += 1;
            }
            if ('\u{ff10}'..='\u{ff19}').contains(&c) {
                has_zen_digit = true;
            }
            if matches!(
                c,
                '一' | '二'
                    | '三'
                    | '四'
                    | '五'
                    | '六'
                    | '七'
                    | '八'
                    | '九'
                    | '十'
                    | '百'
                    | '千'
                    | '万'
                    | '〇'
                    | '零'
            ) {
                has_kanji_digit = true;
            }
        }
    }
    let en_ratio = if total == 0 {
        0.0
    } else {
        en as f32 / total as f32
    };
    let signals = CorpusSignals {
        total_chars: total,
        en_ratio,
        has_zenkaku_digit: has_zen_digit,
        has_kanji_digit,
    };
    let pick = if morpheme_dict_available {
        TokenizerRecommendation::MorphemeNfkc
    } else {
        TokenizerRecommendation::BigramNfkc
    };
    (pick, signals)
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

    #[test]
    fn recommend_picks_bigram_when_no_dict() {
        let (pick, sig) = recommend_tokenizer(["日本語のテキスト"].iter().copied(), false);
        assert_eq!(pick, TokenizerRecommendation::BigramNfkc);
        assert_eq!(sig.en_ratio, 0.0);
    }

    #[test]
    fn recommend_picks_morpheme_when_dict_available() {
        let (pick, _) = recommend_tokenizer(["ACID トランザクション"].iter().copied(), true);
        assert_eq!(pick, TokenizerRecommendation::MorphemeNfkc);
    }

    #[test]
    fn signals_detect_en_zen_kanji() {
        let (_, sig) =
            recommend_tokenizer(["ACID は ACID で、４０時間 / 二十年"].iter().copied(), true);
        assert!(sig.en_ratio > 0.0);
        assert!(sig.has_zenkaku_digit);
        assert!(sig.has_kanji_digit);
    }

    #[test]
    fn signals_empty_input() {
        let (_, sig) = recommend_tokenizer(std::iter::empty::<&str>(), false);
        assert_eq!(sig.total_chars, 0);
        assert_eq!(sig.en_ratio, 0.0);
    }
}
