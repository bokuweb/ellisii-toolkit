//! 文字 bigram トークナイザ。日本語連続部分は **重なり 2-gram**、ASCII / 数字
//! 連続部分はそのまま 1 単語にする。辞書不要で配布が軽く、一定の検索品質が出る。

use ellisii_jp_tokenizer_core::{is_cjk, is_word_char, JpTokenizer};

#[derive(Default, Clone, Copy)]
pub struct CharBigramTokenizer;

impl CharBigramTokenizer {
    pub fn new() -> Self {
        Self
    }
}

impl JpTokenizer for CharBigramTokenizer {
    fn tokenize(&self, text: &str) -> Vec<String> {
        let mut out: Vec<String> = Vec::new();
        let mut buf = String::new();
        let mut buf_is_cjk = false;

        let flush = |buf: &mut String, is_cjk: bool, out: &mut Vec<String>| {
            if buf.is_empty() {
                return;
            }
            if is_cjk {
                let chars: Vec<char> = buf.chars().collect();
                if chars.len() == 1 {
                    out.push(chars[0].to_string());
                } else {
                    for w in chars.windows(2) {
                        out.push(w.iter().collect::<String>());
                    }
                }
            } else {
                out.push(buf.to_lowercase());
            }
            buf.clear();
        };

        for c in text.chars() {
            if !is_word_char(c) {
                flush(&mut buf, buf_is_cjk, &mut out);
                continue;
            }
            let cur_is_cjk = is_cjk(c);
            if !buf.is_empty() && cur_is_cjk != buf_is_cjk {
                flush(&mut buf, buf_is_cjk, &mut out);
            }
            buf.push(c);
            buf_is_cjk = cur_is_cjk;
        }
        flush(&mut buf, buf_is_cjk, &mut out);
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ascii_words_are_lowercased_singletons() {
        let t = CharBigramTokenizer::new();
        assert_eq!(t.tokenize("Hello World"), vec!["hello", "world"]);
    }

    #[test]
    fn cjk_emits_overlapping_bigrams() {
        let t = CharBigramTokenizer::new();
        assert_eq!(
            t.tokenize("東京駅"),
            vec!["東京".to_string(), "京駅".to_string()]
        );
    }

    #[test]
    fn mixed_text_segments_are_separated() {
        let t = CharBigramTokenizer::new();
        let toks = t.tokenize("Rust で 本番 に出す");
        // "で" は 1 文字 CJK のためそのまま、"本番" は 1 つの bigram
        assert!(toks.contains(&"rust".to_string()));
        assert!(toks.contains(&"で".to_string()));
        assert!(toks.contains(&"本番".to_string()));
        assert!(toks.iter().any(|s| s == "に出"));
    }

    #[test]
    fn fts_join_is_space_separated() {
        let t = CharBigramTokenizer::new();
        let s = t.tokenize_for_fts("検索");
        assert_eq!(s, "検索");
        let s2 = t.tokenize_for_fts("東京駅前");
        assert_eq!(s2, "東京 京駅 駅前");
    }
}
