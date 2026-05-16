//! Delarocha (https://github.com/bokuweb/delarocha) を裏で使う日本語トークナイザ。
//!
//! Vibrato system 形式の `system.dic` / `system.dic.zst` を `delarocha` 経由で
//! 読み込み、`JpTokenizer` trait に適合させる。`real` feature を有効にしたとき
//! のみ実体がリンクされる。

use ellisii_jp_tokenizer_core::JpTokenizer;
use std::path::Path;

#[cfg(feature = "real")]
mod imp {
    use super::*;
    use delarocha::{VibratoSystemDictionary, VibratoSystemTokenizer};

    pub struct DelarochaTokenizer {
        tokenizer: VibratoSystemTokenizer,
    }

    impl DelarochaTokenizer {
        /// `system.dic` または `system.dic.zst` を拡張子で判別してロードする。
        pub fn from_path(path: &Path) -> Result<Self, String> {
            let dict = VibratoSystemDictionary::from_path(path)
                .map_err(|e| format!("delarocha dict load: {e}"))?;
            // MeCab 互換で space を無視 (FTS5 入力では空白区切りを使うため必須)。
            let tokenizer = dict
                .into_tokenizer()
                .ignore_space(true)
                .map_err(|e| format!("delarocha ignore_space: {e}"))?;
            Ok(Self { tokenizer })
        }
    }

    impl JpTokenizer for DelarochaTokenizer {
        fn tokenize(&self, text: &str) -> Vec<String> {
            let trimmed: String = text.chars().filter(|c| !c.is_control()).collect();
            if trimmed.is_empty() {
                return vec![];
            }
            let mut worker = self.tokenizer.new_worker();
            worker.tokenize(&trimmed);
            worker
                .token_iter()
                .map(|t| t.surface().to_string())
                .filter(|s| !s.trim().is_empty())
                .collect()
        }
    }
}

#[cfg(not(feature = "real"))]
mod imp {
    use super::*;
    pub struct DelarochaTokenizer;
    impl DelarochaTokenizer {
        pub fn from_path(_path: &Path) -> Result<Self, String> {
            Err("delarocha tokenizer requires `real` feature".to_string())
        }
    }
    impl JpTokenizer for DelarochaTokenizer {
        fn tokenize(&self, _text: &str) -> Vec<String> {
            vec![]
        }
    }
}

pub use imp::DelarochaTokenizer;

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn from_path_errors_when_file_missing() {
        let p = PathBuf::from("/this/path/does/not/exist.dic.zst");
        let res = DelarochaTokenizer::from_path(&p);
        assert!(res.is_err());
    }

    #[cfg(not(feature = "real"))]
    #[test]
    fn stub_from_path_message_mentions_real_feature() {
        let err = DelarochaTokenizer::from_path(&PathBuf::from("x")).unwrap_err();
        assert!(err.contains("real"), "unexpected error: {err}");
    }
}
