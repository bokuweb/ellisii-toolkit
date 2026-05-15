//! Vaporetto ベースの日本語トークナイザ。
//!
//! 参考: https://secon.dev/entry/2026/04/27/080000-sqlite-duckdb-vaporetto/
//!
//! Vaporetto は点予測 + 線形分類で字単位に語境界を決めるモデル。辞書フリーで
//! 配布が軽い (~10MB)。`real` feature を有効にしたときのみ実体がリンクされる。
//!
//! モデルは `bccwj-suw+unidic_pos+pron.model.zst` などを想定し、
//! ファイルパスを与えてロードする。

use ellisii_jp_tokenizer_core::JpTokenizer;
use std::path::Path;

#[cfg(feature = "real")]
mod imp {
    use super::*;
    use std::fs::File;
    use std::io::Read;
    use vaporetto::{Predictor, Sentence};

    pub struct VaporettoTokenizer {
        predictor: Predictor,
    }

    impl VaporettoTokenizer {
        pub fn from_zst(path: &Path) -> Result<Self, String> {
            let mut f = File::open(path).map_err(|e| format!("open {path:?}: {e}"))?;
            let mut buf = Vec::new();
            f.read_to_end(&mut buf).map_err(|e| format!("read: {e}"))?;
            let mut decoder =
                zstd::Decoder::with_buffer(&buf[..]).map_err(|e| format!("zstd: {e}"))?;
            let model = vaporetto::Model::read(&mut decoder).map_err(|e| format!("model: {e}"))?;
            let predictor = Predictor::new(model, false).map_err(|e| format!("predictor: {e}"))?;
            Ok(Self { predictor })
        }
    }

    impl JpTokenizer for VaporettoTokenizer {
        fn tokenize(&self, text: &str) -> Vec<String> {
            let trimmed: String = text.chars().filter(|c| !c.is_control()).collect();
            if trimmed.is_empty() {
                return vec![];
            }
            let mut s = match Sentence::from_raw(trimmed) {
                Ok(s) => s,
                Err(_) => return vec![],
            };
            self.predictor.predict(&mut s);
            s.iter_tokens()
                .map(|t| t.surface().to_string())
                .filter(|t| !t.trim().is_empty())
                .collect()
        }
    }
}

#[cfg(not(feature = "real"))]
mod imp {
    use super::*;

    pub struct VaporettoTokenizer;

    impl VaporettoTokenizer {
        pub fn from_zst(_: &Path) -> Result<Self, String> {
            Err("ellisii-jp-tokenizer-vaporetto built without `real` feature".into())
        }
    }

    impl JpTokenizer for VaporettoTokenizer {
        fn tokenize(&self, _text: &str) -> Vec<String> {
            vec![]
        }
    }
}

pub use imp::VaporettoTokenizer;

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn from_zst_errors_when_file_missing() {
        let p = PathBuf::from("/this/path/does/not/exist.model.zst");
        let res = VaporettoTokenizer::from_zst(&p);
        assert!(res.is_err());
    }

    #[cfg(not(feature = "real"))]
    #[test]
    fn stub_from_zst_message_mentions_real_feature() {
        let err = match VaporettoTokenizer::from_zst(&PathBuf::from("x")) {
            Ok(_) => panic!("expected Err"),
            Err(e) => e,
        };
        assert!(err.contains("real"), "unexpected error: {err}");
    }
}
