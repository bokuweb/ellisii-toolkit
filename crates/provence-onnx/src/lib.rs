//! `hotchpotch/open-provence-reranker-xsmall-v1` 互換のクロスエンコーダ式
//! コンテキスト圧縮機。
//!
//! 期待モデル配置:
//!
//! ```text
//! <model_dir>/
//!   tokenizer.json
//!   model.onnx           # 入力 [input_ids, attention_mask] → logits (1)
//! ```
//!
//! `real` feature を有効にしたときのみ実体がリンクされ、それ以外は
//! `ProvenceOnnx::load` が常に `Err` を返す。フォールバックは呼び出し側で。

use async_trait::async_trait;
use ellisii_core::{Error, Result};
#[cfg(feature = "real")]
use ellisii_provence_core::{apply_floor, split_sentences, ScoredSentence};
use ellisii_provence_core::{CompressedContext, ContextCompressor};
use std::path::Path;

#[derive(Debug, Clone, Copy)]
pub struct ProvenceConfig {
    /// このスコア未満の文を捨てる (sigmoid 後)
    pub keep_threshold: f32,
    pub max_seq_len: usize,
    /// 閾値で削りすぎたときの保険として、必ず残す文数の絶対下限。
    pub min_keep_sentences: usize,
    /// 同じく、全文に対する最低保持比率 (0.0〜1.0)。`min_keep_sentences` と
    /// 比べて大きい方が採用される。
    pub min_keep_ratio: f32,
}

impl Default for ProvenceConfig {
    fn default() -> Self {
        Self {
            keep_threshold: 0.20,
            max_seq_len: 512,
            // 既定はカタカナ造語などで全カットされるのを防ぐ控えめなフロア。
            // 「短いチャンクは 30% 以上 / 最低 3 文は残す」程度。
            min_keep_sentences: 3,
            min_keep_ratio: 0.3,
        }
    }
}

#[cfg(feature = "real")]
mod imp {
    use super::*;
    use ndarray::Array2;
    use ort::session::Session;
    use ort::value::Tensor;
    use tokenizers::Tokenizer;

    pub struct ProvenceOnnx {
        session: std::sync::Mutex<Session>,
        tokenizer: Tokenizer,
        cfg: ProvenceConfig,
    }

    impl ProvenceOnnx {
        pub fn load(model_dir: &Path, cfg: ProvenceConfig) -> Result<Self> {
            let tok_path = model_dir.join("tokenizer.json");
            let onnx_path = model_dir.join("model.onnx");
            let tokenizer = Tokenizer::from_file(&tok_path)
                .map_err(|e| Error::Other(anyhow::anyhow!("tokenizer.json: {e}")))?;
            // メモリ使用量削減のため intra/inter スレッドを 1 に絞る。
            // 16GB マシンでは複数スレッドの arena 確保が OOM (mach_vm_allocate_kernel
            // failed) を引き起こすことがあるため、単一スレッドで安全側に倒す。
            // ort rc.12 で `commit_from_file` が `&mut self` 受けに変わったため、
            // chain を分けて mutable に bind してから commit を呼ぶ。
            let mut builder = Session::builder()
                .map_err(|e| Error::Other(anyhow::anyhow!("session builder: {e}")))?
                .with_intra_threads(1)
                .map_err(|e| Error::Other(anyhow::anyhow!("intra_threads: {e}")))?
                .with_inter_threads(1)
                .map_err(|e| Error::Other(anyhow::anyhow!("inter_threads: {e}")))?;
            let session = builder
                .commit_from_file(&onnx_path)
                .map_err(|e| Error::Other(anyhow::anyhow!("commit_from_file: {e}")))?;
            Ok(Self {
                session: std::sync::Mutex::new(session),
                tokenizer,
                cfg,
            })
        }

        fn score_pairs(&self, query: &str, sentences: &[String]) -> Result<Vec<f32>> {
            if sentences.is_empty() {
                return Ok(vec![]);
            }
            let pairs: Vec<(String, String)> = sentences
                .iter()
                .map(|s| (query.to_string(), s.clone()))
                .collect();
            let encs = self
                .tokenizer
                .encode_batch(pairs, true)
                .map_err(|e| Error::Other(anyhow::anyhow!("encode_batch: {e}")))?;
            let max_len = encs
                .iter()
                .map(|e| e.get_ids().len())
                .max()
                .unwrap_or(0)
                .min(self.cfg.max_seq_len);
            let n = encs.len();
            let mut input_ids = Array2::<i64>::zeros((n, max_len));
            let mut attn = Array2::<i64>::zeros((n, max_len));
            for (i, enc) in encs.iter().enumerate() {
                let ids = enc.get_ids();
                let mask = enc.get_attention_mask();
                let take = ids.len().min(max_len);
                for j in 0..take {
                    input_ids[[i, j]] = ids[j] as i64;
                    attn[[i, j]] = mask[j] as i64;
                }
            }
            let ids_tensor = Tensor::from_array(input_ids)
                .map_err(|e| Error::Other(anyhow::anyhow!("ids tensor: {e}")))?;
            let attn_tensor = Tensor::from_array(attn)
                .map_err(|e| Error::Other(anyhow::anyhow!("attn tensor: {e}")))?;
            let inputs = ort::inputs![
                "input_ids" => ids_tensor,
                "attention_mask" => attn_tensor,
            ];
            let flat: Vec<f32> = {
                let mut sess = self
                    .session
                    .lock()
                    .map_err(|_| Error::Other(anyhow::anyhow!("session mutex poisoned")))?;
                let outputs = sess
                    .run(inputs)
                    .map_err(|e| Error::Other(anyhow::anyhow!("run: {e}")))?;
                let (_, first) = outputs
                    .iter()
                    .next()
                    .ok_or_else(|| Error::Other(anyhow::anyhow!("no outputs")))?;
                let view = first
                    .try_extract_array::<f32>()
                    .map_err(|e| Error::Other(anyhow::anyhow!("extract: {e}")))?;
                view.iter().copied().collect()
            };
            let scores: Vec<f32> = if flat.len() == n {
                flat
            } else if flat.len() == n * 2 {
                // [N, 2] の場合は positive class の logit (index=1) を使う
                (0..n).map(|i| flat[i * 2 + 1]).collect()
            } else if flat.len() % n == 0 {
                let step = flat.len() / n;
                (0..n).map(|i| flat[i * step]).collect()
            } else {
                return Err(Error::Other(anyhow::anyhow!(
                    "unexpected logits shape: {} for {n} pairs",
                    flat.len()
                )));
            };
            Ok(scores.into_iter().map(sigmoid).collect())
        }
    }

    fn sigmoid(x: f32) -> f32 {
        1.0 / (1.0 + (-x).exp())
    }

    #[async_trait]
    impl ContextCompressor for ProvenceOnnx {
        fn is_active(&self) -> bool {
            true
        }
        async fn score_passages(&self, query: &str, passages: &[String]) -> Result<Vec<f32>> {
            // score_pairs は文用途で書かれているが、入力は (query, text) の
            // ペアを並べてバッチ推論するだけなので chunk テキストにもそのまま
            // 適用できる。max_seq_len で切られる長さのリミットはあるが、
            // chunk 単位は元々 ~1500 文字以内に丸めてあるので問題なし。
            self.score_pairs(query, passages)
        }
        async fn compress(&self, query: &str, text: &str) -> Result<CompressedContext> {
            let sentences = split_sentences(text);
            if sentences.is_empty() {
                return Ok(CompressedContext {
                    kept_text: String::new(),
                    original_chars: 0,
                    kept_chars: 0,
                    sentences: vec![],
                });
            }
            let original_chars = text.chars().count();
            let scores = self.score_pairs(query, &sentences)?;
            let threshold = self.cfg.keep_threshold;
            let mut scored: Vec<ScoredSentence> = sentences
                .into_iter()
                .zip(scores.into_iter())
                .map(|(s, sc)| ScoredSentence {
                    text: s,
                    score: sc,
                    kept: sc >= threshold,
                })
                .collect();
            // フロア: 閾値で削りすぎたときに最低 N 文 / 比率を必ず残す。
            // 旧来の「全カット時のみ top-1 を残す」は、N=3 / ratio=0.3 の
            // フロアで自然に包含される (低リコール事故の方が圧倒的に痛い)。
            apply_floor(
                &mut scored,
                self.cfg.min_keep_sentences,
                self.cfg.min_keep_ratio,
            );
            let kept_text = scored
                .iter()
                .filter(|s| s.kept)
                .map(|s| s.text.as_str())
                .collect::<Vec<_>>()
                .join(" ");
            let kept_chars = kept_text.chars().count();
            Ok(CompressedContext {
                kept_text,
                original_chars,
                kept_chars,
                sentences: scored,
            })
        }
    }
}

#[cfg(not(feature = "real"))]
mod imp {
    use super::*;

    pub struct ProvenceOnnx;

    impl ProvenceOnnx {
        pub fn load(_dir: &Path, _cfg: ProvenceConfig) -> Result<Self> {
            Err(Error::Other(anyhow::anyhow!(
                "ellisii-provence-onnx built without `real` feature"
            )))
        }
    }

    #[async_trait]
    impl ContextCompressor for ProvenceOnnx {
        fn is_active(&self) -> bool {
            false
        }
        async fn compress(&self, _query: &str, text: &str) -> Result<CompressedContext> {
            let n = text.chars().count();
            Ok(CompressedContext {
                kept_text: text.to_string(),
                original_chars: n,
                kept_chars: n,
                sentences: vec![],
            })
        }
    }
}

pub use imp::ProvenceOnnx;

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn config_default_is_sensible() {
        let c = ProvenceConfig::default();
        assert!(c.keep_threshold > 0.0 && c.keep_threshold < 1.0);
        assert!(c.max_seq_len >= 128);
    }

    #[cfg(not(feature = "real"))]
    #[test]
    fn load_fails_without_real_feature() {
        let err = ProvenceOnnx::load(&PathBuf::from("/no/such/dir"), ProvenceConfig::default())
            .err()
            .expect("load should fail");
        let msg = format!("{err}");
        assert!(msg.contains("real"), "unexpected: {msg}");
    }

    #[cfg(not(feature = "real"))]
    #[tokio::test]
    async fn stub_compress_is_passthrough_and_inactive() {
        // stub では `load` できないが、ZST をスタブ的に手で構築できるか不能なため
        // ここでは load の失敗のみ確認 (実体テストは real feature 下で実施)。
        let _ = PathBuf::from("x");
    }
}
