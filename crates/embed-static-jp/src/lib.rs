//! `hotchpotch/static-embedding-japanese` 互換のローダ。
//!
//! 静的埋め込みなので Transformer は不要で、語彙→埋め込みベクトルの
//! 単純な lookup と平均プーリングだけで済む。
//!
//! 期待するモデル配置 (HuggingFace `snapshot_download` で入る形を想定):
//!
//! ```text
//! <model_dir>/
//!   tokenizer.json
//!   0_StaticEmbedding/model.safetensors    # tensor `embedding.weight` shape=(vocab, dim)
//! ```
//!
//! `0_StaticEmbedding/` がない場合はルート直下の `model.safetensors` も探す。

use async_trait::async_trait;
use ellisii_core::{Error, Result};
use ellisii_embed_core::Embedder;
use memmap2::Mmap;
use safetensors::SafeTensors;
use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokenizers::Tokenizer;

pub struct StaticJpEmbedder {
    tokenizer: Tokenizer,
    table: Arc<EmbeddingTable>,
}

struct EmbeddingTable {
    data: Vec<f32>,
    vocab: usize,
    dim: usize,
}

impl StaticJpEmbedder {
    pub fn from_dir(model_dir: impl AsRef<Path>) -> Result<Self> {
        let dir = model_dir.as_ref();
        let tok_path = dir.join("tokenizer.json");
        let tokenizer = Tokenizer::from_file(&tok_path)
            .map_err(|e| Error::Embed(format!("tokenizer.json: {e}")))?;
        let weights_path = locate_weights(dir)
            .ok_or_else(|| Error::Embed("model.safetensors not found".into()))?;
        let table = load_embedding_table(&weights_path)?;
        Ok(Self {
            tokenizer,
            table: Arc::new(table),
        })
    }

    /// テスト用: tokenizer と埋め込みテーブルを直接渡す。
    pub fn from_parts(
        tokenizer: Tokenizer,
        table: Vec<f32>,
        vocab: usize,
        dim: usize,
    ) -> Result<Self> {
        if table.len() != vocab * dim {
            return Err(Error::Embed("table shape mismatch".into()));
        }
        Ok(Self {
            tokenizer,
            table: Arc::new(EmbeddingTable {
                data: table,
                vocab,
                dim,
            }),
        })
    }
}

fn locate_weights(dir: &Path) -> Option<PathBuf> {
    for sub in ["0_StaticEmbedding/model.safetensors", "model.safetensors"] {
        let p = dir.join(sub);
        if p.is_file() {
            return Some(p);
        }
    }
    None
}

fn load_embedding_table(path: &Path) -> Result<EmbeddingTable> {
    let file = File::open(path).map_err(Error::Io)?;
    let mmap = unsafe { Mmap::map(&file).map_err(Error::Io)? };
    let st =
        SafeTensors::deserialize(&mmap).map_err(|e| Error::Embed(format!("safetensors: {e}")))?;

    // 候補となる tensor 名を順に試す
    let candidates = [
        "embedding.weight",
        "embeddings.weight",
        "0_StaticEmbedding.embedding.weight",
        "weight",
    ];
    let names: Vec<String> = st.names().into_iter().map(|s| s.to_string()).collect();
    let chosen = candidates
        .iter()
        .find(|n| names.iter().any(|m| m == *n))
        .copied()
        .or_else(|| names.first().map(|s| s.as_str()))
        .ok_or_else(|| Error::Embed("no tensors in safetensors".into()))?;

    let view = st
        .tensor(chosen)
        .map_err(|e| Error::Embed(format!("tensor `{chosen}`: {e}")))?;
    let shape = view.shape();
    if shape.len() != 2 {
        return Err(Error::Embed(format!("expected 2D tensor, got {shape:?}")));
    }
    let vocab = shape[0];
    let dim = shape[1];
    let bytes = view.data();
    if bytes.len() != vocab * dim * 4 {
        return Err(Error::Embed(format!(
            "byte length mismatch: {} vs {}",
            bytes.len(),
            vocab * dim * 4
        )));
    }
    let mut data = vec![0_f32; vocab * dim];
    for (i, chunk) in bytes.chunks_exact(4).enumerate() {
        data[i] = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
    }
    Ok(EmbeddingTable { data, vocab, dim })
}

#[async_trait]
impl Embedder for StaticJpEmbedder {
    fn dim(&self) -> usize {
        self.table.dim
    }

    async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        let table = self.table.clone();
        let encs = self
            .tokenizer
            .encode_batch(texts.to_vec(), false)
            .map_err(|e| Error::Embed(format!("encode: {e}")))?;
        let mut out = Vec::with_capacity(encs.len());
        for enc in encs {
            let ids = enc.get_ids();
            out.push(mean_pool(&table, ids));
        }
        Ok(out)
    }
}

fn mean_pool(table: &EmbeddingTable, ids: &[u32]) -> Vec<f32> {
    let mut acc = vec![0_f32; table.dim];
    let mut n = 0_f32;
    for &id in ids {
        let id = id as usize;
        if id >= table.vocab {
            continue;
        }
        let row = &table.data[id * table.dim..(id + 1) * table.dim];
        for (a, r) in acc.iter_mut().zip(row.iter()) {
            *a += *r;
        }
        n += 1.0;
    }
    if n > 0.0 {
        for a in &mut acc {
            *a /= n;
        }
    }
    // L2 正規化 (cosine 類似度を扱いやすくするため)
    let norm = acc.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for a in &mut acc {
            *a /= norm;
        }
    }
    acc
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokenizers::Tokenizer;

    fn tiny_tokenizer() -> Tokenizer {
        // BPE で a/b/c の 3 トークン
        use tokenizers::models::wordlevel::{WordLevel, WordLevelBuilder};
        use tokenizers::pre_tokenizers::whitespace::Whitespace;
        let mut vocab: ahash::AHashMap<String, u32> = ahash::AHashMap::new();
        vocab.insert("[UNK]".to_string(), 0_u32);
        vocab.insert("a".to_string(), 1);
        vocab.insert("b".to_string(), 2);
        vocab.insert("c".to_string(), 3);
        let model: WordLevel = WordLevelBuilder::new()
            .vocab(vocab)
            .unk_token("[UNK]".into())
            .build()
            .unwrap();
        let mut tok = Tokenizer::new(model);
        tok.with_pre_tokenizer(Some(Whitespace {}));
        tok
    }

    #[tokio::test]
    async fn mean_pools_known_vocab() {
        let dim = 2;
        let vocab = 4;
        // [UNK]=(0,0), a=(1,0), b=(0,1), c=(1,1)
        let table: Vec<f32> = vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let e = StaticJpEmbedder::from_parts(tiny_tokenizer(), table, vocab, dim).unwrap();
        let v = e.embed(&["a b".to_string()]).await.unwrap();
        assert_eq!(v[0].len(), 2);
        // 平均 = (0.5, 0.5) → L2 正規化で 1/sqrt(2) ずつ
        let expect = 1.0 / 2_f32.sqrt();
        assert!((v[0][0] - expect).abs() < 1e-5);
        assert!((v[0][1] - expect).abs() < 1e-5);
    }

    #[tokio::test]
    async fn dim_matches_table() {
        let e = StaticJpEmbedder::from_parts(tiny_tokenizer(), vec![0.0; 4 * 8], 4, 8).unwrap();
        assert_eq!(e.dim(), 8);
    }

    #[test]
    fn rejects_shape_mismatch() {
        let r = StaticJpEmbedder::from_parts(tiny_tokenizer(), vec![1.0, 2.0, 3.0], 4, 8);
        assert!(r.is_err());
    }
}
