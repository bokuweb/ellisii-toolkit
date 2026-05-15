use async_trait::async_trait;
use ellisii_core::Result;
use ellisii_embed_core::Embedder;

/// FNV-1a ハッシュベースの決定的ダミー埋め込み。配線確認のみで、品質は保証しない。
pub struct DummyEmbedder {
    pub dim: usize,
}

impl DummyEmbedder {
    pub fn new(dim: usize) -> Self {
        Self { dim }
    }
}

#[async_trait]
impl Embedder for DummyEmbedder {
    fn dim(&self) -> usize {
        self.dim
    }

    async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        Ok(texts.iter().map(|t| hash_vec(t, self.dim)).collect())
    }
}

fn hash_vec(text: &str, dim: usize) -> Vec<f32> {
    let mut h: u64 = 0xcbf29ce484222325;
    for b in text.as_bytes() {
        h ^= *b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    (0..dim)
        .map(|i| {
            let bits = h.rotate_left((i as u32).wrapping_mul(7));
            (bits as i32 as f32) / (i32::MAX as f32)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn deterministic_for_same_text() {
        let e = DummyEmbedder::new(8);
        let a = e.embed(&["hello".to_string()]).await.unwrap();
        let b = e.embed(&["hello".to_string()]).await.unwrap();
        assert_eq!(a, b);
    }

    #[tokio::test]
    async fn dim_matches() {
        let e = DummyEmbedder::new(16);
        let v = e.embed(&["x".to_string()]).await.unwrap();
        assert_eq!(v[0].len(), 16);
    }
}
