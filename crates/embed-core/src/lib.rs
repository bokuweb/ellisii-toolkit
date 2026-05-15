use async_trait::async_trait;
use ellisii_core::Result;

#[async_trait]
pub trait Embedder: Send + Sync {
    fn dim(&self) -> usize;
    async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>>;
}

#[cfg(test)]
mod tests {
    use super::*;

    struct ZeroEmbed(usize);

    #[async_trait]
    impl Embedder for ZeroEmbed {
        fn dim(&self) -> usize {
            self.0
        }
        async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
            Ok(texts.iter().map(|_| vec![0.0; self.0]).collect())
        }
    }

    #[tokio::test]
    async fn embedder_returns_dim_sized_vectors_per_input() {
        let e = ZeroEmbed(8);
        let out = e.embed(&["a".into(), "b".into(), "c".into()]).await.unwrap();
        assert_eq!(out.len(), 3);
        for v in out {
            assert_eq!(v.len(), e.dim());
        }
    }

    #[tokio::test]
    async fn embedder_handles_empty_input() {
        let e = ZeroEmbed(4);
        assert!(e.embed(&[]).await.unwrap().is_empty());
    }

    #[test]
    fn embedder_is_object_safe() {
        let _: Box<dyn Embedder> = Box::new(ZeroEmbed(2));
    }
}
