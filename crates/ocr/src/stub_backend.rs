use crate::{OcrBackend, OcrBlock};
use async_trait::async_trait;
use ellisii_core::Result;

/// onnx feature が無効なときのフォールバック。常に空を返す。
pub struct StubOcr;

#[async_trait]
impl OcrBackend for StubOcr {
    async fn ocr_image(&self, _path: &str) -> Result<Vec<OcrBlock>> {
        Ok(vec![])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn stub_returns_empty_for_any_path() {
        assert!(StubOcr.ocr_image("/nonexistent.png").await.unwrap().is_empty());
        assert!(StubOcr.ocr_image("").await.unwrap().is_empty());
    }

    #[test]
    fn stub_is_object_safe() {
        let _: Box<dyn OcrBackend> = Box::new(StubOcr);
    }
}
