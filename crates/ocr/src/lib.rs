use async_trait::async_trait;
use ellisii_core::Result;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::sync::Arc;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OcrBlock {
    pub text: String,
    pub bbox: [f32; 4],
    pub page: u32,
    pub confidence: f32,
}

/// 1 ページぶんのラスタ画像 (一時 PNG ファイル)。
/// `_keepalive` は tempdir などの guard を保持して、
/// `path` が指すファイルを `RasterizedPage` のライフタイム内有効に保つ。
#[derive(Clone)]
pub struct RasterizedPage {
    pub page: u32,
    pub path: PathBuf,
    pub _keepalive: Arc<dyn std::any::Any + Send + Sync>,
}

impl std::fmt::Debug for RasterizedPage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RasterizedPage")
            .field("page", &self.page)
            .field("path", &self.path)
            .finish()
    }
}

/// PDF を 1 ページ 1 画像に展開するバックエンド。
/// テキストレイヤなしの PDF (またはスキャン混在 PDF の画像ページだけ) を
/// OCR にかけるための前処理に使う。
#[async_trait]
pub trait PdfRasterizer: Send + Sync {
    /// PDF の総ページ数。スキャン混在 PDF で「テキスト無しページ」を
    /// 算出するために使うので、`rasterize_pages` を呼ぶ前に軽く取れること。
    async fn page_count(&self, path: &str) -> Result<u32>;

    /// `pages` で指定したページ (1-indexed) だけを画像化する。
    /// 空スライスを渡したら空 Vec を返す (render を一切走らせない)。
    async fn rasterize_pages(&self, path: &str, pages: &[u32]) -> Result<Vec<RasterizedPage>>;

    /// `rasterize_pages` のストリーミング版: 1 ページ rasterize する都度 `tx`
    /// に送信する。`tx` 受信側 (= OCR ワーカー) が並列で消費すれば、rasterize
    /// と OCR が重なって全体 throughput が上がる。
    ///
    /// 既定実装は `rasterize_pages` でまとめて作ってから順に送信するだけ
    /// (= ストリーミング効果なし、後方互換のための fallback)。
    /// `PdfiumRasterizer` 等は spawn_blocking 内のループから 1 ページずつ
    /// `blocking_send` する形で override し、PDF を 1 度だけ load しつつ
    /// 順次 hand-off する。
    async fn rasterize_pages_streaming(
        &self,
        path: &str,
        pages: &[u32],
        tx: tokio::sync::mpsc::Sender<RasterizedPage>,
    ) -> Result<()> {
        let rendered = self.rasterize_pages(path, pages).await?;
        for r in rendered {
            if tx.send(r).await.is_err() {
                break;
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct OcrConfig {
    /// DEIM 行検出モデル (.onnx)
    pub det_model: std::path::PathBuf,
    /// PARSeq 認識モデル 100 (.onnx)
    pub model100: std::path::PathBuf,
    pub model30: std::path::PathBuf,
    pub model50: std::path::PathBuf,
    /// charset (.txt)
    pub charset: std::path::PathBuf,
    pub det_conf_threshold: f32,
    pub min_line_confidence: f32,
}

/// OCR バックエンドの抽象。`onnx` feature 有効時に ndlocr 実装が、
/// 無効時はダミーが選ばれる。
#[async_trait]
pub trait OcrBackend: Send + Sync {
    async fn ocr_image(&self, path: &str) -> Result<Vec<OcrBlock>>;

    /// 複数画像をまとめて OCR する batch API。
    ///
    /// 既定実装は `ocr_image` を順番に呼ぶだけ (= no batching effect)。
    /// 実装 (`NdlocrBackend`) は **全ページの行を 1 つの parseq batch に
    /// concat して 1 度の inference で解く** ことで、parseq decoder の
    /// max_seq_len × per_token cost を全行で共有する (= 1 行あたりのコストが
    /// バッチサイズに対してほぼ一定)。typical で 3 ページぶんを 1 batch に
    /// すれば parseq 全体時間が 1/3 近くまで縮む。
    ///
    /// 戻り値は入力 `paths` と同じ順・同じ長さ。1 ページの OCR が失敗した
    /// 場合は **そのページだけ空 Vec** を返し、他のページは継続する。
    /// 構造的な失敗 (画像 load 不能 / モデル未ロード) は `Err` を返す。
    async fn ocr_images_batch(&self, paths: &[&str]) -> Result<Vec<Vec<OcrBlock>>> {
        let mut out = Vec::with_capacity(paths.len());
        for p in paths {
            out.push(self.ocr_image(p).await.unwrap_or_default());
        }
        Ok(out)
    }
}

#[cfg(feature = "onnx")]
pub mod ndlocr_backend;
#[cfg(feature = "onnx")]
pub use ndlocr_backend::NdlocrBackend;

// StubOcr は feature に関係なく公開する。
// 実モデル未配置時のフォールバックや軽量ビルド用。
pub mod stub_backend;
pub use stub_backend::StubOcr;
