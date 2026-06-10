//! builder 経由で注入した OCR バックエンドが、テキストレイヤの無い PDF の
//! ingest フォールバックとして機能することを確認する。
//! 外部モデル不要 (DummyEmbedder + InMemoryStore + Fake OCR/Rasterizer)。
//!
//! ここで検証するのは「SDK builder → Ingestor への配線」だけ。
//! OCR 実体 (`NdlocrBackend`) や rasterizer 実体 (`PdfiumRasterizer`) の
//! 品質は各 crate のテストに任せる。

use std::fmt::Write as _;
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use async_trait::async_trait;
use ellisii_ocr::{OcrBackend, OcrBlock, PdfRasterizer, RasterizedPage};
use ellisii_sdk::{Ellisii, IngestPathOutcome, SearchOptions};

/// 常に 1 ページ扱いで、ダミー PNG のパスを返す rasterizer。
struct FakeRasterizer {
    dir: Arc<tempfile::TempDir>,
}

#[async_trait]
impl PdfRasterizer for FakeRasterizer {
    async fn page_count(&self, _path: &str) -> ellisii_core::Result<u32> {
        Ok(1)
    }

    async fn rasterize_pages(
        &self,
        _path: &str,
        pages: &[u32],
    ) -> ellisii_core::Result<Vec<RasterizedPage>> {
        let mut out = Vec::with_capacity(pages.len());
        for &page in pages {
            let p = self.dir.path().join(format!("page-{page}.png"));
            std::fs::write(&p, b"fake png")
                .map_err(|e| ellisii_core::Error::Parse(format!("write fake png: {e}")))?;
            out.push(RasterizedPage {
                page,
                path: p,
                _keepalive: self.dir.clone(),
            });
        }
        Ok(out)
    }
}

/// 固定の日本語テキストを返す OCR (呼ばれた回数を記録する)。
struct FakeOcr {
    calls: Arc<AtomicUsize>,
}

#[async_trait]
impl OcrBackend for FakeOcr {
    async fn ocr_image(&self, _path: &str) -> ellisii_core::Result<Vec<OcrBlock>> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        Ok(vec![OcrBlock {
            text: "秘密保持義務は契約終了後も3年間存続するものとします。\
                   本条はスキャン文書から光学文字認識で抽出された条項です。"
                .to_string(),
            bbox: [0.0, 0.0, 1.0, 1.0],
            page: 1,
            confidence: 0.99,
        }])
    }
}

/// テキストレイヤを一切持たない最小 PDF (1 ページ、描画なし)。
/// pdf_extract は空テキストを返すため、OCR フォールバックの入口になる。
fn build_textless_pdf(path: &Path) {
    let objects = [
        "<< /Type /Catalog /Pages 2 0 R >>".to_string(),
        "<< /Type /Pages /Kids [3 0 R] /Count 1 >>".to_string(),
        "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R >>".to_string(),
        // テキスト描画オペレータを含まない空のコンテンツストリーム
        "<< /Length 0 >>\nstream\n\nendstream".to_string(),
    ];
    let mut pdf = String::from("%PDF-1.4\n");
    let mut offsets = Vec::with_capacity(objects.len());
    for (i, obj) in objects.iter().enumerate() {
        offsets.push(pdf.len());
        let _ = write!(pdf, "{} 0 obj\n{obj}\nendobj\n", i + 1);
    }
    let xref_pos = pdf.len();
    let _ = write!(pdf, "xref\n0 {}\n", objects.len() + 1);
    pdf.push_str("0000000000 65535 f \n");
    for off in offsets {
        let _ = writeln!(pdf, "{off:010} 00000 n ");
    }
    let _ = write!(
        pdf,
        "trailer\n<< /Size {} /Root 1 0 R >>\nstartxref\n{xref_pos}\n%%EOF\n",
        objects.len() + 1
    );
    std::fs::write(path, pdf).unwrap();
}

#[tokio::test]
async fn scanned_pdf_is_indexed_via_injected_ocr() {
    let tmp = tempfile::tempdir().unwrap();
    let pdf = tmp.path().join("scan.pdf");
    build_textless_pdf(&pdf);

    let calls = Arc::new(AtomicUsize::new(0));
    let ellisii = Ellisii::builder()
        .with_embedder_dummy(16)
        .with_store_memory()
        .with_ocr(Arc::new(FakeOcr {
            calls: calls.clone(),
        }))
        .with_pdf_rasterizer(Arc::new(FakeRasterizer {
            dir: Arc::new(tempfile::tempdir().unwrap()),
        }))
        .build()
        .unwrap();

    let outcome = ellisii.index_file(&pdf).await.unwrap();
    let report = match outcome {
        IngestPathOutcome::Ingested(r) => r,
        IngestPathOutcome::Unchanged => panic!("expected ingest"),
    };
    assert!(report.chunks_stored >= 1, "report={report:?}");
    assert!(
        calls.load(Ordering::SeqCst) >= 1,
        "OCR バックエンドが呼ばれていない"
    );

    // OCR で抽出したテキストが検索でヒットする
    let hits = ellisii
        .search("秘密保持義務の存続期間", SearchOptions::default())
        .await
        .unwrap();
    assert!(
        hits.iter().any(|h| h.chunk.text.contains("3年間存続")),
        "hits={hits:?}"
    );
}

#[tokio::test]
async fn textless_pdf_yields_zero_chunks_without_ocr() {
    // OCR 未設定なら従来どおり 0 chunks (フォールバックは opt-in)。
    let tmp = tempfile::tempdir().unwrap();
    let pdf = tmp.path().join("scan.pdf");
    build_textless_pdf(&pdf);

    let ellisii = Ellisii::builder()
        .with_embedder_dummy(16)
        .with_store_memory()
        .build()
        .unwrap();
    let outcome = ellisii.index_file(&pdf).await.unwrap();
    let report = match outcome {
        IngestPathOutcome::Ingested(r) => r,
        IngestPathOutcome::Unchanged => panic!("expected ingest"),
    };
    assert_eq!(report.chunks_stored, 0, "report={report:?}");
}
