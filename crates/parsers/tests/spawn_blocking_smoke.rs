//! parser facade が tokio async コンテキストから呼ばれても
//! nested-runtime panic を起こさないことの smoke test。
//!
//! 実 PDF は要らない — `parse()` が `spawn_blocking` 経由で sync パーサに
//! 落ちることだけ検証する。存在しないパスを渡し、エラーで返る (panic しない)
//! ことをアサート。
//!
//! 実 pdfium バイナリ DL を伴う E2E テストは `#[ignore]` 印で別管理 (将来)。

#[tokio::test]
async fn parse_missing_pdf_returns_err_not_panic() {
    let res = ellisii_parsers::parse("/nonexistent-test-file.pdf").await;
    assert!(res.is_err(), "should return Err for missing path");
    let msg = format!("{}", res.err().unwrap());
    // panic ではなく Error::Parse もしくは Error::Io のメッセージが来る
    assert!(!msg.is_empty());
}

#[tokio::test]
async fn parse_missing_text_returns_err_not_panic() {
    let res = ellisii_parsers::parse("/nonexistent-test-file.txt").await;
    assert!(res.is_err());
}

/// 実 PDF を pdfium 経由で読めることの E2E 検証。
/// 初回は pdfium バイナリ (~10MB) を `~/.cache/pdf2md/` に DL するため時間がかかる。
/// CI では `cargo test -- --ignored` で別途実行。
#[tokio::test]
#[ignore]
async fn parse_real_pdf_via_pdfium() {
    let path = "/tmp/test.pdf";
    if !std::path::Path::new(path).exists() {
        eprintln!("skipping: {path} not found (run `echo hi | cupsfilter > /tmp/test.pdf`)");
        return;
    }
    let res = ellisii_parsers::parse(path).await;
    assert!(res.is_ok(), "parse failed: {:?}", res.err());
    let doc = res.unwrap();
    assert!(!doc.blocks.is_empty(), "should extract at least 1 block");
}

/// pdfium-auto が async コンテキスト下から panic せず動くこと
/// (= reqwest::blocking が tokio runtime と衝突しない) を確かめる。
#[tokio::test]
#[ignore]
async fn pdfium_auto_binds_inside_tokio() {
    let res = tokio::task::spawn_blocking(|| {
        pdfium_auto::bind_pdfium_silent()
            .map(|_| ())
            .map_err(|e| format!("{e}"))
    })
    .await;
    let bind = res.expect("spawn_blocking join");
    eprintln!("bind_pdfium_silent: {:?}", &bind);
    assert!(bind.is_ok(), "pdfium bind failed: {:?}", bind.err());
}
