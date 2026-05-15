//! 各 parser-* crate を束ねて MIME 種別から dispatch する facade。

pub use ellisii_parsers_core::{detect_kind, ParsedBlock, ParsedDocument};

use ellisii_core::{Error, Result, SourceKind};

pub async fn parse(path: &str) -> Result<ParsedDocument> {
    let kind =
        detect_kind(path).ok_or_else(|| Error::Parse(format!("unknown file type: {path}")))?;
    // 各パーサは同期 I/O。pdfium-auto は reqwest::blocking を使うため、
    // tokio の async コンテキスト内から直接呼ぶと nested runtime で panic する。
    // 全パーサ共通に spawn_blocking で逃がす (他パーサも sync なので副作用なし)。
    let path_owned = path.to_string();
    let blocks = tokio::task::spawn_blocking(move || -> Result<Vec<ParsedBlock>> {
        match kind {
            SourceKind::Pdf => ellisii_parser_pdf::parse(&path_owned),
            SourceKind::Docx => ellisii_parser_docx::parse(&path_owned),
            SourceKind::Xlsx => ellisii_parser_xlsx::parse(&path_owned),
            SourceKind::Pptx => ellisii_parser_pptx::parse(&path_owned),
            SourceKind::Markdown => ellisii_parser_markdown::parse(&path_owned),
            SourceKind::Text => ellisii_parser_text::parse(&path_owned),
            SourceKind::Image => Ok(vec![]), // OCR レイヤで処理
            SourceKind::Audio => {
                // Phase 2 (Meeting Recorder) で導入。STT には Transcriber 実装を
                // 明示的に渡す必要があるため、汎用 dispatch では handle しない。
                // SDK 側で SourceKind::Audio を見たら `ellisii_parser_audio::
                // parse_audio(...)` を別経路で呼び出す (Phase 4 で wiring 予定)。
                Err(Error::Parse(format!(
                    "audio parsing requires explicit Transcriber setup; \
                     use ellisii_parser_audio::parse_audio() directly for: {path_owned}"
                )))
            }
        }
    })
    .await
    .map_err(|e| Error::Parse(format!("parse task join: {e}")))??;
    Ok(ParsedDocument { blocks, kind })
}
