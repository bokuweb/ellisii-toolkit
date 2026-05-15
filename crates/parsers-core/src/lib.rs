use ellisii_core::SourceKind;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParsedBlock {
    pub text: String,
    pub heading_path: Vec<String>,
    pub page: Option<u32>,
    pub bbox: Option<[f32; 4]>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParsedDocument {
    pub blocks: Vec<ParsedBlock>,
    pub kind: SourceKind,
}

pub fn detect_kind(path: &str) -> Option<SourceKind> {
    let lower = path.to_lowercase();
    if lower.ends_with(".pdf") {
        Some(SourceKind::Pdf)
    } else if lower.ends_with(".docx") {
        Some(SourceKind::Docx)
    } else if lower.ends_with(".xlsx") {
        Some(SourceKind::Xlsx)
    } else if lower.ends_with(".pptx") {
        Some(SourceKind::Pptx)
    } else if lower.ends_with(".md") || lower.ends_with(".markdown") {
        Some(SourceKind::Markdown)
    } else if lower.ends_with(".txt") {
        Some(SourceKind::Text)
    } else if [".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff", ".bmp"]
        .iter()
        .any(|e| lower.ends_with(e))
    {
        Some(SourceKind::Image)
    } else if [".wav", ".mp3", ".m4a", ".flac", ".ogg"]
        .iter()
        .any(|e| lower.ends_with(e))
    {
        Some(SourceKind::Audio)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_common_extensions() {
        assert_eq!(detect_kind("a.pdf"), Some(SourceKind::Pdf));
        assert_eq!(detect_kind("A.PNG"), Some(SourceKind::Image));
        assert_eq!(detect_kind("foo.docx"), Some(SourceKind::Docx));
        assert_eq!(detect_kind("readme.md"), Some(SourceKind::Markdown));
        assert_eq!(detect_kind("unknown.bin"), None);
    }

    #[test]
    fn detects_audio_variants() {
        for ext in ["wav", "mp3", "m4a", "flac", "ogg"] {
            let p = format!("clip.{}", ext);
            assert_eq!(detect_kind(&p), Some(SourceKind::Audio), "ext: {}", ext);
            let upper = format!("clip.{}", ext.to_uppercase());
            assert_eq!(detect_kind(&upper), Some(SourceKind::Audio));
        }
    }

    #[test]
    fn detects_all_image_variants() {
        for ext in ["png", "jpg", "jpeg", "webp", "tif", "tiff", "bmp"] {
            let p = format!("img.{}", ext);
            assert_eq!(detect_kind(&p), Some(SourceKind::Image), "ext: {}", ext);
            let upper = format!("img.{}", ext.to_uppercase());
            assert_eq!(detect_kind(&upper), Some(SourceKind::Image));
        }
    }

    #[test]
    fn detects_xlsx_pptx_text_markdown_long_form() {
        assert_eq!(detect_kind("a.xlsx"), Some(SourceKind::Xlsx));
        assert_eq!(detect_kind("a.pptx"), Some(SourceKind::Pptx));
        assert_eq!(detect_kind("a.txt"), Some(SourceKind::Text));
        assert_eq!(detect_kind("a.markdown"), Some(SourceKind::Markdown));
    }

    #[test]
    fn detect_kind_handles_full_path_and_no_ext() {
        assert_eq!(detect_kind("/tmp/dir/sub/file.PDF"), Some(SourceKind::Pdf));
        assert_eq!(detect_kind("noext"), None);
        assert_eq!(detect_kind(""), None);
    }
}
