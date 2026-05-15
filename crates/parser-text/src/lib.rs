use ellisii_core::{Error, Result};
use ellisii_parsers_core::ParsedBlock;

pub fn parse(path: &str) -> Result<Vec<ParsedBlock>> {
    let bytes = std::fs::read(path).map_err(Error::Io)?;
    Ok(parse_bytes(&bytes))
}

pub fn parse_bytes(bytes: &[u8]) -> Vec<ParsedBlock> {
    let (text, _, had_errors) = encoding_rs::UTF_8.decode(bytes);
    let text = if had_errors {
        let (s, _, _) = encoding_rs::SHIFT_JIS.decode(bytes);
        s.into_owned()
    } else {
        text.into_owned()
    };
    text.split("\n\n")
        .filter(|s| !s.trim().is_empty())
        .map(|s| ParsedBlock {
            text: s.trim().to_string(),
            heading_path: vec![],
            page: None,
            bbox: None,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::parse_bytes;

    #[test]
    fn splits_on_blank_lines() {
        let blocks = parse_bytes(b"foo\nbar\n\nbaz\n\n\nqux");
        assert_eq!(blocks.len(), 3);
        assert_eq!(blocks[0].text, "foo\nbar");
        assert_eq!(blocks[1].text, "baz");
        assert_eq!(blocks[2].text, "qux");
    }

    #[test]
    fn falls_back_to_shift_jis() {
        let sjis = [0x82, 0xA0, 0x82, 0xA2, 0x82, 0xA4];
        let blocks = parse_bytes(&sjis);
        assert_eq!(blocks[0].text, "あいう");
    }

    #[test]
    fn empty_or_whitespace_only_yields_no_blocks() {
        assert!(parse_bytes(b"").is_empty());
        assert!(parse_bytes(b"   \n\n   \n  ").is_empty());
    }

    #[test]
    fn trims_per_block_whitespace() {
        let blocks = parse_bytes(b"   foo   \n\n   bar\n");
        assert_eq!(blocks.len(), 2);
        assert_eq!(blocks[0].text, "foo");
        assert_eq!(blocks[1].text, "bar");
    }

    #[test]
    fn block_metadata_is_clean() {
        let blocks = parse_bytes(b"x");
        assert_eq!(blocks.len(), 1);
        assert!(blocks[0].heading_path.is_empty());
        assert!(blocks[0].page.is_none());
        assert!(blocks[0].bbox.is_none());
    }

    #[test]
    fn utf8_input_is_not_misdetected_as_sjis() {
        let blocks = parse_bytes("日本語テキスト\n\nABC".as_bytes());
        assert_eq!(blocks.len(), 2);
        assert_eq!(blocks[0].text, "日本語テキスト");
        assert_eq!(blocks[1].text, "ABC");
    }

    #[test]
    fn parse_reads_file_from_disk() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("a.txt");
        std::fs::write(&path, "alpha\n\nbeta").unwrap();
        let blocks = super::parse(path.to_str().unwrap()).unwrap();
        assert_eq!(blocks.len(), 2);
        assert_eq!(blocks[0].text, "alpha");
        assert_eq!(blocks[1].text, "beta");
    }
}
