use ellisii_core::{Error, Result};
use ellisii_parsers_core::ParsedBlock;
use std::io::Read;

pub fn parse(path: &str) -> Result<Vec<ParsedBlock>> {
    let f = std::fs::File::open(path).map_err(Error::Io)?;
    let mut zip = zip::ZipArchive::new(f).map_err(|e| Error::Parse(format!("zip: {e}")))?;
    let mut xml = String::new();
    zip.by_name("word/document.xml")
        .map_err(|e| Error::Parse(format!("docx: missing document.xml: {e}")))?
        .read_to_string(&mut xml)
        .map_err(Error::Io)?;
    Ok(extract_blocks(&xml))
}

pub fn extract_blocks(xml: &str) -> Vec<ParsedBlock> {
    use quick_xml::events::Event;
    use quick_xml::reader::Reader;

    let mut reader = Reader::from_str(xml);
    reader.config_mut().trim_text(true);
    let mut buf = Vec::new();
    let mut in_t = false;
    let mut in_p = false;
    let mut current = String::new();
    let mut blocks = Vec::new();

    loop {
        match reader.read_event_into(&mut buf) {
            Ok(Event::Start(e)) => match e.name().local_name().as_ref() {
                b"t" => in_t = true,
                b"p" => in_p = true,
                _ => {}
            },
            Ok(Event::End(e)) => match e.name().local_name().as_ref() {
                b"t" => in_t = false,
                b"p" => {
                    if in_p && !current.trim().is_empty() {
                        blocks.push(ParsedBlock {
                            text: std::mem::take(&mut current).trim().to_string(),
                            heading_path: vec![],
                            page: None,
                            bbox: None,
                        });
                    }
                    current.clear();
                    in_p = false;
                }
                _ => {}
            },
            Ok(Event::Text(t)) if in_t => match t.decode() {
                Ok(s) => current.push_str(&s),
                Err(_) => current.push_str(std::str::from_utf8(t.as_ref()).unwrap_or("")),
            },
            Ok(Event::GeneralRef(r)) if in_t => {
                let name = std::str::from_utf8(r.as_ref()).unwrap_or("");
                if let Some(ch) = decode_entity(name) {
                    current.push(ch);
                }
            }
            Ok(Event::Eof) | Err(_) => break,
            _ => {}
        }
        buf.clear();
    }
    blocks
}

fn decode_entity(name: &str) -> Option<char> {
    match name {
        "amp" => Some('&'),
        "lt" => Some('<'),
        "gt" => Some('>'),
        "quot" => Some('"'),
        "apos" => Some('\''),
        n if n.starts_with("#x") || n.starts_with("#X") => u32::from_str_radix(&n[2..], 16)
            .ok()
            .and_then(char::from_u32),
        n if n.starts_with('#') => n[1..].parse::<u32>().ok().and_then(char::from_u32),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::extract_blocks;

    #[test]
    fn extracts_single_paragraph() {
        let xml = r#"<?xml version="1.0"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:body>
    <w:p><w:r><w:t>こんにちは</w:t></w:r></w:p>
  </w:body>
</w:document>"#;
        let blocks = extract_blocks(xml);
        assert_eq!(blocks.len(), 1);
        assert_eq!(blocks[0].text, "こんにちは");
    }

    #[test]
    fn skips_empty_paragraphs() {
        let xml = r#"<w:document xmlns:w="x">
            <w:p></w:p>
            <w:p><w:r><w:t>本文</w:t></w:r></w:p>
        </w:document>"#;
        let blocks = extract_blocks(xml);
        assert_eq!(blocks.len(), 1);
    }

    #[test]
    fn unescapes_entities() {
        let xml =
            r#"<w:document xmlns:w="x"><w:p><w:r><w:t>A&amp;B</w:t></w:r></w:p></w:document>"#;
        let blocks = extract_blocks(xml);
        assert_eq!(blocks[0].text, "A&B");
    }

    #[test]
    fn multiple_paragraphs_become_separate_blocks() {
        let xml = r#"<w:document xmlns:w="x">
            <w:p><w:r><w:t>p1</w:t></w:r></w:p>
            <w:p><w:r><w:t>p2</w:t></w:r></w:p>
            <w:p><w:r><w:t>p3</w:t></w:r></w:p>
        </w:document>"#;
        let blocks = extract_blocks(xml);
        assert_eq!(blocks.len(), 3);
        assert_eq!(blocks[0].text, "p1");
        assert_eq!(blocks[2].text, "p3");
    }

    #[test]
    fn concatenates_runs_within_one_paragraph() {
        // trim_text(true) is enabled, so whitespace-only <w:t> nodes are dropped.
        // 単語境界の維持は呼び出し側の責務。
        let xml = r#"<w:document xmlns:w="x">
            <w:p>
              <w:r><w:t>hello,</w:t></w:r>
              <w:r><w:t>world</w:t></w:r>
            </w:p>
        </w:document>"#;
        let blocks = extract_blocks(xml);
        assert_eq!(blocks.len(), 1);
        assert_eq!(blocks[0].text, "hello,world");
    }

    #[test]
    fn numeric_entity_decoding() {
        let xml = r#"<w:document xmlns:w="x">
            <w:p><w:r><w:t>A&#65;B</w:t></w:r></w:p>
            <w:p><w:r><w:t>X&#x41;Y</w:t></w:r></w:p>
        </w:document>"#;
        let blocks = extract_blocks(xml);
        assert_eq!(blocks[0].text, "AAB");
        assert_eq!(blocks[1].text, "XAY");
    }

    #[test]
    fn block_metadata_is_clean() {
        let xml = r#"<w:document xmlns:w="x"><w:p><w:r><w:t>x</w:t></w:r></w:p></w:document>"#;
        let blocks = extract_blocks(xml);
        assert!(blocks[0].heading_path.is_empty());
        assert!(blocks[0].page.is_none());
        assert!(blocks[0].bbox.is_none());
    }
}
