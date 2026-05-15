// Restructuring the Event match arms changes parser semantics — keep the
// nested `if` checks readable and suppress the style lint locally.
#![allow(clippy::collapsible_match, clippy::collapsible_if)]

use ellisii_core::{Error, Result};
use ellisii_parsers_core::ParsedBlock;
use pulldown_cmark::{Event, HeadingLevel, Parser, Tag, TagEnd};

pub fn parse(path: &str) -> Result<Vec<ParsedBlock>> {
    let raw = std::fs::read_to_string(path).map_err(Error::Io)?;
    Ok(parse_str(&raw))
}

pub fn parse_str(raw: &str) -> Vec<ParsedBlock> {
    let parser = Parser::new(raw);
    let mut blocks: Vec<ParsedBlock> = Vec::new();
    let mut heading_stack: Vec<(u32, String)> = Vec::new();
    let mut current = String::new();
    let mut in_heading: Option<HeadingLevel> = None;

    fn flush(blocks: &mut Vec<ParsedBlock>, stack: &[(u32, String)], buf: &mut String) {
        let trimmed = buf.trim();
        if !trimmed.is_empty() {
            blocks.push(ParsedBlock {
                text: trimmed.to_string(),
                heading_path: stack.iter().map(|(_, t)| t.clone()).collect(),
                page: None,
                bbox: None,
            });
        }
        buf.clear();
    }

    for ev in parser {
        match ev {
            Event::Start(Tag::Heading { level, .. }) => {
                flush(&mut blocks, &heading_stack, &mut current);
                in_heading = Some(level);
            }
            Event::End(TagEnd::Heading(level)) => {
                let depth = heading_level(level);
                let title = std::mem::take(&mut current).trim().to_string();
                heading_stack.retain(|(d, _)| *d < depth);
                heading_stack.push((depth, title));
                in_heading = None;
            }
            Event::End(TagEnd::Paragraph) | Event::End(TagEnd::Item) => {
                if in_heading.is_none() {
                    current.push('\n');
                }
            }
            Event::Text(t) | Event::Code(t) => current.push_str(&t),
            Event::SoftBreak | Event::HardBreak => current.push(' '),
            Event::End(TagEnd::List(_)) => flush(&mut blocks, &heading_stack, &mut current),
            _ => {}
        }
    }
    flush(&mut blocks, &heading_stack, &mut current);
    blocks
}

fn heading_level(level: HeadingLevel) -> u32 {
    match level {
        HeadingLevel::H1 => 1,
        HeadingLevel::H2 => 2,
        HeadingLevel::H3 => 3,
        HeadingLevel::H4 => 4,
        HeadingLevel::H5 => 5,
        HeadingLevel::H6 => 6,
    }
}

#[cfg(test)]
mod tests {
    use super::parse_str;

    #[test]
    fn captures_heading_path() {
        let blocks = parse_str("# A\n## B\n本文1\n## C\n本文2");
        assert_eq!(blocks.len(), 2);
        assert_eq!(
            blocks[0].heading_path,
            vec!["A".to_string(), "B".to_string()]
        );
        assert_eq!(
            blocks[1].heading_path,
            vec!["A".to_string(), "C".to_string()]
        );
    }

    #[test]
    fn h2_after_h1_does_not_inherit_sibling() {
        let blocks = parse_str("# A\n## B1\nx\n## B2\ny");
        assert_eq!(
            blocks[0].heading_path,
            vec!["A".to_string(), "B1".to_string()]
        );
        assert_eq!(
            blocks[1].heading_path,
            vec!["A".to_string(), "B2".to_string()]
        );
    }

    #[test]
    fn empty_or_whitespace_input_yields_no_blocks() {
        assert!(parse_str("").is_empty());
        assert!(parse_str("   \n\n  ").is_empty());
    }

    #[test]
    fn going_up_levels_pops_deeper_headings() {
        let blocks = parse_str("# A\n## B\ntext\n# C\nbody");
        assert_eq!(blocks.len(), 2);
        assert_eq!(
            blocks[0].heading_path,
            vec!["A".to_string(), "B".to_string()]
        );
        assert_eq!(blocks[1].heading_path, vec!["C".to_string()]);
    }

    #[test]
    fn list_items_collapse_into_block() {
        let blocks = parse_str("# H\n- a\n- b\n- c");
        assert!(!blocks.is_empty());
        assert_eq!(blocks[0].heading_path, vec!["H".to_string()]);
        assert!(blocks[0].text.contains('a'));
        assert!(blocks[0].text.contains('b'));
        assert!(blocks[0].text.contains('c'));
    }

    #[test]
    fn inline_code_is_captured_as_text() {
        let blocks = parse_str("# H\n本文 `code` 続き");
        assert_eq!(blocks.len(), 1);
        assert!(blocks[0].text.contains("code"));
    }

    #[test]
    fn no_heading_gives_empty_heading_path() {
        // 連続パラグラフは見出しが切れない限り単一ブロックに連結される。
        let blocks = parse_str("just a paragraph\n\nsecond para");
        assert_eq!(blocks.len(), 1);
        assert!(blocks[0].heading_path.is_empty());
        assert!(blocks[0].text.contains("just a paragraph"));
        assert!(blocks[0].text.contains("second para"));
    }

    #[test]
    fn deep_heading_path_h1_to_h3() {
        let blocks = parse_str("# A\n## B\n### C\nbody");
        assert_eq!(blocks.len(), 1);
        assert_eq!(
            blocks[0].heading_path,
            vec!["A".to_string(), "B".to_string(), "C".to_string()]
        );
    }
}
