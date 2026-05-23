// Restructuring the quick-xml event match arms changes the streaming
// parser's state machine — keep the nested `if` checks readable and
// suppress the style lint locally.
#![allow(clippy::collapsible_match, clippy::collapsible_if)]

use ellisii_core::{Error, Result};
use ellisii_parsers_core::ParsedBlock;
use std::io::Read;

pub fn parse(path: &str) -> Result<Vec<ParsedBlock>> {
    let f = std::fs::File::open(path).map_err(Error::Io)?;
    let mut zip = zip::ZipArchive::new(f).map_err(|e| Error::Parse(format!("zip: {e}")))?;
    let mut slide_files: Vec<String> = (0..zip.len())
        .filter_map(|i| zip.by_index(i).ok().map(|f| f.name().to_string()))
        .filter(|n| n.starts_with("ppt/slides/slide") && n.ends_with(".xml"))
        .collect();
    slide_files.sort_by_key(|n| extract_slide_number(n).unwrap_or(u32::MAX));

    let mut out = Vec::new();
    for (idx, name) in slide_files.iter().enumerate() {
        let slide_num_in_archive = extract_slide_number(name);
        let display_num = idx as u32 + 1;
        let mut xml = String::new();
        zip.by_name(name)
            .map_err(|e| Error::Parse(format!("pptx slide read: {e}")))?
            .read_to_string(&mut xml)
            .map_err(Error::Io)?;
        let text = extract_a_t(&xml);
        if !text.trim().is_empty() {
            out.push(ParsedBlock {
                text,
                heading_path: vec![format!("Slide {display_num}")],
                page: Some(display_num),
                bbox: None,
            });
        }
        // Speaker notes: `ppt/notesSlides/notesSlideN.xml` の N が `slideN.xml` の N と
        // 1-origin で対応する。プレゼン発表者用の補足説明で、本文には出ていない詳細
        // 情報を含むことが多いので RAG では別 block として残す方が hit 率が上がる。
        // 対応 notesSlide が存在しないケース (notes 未作成) はサイレントに skip。
        if let Some(num) = slide_num_in_archive {
            let notes_name = format!("ppt/notesSlides/notesSlide{num}.xml");
            if let Ok(mut nf) = zip.by_name(&notes_name) {
                let mut nxml = String::new();
                if nf.read_to_string(&mut nxml).is_ok() {
                    let notes = extract_notes_body(&nxml);
                    if !notes.trim().is_empty() {
                        out.push(ParsedBlock {
                            text: notes,
                            heading_path: vec![format!("Slide {display_num}"), "Notes".into()],
                            page: Some(display_num),
                            bbox: None,
                        });
                    }
                }
            }
        }
    }
    Ok(out)
}

/// スライド (= page) ごとに 1 entry、全文を string にまとめて返す。
///
/// 戻り値の Vec の index は 0-origin の slide index に対応 (= `out[0]` が 1 枚目)。
/// 1 slide の string は
///   - 本文 (`<a:t>` の連結)
///   - speaker notes (あれば本文の後に空行+`Notes:\n` 区切りで追記)
///
/// を `\n` で結合した形。notes が無い slide はそのまま本文だけ。
/// 本文が空 (画像 only など) でも entry 自体は push し、ファイル内の slide 順序と
/// Vec index がずれないようにしてある。
pub fn slide_texts(path: &str) -> Result<Vec<String>> {
    let f = std::fs::File::open(path).map_err(Error::Io)?;
    let mut zip = zip::ZipArchive::new(f).map_err(|e| Error::Parse(format!("zip: {e}")))?;
    let mut slide_files: Vec<String> = (0..zip.len())
        .filter_map(|i| zip.by_index(i).ok().map(|f| f.name().to_string()))
        .filter(|n| n.starts_with("ppt/slides/slide") && n.ends_with(".xml"))
        .collect();
    slide_files.sort_by_key(|n| extract_slide_number(n).unwrap_or(u32::MAX));

    let mut out = Vec::with_capacity(slide_files.len());
    for name in &slide_files {
        let slide_num_in_archive = extract_slide_number(name);
        let mut xml = String::new();
        zip.by_name(name)
            .map_err(|e| Error::Parse(format!("pptx slide read: {e}")))?
            .read_to_string(&mut xml)
            .map_err(Error::Io)?;
        let body = extract_a_t(&xml);

        let notes = slide_num_in_archive
            .and_then(|num| {
                let notes_name = format!("ppt/notesSlides/notesSlide{num}.xml");
                let mut nf = zip.by_name(&notes_name).ok()?;
                let mut nxml = String::new();
                nf.read_to_string(&mut nxml).ok()?;
                let n = extract_notes_body(&nxml);
                if n.trim().is_empty() { None } else { Some(n) }
            });

        let combined = match notes {
            Some(n) if body.trim().is_empty() => format!("Notes:\n{n}"),
            Some(n) => format!("{body}\n\nNotes:\n{n}"),
            None => body,
        };
        out.push(combined);
    }
    Ok(out)
}

/// `ppt/slides/slide12.xml` → `Some(12)`、形式が違えば `None`。
fn extract_slide_number(name: &str) -> Option<u32> {
    let stem = name
        .trim_start_matches("ppt/slides/slide")
        .trim_end_matches(".xml");
    stem.parse::<u32>().ok()
}

/// notesSlide XML から speaker notes 本文だけを抽出する。
///
/// 通常 notesSlide には:
/// - スライド画像のプレースホルダ
/// - スライド番号 (`<a:fld type="slidenum">` で展開される `<a:t>1</a:t>` 等)
/// - notes 本文 (txBody 配下の `<a:t>`)
///
/// が並んでいる。`<a:t>` を全部拾うとスライド番号 1, 2, 3 ... も notes として
/// 取り込まれてしまうので、ここでは `placeholder type="sldNum"` を持つ
/// `<p:sp>` 配下の text を **skip** する単純な state machine を組む。
pub fn extract_notes_body(xml: &str) -> String {
    use quick_xml::events::Event;
    use quick_xml::reader::Reader;

    let mut reader = Reader::from_str(xml);
    reader.config_mut().trim_text(true);
    let mut buf = Vec::new();
    let mut in_t = false;
    // shape stack: 各 `<p:sp>` 開始でスライド番号 placeholder かどうかを push
    let mut sp_stack_is_slidenum: Vec<bool> = Vec::new();
    // 直近の `<p:nvSpPr>` 配下で見た placeholder type 属性
    let mut current_sp_is_slidenum: Option<bool> = None;
    let mut out = String::new();

    /// `<p:ph type="X"/>` の type 属性値を見て skip 対象か判定するヘルパ。
    fn ph_is_skip_target(e: &quick_xml::events::BytesStart<'_>) -> bool {
        for a in e.attributes().flatten() {
            if a.key.local_name().as_ref() == b"type" {
                let v = std::str::from_utf8(a.value.as_ref()).unwrap_or("");
                if v == "sldNum" || v == "dt" || v == "ftr" {
                    return true;
                }
            }
        }
        false
    }

    loop {
        match reader.read_event_into(&mut buf) {
            Ok(Event::Start(e)) => {
                let local = e.name().local_name();
                let local_str = std::str::from_utf8(local.as_ref()).unwrap_or("");
                match local_str {
                    "sp" => {
                        // 新しい shape 開始 — placeholder 検査をリセット
                        current_sp_is_slidenum = Some(false);
                    }
                    "ph" => {
                        if ph_is_skip_target(&e) {
                            current_sp_is_slidenum = Some(true);
                        }
                    }
                    "txBody" => {
                        // shape の本文に入る — その shape が slidenum なら以後の <a:t> を skip。
                        sp_stack_is_slidenum.push(current_sp_is_slidenum.unwrap_or(false));
                    }
                    "t" => {
                        in_t = true;
                    }
                    _ => {}
                }
            }
            Ok(Event::End(e)) => {
                let local = e.name().local_name();
                match local.as_ref() {
                    b"sp" => {
                        current_sp_is_slidenum = None;
                    }
                    b"txBody" => {
                        sp_stack_is_slidenum.pop();
                    }
                    b"t" => in_t = false,
                    b"p" => out.push('\n'),
                    _ => {}
                }
            }
            Ok(Event::Empty(e)) => {
                // `<p:ph type="sldNum"/>` のような self-closing 要素も拾う。
                if e.name().local_name().as_ref() == b"ph" && ph_is_skip_target(&e) {
                    current_sp_is_slidenum = Some(true);
                }
            }
            Ok(Event::Text(t)) if in_t => {
                if sp_stack_is_slidenum.last().copied().unwrap_or(false) {
                    // slidenum / footer / datetime placeholder — skip
                } else {
                    match t.decode() {
                        Ok(s) => out.push_str(&s),
                        Err(_) => out.push_str(std::str::from_utf8(t.as_ref()).unwrap_or("")),
                    }
                }
            }
            Ok(Event::GeneralRef(r)) if in_t => {
                if !sp_stack_is_slidenum.last().copied().unwrap_or(false) {
                    let name = std::str::from_utf8(r.as_ref()).unwrap_or("");
                    if let Some(ch) = decode_entity(name) {
                        out.push(ch);
                    }
                }
            }
            Ok(Event::Eof) | Err(_) => break,
            _ => {}
        }
        buf.clear();
    }
    out.trim().to_string()
}

pub fn extract_a_t(xml: &str) -> String {
    use quick_xml::events::Event;
    use quick_xml::reader::Reader;

    let mut reader = Reader::from_str(xml);
    reader.config_mut().trim_text(true);
    let mut buf = Vec::new();
    let mut in_t = false;
    let mut out = String::new();

    loop {
        match reader.read_event_into(&mut buf) {
            Ok(Event::Start(e)) => {
                if e.name().local_name().as_ref() == b"t" {
                    in_t = true;
                }
            }
            Ok(Event::End(e)) => match e.name().local_name().as_ref() {
                b"t" => in_t = false,
                b"p" => out.push('\n'),
                _ => {}
            },
            Ok(Event::Text(t)) if in_t => match t.decode() {
                Ok(s) => out.push_str(&s),
                Err(_) => out.push_str(std::str::from_utf8(t.as_ref()).unwrap_or("")),
            },
            Ok(Event::GeneralRef(r)) if in_t => {
                let name = std::str::from_utf8(r.as_ref()).unwrap_or("");
                if let Some(ch) = decode_entity(name) {
                    out.push(ch);
                }
            }
            Ok(Event::Eof) | Err(_) => break,
            _ => {}
        }
        buf.clear();
    }
    out.trim().to_string()
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
    use super::extract_a_t;

    #[test]
    fn collects_a_t_in_order() {
        let xml = r#"<p:sld xmlns:a="x" xmlns:p="y">
            <p:txBody><a:p><a:r><a:t>title</a:t></a:r></a:p>
            <a:p><a:r><a:t>body1</a:t></a:r></a:p></p:txBody>
        </p:sld>"#;
        let out = extract_a_t(xml);
        assert!(out.contains("title"));
        assert!(out.contains("body1"));
    }

    #[test]
    fn p_tag_inserts_newline_separator() {
        let xml = r#"<p:sld xmlns:a="x" xmlns:p="y">
            <a:p><a:r><a:t>one</a:t></a:r></a:p>
            <a:p><a:r><a:t>two</a:t></a:r></a:p>
        </p:sld>"#;
        let out = extract_a_t(xml);
        let lines: Vec<&str> = out.split('\n').filter(|l| !l.is_empty()).collect();
        assert_eq!(lines, vec!["one", "two"]);
    }

    #[test]
    fn empty_xml_yields_empty_string() {
        assert_eq!(extract_a_t(""), "");
        assert_eq!(extract_a_t("<p:sld/>"), "");
    }

    #[test]
    fn entity_decoding_amp_and_numeric() {
        let xml = r#"<p:sld><a:p><a:r><a:t>A&amp;B&#65;</a:t></a:r></a:p></p:sld>"#;
        let out = extract_a_t(xml);
        assert!(out.contains("A&BA"));
    }

    #[test]
    fn ignores_text_outside_of_t_tag() {
        let xml = r#"<p:sld><a:p>OUTSIDE<a:r><a:t>IN</a:t></a:r>OUT2</a:p></p:sld>"#;
        let out = extract_a_t(xml);
        assert_eq!(out.trim(), "IN");
    }

    // ─── speaker notes (extract_notes_body) ──────────────────────────────────

    use super::extract_notes_body;

    #[test]
    fn notes_body_skips_slide_number_placeholder() {
        // 実 PowerPoint が吐く notesSlide の最小サンプル。
        // shape 1: sldNum placeholder (中身は "12") → skip 対象
        // shape 2: 通常 txBody に notes 本文 — これだけ拾われるべき
        let xml = r#"<p:notes xmlns:p="x" xmlns:a="y">
            <p:cSld><p:spTree>
                <p:sp>
                    <p:nvSpPr><p:nvPr><p:ph type="sldNum"/></p:nvPr></p:nvSpPr>
                    <p:txBody><a:p><a:r><a:t>12</a:t></a:r></a:p></p:txBody>
                </p:sp>
                <p:sp>
                    <p:nvSpPr><p:nvPr><p:ph type="body" idx="1"/></p:nvPr></p:nvSpPr>
                    <p:txBody>
                        <a:p><a:r><a:t>これは発表者ノートです。</a:t></a:r></a:p>
                        <a:p><a:r><a:t>追加の説明をここに書きます。</a:t></a:r></a:p>
                    </p:txBody>
                </p:sp>
            </p:spTree></p:cSld>
        </p:notes>"#;
        let out = extract_notes_body(xml);
        assert!(out.contains("これは発表者ノート"), "got: {out}");
        assert!(out.contains("追加の説明"), "got: {out}");
        assert!(!out.contains("12"), "slide number must be skipped: {out}");
    }

    #[test]
    fn notes_body_returns_empty_for_no_notes() {
        // notes 本文が無い (sldNum しかない) ケース → 空文字
        let xml = r#"<p:notes xmlns:p="x" xmlns:a="y">
            <p:cSld><p:spTree>
                <p:sp>
                    <p:nvSpPr><p:nvPr><p:ph type="sldNum"/></p:nvPr></p:nvSpPr>
                    <p:txBody><a:p><a:r><a:t>5</a:t></a:r></a:p></p:txBody>
                </p:sp>
            </p:spTree></p:cSld>
        </p:notes>"#;
        let out = extract_notes_body(xml);
        assert_eq!(out, "");
    }

    #[test]
    fn notes_body_skips_footer_and_datetime_placeholders() {
        // 一部のテンプレートは footer / datetime 用 placeholder にも文字列を
        // 入れてくる。これらも RAG signal として有用ではないので skip する。
        let xml = r#"<p:notes xmlns:p="x" xmlns:a="y">
            <p:cSld><p:spTree>
                <p:sp>
                    <p:nvSpPr><p:nvPr><p:ph type="ftr"/></p:nvPr></p:nvSpPr>
                    <p:txBody><a:p><a:r><a:t>Confidential</a:t></a:r></a:p></p:txBody>
                </p:sp>
                <p:sp>
                    <p:nvSpPr><p:nvPr><p:ph type="dt"/></p:nvPr></p:nvSpPr>
                    <p:txBody><a:p><a:r><a:t>2026-05-10</a:t></a:r></a:p></p:txBody>
                </p:sp>
                <p:sp>
                    <p:nvSpPr><p:nvPr><p:ph type="body"/></p:nvPr></p:nvSpPr>
                    <p:txBody><a:p><a:r><a:t>本物のノート</a:t></a:r></a:p></p:txBody>
                </p:sp>
            </p:spTree></p:cSld>
        </p:notes>"#;
        let out = extract_notes_body(xml);
        assert!(out.contains("本物のノート"));
        assert!(!out.contains("Confidential"));
        assert!(!out.contains("2026-05-10"));
    }

    #[test]
    fn notes_body_decodes_entities() {
        let xml = r#"<p:notes xmlns:p="x" xmlns:a="y">
            <p:cSld><p:spTree><p:sp>
                <p:nvSpPr><p:nvPr><p:ph type="body"/></p:nvPr></p:nvSpPr>
                <p:txBody><a:p><a:r><a:t>A&amp;B&#65;</a:t></a:r></a:p></p:txBody>
            </p:sp></p:spTree></p:cSld>
        </p:notes>"#;
        let out = extract_notes_body(xml);
        assert!(out.contains("A&BA"), "got: {out}");
    }
}
