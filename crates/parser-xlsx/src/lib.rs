use calamine::{open_workbook_auto, Data, Reader};
use ellisii_core::{Error, Result};
use ellisii_parsers_core::ParsedBlock;

const ROWS_PER_BLOCK: usize = 20;

pub fn parse(path: &str) -> Result<Vec<ParsedBlock>> {
    let mut wb = open_workbook_auto(path).map_err(|e| Error::Parse(format!("xlsx: {e}")))?;
    let sheets: Vec<String> = wb.sheet_names().to_vec();
    let mut out = Vec::new();
    for sheet in sheets {
        let Ok(range) = wb.worksheet_range(&sheet) else {
            continue;
        };
        let rows: Vec<Vec<String>> = range
            .rows()
            .map(|r| r.iter().map(cell_to_string).collect())
            .collect();
        out.extend(rows_to_blocks(&sheet, &rows));
    }
    Ok(out)
}

/// シートごとに 1 entry、全文を string にまとめて返す。
///
/// 戻り値の Vec の順序は `Workbook::sheet_names()` の順 (= ファイルに並んでいる順)。
/// `worksheet_range` が失敗したシートは空文字 entry を入れることで「i 番目 = シート i」の
/// 対応を呼び出し側で保てるようにしてある — 抜けると下流の page index 付けが崩れる。
///
/// 1 シートのフォーマットは
///   - 行は `\n` で結合
///   - セルは " | " で結合
///   - 完全に空の行は skip
///
/// で、`parse()` が出す `ParsedBlock.text` と同じ join 規約に揃えている。
pub fn sheet_texts(path: &str) -> Result<Vec<String>> {
    let mut wb = open_workbook_auto(path).map_err(|e| Error::Parse(format!("xlsx: {e}")))?;
    let sheets: Vec<String> = wb.sheet_names().to_vec();
    let mut out = Vec::with_capacity(sheets.len());
    for sheet in &sheets {
        let Ok(range) = wb.worksheet_range(sheet) else {
            out.push(String::new());
            continue;
        };
        let mut text = String::new();
        for row in range.rows() {
            let cells: Vec<String> = row.iter().map(cell_to_string).collect();
            if cells.iter().all(|c| c.trim().is_empty()) {
                continue;
            }
            if !text.is_empty() {
                text.push('\n');
            }
            text.push_str(&cells.join(" | "));
        }
        out.push(text);
    }
    Ok(out)
}

pub fn rows_to_blocks(sheet: &str, rows: &[Vec<String>]) -> Vec<ParsedBlock> {
    if rows.is_empty() {
        return vec![];
    }
    let header = rows[0].clone();
    let body = &rows[1..];
    if body.is_empty() {
        return vec![ParsedBlock {
            text: header.join(" | "),
            heading_path: vec![sheet.to_string()],
            page: None,
            bbox: None,
        }];
    }
    let mut out = Vec::new();
    for (i, chunk) in body.chunks(ROWS_PER_BLOCK).enumerate() {
        let mut text = header.join(" | ");
        text.push('\n');
        for row in chunk {
            if row.iter().all(|c| c.trim().is_empty()) {
                continue;
            }
            text.push_str(&row.join(" | "));
            text.push('\n');
        }
        out.push(ParsedBlock {
            text: text.trim_end().to_string(),
            heading_path: vec![
                sheet.to_string(),
                format!("rows {}", i * ROWS_PER_BLOCK + 2),
            ],
            page: None,
            bbox: None,
        });
    }
    out
}

fn cell_to_string(c: &Data) -> String {
    match c {
        Data::Empty => String::new(),
        Data::String(s) => s.clone(),
        Data::Float(f) => format_float(*f),
        Data::Int(i) => i.to_string(),
        Data::Bool(b) => b.to_string(),
        Data::Error(e) => format!("#ERR({e:?})"),
        Data::DateTime(d) => d.to_string(),
        Data::DateTimeIso(s) | Data::DurationIso(s) => s.clone(),
    }
}

fn format_float(f: f64) -> String {
    if f.fract() == 0.0 && f.abs() < 1e15 {
        format!("{}", f as i64)
    } else {
        f.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::rows_to_blocks;

    #[test]
    fn header_prepended_to_each_chunk() {
        let header = vec!["A".to_string(), "B".to_string()];
        let mut rows = vec![header.clone()];
        for i in 0..45 {
            rows.push(vec![format!("v{i}"), format!("w{i}")]);
        }
        let blocks = rows_to_blocks("Sheet1", &rows);
        assert_eq!(blocks.len(), 3); // 20 + 20 + 5
        for b in &blocks {
            assert!(b.text.starts_with("A | B"));
            assert_eq!(b.heading_path[0], "Sheet1");
        }
    }

    #[test]
    fn header_only_returns_single_block() {
        let rows = vec![vec!["a".into(), "b".into()]];
        let blocks = rows_to_blocks("S", &rows);
        assert_eq!(blocks.len(), 1);
    }

    #[test]
    fn empty_rows_returns_no_blocks() {
        let rows: Vec<Vec<String>> = vec![];
        assert!(rows_to_blocks("S", &rows).is_empty());
    }

    #[test]
    fn entirely_empty_body_rows_are_skipped() {
        let rows = vec![
            vec!["A".into(), "B".into()],
            vec!["".into(), "  ".into()],
            vec!["x".into(), "y".into()],
            vec!["".into(), "".into()],
        ];
        let blocks = rows_to_blocks("S", &rows);
        assert_eq!(blocks.len(), 1);
        let body: Vec<&str> = blocks[0].text.lines().collect();
        assert_eq!(body.len(), 2);
        assert!(body[0].contains("A | B"));
        assert!(body[1].contains("x | y"));
    }

    #[test]
    fn heading_path_row_offset_starts_at_2() {
        let mut rows = vec![vec!["H".into()]];
        for i in 0..25 {
            rows.push(vec![format!("v{i}")]);
        }
        let blocks = rows_to_blocks("Sheet2", &rows);
        assert_eq!(blocks.len(), 2);
        assert_eq!(blocks[0].heading_path[0], "Sheet2");
        assert_eq!(blocks[0].heading_path[1], "rows 2");
        assert_eq!(blocks[1].heading_path[1], "rows 22");
    }
}
