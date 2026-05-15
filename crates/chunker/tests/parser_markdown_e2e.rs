//! parser-markdown → chunker の e2e。Run 28 で「実 Markdown ingest を通すと
//! heading_path[-1] が section title (= title-as-caption フォールバックの素材)
//! になっている」と仮説を立てた。本テストでその不変条件を回帰として locking する。
//!
//! caption rerank の heading fallback (sdk: `apply_heading_index` 経由) は
//! `Chunk::heading_path` の最後の segment を caption と同じ重みで query 比較する。
//! ここが空 / `doc_id` 等の無関係な値だと caption rerank の効きが消えるため、
//! parser-markdown 経由の典型 markdown で last segment が見出し title を保持する
//! ことを保証する。

use ellisii_chunker::{chunk, ChunkConfig};
use ellisii_core::SourceKind;
use ellisii_parsers_core::ParsedDocument;
use uuid::Uuid;

#[test]
fn parser_markdown_to_chunker_preserves_heading_path() {
    // 典型的な Wikipedia ライク markdown
    let raw = "\
# データベース概論

## ACID

ACIDとはトランザクションの 4 つの性質、原子性・一貫性・独立性・永続性のこと。
信頼性のあるトランザクションシステムが備えるべき要件を表す。

## B木

B木はディスク上のデータ構造で、データベース管理システムの索引で広く使われる。
ブロック単位のランダムアクセスに適した木構造として知られる。
";

    let blocks = ellisii_parser_markdown::parse_str(raw);
    assert!(!blocks.is_empty(), "parser should yield at least one block");

    // すべての block が H1+H2 の 2 段 heading_path を持つことを確認 (parser 側の不変条件)
    for b in &blocks {
        assert_eq!(
            b.heading_path.len(),
            2,
            "expected H1 + H2 chain, got: {:?}",
            b.heading_path
        );
        assert_eq!(b.heading_path[0], "データベース概論");
    }

    let doc = ParsedDocument {
        kind: SourceKind::Markdown,
        blocks,
    };
    let chunks = chunk(&doc, Uuid::nil(), ChunkConfig::default());
    assert!(
        !chunks.is_empty(),
        "chunker should yield at least one chunk"
    );

    // 不変条件: chunk の heading_path[-1] は section title (H2 = ACID / B木)
    // ここが落ちると Run 28 で確認した「heading rerank フォールバックの素材」が
    // 失われ、caption rerank の効きが Markdown corpus で消える。
    for c in &chunks {
        let last = c
            .heading_path
            .last()
            .unwrap_or_else(|| panic!("missing heading_path: {c:?}"));
        assert!(
            last == "ACID" || last == "B木",
            "expected last segment = section title, got {last:?}"
        );
    }

    // 章 (H1) も heading_path のどこかに残っていること
    for c in &chunks {
        assert!(
            c.heading_path.iter().any(|h| h == "データベース概論"),
            "lost H1 chapter from heading_path: {:?}",
            c.heading_path
        );
    }

    // 各 H2 ごとに少なくとも 1 chunk
    let acid_chunks: Vec<_> = chunks
        .iter()
        .filter(|c| c.heading_path.last().map(|s| s.as_str()) == Some("ACID"))
        .collect();
    let btree_chunks: Vec<_> = chunks
        .iter()
        .filter(|c| c.heading_path.last().map(|s| s.as_str()) == Some("B木"))
        .collect();
    assert!(!acid_chunks.is_empty(), "no chunk for H2 ACID");
    assert!(!btree_chunks.is_empty(), "no chunk for H2 B木");
}

#[test]
fn parser_markdown_with_h3_uses_deepest_heading_as_last() {
    // H1 → H2 → H3 のとき heading_path の最後は H3 (= 最も具体的な section title)
    let raw = "# A\n## B\n### C\n本文です。\n本文 2。\n";
    let blocks = ellisii_parser_markdown::parse_str(raw);
    let doc = ParsedDocument {
        kind: SourceKind::Markdown,
        blocks,
    };
    let chunks = chunk(&doc, Uuid::nil(), ChunkConfig::default());
    assert!(!chunks.is_empty());
    for c in &chunks {
        assert_eq!(
            c.heading_path.last().map(|s| s.as_str()),
            Some("C"),
            "deepest heading should be last segment"
        );
    }
}

#[test]
fn parser_markdown_no_heading_yields_empty_heading_path() {
    // heading が無い markdown では heading_path が空のまま (caption rerank は noop)
    let raw = "ただの段落。\n\n別の段落。\n";
    let blocks = ellisii_parser_markdown::parse_str(raw);
    let doc = ParsedDocument {
        kind: SourceKind::Markdown,
        blocks,
    };
    let chunks = chunk(&doc, Uuid::nil(), ChunkConfig::default());
    assert!(!chunks.is_empty());
    for c in &chunks {
        assert!(
            c.heading_path.is_empty(),
            "no-heading markdown should leave heading_path empty, got {:?}",
            c.heading_path
        );
    }
}
