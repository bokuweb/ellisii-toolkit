pub mod caption;

use serde::{Deserialize, Serialize};
use thiserror::Error;
use uuid::Uuid;

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug, Error)]
pub enum Error {
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("parse: {0}")]
    Parse(String),
    #[error("ocr: {0}")]
    Ocr(String),
    #[error("embed: {0}")]
    Embed(String),
    #[error("store: {0}")]
    Store(String),
    #[error("llm: {0}")]
    Llm(String),
    #[error(transparent)]
    Other(#[from] anyhow::Error),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SourceKind {
    Pdf,
    Image,
    Docx,
    Xlsx,
    Pptx,
    Markdown,
    Text,
    /// Audio file (wav / mp3 / m4a / flac / ogg). Speech-to-text is required
    /// before chunking; see `crates/parser-audio` (Meeting Recorder Phase 2).
    Audio,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Source {
    pub id: Uuid,
    pub notebook_id: Uuid,
    pub path: String,
    pub kind: SourceKind,
    pub title: String,
    pub status: SourceStatus,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SourceStatus {
    Pending,
    Parsing,
    Embedding,
    Ready,
    Failed,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Chunk {
    pub id: Uuid,
    pub source_id: Uuid,
    pub ord: u32,
    pub text: String,
    pub heading_path: Vec<String>,
    pub page: Option<u32>,
    pub bbox: Option<[f32; 4]>,
    pub summary: Option<String>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "lowercase")]
pub enum HitSource {
    /// ベクトル類似度のみでヒット
    #[default]
    Vector,
    /// FTS / BM25 キーワードマッチでヒット
    Keyword,
    /// 両方でヒット (RRF 統合後)
    Hybrid,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchHit {
    pub chunk: Chunk,
    pub score: f32,
    #[serde(default)]
    pub source: HitSource,
}

/// retrieve 段で top_k に並べる前に弾きたい「ノイズ chunk」を判定する。
///
/// chunker 側の ingest フィルタは「内容語 4 文字未満」しか弾かないので、
/// "もしれません。" "return:" のような短いがロジック上は内容語を含む断片や、
/// 目次行 ("15.6 リストの長さの制限……11"), 章タイトル断片
/// ("クエリのアンチパターン") が DB に残り、ハイブリッド検索の上位を
/// 占拠してしまう。これらは LLM コンテキストに入っても解説の役に立たないので、
/// retrieve 結果から除外する。
///
/// 真と判定するパターン:
/// 1. 内容語 (アルファベット / かな / 漢字) が 25 文字未満
///    - 章タイトル断片 ("アプリケーション開発のアンチパターン" = 18 chars) や
///      節見出しだけの chunk もここで弾く。
/// 2. 「leader 文字 (`…` `・` `─`) が全体の 30% 以上」かつ全体 60 文字未満
/// 3. 行頭が「数字.数字...」で始まる目次行が 2 行以上を占める
///
/// 偽と判定したいケース:
/// - 通常の本文 (50 文字以上の解説文)
/// - 概念定義を含む比較的長い見出し付き本文 (内容語 25 文字以上)
pub fn is_retrieval_noise(text: &str) -> bool {
    let trimmed = text.trim();
    let total = trimmed.chars().count();
    if total == 0 {
        return true;
    }

    let content_chars = trimmed
        .chars()
        .filter(|c| {
            c.is_alphabetic()
                || ('一'..='龥').contains(c)
                || ('ぁ'..='ん').contains(c)
                || ('ァ'..='ヶ').contains(c)
        })
        .count();
    if content_chars < 25 {
        return true;
    }

    // leader 系の OCR/組版ノイズ文字
    let leader_chars = trimmed
        .chars()
        .filter(|c| matches!(*c, '…' | '・' | '─' | '━' | '.' | '·'))
        .count();
    if total < 60 && leader_chars * 10 >= total * 3 {
        return true;
    }

    // 目次行検出: 行頭が "1." "1.2" "1.2.3" 等の数値ドット番号で始まる行が 2 行以上。
    let toc_like_lines = trimmed
        .lines()
        .filter(|line| {
            let l = line.trim_start();
            let mut chars = l.chars();
            let first = match chars.next() {
                Some(c) if c.is_ascii_digit() => c,
                _ => return false,
            };
            let _ = first;
            // 先頭から「数字 (とドット) のみ」が 2 文字以上続けば目次行とみなす
            let head: String = l.chars().take(8).collect();
            head.starts_with(|c: char| c.is_ascii_digit()) && head.contains('.')
        })
        .count();
    if toc_like_lines >= 2 {
        return true;
    }

    false
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_chunk() -> Chunk {
        Chunk {
            id: Uuid::new_v4(),
            source_id: Uuid::new_v4(),
            ord: 0,
            text: "本文".into(),
            heading_path: vec!["第1章".into(), "1.1".into()],
            page: Some(3),
            bbox: Some([0.0, 0.0, 100.0, 50.0]),
            summary: None,
        }
    }

    #[test]
    fn source_kind_serializes_lowercase() {
        let json = serde_json::to_string(&SourceKind::Pdf).unwrap();
        assert_eq!(json, "\"pdf\"");
        let parsed: SourceKind = serde_json::from_str("\"markdown\"").unwrap();
        assert_eq!(parsed, SourceKind::Markdown);
    }

    #[test]
    fn source_status_roundtrip() {
        for s in [
            SourceStatus::Pending,
            SourceStatus::Parsing,
            SourceStatus::Embedding,
            SourceStatus::Ready,
            SourceStatus::Failed,
        ] {
            let json = serde_json::to_string(&s).unwrap();
            let back: SourceStatus = serde_json::from_str(&json).unwrap();
            assert_eq!(format!("{back:?}"), format!("{s:?}"));
        }
    }

    #[test]
    fn hit_source_default_is_vector() {
        assert_eq!(HitSource::default(), HitSource::Vector);
    }

    #[test]
    fn search_hit_defaults_source_when_missing() {
        let chunk = sample_chunk();
        let json = serde_json::json!({
            "chunk": chunk,
            "score": 0.42_f32,
        });
        let hit: SearchHit = serde_json::from_value(json).unwrap();
        assert_eq!(hit.source, HitSource::Vector);
        assert!((hit.score - 0.42).abs() < 1e-6);
    }

    #[test]
    fn error_displays_with_prefix() {
        let e = Error::Parse("boom".into());
        assert_eq!(e.to_string(), "parse: boom");
        let e = Error::Embed("dim".into());
        assert_eq!(e.to_string(), "embed: dim");
    }

    #[test]
    fn io_error_converts_into_error() {
        let io = std::io::Error::new(std::io::ErrorKind::NotFound, "x");
        let e: Error = io.into();
        assert!(matches!(e, Error::Io(_)));
    }

    #[test]
    fn retrieval_noise_keeps_normal_body_text() {
        let body = "「IDリクワイアド(とりあえず10)」アンチパターンの兆候です。記述的な名前の代わりに「id」を列名に使うことを正当化しようとする発言を見聞きすることがあります";
        assert!(!is_retrieval_noise(body));
    }

    #[test]
    fn retrieval_noise_keeps_section_heading() {
        // 章タイトル + 短い導入は本文として残す (内容語 25 文字以上)
        let heading = "3章 IDリクワイアド(とりあえずID) 2つのテーブルを結合する際に留意すべき重要なポイントです";
        assert!(!is_retrieval_noise(heading));
    }

    #[test]
    fn retrieval_noise_drops_short_orphan_fragments() {
        assert!(is_retrieval_noise("もしれません。"));
        assert!(is_retrieval_noise("return:"));
        assert!(is_retrieval_noise("ことになります。"));
        assert!(is_retrieval_noise("モデルのテスト"));
        // 章タイトルだけの断片も本文 chunk として扱わない
        assert!(is_retrieval_noise("アプリケーション開発のアンチパターン"));
        assert!(is_retrieval_noise("クエリのアンチパターン"));
        assert!(is_retrieval_noise("18.4アンチパターンを用いてもよい場合"));
    }

    #[test]
    fn retrieval_noise_drops_toc_lines() {
        let toc = "15.6リストの長さの制限\n15.7交差テーブルの他のメリット………11\n2章ナイーブツリー(素朴な木)………1";
        assert!(is_retrieval_noise(toc));
    }

    #[test]
    fn retrieval_noise_drops_leader_dominant_short_fragments() {
        // 「3.2.5 複合キーは使いにくい …………」のような目次断片
        assert!(is_retrieval_noise("3.2.5複合キーは使いにくい"));
        // 内容語を増やせば素通り (= 章節名そのものではなく説明文)
        let body = "3.2.5 複合キーは使いにくいので、いっさい使わないという開発者もいます";
        assert!(!is_retrieval_noise(body));
    }

    #[test]
    fn retrieval_noise_drops_pure_dots_and_leaders() {
        assert!(is_retrieval_noise("………………"));
        assert!(is_retrieval_noise("・・・・・・"));
    }

    #[test]
    fn chunk_serde_roundtrip() {
        let c = sample_chunk();
        let json = serde_json::to_string(&c).unwrap();
        let back: Chunk = serde_json::from_str(&json).unwrap();
        assert_eq!(back.id, c.id);
        assert_eq!(back.heading_path, c.heading_path);
        assert_eq!(back.page, c.page);
    }
}
