//! Citation verification (Run 63 で `src-tauri/src/lib.rs::verify_citations` から
//! ライブラリ層へ昇格)。
//!
//! LLM が生成した回答テキストから `[1]` 〜 `[NN]` 形式の citation marker を抽出し、
//! 各マーカーが retrieval hits の有効範囲か / 出典 chunk に **語彙的に支持** されて
//! いるかを判定する。
//!
//! このモジュールは **ストリーミング外** の後段検証用 (= 生成完了後の self-check)。
//! 重い faithfulness / Ragas 系評価は `crates/rag-answer-eval` に置き、本モジュールは
//! 軽量な「unsupported citation 検出」だけを扱う。
//!
//! ## 用語
//! - **citation marker**: 回答中の `[N]` トークン (N は 1〜99 の正整数)
//! - **supported**: marker `N` の周辺 sentence が `hits[N-1]` の chunk text と
//!   Jaccard >= 0.05 で語彙重複している状態。日本語 (CJK 連続 2+ 文字) と
//!   英数字 (2+ chars) を 1 token として比較
//! - **unsupported**: 範囲外 (N=0 or N > hits.len) または overlap 不足
//!
//! ## 利用例
//!
//! ```ignore
//! let stats = ellisii_rag::citation::verify_citations(answer_text, &hits);
//! if stats.total > 0 && stats.unsupported >= stats.total / 2 {
//!     // 過半数が unsupported → hallucination 疑い、UI で警告
//! }
//! ```
//!
//! production の `src-tauri::run_stream` は `Verification` event でこの結果を
//! UI に流している。

use ellisii_core::SearchHit;
use regex::Regex;
use std::sync::OnceLock;

/// Citation 検証結果。`total` のうち `unsupported` 件は範囲外 / 出典との重複不足。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct CitationStats {
    /// 回答中で観測された `[N]` マーカー総数 (重複含む)。
    pub total: usize,
    /// 範囲外 (N=0 or N > hits.len) または出典 chunk と overlap 不足の数。
    pub unsupported: usize,
}

impl CitationStats {
    pub fn supported(&self) -> usize {
        self.total.saturating_sub(self.unsupported)
    }

    /// `unsupported / total` の比率 (0.0..=1.0)。`total == 0` のときは
    /// `0.0` (= 「無 citation だが unsupported とも判定できない」)。
    ///
    /// 高いほど LLM が **存在しない / 関係ない出典** を引いていることを示す。
    /// production の faithfulness ratio gate (Run 66) に使う。
    pub fn unsupported_ratio(&self) -> f32 {
        if self.total == 0 {
            0.0
        } else {
            self.unsupported as f32 / self.total as f32
        }
    }

    /// `unsupported_ratio` が `DEFAULT_UNSUPPORTED_RATIO_THRESHOLD` 以上か。
    /// 閾値超 = 「回答内引用の **過半数** が裏付け取れず、hallucination 疑い」。
    /// UI 警告 / 自動 hide / faithfulness gate の判定に使う。
    pub fn is_unsupported_high(&self) -> bool {
        self.unsupported_ratio() >= DEFAULT_UNSUPPORTED_RATIO_THRESHOLD
    }
}

/// `CitationStats::unsupported_ratio()` の **既定閾値** (Run 66)。これ以上で
/// 「回答内引用の過半数が裏付け取れない」と判定する。
///
/// 値 `0.5` は経験則: half-or-more の unsupported は LLM が架空 marker を吐いて
/// いるか出典 chunk が無関係な可能性が高く、user に表示する前に gate すべきレベル。
/// callsite で必要なら override 可能 (constant ではなく独自閾値を使う)。
pub const DEFAULT_UNSUPPORTED_RATIO_THRESHOLD: f32 = 0.5;

/// 回答テキストから `[N]` 形式の citation marker を抽出し、各 marker が
/// `hits[N-1]` に語彙的に支持されるかを判定する。詳細はモジュール docstring。
///
/// 戻り値: (total, unsupported) 互換のため tuple-friendly な
/// [`CitationStats`] を返す。
pub fn verify_citations(answer: &str, hits: &[SearchHit]) -> CitationStats {
    let cite_re = cite_regex();
    let mut stats = CitationStats::default();
    for m in cite_re.find_iter(answer) {
        let Ok(n) = m.as_str()[1..m.as_str().len() - 1].parse::<usize>() else {
            continue;
        };
        if n == 0 || n > hits.len() {
            // 範囲外引用は unsupported として加算 (LLM が架空 [N] を吐いた)
            stats.total += 1;
            stats.unsupported += 1;
            continue;
        }
        stats.total += 1;
        let chunk_text = &hits[n - 1].chunk.text;
        let sentence = surrounding_sentence(answer, m.start());
        if !sentence_overlaps_chunk(&sentence, chunk_text) {
            stats.unsupported += 1;
        }
    }
    stats
}

/// 回答中の citation marker をすべて列挙する (重複なし、出現順)。
/// `[1] [2] [1]` → `[1, 2]` を返す。範囲外も含む (= 検出のみ、判定はしない)。
pub fn extract_citation_ids(answer: &str) -> Vec<usize> {
    let cite_re = cite_regex();
    let mut seen: Vec<usize> = Vec::new();
    for m in cite_re.find_iter(answer) {
        if let Ok(n) = m.as_str()[1..m.as_str().len() - 1].parse::<usize>() {
            if !seen.contains(&n) {
                seen.push(n);
            }
        }
    }
    seen
}

fn cite_regex() -> &'static Regex {
    static CITE_RE: OnceLock<Regex> = OnceLock::new();
    CITE_RE.get_or_init(|| Regex::new(r"\[(\d{1,2})\]").expect("cite re"))
}

/// `start` の byte オフセットを含む sentence を抽出する。区切りは `。` `.` `!`
/// `?` `\n` のいずれか。
pub fn surrounding_sentence(text: &str, start: usize) -> String {
    fn is_sep(c: char) -> bool {
        matches!(c, '。' | '.' | '!' | '?' | '\n')
    }
    let mut left = 0usize;
    let mut right = text.len();
    let mut passed_start = false;
    for (i, c) in text.char_indices() {
        if is_sep(c) {
            if i < start {
                left = i + c.len_utf8();
            } else if !passed_start {
                right = i;
                passed_start = true;
            }
        }
    }
    text[left..right].trim().to_string()
}

/// sentence と chunk text の Jaccard 重複が 0.05 以上なら "supported"。
/// sentence にトークンが無い (記号や [N] 単独など) なら判定不能 → 支持扱い。
pub fn sentence_overlaps_chunk(sentence: &str, chunk: &str) -> bool {
    let s = tokens_for_overlap(sentence);
    if s.is_empty() {
        return true;
    }
    let c = tokens_for_overlap(chunk);
    if c.is_empty() {
        return false;
    }
    let inter = s.iter().filter(|t| c.contains(*t)).count();
    let jac = inter as f32 / s.len() as f32;
    jac >= 0.05
}

/// 各 `[N]` citation marker に対して、回答中の「どの sentence が」chunk 内の
/// 「どの sentence を」引用しているかを返す (Run 65)。
///
/// UI 側で「この一文がこの chunk 内のここを参照しています」と精密 highlight
/// するために使う。`verify_citations` が全体カウントしか返さないのに対し、
/// 本関数は marker ごとに 1 件の [`CitedSpan`] を返す。
#[derive(Debug, Clone, PartialEq)]
pub struct CitedSpan {
    /// 回答中の citation marker そのもの (e.g. "[1]")。
    pub marker: String,
    /// `N` の値 (1-indexed)。`hit_index = n - 1` で `hits` を引く。
    pub n: usize,
    /// 0-indexed chunk position in hits (`hits[hit_index]`)。範囲外なら None。
    pub hit_index: Option<usize>,
    /// `[N]` が出現した位置を含む回答中の sentence。
    pub answer_sentence: String,
    /// answer_sentence と最も語彙的に重なる chunk 内 sentence。chunk が空 /
    /// hit_index が None / overlap=0 なら None。
    pub chunk_sentence: Option<String>,
    /// chunk_sentence の `chunk.text` 内 byte range `(start, end)`。UI 側
    /// で部分文字列を切り出すのに使う。
    pub chunk_span: Option<(usize, usize)>,
    /// answer_sentence と chunk_sentence の Jaccard 重複 (0.0..=1.0)。
    /// 0.0 は overlap 不在 (= unsupported 相当)、高ければ高いほど信頼度が高い。
    pub overlap: f32,
}

/// 回答テキスト中の各 `[N]` に対して [`CitedSpan`] を構築する。
///
/// アルゴリズム:
/// 1. `[N]` marker を全件抽出 (出現順、重複は出現ごとに別 CitedSpan として保持)
/// 2. 各 marker について:
///    - 周辺 sentence (`surrounding_sentence`) を answer_sentence にする
///    - `hits[N-1]` の chunk.text を sentence 分割
///    - 各 chunk sentence と answer_sentence の Jaccard を計算
///    - 最大 Jaccard の chunk sentence を採用、byte range も返す
/// 3. `N` が範囲外なら hit_index=None / chunk_sentence=None / overlap=0.0
///
/// 同じ `[N]` が複数回出現したら、その都度それぞれ別 CitedSpan が返る (回答内の
/// **各文ごと**に highlight する用)。dedup したい呼び出し側は `marker` /
/// `chunk_span` でユニーク化する。
pub fn span_citations(answer: &str, hits: &[SearchHit]) -> Vec<CitedSpan> {
    let cite_re = cite_regex();
    let mut out = Vec::new();
    for m in cite_re.find_iter(answer) {
        let marker = m.as_str().to_string();
        let Ok(n) = m.as_str()[1..m.as_str().len() - 1].parse::<usize>() else {
            continue;
        };
        let answer_sentence = surrounding_sentence(answer, m.start());
        let hit_index = if n >= 1 && n <= hits.len() {
            Some(n - 1)
        } else {
            None
        };
        let (chunk_sentence, chunk_span, overlap) = match hit_index {
            Some(idx) => best_matching_sentence(&answer_sentence, &hits[idx].chunk.text),
            None => (None, None, 0.0),
        };
        out.push(CitedSpan {
            marker,
            n,
            hit_index,
            answer_sentence,
            chunk_sentence,
            chunk_span,
            overlap,
        });
    }
    out
}

/// chunk text を sentence に分割し、answer_sentence と Jaccard 最大の sentence を返す。
/// 戻り値: (chunk_sentence, byte_span, overlap)。一致が無ければ全部 None / 0.0。
fn best_matching_sentence(
    answer_sentence: &str,
    chunk: &str,
) -> (Option<String>, Option<(usize, usize)>, f32) {
    let a_tokens = tokens_for_overlap(answer_sentence);
    if a_tokens.is_empty() {
        return (None, None, 0.0);
    }
    let mut best: Option<(String, (usize, usize), f32)> = None;
    for (start, end, sentence) in split_sentences_with_spans(chunk) {
        let s_tokens = tokens_for_overlap(sentence);
        if s_tokens.is_empty() {
            continue;
        }
        let inter = a_tokens.iter().filter(|t| s_tokens.contains(*t)).count() as f32;
        let union = a_tokens.len() as f32 + s_tokens.len() as f32 - inter;
        let jac = if union > 0.0 { inter / union } else { 0.0 };
        if best.as_ref().map(|b| jac > b.2).unwrap_or(true) {
            best = Some((sentence.to_string(), (start, end), jac));
        }
    }
    match best {
        Some((s, span, jac)) if jac > 0.0 => (Some(s), Some(span), jac),
        _ => (None, None, 0.0),
    }
}

/// chunk text を sentence 区切り (`。` `.` `!` `?` `\n`) で分割し、各 sentence の
/// `(byte_start, byte_end, &str)` を返す。区切り文字は前の sentence の末尾に含める。
/// 空白だけの sentence は除外。
pub fn split_sentences_with_spans(text: &str) -> Vec<(usize, usize, &str)> {
    let mut out = Vec::new();
    let mut left = 0usize;
    for (i, c) in text.char_indices() {
        if matches!(c, '。' | '.' | '!' | '?' | '\n') {
            let end = i + c.len_utf8();
            let seg = &text[left..end];
            if !seg.trim().is_empty() {
                out.push((left, end, seg));
            }
            left = end;
        }
    }
    if left < text.len() {
        let seg = &text[left..];
        if !seg.trim().is_empty() {
            out.push((left, text.len(), seg));
        }
    }
    out
}

/// テキストから重複比較用のトークン列を抽出する。CJK は 2 文字以上の連続、
/// 英数字も 2 chars 以上を採用、小文字化して dedup する。
pub fn tokens_for_overlap(text: &str) -> Vec<String> {
    static TOK_RE: OnceLock<Regex> = OnceLock::new();
    let re =
        TOK_RE.get_or_init(|| Regex::new(r"[\p{Han}]{2,}|[A-Za-z0-9０-９]{2,}").expect("tok re"));
    let mut out: Vec<String> = re
        .find_iter(text)
        .map(|m| m.as_str().to_lowercase())
        .collect();
    out.sort();
    out.dedup();
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use ellisii_core::{Chunk, HitSource};
    use uuid::Uuid;

    fn mk_hit(text: &str) -> SearchHit {
        SearchHit {
            chunk: Chunk {
                id: Uuid::new_v4(),
                source_id: Uuid::nil(),
                ord: 0,
                text: text.to_string(),
                heading_path: vec![],
                page: None,
                bbox: None,
                summary: None,
            },
            score: 0.5,
            source: HitSource::Vector,
        }
    }

    #[test]
    fn citation_stats_unsupported_ratio_basic() {
        let s = CitationStats {
            total: 4,
            unsupported: 1,
        };
        assert!((s.unsupported_ratio() - 0.25).abs() < 1e-6);
        assert_eq!(s.supported(), 3);
        assert!(!s.is_unsupported_high());
    }

    #[test]
    fn citation_stats_unsupported_ratio_zero_total() {
        let s = CitationStats::default();
        assert_eq!(s.unsupported_ratio(), 0.0);
        assert!(!s.is_unsupported_high());
    }

    #[test]
    fn citation_stats_unsupported_ratio_high() {
        let s = CitationStats {
            total: 2,
            unsupported: 1,
        };
        // ratio = 0.5 == threshold → is_high = true
        assert!((s.unsupported_ratio() - 0.5).abs() < 1e-6);
        assert!(s.is_unsupported_high());
    }

    #[test]
    fn citation_stats_unsupported_ratio_full_unsupported() {
        let s = CitationStats {
            total: 3,
            unsupported: 3,
        };
        assert_eq!(s.unsupported_ratio(), 1.0);
        assert!(s.is_unsupported_high());
        assert_eq!(s.supported(), 0);
    }

    #[test]
    fn verify_citations_supported() {
        let hits = vec![mk_hit(
            "第94条 相手方と通謀してした虚偽の意思表示は、無効とする。",
        )];
        let answer = "通謀虚偽表示は無効です [1]。";
        let s = verify_citations(answer, &hits);
        assert_eq!(s.total, 1);
        assert_eq!(s.unsupported, 0);
        assert_eq!(s.supported(), 1);
    }

    #[test]
    fn verify_citations_out_of_range_marked_unsupported() {
        let hits = vec![mk_hit("ある条文")];
        let answer = "本文 [3] 何かが書かれている。";
        let s = verify_citations(answer, &hits);
        assert_eq!(s.total, 1);
        assert_eq!(s.unsupported, 1);
    }

    #[test]
    fn verify_citations_no_overlap_marked_unsupported() {
        let hits = vec![mk_hit("food court hours of operation")];
        let answer = "通謀虚偽表示は無効です [1]。";
        let s = verify_citations(answer, &hits);
        assert_eq!(s.total, 1);
        assert_eq!(s.unsupported, 1);
    }

    #[test]
    fn verify_citations_no_citations_returns_zero() {
        let hits = vec![mk_hit("ある条文")];
        let answer = "引用記号がない自由応答。";
        let s = verify_citations(answer, &hits);
        assert_eq!(s.total, 0);
        assert_eq!(s.unsupported, 0);
    }

    #[test]
    fn verify_citations_multi_hits_partial_unsupported() {
        let hits = vec![
            mk_hit("第94条 通謀虚偽表示は無効とする。"),
            mk_hit("XYZ unrelated random sentence"),
        ];
        let answer = "通謀虚偽表示は無効 [1]。一方で別件の話 [2]。";
        let s = verify_citations(answer, &hits);
        assert_eq!(s.total, 2);
        assert_eq!(s.unsupported, 1);
    }

    #[test]
    fn extract_citation_ids_dedupes_and_preserves_order() {
        assert_eq!(extract_citation_ids("最初 [2] 次 [1] 再度 [2]"), vec![2, 1]);
    }

    #[test]
    fn extract_citation_ids_skips_invalid() {
        assert_eq!(extract_citation_ids("普通の本文。"), Vec::<usize>::new());
        // 3 桁は cite_regex がマッチしない (\d{1,2}) → skip
        assert_eq!(extract_citation_ids("ref [100]"), Vec::<usize>::new());
    }

    #[test]
    fn surrounding_sentence_picks_segment_around_offset() {
        let text = "前文。中段 [1] です。後段。";
        let idx = text.find("[1]").unwrap();
        let s = surrounding_sentence(text, idx);
        assert_eq!(s, "中段 [1] です");
    }

    #[test]
    fn split_sentences_with_spans_basic() {
        let text = "第一文。第二文。第三文。";
        let sentences: Vec<&str> = split_sentences_with_spans(text)
            .into_iter()
            .map(|(_, _, s)| s)
            .collect();
        assert_eq!(sentences, vec!["第一文。", "第二文。", "第三文。"]);
    }

    #[test]
    fn split_sentences_with_spans_returns_correct_byte_ranges() {
        let text = "abc。defg。";
        let spans = split_sentences_with_spans(text);
        // "abc。" は bytes 0..6 (abc=3, 。=3 bytes UTF-8)
        let (s1, e1, t1) = spans[0];
        let (s2, e2, t2) = spans[1];
        assert_eq!(t1, "abc。");
        assert_eq!(&text[s1..e1], "abc。");
        assert_eq!(t2, "defg。");
        assert_eq!(&text[s2..e2], "defg。");
    }

    #[test]
    fn span_citations_finds_best_chunk_sentence() {
        let hits = vec![mk_hit(
            "第94条 通謀虚偽表示は無効とする。当事者間では効力を生じない。第三者には対抗できない。",
        )];
        let answer = "通謀虚偽表示は無効です [1]。";
        let spans = span_citations(answer, &hits);
        assert_eq!(spans.len(), 1);
        let span = &spans[0];
        assert_eq!(span.marker, "[1]");
        assert_eq!(span.n, 1);
        assert_eq!(span.hit_index, Some(0));
        assert!(
            span.chunk_sentence.is_some(),
            "should match a chunk sentence"
        );
        // 最大 overlap は 1 文目 (「通謀虚偽表示は無効とする。」)
        let chunk_s = span.chunk_sentence.as_ref().unwrap();
        assert!(
            chunk_s.contains("通謀虚偽表示"),
            "expected best-match sentence to contain key term, got {chunk_s:?}"
        );
        assert!(span.overlap > 0.0);
        // chunk_span が実際に chunk_sentence と一致する
        let (start, end) = span.chunk_span.unwrap();
        assert_eq!(&hits[0].chunk.text[start..end], chunk_s);
    }

    #[test]
    fn span_citations_out_of_range_returns_none_fields() {
        let hits = vec![mk_hit("ある条文")];
        let answer = "本文 [3] 何かが書かれている。";
        let spans = span_citations(answer, &hits);
        assert_eq!(spans.len(), 1);
        assert_eq!(spans[0].n, 3);
        assert_eq!(spans[0].hit_index, None);
        assert_eq!(spans[0].chunk_sentence, None);
        assert_eq!(spans[0].chunk_span, None);
        assert_eq!(spans[0].overlap, 0.0);
    }

    #[test]
    fn span_citations_no_overlap_returns_zero() {
        let hits = vec![mk_hit("apple banana orange")];
        let answer = "通謀虚偽表示は無効 [1]。";
        let spans = span_citations(answer, &hits);
        assert_eq!(spans.len(), 1);
        // tokens for "通謀虚偽表示は無効" は CJK 連続だが apple banana orange と
        // 重ならない → overlap=0、chunk_sentence=None
        assert_eq!(spans[0].overlap, 0.0);
        assert_eq!(spans[0].chunk_sentence, None);
    }

    #[test]
    fn span_citations_multiple_markers_yield_multiple_spans() {
        let hits = vec![
            mk_hit("通謀虚偽表示は無効とする。"),
            mk_hit("代理権の消滅事由を列挙する。"),
        ];
        let answer = "通謀虚偽表示は無効 [1]。代理権の消滅 [2]。";
        let spans = span_citations(answer, &hits);
        assert_eq!(spans.len(), 2);
        assert_eq!(spans[0].marker, "[1]");
        assert_eq!(spans[0].n, 1);
        assert_eq!(spans[0].hit_index, Some(0));
        assert!(spans[0].overlap > 0.0);
        assert_eq!(spans[1].marker, "[2]");
        assert_eq!(spans[1].n, 2);
        assert_eq!(spans[1].hit_index, Some(1));
        assert!(spans[1].overlap > 0.0);
    }
}
