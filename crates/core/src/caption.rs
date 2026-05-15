//! Chunk-leading caption の抽出ヘルパ。
//!
//! 法令系 / 条文構造をもつ日本語文書 (例: 横浜市市税条例) の各 chunk は典型的に
//! `(...見出し...)\n第N条 …本文…` で始まる。検索後の rerank で「クエリと caption の
//! Jaccard 一致度」を boost に使うため、chunk 先頭から caption を取り出す純粋関数を
//! 提供する。
//!
//! - [`extract_caption`]: chunk 先頭の `(...)` 見出しを取り出す厳格版。改正注記
//!   (`(平18条例70・一部改正)` 等) は意味的見出しでないので skip して 2 つめを採る。
//! - [`extract_caption_or_lead`]: caption が無いチャンクでも `第N条(のM)?` の本文先頭を
//!   fallback として返す。改正注記から始まる「caption 無し」chunk の救済用。
//!
//! `crates/rag` (rerank 側) と `crates/store-sqlite` (caption index 側) の両方から
//! 共通参照する。store→rag を直接引きたくない都合で `core` に置く。

/// chunk テキストの先頭 (空白 trim 後) が `(...)` で始まっていれば、その中身を返す。
/// 改正注記 (例: `(平18条例70・一部改正)`、`(令5条例16・全改)`) は意味的な見出しでは
/// ないので skip して、続く 2 つめの `(...)` を見る。それが無ければ `None`。
pub fn extract_caption(text: &str) -> Option<&str> {
    let t = text.trim_start();
    let rest = t.strip_prefix('(')?;
    let end = rest.find(')')?;
    let first = &rest[..end];
    if is_revision_note(first) {
        let after = rest.get(end + ')'.len_utf8()..)?.trim_start();
        let rest2 = after.strip_prefix('(')?;
        let end2 = rest2.find(')')?;
        let second = &rest2[..end2];
        if is_revision_note(second) {
            return None;
        }
        Some(second)
    } else {
        Some(first)
    }
}

/// `extract_caption` の上に乗せる fallback。caption (`(...)`) が無いチャンクでも、
/// `第N条 <body>` パターンならその本文先頭を疑似 caption として返す。
///
/// 動機: 条例 chunk の先頭が改正注記 (`(平18条例70・一部改正)`) **だけ** で、
/// 続く `(...)` 見出しが無い構造では、`extract_caption` は `None` を返すため
/// caption rerank が無効化される (yokohama golden [15] の構造的失敗)。本文側に
/// 「横浜市が課する普通税は…」のようにクエリ語彙が直接出てくるケースが多いため、
/// 本文先頭を 80 char までで切って疑似 caption として使う。
pub fn extract_caption_or_lead(text: &str) -> Option<&str> {
    if let Some(c) = extract_caption(text) {
        return Some(c);
    }
    extract_article_body_lead(text)
}

/// 改正注記スキップ後に `第N条(のM)?` で始まる本文先頭を最大 80 char (or `。`/改行 まで) 取り出す。
pub fn extract_article_body_lead(text: &str) -> Option<&str> {
    let mut t = text.trim_start();
    if let Some(rest) = t.strip_prefix('(') {
        if let Some(end) = rest.find(')') {
            let inside = &rest[..end];
            if is_revision_note(inside) {
                t = rest.get(end + ')'.len_utf8()..)?.trim_start();
            }
        }
    }
    let after_dai = t.strip_prefix('第')?;
    let mut end_num = 0usize;
    for (i, c) in after_dai.char_indices() {
        if c.is_ascii_digit() || is_jp_numeral(c) {
            end_num = i + c.len_utf8();
        } else {
            break;
        }
    }
    if end_num == 0 {
        return None;
    }
    let after_jou = after_dai[end_num..].strip_prefix('条')?;
    let after_branch = if let Some(rest) = after_jou.strip_prefix('の') {
        let mut end_b = 0usize;
        for (i, c) in rest.char_indices() {
            if c.is_ascii_digit() || is_jp_numeral_basic(c) {
                end_b = i + c.len_utf8();
            } else {
                break;
            }
        }
        if end_b == 0 { after_jou } else { &rest[end_b..] }
    } else {
        after_jou
    };
    let body = after_branch.trim_start_matches(|c: char| c.is_whitespace() || c == '　');
    let mut end_idx = body.len();
    let mut taken = 0usize;
    for (i, c) in body.char_indices() {
        if matches!(c, '。' | '\n') {
            end_idx = i;
            break;
        }
        taken += 1;
        if taken >= 80 {
            end_idx = i + c.len_utf8();
            break;
        }
    }
    let lead = body[..end_idx].trim_end();
    if lead.is_empty() {
        None
    } else {
        Some(lead)
    }
}

fn is_jp_numeral(c: char) -> bool {
    matches!(
        c,
        '０'..='９' | '〇' | '一' | '二' | '三' | '四' | '五' | '六' |
        '七' | '八' | '九' | '十' | '百' | '千' | '万' | '・'
    )
}

fn is_jp_numeral_basic(c: char) -> bool {
    matches!(
        c,
        '０'..='９' | '〇' | '一' | '二' | '三' | '四' | '五' | '六' |
        '七' | '八' | '九' | '十'
    )
}

/// `caption` の char-bigram 集合に **無い** bigram が `body` にどれだけ含まれるか
/// (0.0..=1.0)。高いほど「body が caption の見出しを言い換えたり、別語彙で
/// 概念を導入している」 = paraphrase-friendly 文書の傾向。
///
/// 動機: jp-tokkyo-hou (特許法) のような corpus は body 側に概念定義 (「発明
/// とは…」) が出てくるので vocab novelty が高く、LLM rewriter (paraphrase 経由
/// の検索) が +6.2pt MRR と効く。逆に yokohama (横浜市市税条例) のような
/// 「税率は◯円」と数値・固有名詞中心の corpus は novelty が低く、rewriter は
/// 効きにくい。corpus-side で precompute できる signal なので、起動時に一度
/// 計算しておき rewriter ルーティング (= `multi_query_max_variants` を 0 にするか
/// どうか) の判断に使える。
///
/// 空 / 1 文字以下の body は `0.0` を返す (シグナル無効)。
pub fn body_vocab_novelty(caption: &str, body: &str) -> f32 {
    let cap = char_bigrams(caption);
    let bod = char_bigrams(body);
    if bod.is_empty() {
        return 0.0;
    }
    let novel = bod.iter().filter(|b| !cap.contains(*b)).count();
    novel as f32 / bod.len() as f32
}

fn char_bigrams(s: &str) -> std::collections::HashSet<String> {
    let chars: Vec<char> = s.chars().filter(|c| !c.is_whitespace()).collect();
    let mut out = std::collections::HashSet::new();
    for w in chars.windows(2) {
        out.insert(w.iter().collect::<String>());
    }
    out
}

/// chunk body の中の **定義語** を抽出する。
///
/// 条文・法令系の文書では `「X」という。` や `「X」をいう。` のパターンで本文中に
/// 定義語を導入することが多い (例: `第129条 …事業所 (以下本節において「事業所等」
/// という。) において…`)。caption だけでは捕捉できないこれらの定義語を
/// 取り出して caption rerank の補助信号として使う想定 (Run 41)。
///
/// 抽出条件: `「(.+?)」` のうち、続く 30 char 以内に `という` または `をいう`
/// が現れるもの。重複は除去 (最初の出現順を保持)。
pub fn extract_defined_terms(text: &str) -> Vec<&str> {
    let mut out: Vec<&str> = Vec::new();
    let mut cursor = 0usize;
    while cursor < text.len() {
        let Some(open_rel) = text[cursor..].find('「') else {
            break;
        };
        let open = cursor + open_rel + '「'.len_utf8();
        let Some(close_rel) = text[open..].find('」') else {
            break;
        };
        let close = open + close_rel;
        let term = &text[open..close];
        let after = close + '」'.len_utf8();
        // `starts_with` だけで判定するので lookahead 切り出しは不要。
        let tail = &text[after..];
        if !term.is_empty()
            && (tail.starts_with("という") || tail.starts_with("をいう"))
            && !out.contains(&term)
        {
            out.push(term);
        }
        cursor = after;
    }
    out
}

/// `(平XX条例YY・…改正)` 系の改正注記を識別する単純なヒューリスティック。
pub fn is_revision_note(s: &str) -> bool {
    s.contains("条例")
        && (s.contains("改正")
            || s.contains("全改")
            || s.contains("追加")
            || s.contains("削除")
            || s.contains("旧"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extract_caption_basic() {
        assert_eq!(extract_caption("(入湯税の税率)\n\n第123条 ..."), Some("入湯税の税率"));
        assert_eq!(extract_caption("  \n  (たばこ税の税率) 第85条"), Some("たばこ税の税率"));
    }

    #[test]
    fn extract_caption_returns_none_when_no_paren() {
        assert_eq!(extract_caption("第3条 市税として課する..."), None);
        assert_eq!(extract_caption(""), None);
    }

    #[test]
    fn extract_caption_takes_first_paren_only() {
        assert_eq!(extract_caption("(A)(B) 本文"), Some("A"));
    }

    #[test]
    fn extract_caption_skips_revision_note_and_uses_second_paren() {
        assert_eq!(
            extract_caption("(平18条例70・一部改正)\n(入湯税の税率)\n第123条"),
            Some("入湯税の税率")
        );
        assert_eq!(extract_caption("(令5条例16・全改) 本文"), None);
        assert_eq!(
            extract_caption("(平8条例24・追加)(均等割の税率の軽減)第26条"),
            Some("均等割の税率の軽減")
        );
    }

    #[test]
    fn extract_defined_terms_picks_quoted_term_before_toiu() {
        // 条文での典型: (以下本節において「X」という。)
        let t = "第129条 事業所税は…事務所又は事業所 (以下本節において「事業所等」という。) において…";
        let terms = extract_defined_terms(t);
        assert_eq!(terms, vec!["事業所等"]);
    }

    #[test]
    fn extract_defined_terms_handles_multiple_definitions() {
        let t = "「Aさん」という。次に「Bさん」という。さらに「Cさん」をいう。";
        let terms = extract_defined_terms(t);
        assert_eq!(terms, vec!["Aさん", "Bさん", "Cさん"]);
    }

    #[test]
    fn extract_defined_terms_dedupes() {
        let t = "「事業所等」という。… 「事業所等」をいう。";
        let terms = extract_defined_terms(t);
        assert_eq!(terms, vec!["事業所等"]);
    }

    #[test]
    fn extract_defined_terms_skips_quotes_without_definition_marker() {
        // 引用句であって定義句でないものは拾わない
        let t = "「猫」が好きだ。「犬」も好きだ。";
        let terms = extract_defined_terms(t);
        assert!(terms.is_empty(), "got {terms:?}");
    }

    #[test]
    fn extract_defined_terms_handles_empty_or_short_input() {
        assert!(extract_defined_terms("").is_empty());
        assert!(extract_defined_terms("「").is_empty());
        assert!(extract_defined_terms("「あ」").is_empty()); // 定義マーカーなし
    }

    #[test]
    fn or_lead_falls_back_after_revision_note() {
        let t = "(平18条例70・一部改正)\n\n第3条 横浜市が課する普通税は、市民税、固定資産税、軽自動車税、市たばこ税及び事業所税とする。";
        let cap = extract_caption_or_lead(t).unwrap();
        assert!(cap.starts_with("横浜市が課する普通税"), "got: {cap}");
        assert!(!cap.contains('。'));
    }

    #[test]
    fn or_lead_uses_paren_when_present() {
        assert_eq!(
            extract_caption_or_lead("(入湯税の税率)\n第123条 入湯税の税率は100円とする"),
            Some("入湯税の税率")
        );
    }

    #[test]
    fn or_lead_handles_article_branch_no_revision() {
        let cap = extract_caption_or_lead("第3条の2 ふるさと納税の控除は次のとおりとする").unwrap();
        assert!(cap.starts_with("ふるさと納税"), "got: {cap}");
    }

    #[test]
    fn or_lead_caps_at_80_chars_or_period() {
        let body = "あ".repeat(200);
        let t = format!("第1条 {body}");
        let cap = extract_caption_or_lead(&t).unwrap();
        assert!(cap.chars().count() <= 80);
    }

    #[test]
    fn or_lead_returns_none_for_non_article_text() {
        assert_eq!(extract_caption_or_lead("ただの本文です"), None);
        assert_eq!(extract_caption_or_lead(""), None);
    }

    #[test]
    fn or_lead_handles_jp_kanji_numbers() {
        let cap = extract_caption_or_lead("第百二十三条 入湯税は次のとおりとする").unwrap();
        assert!(cap.starts_with("入湯税"), "got: {cap}");
    }

    #[test]
    fn revision_note_recognises_common_forms() {
        assert!(is_revision_note("平18条例70・一部改正"));
        assert!(is_revision_note("令5条例16・全改"));
        assert!(is_revision_note("平8条例24・追加"));
        assert!(is_revision_note("令1条例3・削除"));
        assert!(!is_revision_note("入湯税の税率"));
        assert!(!is_revision_note("市税として課するものは"));
    }

    #[test]
    fn body_vocab_novelty_is_zero_for_empty_body() {
        assert_eq!(body_vocab_novelty("入湯税の税率", ""), 0.0);
        assert_eq!(body_vocab_novelty("", ""), 0.0);
    }

    #[test]
    fn body_vocab_novelty_is_zero_when_body_subset_of_caption() {
        // body 全 bigram が caption に含まれる → novelty = 0
        let n = body_vocab_novelty("入湯税の税率", "入湯税");
        assert_eq!(n, 0.0);
    }

    #[test]
    fn body_vocab_novelty_is_one_when_body_disjoint_from_caption() {
        // body の bigram が一切 caption と重ならない → novelty = 1.0
        let n = body_vocab_novelty("AAA", "BCD");
        assert!((n - 1.0).abs() < 1e-6, "got {n}");
    }

    #[test]
    #[test]
    fn body_vocab_novelty_paraphrase_case_is_higher_than_literal() {
        // 特許法風: caption が短く、body が caption に無い概念語彙を持つ → 高 novelty
        let para = body_vocab_novelty("発明", "自然法則を利用した技術的思想の創作のうち高度のもの");
        // 横浜条例風: caption と body が同じ語彙で構成 → 相対的に低 novelty
        let lit = body_vocab_novelty("入湯税の税率", "入湯税の税率は100円とする");
        assert!(
            para > lit + 0.2,
            "paraphrase corpus must have higher novelty than literal one: paraphrase={para}, literal={lit}"
        );
        assert!(para > 0.85, "paraphrase novelty too low: {para}");
        assert!(lit < 0.7, "literal novelty too high: {lit}");
    }

    // 抽出結果は元文字列の slice なので、入力文字列の連続部分でなければならない。
    // panic しないことも併せて確認する基本健全性テスト。
    #[test]
    fn caption_or_lead_result_is_substring_of_input() {
        let cases = [
            "",
            "(",
            ")",
            "()",
            "()()",
            "第条",
            "第123",
            "第3条 ",
            "(改正)条文",
            "(普通税)\n第3条 X",
            "(平18条例70・一部改正)\n第3条 横浜市が課する",
        ];
        for s in cases {
            if let Some(out) = extract_caption_or_lead(s) {
                assert!(s.contains(out), "expected `{out}` in input `{s}`");
            }
        }
    }
}
