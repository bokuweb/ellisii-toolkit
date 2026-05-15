//! `looks_specific_query` (src-tauri の routing 関数) を 4 つの fixture
//! query に適用し、production の routing が「rewriter が効く domain で
//! expansion を発火し、害になる domain で skip するか」を検証する。
//!
//! src-tauri は private fn なので test で同等の実装を再現して評価する。
//! 同期が外れたら test で気づける。
//!
//! 期待する結果 (これまでの 4 domain 計測から):
//! - 民法 hard / CS Wiki hard: rewriter が効く → 大半が non-specific (expansion 発火)
//! - SQL antipatterns / jp-patents: rewriter が不要 / 害 → 大半が specific (expansion skip)
//!
//! `cargo test -p ellisii-rag-eval-cli --test validate_router -- --nocapture`
//! (この test は LLM 不要なので `#[ignore]` 不要)

use ellisii_rag::eval::GoldenSet;
use std::path::PathBuf;

/// 旧版 (= バグあり): kanji_count>=4 で過剰判定するバージョン。
/// 比較用に残してある (alignment 改善幅の数値根拠)。
fn looks_specific_query_legacy(query: &str) -> bool {
    if has_article_id(query) {
        return true;
    }
    if has_quoted_phrase(query) {
        return true;
    }
    if has_urlish(query) {
        return true;
    }
    if query.chars().count() >= 50 {
        return true;
    }
    let kanji_count = query
        .chars()
        .filter(|c| matches!(c, '\u{4E00}'..='\u{9FFF}'))
        .count();
    kanji_count >= 4
}

/// 現本番 (新版): src-tauri/src/lib.rs の `looks_specific_query` と同期。
/// kanji_count 廃止 + has_code_snippet 追加。
fn looks_specific_query(query: &str) -> bool {
    if has_article_id(query) {
        return true;
    }
    if has_quoted_phrase(query) {
        return true;
    }
    if has_urlish(query) {
        return true;
    }
    if has_code_snippet(query) {
        return true;
    }
    if query.chars().count() >= 50 {
        return true;
    }
    false
}

/// SQL / 識別子 / 関数呼び出しなど code-like なマーカーを検出。
fn has_code_snippet(query: &str) -> bool {
    let chars: Vec<char> = query.chars().collect();

    // (1) 連続する大文字 / アンダースコア >= 4: SELECT, ORDER, ENUM, AUTO_INCREMENT
    let mut upper_run = 0usize;
    for c in &chars {
        if c.is_ascii_uppercase() || *c == '_' {
            upper_run += 1;
            if upper_run >= 4 {
                return true;
            }
        } else {
            upper_run = 0;
        }
    }

    // (2) ASCII 記号 (), {}, ;, =, %, <, > を 2 つ以上 (SQL/コードっぽい)
    let punct = chars
        .iter()
        .filter(|c| matches!(c, '(' | ')' | '{' | '}' | ';' | '=' | '%' | '<' | '>'))
        .count();
    if punct >= 2 {
        return true;
    }

    // (3) snake_case 識別子 (lowercase + _): parent_id, foreign_key
    // ASCII 文字+アンダースコアの連続中に _ が含まれていれば識別子
    let mut tok_start: Option<usize> = None;
    for i in 0..chars.len() {
        let is_id_char = chars[i].is_ascii_alphanumeric() || chars[i] == '_';
        match (tok_start, is_id_char) {
            (None, true) => tok_start = Some(i),
            (Some(s), false) | (Some(s), _) if !is_id_char => {
                let tok: String = chars[s..i].iter().collect();
                if tok.contains('_') && tok.chars().count() >= 4 {
                    return true;
                }
                tok_start = None;
            }
            _ => {}
        }
    }
    if let Some(s) = tok_start {
        let tok: String = chars[s..].iter().collect();
        if tok.contains('_') && tok.chars().count() >= 4 {
            return true;
        }
    }

    // (4) Title-case 識別子 ≥5 chars: MergeRequest, ActiveRecord, JavaScript
    // 簡易: ASCII alpha 連続中に「先頭大文字 + 後半小文字」と「途中で大文字復活」のパターン
    let mut camel_run = 0usize;
    let mut had_upper_in_middle = false;
    let mut started_upper = false;
    for c in &chars {
        if c.is_ascii_alphabetic() {
            if camel_run == 0 {
                started_upper = c.is_ascii_uppercase();
                camel_run = 1;
            } else {
                if c.is_ascii_uppercase() && started_upper && camel_run >= 2 {
                    had_upper_in_middle = true;
                }
                camel_run += 1;
            }
        } else {
            if started_upper && had_upper_in_middle && camel_run >= 5 {
                return true;
            }
            camel_run = 0;
            had_upper_in_middle = false;
            started_upper = false;
        }
    }
    if started_upper && had_upper_in_middle && camel_run >= 5 {
        return true;
    }

    false
}

fn has_article_id(query: &str) -> bool {
    // 第 N 条 / Article N / Section N / Sec. N
    let chars: Vec<char> = query.chars().collect();
    for i in 0..chars.len() {
        if chars[i] == '第' {
            // 後続が数字 (半角/全角/漢数字) で「条」を含む
            let mut j = i + 1;
            let mut found_digit = false;
            while j < chars.len() {
                let c = chars[j];
                if c.is_ascii_digit()
                    || ('０'..='９').contains(&c)
                    || "一二三四五六七八九十百千".contains(c)
                {
                    found_digit = true;
                    j += 1;
                } else {
                    break;
                }
            }
            if found_digit && j < chars.len() && chars[j] == '条' {
                return true;
            }
        }
    }
    let lower = query.to_lowercase();
    if let Some(idx) = lower.find("article ") {
        let rest = &query[idx + "article ".len()..];
        if rest.chars().next().map_or(false, |c| {
            c.is_ascii_digit() || matches!(c, 'I' | 'V' | 'X' | 'L' | 'C' | 'D' | 'M')
        }) {
            return true;
        }
    }
    if let Some(idx) = lower.find("section ") {
        let rest = &query[idx + "section ".len()..];
        if rest.chars().next().map_or(false, |c| c.is_ascii_digit()) {
            return true;
        }
    }
    if let Some(idx) = lower.find("sec.") {
        let rest = &query[idx + "sec.".len()..];
        if rest
            .trim_start()
            .chars()
            .next()
            .map_or(false, |c| c.is_ascii_digit())
        {
            return true;
        }
    }
    false
}

fn has_quoted_phrase(query: &str) -> bool {
    // 「2 文字以上」を引用符で囲んでいる
    fn check(s: &str, open: char, close: char) -> bool {
        let mut chars = s.chars();
        while let Some(c) = chars.next() {
            if c == open {
                let mut count = 0;
                for c2 in chars.by_ref() {
                    if c2 == close {
                        return count >= 2;
                    }
                    count += 1;
                }
            }
        }
        false
    }
    check(query, '"', '"')
        || check(query, '「', '」')
        || check(query, '『', '』')
        || check(query, '\u{201C}', '\u{201D}')
}

fn has_urlish(query: &str) -> bool {
    if query.contains("http://") || query.contains("https://") {
        return true;
    }
    // 簡易メール検出: x@y.z
    if let Some(at) = query.find('@') {
        let after = &query[at + 1..];
        if after.contains('.')
            && after
                .chars()
                .take_while(|c| !c.is_whitespace())
                .any(|c| c == '.')
        {
            return true;
        }
    }
    false
}

fn load_golden(domain: &str) -> GoldenSet {
    let base = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .join("rag/tests/fixtures/eval")
        .join(domain);
    serde_json::from_str(&std::fs::read_to_string(base.join("golden.json")).unwrap()).unwrap()
}

#[derive(Debug)]
struct DomainExpectation {
    domain: &'static str,
    /// 「rewriter が効く domain か」(true = expansion を発火させたい)
    expansion_helps: bool,
    /// rewriter ROI (前回計測の MRR Δ)
    measured_mrr_delta: f32,
}

#[test]
fn validate_router_alignment() {
    // 4 domain の rewriter ROI (PR #56 / #61 / #63 / #64 で計測済み)
    let expectations = &[
        DomainExpectation {
            domain: "jp-civil-law-hard",
            expansion_helps: true,
            measured_mrr_delta: 0.251,
        },
        DomainExpectation {
            domain: "jp-cs-wiki-hard",
            expansion_helps: true,
            measured_mrr_delta: 0.090,
        },
        DomainExpectation {
            domain: "sql-antipatterns",
            expansion_helps: false,
            measured_mrr_delta: -0.051,
        },
        DomainExpectation {
            domain: "jp-patents",
            // 微改善 (+0.010)、ROI が極めて低いので skip でも実用差は小さい
            expansion_helps: false,
            measured_mrr_delta: 0.010,
        },
    ];

    fn run(label: &str, exps: &[DomainExpectation], judge: fn(&str) -> bool) -> (usize, usize) {
        println!("\n=== {label} ===\n");
        println!(
            "  domain                  total  specific  non-spec  spec%  rewriter desired  measured Δ MRR"
        );
        let mut total_aligned = 0usize;
        let mut total_misaligned = 0usize;

        for exp in exps {
            let golden = load_golden(exp.domain);
            let n = golden.items.len();
            let mut spec = 0;
            let mut non_spec = 0;
            for item in &golden.items {
                let is_spec = judge(&item.query);
                if is_spec {
                    spec += 1;
                    if exp.expansion_helps {
                        total_misaligned += 1;
                    } else {
                        total_aligned += 1;
                    }
                } else {
                    non_spec += 1;
                    if !exp.expansion_helps {
                        total_misaligned += 1;
                    } else {
                        total_aligned += 1;
                    }
                }
            }
            let spec_pct = (spec as f32 / n as f32) * 100.0;
            let want = if exp.expansion_helps {
                "expand (non-spec)"
            } else {
                "skip (spec)     "
            };
            println!(
                "  {:<22}  {:>5}  {:>8}  {:>8}  {:>4.1}%  {}  {:+.3}",
                exp.domain, n, spec, non_spec, spec_pct, want, exp.measured_mrr_delta,
            );
        }
        let total = total_aligned + total_misaligned;
        println!(
            "  aligned: {}/{} ({:.1}%)  misaligned: {}",
            total_aligned,
            total,
            100.0 * total_aligned as f32 / total as f32,
            total_misaligned
        );
        (total_aligned, total_misaligned)
    }

    let (a_legacy, _m1) = run(
        "legacy router (kanji_count>=4 のバグあり)",
        expectations,
        looks_specific_query_legacy,
    );
    let (a_new, _m2) = run(
        "現本番 router (kanji 廃止 + code 検出)",
        expectations,
        looks_specific_query,
    );

    println!(
        "\n  alignment 改善: legacy {} → new {} (+{})",
        a_legacy,
        a_new,
        a_new as i32 - a_legacy as i32
    );
    // 本番ロジック (= 新版) が legacy より明確に良いことを保証する regression test
    assert!(
        a_new > a_legacy,
        "new router should align better than legacy ({} vs {})",
        a_new,
        a_legacy
    );
}

#[test]
fn looks_specific_known_positives() {
    assert!(looks_specific_query("民法第709条について教えて"));
    assert!(looks_specific_query("Article 5 of the Constitution"));
    assert!(looks_specific_query("\"hello world\" を検索"));
    assert!(looks_specific_query("「契約解除」とは"));
    assert!(looks_specific_query("https://example.com を要約"));
    assert!(looks_specific_query("user@example.com に送って"));
    // 50 文字
    assert!(looks_specific_query(&"あ".repeat(50)));
    // SQL keyword (4+ 大文字連続)
    assert!(looks_specific_query("SELECT で売上を取る"));
    assert!(looks_specific_query("ORDER BY RAND() を使う"));
    // 記号塊
    assert!(looks_specific_query("(key, value) ペア"));
    // snake_case 識別子
    assert!(looks_specific_query("parent_id の使い方"));
    // CamelCase 識別子
    assert!(looks_specific_query("MergeRequest との紐付け"));
}

#[test]
fn looks_specific_known_negatives() {
    assert!(!looks_specific_query("猫が好き"));
    assert!(!looks_specific_query("天気は?"));
    assert!(!looks_specific_query("Hello"));
    // 漢字 3 個 + その他
    assert!(!looks_specific_query("猫犬鳥"));
    // 漢字 4 個以上だけでは specific ではない (legacy バグ)。これらは
    // むしろ rewriter で expand したいケース (民法 hard 系の自然文)。
    assert!(!looks_specific_query("不法行為損害賠償請求"));
    assert!(!looks_specific_query(
        "中学生が買ったゲーム機を親はキャンセルできるか"
    ));
    assert!(!looks_specific_query(
        "脅されて結ばされた契約はどう扱われるか"
    ));
    // 短い ASCII (DB, ID, MRI 等) だけでは specific 判定しない
    assert!(!looks_specific_query("DB に保存するパスワード"));
    assert!(!looks_specific_query("MRI 検査の時間を短くしたい"));
}
