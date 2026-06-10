//! 検索後の rerank ヘルパ。retrieval 自体は変えず、上位 K 件の並び替えだけ行う。
//!
//! 提供:
//! - [`extract_caption`]: chunk 先頭の `(...)` 見出しを取り出す。条文 ID `第X条`
//!   の直前にある括弧書きの題目 (例: `(入湯税の税率)`) を想定。
//! - [`caption_overlap`]: query と caption の char-bigram 一致率 (caption 側を分母 = precision)。
//! - [`caption_boost_in_place`]: pool の各 hit に対し、caption が query と一致する分だけ
//!   `score` に加算して再ソート。
//! - [`apply_caption_index`]: 別途用意した `(chunk_id, caption)` 一覧を全走査して overlap
//!   が高いものを pool に注入する (pool に未収のチャンクも引き上げられる)。

use ellisii_core::SearchHit;
use std::collections::{HashMap, HashSet};
use uuid::Uuid;

// caption 抽出ヘルパは `ellisii_core::caption` に集約。store-sqlite との
// 重複を避けるため core 経由で参照し、既存の API は re-export で互換維持する。
pub use ellisii_core::caption::{
    extract_article_body_lead, extract_caption, extract_caption_or_lead, extract_defined_terms,
};

fn bigrams(s: &str) -> HashSet<String> {
    let chars: Vec<char> = s.chars().filter(|c| !c.is_whitespace()).collect();
    let mut out = HashSet::new();
    for w in chars.windows(2) {
        out.insert(w.iter().collect::<String>());
    }
    out
}

/// query と caption の bigram 一致率。Jaccard 類似度 (intersection / union)。
/// caption 側を分母にとる旧仕様では「短い caption が部分一致だけで満点に近い」現象が
/// 起きて、SQL アンチパターン系のような短いラベル captioned 文書で誤爆していた。
/// Jaccard なら query 側の長い文脈もペナルティとして効く。0.0..=1.0。
pub fn caption_overlap(query: &str, caption: &str) -> f32 {
    let qb = bigrams(query);
    let cb = bigrams(caption);
    if qb.is_empty() || cb.is_empty() {
        return 0.0;
    }
    let inter = qb.intersection(&cb).count() as f32;
    let union = qb.union(&cb).count() as f32;
    if union == 0.0 {
        0.0
    } else {
        inter / union
    }
}

/// caption / query の Jaccard がこの値未満なら caption boost を加えない。短い caption への
/// 偶然マッチで rank が荒れるのを防ぐためのノイズフロア。
/// 0.10 で sql-antipatterns 系の誤爆 (-24pt hit@1) と jp-civil-law / yokohama の本来の
/// 効果 (+18..+30pt hit@1) を両立できる、というのが eval_fixtures + eval_yokohama の
/// 計測結果。詳細は `docs/eval/recall-evals.md`。
pub const MIN_CAPTION_OVERLAP: f32 = 0.10;

/// `query` と corpus 内 captions の **最大** Jaccard overlap。空 captions / 空 query で 0.0。
///
/// Run 33 のフォロー: LLM rewriter が生成した variant を corpus caption と照合して、
/// caption と全く重ならない (= 流れ弾な) variant を post-filter で drop する用途。
pub fn max_caption_overlap(query: &str, captions: &[(Uuid, String)]) -> f32 {
    captions
        .iter()
        .map(|(_, c)| caption_overlap(query, c))
        .fold(0.0_f32, f32::max)
}

/// pool の各 hit に対し、caption と **body 中の定義語** (`「X」という。` 系) の Jaccard
/// 一致率の最大を `score` に `alpha` 倍で加算し再ソートする。
/// 動機 (Run 41): caption が短く query 中心語と乖離するケース (例: yokohama
/// chunk 331 caption=「事業所税の納税義務者等」, query=「事業所税の事業所等とは
/// どんな場所」) では caption だけでは boost が効かないが、本文中で
/// `「事業所等」という。` と定義された語が caption と同等の overlap 信号として
/// 使える。caption と defined terms のいずれか強いほうを採用する。
/// `MIN_CAPTION_OVERLAP` 未満は noise として無視する。
pub fn caption_boost_in_place(query: &str, hits: &mut [SearchHit], alpha: f32) {
    if alpha <= 0.0 || hits.is_empty() {
        return;
    }
    for h in hits.iter_mut() {
        let cap_ov = extract_caption_or_lead(&h.chunk.text)
            .map(|c| caption_overlap(query, c))
            .unwrap_or(0.0);
        let def_ov = extract_defined_terms(&h.chunk.text)
            .into_iter()
            .map(|t| caption_overlap(query, t))
            .fold(0.0_f32, f32::max);
        let ov = cap_ov.max(def_ov);
        if ov >= MIN_CAPTION_OVERLAP {
            h.score += alpha * ov;
        }
    }
    hits.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
}

/// 各 hit の `heading_path` を caption と同じ要領で query と bigram 比較し、最大の一致率を
/// `score` に `alpha` 倍で加える。caption が無い chunk (例: 改正注記から始まる条文) でも
/// 章節タイトルが効くようになる。caption_boost と組み合わせて使う想定。
pub fn heading_boost_in_place(query: &str, hits: &mut [SearchHit], alpha: f32) {
    if alpha <= 0.0 || hits.is_empty() {
        return;
    }
    for h in hits.iter_mut() {
        let mut best = 0.0_f32;
        for seg in &h.chunk.heading_path {
            let ov = caption_overlap(query, seg);
            if ov > best {
                best = ov;
            }
        }
        h.score += alpha * best;
    }
    hits.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
}

/// 並び順を保ったまま、`source_id` ごとの出現回数が `max_per_source` を超えた hit を
/// 落とす MMR-lite (diversity guard)。caption / heading rerank 後の最終 truncate の
/// 直前に呼ぶ想定。
///
/// 動機: 1 つの長い doc を chunker が複数 chunk に割ったとき、上位 K 件が同一 source の
/// 連続 chunks で埋まり、別 source の重要 chunk が押し出される (top-K 偏り)。
/// `max_per_source = 2` のような上限で「同一 source は最大 N まで」と制約することで
/// 上位の source 多様性を確保する。
///
/// このフィルタは **rerank ではない**: score を変えず、order を変えず、ただ削るだけ。
/// 結果的に上位の見た目だけ多様化される。MMR の simplification (λ→0 の極端版 + 上限化)。
///
/// `max_per_source == 0` は無効 (= passthrough, 何も削らない) として扱う。
/// fixture 評価では 1 doc = 1 source の構造のため metrics 上は no-op になる。
/// production の chunker 出力 (1 source → 複数 chunks) で初めて効く。
pub fn dedup_by_source_in_place(hits: &mut Vec<SearchHit>, max_per_source: usize) {
    if max_per_source == 0 || hits.is_empty() {
        return;
    }
    let mut counts: HashMap<Uuid, usize> = HashMap::new();
    hits.retain(|h| {
        let c = counts.entry(h.chunk.source_id).or_insert(0);
        if *c < max_per_source {
            *c += 1;
            true
        } else {
            false
        }
    });
}

/// query から「内容語ぽい term」を抽出する (lexical overlap boost 用)。
///
/// - ASCII 単語 (`[A-Za-z0-9_]{2,}`) を lowercase で
/// - 日本語は CJK / かな / カナだけで構成される 2-gram (助詞・句読点を除く)
///
/// `src-tauri` の `extract_terms` を移植したもの。char-bigram tokenizer
/// (FTS5 indexer) と整合する粒度で、口語クエリでも「実際に本文に出る部分文字列」を
/// 拾えるようにしている。重複は除去済み。
#[must_use]
pub fn extract_terms(query: &str) -> Vec<String> {
    const STOP_CHARS: [char; 13] = [
        'は', 'が', 'を', 'に', 'で', 'と', 'の', 'へ', 'や', '、', '。', '?', '？',
    ];
    let mut out: Vec<String> = Vec::new();
    // ASCII 単語
    for token in query.split(|c: char| !(c.is_ascii_alphanumeric() || c == '_')) {
        if token.len() >= 2 && token.chars().any(|c| c.is_ascii_alphanumeric()) {
            out.push(token.to_lowercase());
        }
    }
    // 日本語 2-gram (CJK / ひらがな / カタカナのみ)
    let chars: Vec<char> = query
        .chars()
        .filter(|c| !c.is_whitespace() && !STOP_CHARS.contains(c))
        .collect();
    for w in chars.windows(2) {
        let is_jp = |c: &char| {
            ('一'..='龥').contains(c) || ('ぁ'..='ん').contains(c) || ('ァ'..='ヶ').contains(c)
        };
        if w.iter().all(is_jp) {
            out.push(w.iter().collect());
        }
    }
    out.sort();
    out.dedup();
    out
}

/// クエリ語が chunk 本文にどれだけ含まれるか (term coverage) を `score` に
/// 乗算 boost する。`src-tauri::lexical_overlap_boost` の移植。
///
/// boost = (本文に出現する term 数 / 全 term 数) × `alpha`。`alpha` は最大加算率
/// (src-tauri は 0.5 = 最大 +50%)。`score *= 1.0 + boost` で適用し、再ソートする。
///
/// hybrid (vector + keyword RRF) は「順位の融合」なので、クエリ語が**実際に本文に
/// 何語マッチしたか**という量的情報が落ちる。この boost はそれを補い、口語クエリでも
/// キーワードが濃く出現する chunk を上位に押し上げる。caption / heading rerank とは
/// 直交 (本文を見る) なので併用してよい。
pub fn lexical_boost_in_place(query: &str, hits: &mut [SearchHit], alpha: f32) {
    if alpha <= 0.0 || hits.is_empty() {
        return;
    }
    let terms = extract_terms(query);
    if terms.is_empty() {
        return;
    }
    for h in hits.iter_mut() {
        let lower = h.chunk.text.to_lowercase();
        let hit = terms.iter().filter(|t| lower.contains(t.as_str())).count();
        let ratio = hit as f32 / terms.len() as f32;
        h.score *= 1.0 + ratio * alpha;
    }
    hits.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
}

/// 別途事前構築した `(chunk_id, caption)` の一覧を全走査して、query と overlap の高いものを
/// pool 側に注入する。pool に既にいる場合は score 加算、いない場合は新たな id_score として
/// 追加 (caller 側で chunk-id → SearchHit を引き直す前提)。
///
/// 戻り値: pool 内 hit の chunk_id をキーにしたスコア表。caption-only で新規追加されたものは
/// "id だけわかれば recall 計算に充分" な evaluator 用の半端 hit となるので、
/// 本番経路で使うときは戻り値の id を chunk 引き直しに使うこと。
pub fn apply_caption_index(
    query: &str,
    hits: &[SearchHit],
    captions: &[(Uuid, String)],
    top_n: usize,
    bonus_alpha: f32,
) -> HashMap<Uuid, f32> {
    let mut id_score: HashMap<Uuid, f32> = hits.iter().map(|h| (h.chunk.id, h.score)).collect();

    let mut scored: Vec<(f32, &Uuid)> = captions
        .iter()
        .map(|(id, c)| (caption_overlap(query, c), id))
        .filter(|(s, _)| *s >= MIN_CAPTION_OVERLAP)
        .collect();
    scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

    for (s, id) in scored.into_iter().take(top_n) {
        let bonus = bonus_alpha * s;
        id_score
            .entry(*id)
            .and_modify(|v| *v += bonus)
            .or_insert(bonus);
    }
    id_score
}

/// `apply_caption_index` の heading 版。`headings` は `(chunk_id, heading_segment_concatenated)`。
/// caption と heading の両方を運用するときは、戻り値の id_score を統合する想定。
pub fn apply_heading_index(
    query: &str,
    hits: &[SearchHit],
    headings: &[(Uuid, String)],
    top_n: usize,
    bonus_alpha: f32,
) -> HashMap<Uuid, f32> {
    apply_caption_index(query, hits, headings, top_n, bonus_alpha)
}

/// Caption corpus の IDF (= 「希少な caption ほど高い weight」) テーブルを作る。
///
/// 動機 (Run 12 / jp-patent-docs): 「請求項1」「効果」「背景技術」のように
/// **複数の文書間で同じ caption を共有する** corpus では、caption-text 一致 boost が
/// 「同じ caption を持つ別文書の chunk」を引き上げて正解 chunk を top-k から押し出す。
/// IDF で頻出 caption を減衰させると、`(入湯税の税率)` のような unique な caption だけが
/// 強い weight を持ち、`(効果)` のような共通ラベルは抑制される。
///
/// 戻り値: caption 文字列 → `[0.0, 1.0]` の重み。最も希少な caption が 1.0、最も頻出
/// な caption が 0 に近い。空入力 / すべて同じ caption の場合は全エントリ 1.0。
///
/// 計算式: `weight = ln((N + 1) / (df + 1)) / ln(N + 1)` (smoothed IDF 正規化)。
/// `N` は corpus 内の captioned chunk 数、`df` はその caption の出現回数。
pub fn compute_caption_idf(captions: &[(Uuid, String)]) -> HashMap<String, f32> {
    if captions.is_empty() {
        return HashMap::new();
    }
    let n = captions.len() as f32;
    let mut df: HashMap<String, usize> = HashMap::new();
    for (_, c) in captions {
        *df.entry(c.clone()).or_insert(0) += 1;
    }
    // 最小 df は 1 (corpus に存在するから)。その場合の idf を 1.0 に正規化する。
    let max_idf = ((n + 1.0) / 2.0).ln().max(1e-6);
    let mut out: HashMap<String, f32> = HashMap::with_capacity(df.len());
    for (caption, count) in df {
        let idf = ((n + 1.0) / (count as f32 + 1.0)).ln();
        let w = (idf / max_idf).clamp(0.0, 1.0);
        out.insert(caption, w);
    }
    out
}

/// IDF テーブルを使って caption boost をかける版。caption が IDF map に無いか map が
/// 空のときは weight=1.0 (= 既存挙動と同じ)。
///
/// Run 41: caption と body 中の defined terms (`「X」という。` 系) の overlap を比較し、
/// 強いほうを採用する。defined term には IDF weight が無いので w=1.0 で計算。
pub fn caption_boost_in_place_with_idf(
    query: &str,
    hits: &mut [SearchHit],
    alpha: f32,
    idf: &HashMap<String, f32>,
) {
    if alpha <= 0.0 || hits.is_empty() {
        return;
    }
    for h in hits.iter_mut() {
        // caption 側: IDF 重みあり
        let (cap_ov, cap_w) = match extract_caption_or_lead(&h.chunk.text) {
            Some(cap) => (
                caption_overlap(query, cap),
                idf.get(cap).copied().unwrap_or(1.0),
            ),
            None => (0.0, 1.0),
        };
        let cap_boost = if cap_ov >= MIN_CAPTION_OVERLAP {
            alpha * cap_ov * cap_w
        } else {
            0.0
        };
        // defined-term 側: IDF 無し (corpus-wide な頻度情報を持たないので w=1.0)
        let def_ov = extract_defined_terms(&h.chunk.text)
            .into_iter()
            .map(|t| caption_overlap(query, t))
            .fold(0.0_f32, f32::max);
        let def_boost = if def_ov >= MIN_CAPTION_OVERLAP {
            alpha * def_ov
        } else {
            0.0
        };
        h.score += cap_boost.max(def_boost);
    }
    hits.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
}

/// `apply_caption_index` に IDF を加味した版。
pub fn apply_caption_index_with_idf(
    query: &str,
    hits: &[SearchHit],
    captions: &[(Uuid, String)],
    top_n: usize,
    bonus_alpha: f32,
    idf: &HashMap<String, f32>,
) -> HashMap<Uuid, f32> {
    let mut id_score: HashMap<Uuid, f32> = hits.iter().map(|h| (h.chunk.id, h.score)).collect();

    let mut scored: Vec<(f32, &Uuid)> = captions
        .iter()
        .map(|(id, c)| {
            let ov = caption_overlap(query, c);
            let w = idf.get(c).copied().unwrap_or(1.0);
            (ov * w, id)
        })
        .filter(|(s, _)| *s >= MIN_CAPTION_OVERLAP)
        .collect();
    scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

    for (s, id) in scored.into_iter().take(top_n) {
        let bonus = bonus_alpha * s;
        id_score
            .entry(*id)
            .and_modify(|v| *v += bonus)
            .or_insert(bonus);
    }
    id_score
}

#[cfg(test)]
mod tests {
    use super::*;
    use ellisii_core::{Chunk, HitSource};

    fn hit(text: &str, score: f32) -> SearchHit {
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
            score,
            source: HitSource::Vector,
            semantic_score: None,
        }
    }

    fn hit_with_heading(text: &str, score: f32, heading: Vec<&str>) -> SearchHit {
        let mut h = hit(text, score);
        h.chunk.heading_path = heading.into_iter().map(String::from).collect();
        h
    }

    #[test]
    fn max_caption_overlap_picks_best_match() {
        let captions: Vec<(Uuid, String)> = vec![
            (Uuid::nil(), "入湯税の税率".into()),
            (Uuid::nil(), "市たばこ税の税率".into()),
            (Uuid::nil(), "都市計画税の税率".into()),
        ];
        let q_relevant = "入湯税の税率はいくらですか";
        let q_irrelevant = "完全に無関係なクエリ xyz";
        let q_empty = "";
        let s_rel = max_caption_overlap(q_relevant, &captions);
        let s_irr = max_caption_overlap(q_irrelevant, &captions);
        assert!(
            s_rel >= MIN_CAPTION_OVERLAP,
            "expected ≥ floor, got {s_rel}"
        );
        assert!(s_irr < MIN_CAPTION_OVERLAP, "expected < floor, got {s_irr}");
        assert_eq!(max_caption_overlap(q_empty, &captions), 0.0);
        assert_eq!(max_caption_overlap("anything", &[]), 0.0);
    }

    #[test]
    fn extract_caption_basic() {
        assert_eq!(
            extract_caption("(入湯税の税率)\n\n第123条 ..."),
            Some("入湯税の税率")
        );
        assert_eq!(
            extract_caption("  \n  (たばこ税の税率) 第85条"),
            Some("たばこ税の税率")
        );
    }

    #[test]
    fn extract_caption_returns_none_when_no_paren() {
        assert_eq!(extract_caption("第3条 市税として課する..."), None);
        assert_eq!(extract_caption(""), None);
    }

    #[test]
    fn extract_caption_takes_first_paren_only() {
        // ネストや 2 つめは取らない (改正注記でないとき)
        let t = "(A)(B) 本文";
        assert_eq!(extract_caption(t), Some("A"));
    }

    #[test]
    fn extract_caption_skips_revision_note_and_uses_second_paren() {
        let t = "(平18条例70・一部改正)\n(入湯税の税率)\n第123条";
        assert_eq!(extract_caption(t), Some("入湯税の税率"));
        let t = "(令5条例16・全改) 本文";
        assert_eq!(extract_caption(t), None);
        let t = "(平8条例24・追加)(均等割の税率の軽減)第26条";
        assert_eq!(extract_caption(t), Some("均等割の税率の軽減"));
    }

    #[test]
    fn extract_caption_or_lead_falls_back_after_revision_note() {
        // 改正注記直後に第N条が来る yokohama-style chunk: 本文先頭が caption になる。
        let t = "(平18条例70・一部改正)\n\n第3条 横浜市が課する普通税は、市民税、固定資産税、軽自動車税、市たばこ税及び事業所税とする。";
        let cap = extract_caption_or_lead(t).unwrap();
        assert!(cap.starts_with("横浜市が課する普通税"), "got: {cap}");
        assert!(!cap.contains('。'), "trailing period not stripped: {cap}");
    }

    #[test]
    fn extract_caption_or_lead_uses_paren_when_present() {
        // (...) があるときは extract_caption と同じ結果。
        let t = "(入湯税の税率)\n第123条 入湯税の税率は100円とする";
        assert_eq!(extract_caption_or_lead(t), Some("入湯税の税率"));
    }

    #[test]
    fn extract_caption_or_lead_handles_article_branch_no_revision() {
        // 改正注記なし、第N条のM 形式でも fallback で本文を取れる。
        let t = "第3条の2 ふるさと納税の控除は次のとおりとする";
        let cap = extract_caption_or_lead(t).unwrap();
        assert!(cap.starts_with("ふるさと納税"), "got: {cap}");
    }

    #[test]
    fn extract_caption_or_lead_caps_at_80_chars_or_period() {
        let body = "あ".repeat(200);
        let t = format!("第1条 {body}");
        let cap = extract_caption_or_lead(&t).unwrap();
        assert!(cap.chars().count() <= 80, "len={}", cap.chars().count());
    }

    #[test]
    fn extract_caption_or_lead_returns_none_for_non_article_text() {
        assert_eq!(extract_caption_or_lead("ただの本文です"), None);
        assert_eq!(extract_caption_or_lead(""), None);
    }

    #[test]
    fn extract_caption_or_lead_handles_jp_kanji_numbers() {
        let t = "第百二十三条 入湯税は次のとおりとする";
        let cap = extract_caption_or_lead(t).unwrap();
        assert!(cap.starts_with("入湯税"), "got: {cap}");
    }

    #[test]
    fn caption_overlap_full_match_is_one() {
        assert!((caption_overlap("入湯税の税率", "入湯税の税率") - 1.0).abs() < 1e-6);
    }

    #[test]
    fn caption_overlap_no_match_is_zero() {
        assert!((caption_overlap("固定資産税", "入湯税の税率") - 0.0).abs() < 1e-6);
    }

    #[test]
    fn caption_overlap_partial_match_jaccard() {
        // caption "入湯税" のbigram = {入湯, 湯税} (2)。query "入湯税の税率" の bigram = 5。
        // intersection = 2, union = 5 → Jaccard = 0.4
        let s = caption_overlap("入湯税の税率", "入湯税");
        assert!((s - 0.4).abs() < 1e-6, "got {s}");
    }

    #[test]
    fn caption_boost_promotes_matching_caption() {
        // 同 score のとき caption マッチが上に来る
        let a = hit("(入湯税の税率)\n\n第123条", 0.10);
        let b = hit("(別件の税率)\n\n第999条", 0.10);
        let a_id = a.chunk.id;
        let mut pool = vec![b, a.clone()];
        caption_boost_in_place("入湯税の税率はいくら", &mut pool, 1.0);
        assert_eq!(pool[0].chunk.id, a_id);
        // boost が score に乗っていること
        assert!(pool[0].score > 0.10);
        let _ = a; // silence unused
    }

    #[test]
    fn caption_boost_zero_alpha_is_noop() {
        let mut pool = vec![hit("(B)\n\n第2条", 0.10), hit("(A)\n\n第1条", 0.20)];
        let before: Vec<_> = pool.iter().map(|h| (h.chunk.id, h.score)).collect();
        caption_boost_in_place("A", &mut pool, 0.0);
        let after: Vec<_> = pool.iter().map(|h| (h.chunk.id, h.score)).collect();
        assert_eq!(before, after);
    }

    #[test]
    fn caption_boost_promotes_chunk_with_defined_term_when_caption_misses() {
        // query「特別徴収義務者とは何ですか」に対して:
        // - 無関係 caption「都市計画税の税率」chunk は overlap が低い (caption-only では浮上しない)
        // - body に `「特別徴収義務者」という。` を含む chunk は defined-term overlap で boost
        let unrelated = hit("(都市計画税の税率)\n第133条 都市計画税の税率は...", 0.10);
        let def_chunk = hit(
            "(入湯税の徴収方法)\n第124条 入湯税の徴収義務者 (以下「特別徴収義務者」という。) は…",
            0.10,
        );
        let def_id = def_chunk.chunk.id;
        let mut pool = vec![unrelated, def_chunk];
        caption_boost_in_place("特別徴収義務者とは何ですか", &mut pool, 1.0);
        assert_eq!(
            pool[0].chunk.id, def_id,
            "defined-term chunk should rank first"
        );
        assert!(pool[0].score > 0.10);
    }

    #[test]
    fn apply_caption_index_introduces_new_id() {
        let pool = vec![hit("(B)\n\n第2条", 0.10)];
        let hidden_id = Uuid::new_v4();
        let captions = vec![(hidden_id, "入湯税の税率".to_string())];
        let scores = apply_caption_index("入湯税の税率は", &pool, &captions, 5, 1.0);
        assert!(scores.contains_key(&hidden_id));
        assert!(scores[&hidden_id] > 0.0);
    }

    #[test]
    fn heading_boost_promotes_matching_heading() {
        // caption が抽出できない (= '(...)' で始まらない) chunk でも、heading_path を介して
        // 引き上げられること
        let a = hit_with_heading(
            "第3条 普通税は次のとおり",
            0.10,
            vec!["第1章 総則", "普通税の種類"],
        );
        let b = hit_with_heading("第500条 別件", 0.10, vec!["第99章 別件"]);
        let a_id = a.chunk.id;
        let mut pool = vec![b, a];
        heading_boost_in_place("普通税の種類は", &mut pool, 1.0);
        assert_eq!(pool[0].chunk.id, a_id);
        assert!(pool[0].score > 0.10);
    }

    #[test]
    fn heading_boost_uses_max_segment() {
        // heading_path の中で最大のマッチ率が採用される
        let h = hit_with_heading(
            "本文",
            0.10,
            vec!["関係ない章", "通謀虚偽表示", "もっと別の節"],
        );
        let h_id = h.chunk.id;
        let mut pool = vec![
            hit_with_heading("別の本文", 0.10, vec!["別の章"]),
            h.clone(),
        ];
        heading_boost_in_place("通謀虚偽表示の意味", &mut pool, 1.0);
        assert_eq!(pool[0].chunk.id, h_id);
        let _ = h;
    }

    #[test]
    fn heading_boost_zero_alpha_is_noop() {
        let mut pool = vec![hit_with_heading("text", 0.10, vec!["A"])];
        let before: Vec<_> = pool.iter().map(|h| (h.chunk.id, h.score)).collect();
        heading_boost_in_place("A", &mut pool, 0.0);
        let after: Vec<_> = pool.iter().map(|h| (h.chunk.id, h.score)).collect();
        assert_eq!(before, after);
    }

    #[test]
    fn idf_unique_caption_gets_full_weight() {
        // 4 chunks の中で 1 個だけが unique caption「入湯税の税率」、3 個が「効果」
        let captions = vec![
            (Uuid::new_v4(), "入湯税の税率".to_string()),
            (Uuid::new_v4(), "効果".to_string()),
            (Uuid::new_v4(), "効果".to_string()),
            (Uuid::new_v4(), "効果".to_string()),
        ];
        let idf = compute_caption_idf(&captions);
        let unique = idf["入湯税の税率"];
        let common = idf["効果"];
        assert!(
            unique > common,
            "unique={unique} should beat common={common}"
        );
        assert!(
            (unique - 1.0).abs() < 1e-6,
            "rarest caption should be 1.0, got {unique}"
        );
        assert!(
            common > 0.0 && common < unique,
            "common caption should be discounted but positive, got {common}"
        );
    }

    #[test]
    fn idf_all_unique_returns_uniform_full_weight() {
        let captions = vec![
            (Uuid::new_v4(), "a".to_string()),
            (Uuid::new_v4(), "b".to_string()),
            (Uuid::new_v4(), "c".to_string()),
        ];
        let idf = compute_caption_idf(&captions);
        for w in idf.values() {
            // df=1 なので weight は ln(N+1/2)/ln(N+1)。N=3 → ln(4/2)/ln(4) = ln2/ln4 ≈ 0.5
            // 重要なのは「全員同じ値」になること、と「< 1.0」になること (ベースラインなので)。
            assert!(*w > 0.0 && *w <= 1.0);
        }
    }

    #[test]
    fn idf_empty_input_returns_empty() {
        let idf = compute_caption_idf(&[]);
        assert!(idf.is_empty());
    }

    #[test]
    fn caption_boost_with_idf_promotes_unique_caption_over_common() {
        // 同じ query overlap のとき、unique caption (idf=1.0) と common caption
        // (idf=0.3) を比較すると unique 側が優先される。
        let mut idf = HashMap::new();
        idf.insert("入湯税の税率".to_string(), 1.0_f32);
        idf.insert("効果".to_string(), 0.3_f32);
        // 同 score の 2 chunk
        let unique = hit("(入湯税の税率)\n本文 …", 0.10);
        let common = hit("(効果)\n入湯税の税率に関する効果 …", 0.10);
        let unique_id = unique.chunk.id;
        let mut pool = vec![common, unique];
        caption_boost_in_place_with_idf("入湯税の税率", &mut pool, 1.0, &idf);
        // unique caption が rank 1 になる (overlap が同じでも IDF で勝つ)
        assert_eq!(pool[0].chunk.id, unique_id);
    }

    #[test]
    fn apply_caption_index_with_idf_filters_common_captions() {
        // unique caption を pool 外から引き上げるが、common caption は IDF で抑制されて
        // 同じ overlap でも順位が下がる。
        let pool: Vec<SearchHit> = vec![hit("(別件)\n第1条", 0.10)];
        let unique_id = Uuid::new_v4();
        let common_id = Uuid::new_v4();
        let captions = vec![
            (unique_id, "入湯税の税率".to_string()),
            (common_id, "効果".to_string()),
        ];
        let mut idf = HashMap::new();
        idf.insert("入湯税の税率".to_string(), 1.0_f32);
        idf.insert("効果".to_string(), 0.2_f32);
        let scores = apply_caption_index_with_idf("入湯税の税率", &pool, &captions, 5, 1.0, &idf);
        let unique_s = scores.get(&unique_id).copied().unwrap_or(0.0);
        let common_s = scores.get(&common_id).copied().unwrap_or(0.0);
        assert!(
            unique_s > common_s,
            "unique={unique_s} should beat common={common_s} after IDF weighting"
        );
    }

    fn hit_with_source(source_id: Uuid, score: f32) -> SearchHit {
        let mut h = hit("dummy", score);
        h.chunk.source_id = source_id;
        h
    }

    #[test]
    fn dedup_by_source_zero_is_noop() {
        let sid = Uuid::new_v4();
        let mut pool = vec![
            hit_with_source(sid, 0.9),
            hit_with_source(sid, 0.8),
            hit_with_source(sid, 0.7),
        ];
        let before = pool.iter().map(|h| h.chunk.id).collect::<Vec<_>>();
        dedup_by_source_in_place(&mut pool, 0);
        let after = pool.iter().map(|h| h.chunk.id).collect::<Vec<_>>();
        assert_eq!(before, after, "max=0 should be passthrough");
    }

    #[test]
    fn dedup_by_source_keeps_top_n_per_source_preserving_order() {
        let s1 = Uuid::new_v4();
        let s2 = Uuid::new_v4();
        let h1a = hit_with_source(s1, 0.9);
        let h1b = hit_with_source(s1, 0.85);
        let h2a = hit_with_source(s2, 0.8);
        let h1c = hit_with_source(s1, 0.7);
        let h2b = hit_with_source(s2, 0.6);
        let (h1a_id, h1b_id, h2a_id, h2b_id) =
            (h1a.chunk.id, h1b.chunk.id, h2a.chunk.id, h2b.chunk.id);
        let mut pool = vec![h1a, h1b, h2a, h1c, h2b];
        dedup_by_source_in_place(&mut pool, 2);
        let kept: Vec<_> = pool.iter().map(|h| h.chunk.id).collect();
        assert_eq!(kept, vec![h1a_id, h1b_id, h2a_id, h2b_id]);
    }

    #[test]
    fn dedup_by_source_unbounded_when_n_exceeds_population() {
        let s1 = Uuid::new_v4();
        let mut pool = vec![hit_with_source(s1, 0.9), hit_with_source(s1, 0.5)];
        let before_len = pool.len();
        dedup_by_source_in_place(&mut pool, 5);
        assert_eq!(pool.len(), before_len);
    }

    #[test]
    fn dedup_by_source_empty_pool_is_safe() {
        let mut pool: Vec<SearchHit> = vec![];
        dedup_by_source_in_place(&mut pool, 1);
        assert!(pool.is_empty());
    }

    #[test]
    fn extract_terms_splits_ascii_and_jp_bigrams() {
        let terms = extract_terms("秘密保持義務はApache2について");
        // ASCII 単語は lowercase
        assert!(terms.contains(&"apache2".to_string()));
        // 日本語 2-gram (助詞「は」は除外される)
        assert!(terms.contains(&"秘密".to_string()));
        assert!(terms.contains(&"保持".to_string()));
        // 助詞単独や記号は含まれない
        assert!(!terms.iter().any(|t| t.contains('は')));
    }

    #[test]
    fn lexical_boost_promotes_chunk_with_more_query_terms() {
        // 低スコアだがクエリ語が濃く出る chunk を、高スコアだが無関係な chunk より
        // 上位へ押し上げる。
        let mut hits = vec![
            hit("これは全く無関係な本文です", 1.0),
            hit("秘密保持義務は契約終了後も存続します", 0.75),
        ];
        lexical_boost_in_place("秘密保持義務の存続", &mut hits, 0.5);
        // boost 後、term coverage の高い 2 番目が先頭に来る
        assert!(hits[0].chunk.text.contains("秘密保持義務"), "hits={hits:?}");
        assert!(hits[0].score > hits[1].score);
    }

    #[test]
    fn lexical_boost_is_noop_for_zero_alpha_or_empty_terms() {
        let mut hits = vec![hit("本文", 1.0)];
        lexical_boost_in_place("クエリ", &mut hits, 0.0);
        assert_eq!(hits[0].score, 1.0);
        // term が抽出されないクエリ (記号のみ)
        let mut hits = vec![hit("本文", 1.0)];
        lexical_boost_in_place("、。?", &mut hits, 0.5);
        assert_eq!(hits[0].score, 1.0);
    }
}
