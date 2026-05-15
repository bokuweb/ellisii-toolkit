//! コンテキスト圧縮 (Provence) の抽象。
//!
//! 参考: https://secon.dev/entry/2025/10/31/100000-open-provence-release/
//!       https://huggingface.co/hotchpotch/open-provence-reranker-xsmall-v1
//!
//! 与えられた `query` に対し、文書 `text` を **文単位で関連度スコアリング** し、
//! 閾値以下を捨てて連結したテキストを返す。LLM に投入する前に呼ぶことで、
//! 30〜95% のトークン削減が可能 (記事の数値、長文 QA で 80-95%)。

use async_trait::async_trait;
use ellisii_core::Result;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScoredSentence {
    pub text: String,
    pub score: f32,
    pub kept: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressedContext {
    pub kept_text: String,
    pub original_chars: usize,
    pub kept_chars: usize,
    pub sentences: Vec<ScoredSentence>,
}

impl CompressedContext {
    /// 0.0〜1.0、保持された文字の比率
    pub fn ratio(&self) -> f32 {
        if self.original_chars == 0 {
            return 1.0;
        }
        self.kept_chars as f32 / self.original_chars as f32
    }
}

#[async_trait]
pub trait ContextCompressor: Send + Sync {
    /// 圧縮機が実体を持つかどうか (UI 表示用)
    fn is_active(&self) -> bool;

    async fn compress(&self, query: &str, text: &str) -> Result<CompressedContext>;

    /// チャンク単位の cross-encoder rerank。
    ///
    /// 与えられた passages を query との関連度で 0-1 のスコアに変換して返す
    /// (順序は入力と一致)。実装が無い (= passthrough) ときは全て 1.0 を返す
    /// ことで「rerank を skip して元順序を尊重する」挙動になる。
    ///
    /// 使い方: hybrid 検索 + RRF で得た top-K に対して呼び、結果のスコアで
    /// 並び替えてから top-N を LLM に渡す。Provence は文単位 cross-encoder
    /// だが、同モデルを passage 単位で適用しても十分にリランカとして働く
    /// (=「同じ tokenizer + classifier でテキストの長さが変わるだけ」)。
    async fn score_passages(&self, _query: &str, passages: &[String]) -> Result<Vec<f32>> {
        Ok(vec![1.0; passages.len()])
    }
}

/// 閾値カットの後に「最低 `min_keep` 文 (または全文の `min_keep_ratio`)」を保証する。
///
/// 動機: Provence の `keep_threshold` は cross-encoder のスコアに依存するため、
/// クエリ語彙が珍しい (例: カタカナ造語の専門用語) と全文が閾値を割って
/// ほぼ全カットされ、LLM が「該当情報なし」と答えてしまう失敗モードがある。
/// 低リコールの事故を防ぐフロアとして、スコア降順で上位を昇格させる。
///
/// `min_keep` は文数の絶対下限、`min_keep_ratio` (0.0〜1.0) は割合の下限。
/// どちらか大きい方を採用する。既に閾値で上回っている文は据え置き。
pub fn apply_floor(scored: &mut [ScoredSentence], min_keep: usize, min_keep_ratio: f32) {
    let total = scored.len();
    if total == 0 {
        return;
    }
    let ratio_floor = (total as f32 * min_keep_ratio.clamp(0.0, 1.0)).ceil() as usize;
    let target = min_keep.max(ratio_floor).min(total);
    let already = scored.iter().filter(|s| s.kept).count();
    if already >= target {
        return;
    }
    let need = target - already;
    // スコア降順で「まだ kept でない」文のインデックスを集めて上位 `need` 件を昇格。
    let mut idxs: Vec<usize> = (0..total).filter(|&i| !scored[i].kept).collect();
    idxs.sort_by(|&a, &b| {
        scored[b]
            .score
            .partial_cmp(&scored[a].score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    for &i in idxs.iter().take(need) {
        scored[i].kept = true;
    }
}

/// 文に分割。日本語は `。!?！？\n` を区切りに、英語/その他は ASCII 句読点で分ける素朴版。
/// Provence の入力単位はセンテンスでよい (記事の説明と一致)。
pub fn split_sentences(text: &str) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    let mut buf = String::new();
    for c in text.chars() {
        buf.push(c);
        let is_terminal = matches!(c, '。' | '！' | '？' | '!' | '?' | '\n');
        if is_terminal {
            let trimmed = buf.trim();
            if !trimmed.is_empty() {
                out.push(trimmed.to_string());
            }
            buf.clear();
        }
    }
    let trimmed = buf.trim();
    if !trimmed.is_empty() {
        out.push(trimmed.to_string());
    }
    // ASCII の "." は数値などで誤分割しやすいので適用しない。長文ではこれで十分。
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splits_japanese_sentences_on_jp_period() {
        let s = split_sentences("吾輩は猫である。名前はまだ無い。どこで生れたか頓と見当がつかぬ。");
        assert_eq!(s.len(), 3);
        assert!(s[0].starts_with("吾輩"));
    }

    #[test]
    fn splits_on_newlines() {
        let s = split_sentences("line1\nline2\n\nline3");
        assert_eq!(s, vec!["line1", "line2", "line3"]);
    }

    #[test]
    fn handles_question_and_exclamation() {
        let s = split_sentences("そうですか？はい！");
        assert_eq!(s.len(), 2);
    }

    #[test]
    fn ratio_handles_zero_original() {
        let c = CompressedContext {
            kept_text: "".into(),
            original_chars: 0,
            kept_chars: 0,
            sentences: vec![],
        };
        assert_eq!(c.ratio(), 1.0);
    }

    #[test]
    fn ratio_calculates_proportion() {
        let c = CompressedContext {
            kept_text: "x".into(),
            original_chars: 100,
            kept_chars: 25,
            sentences: vec![],
        };
        assert!((c.ratio() - 0.25).abs() < 1e-6);
    }

    #[test]
    fn split_empty_string_yields_no_sentences() {
        assert!(split_sentences("").is_empty());
        assert!(split_sentences("   \t  ").is_empty());
    }

    #[test]
    fn split_handles_text_without_terminator() {
        let s = split_sentences("最後にピリオドなし");
        assert_eq!(s, vec!["最後にピリオドなし"]);
    }

    #[test]
    fn split_keeps_terminator_in_sentence() {
        let s = split_sentences("はい。");
        assert_eq!(s.len(), 1);
        assert!(s[0].ends_with('。'));
    }

    fn s(text: &str, score: f32, kept: bool) -> ScoredSentence {
        ScoredSentence { text: text.into(), score, kept }
    }

    #[test]
    fn apply_floor_promotes_top_when_threshold_drops_everything() {
        // 全文が閾値未満で kept=false。min_keep=3 なら上位 3 文が昇格する。
        let mut v = vec![
            s("a", 0.10, false),
            s("b", 0.05, false),
            s("c", 0.18, false),
            s("d", 0.02, false),
            s("e", 0.15, false),
        ];
        apply_floor(&mut v, 3, 0.0);
        let kept: Vec<_> = v.iter().filter(|x| x.kept).map(|x| x.text.clone()).collect();
        assert_eq!(kept.len(), 3);
        // 上位スコアは c(0.18), e(0.15), a(0.10)
        assert!(kept.contains(&"c".to_string()));
        assert!(kept.contains(&"e".to_string()));
        assert!(kept.contains(&"a".to_string()));
    }

    #[test]
    fn apply_floor_keeps_existing_kept_intact() {
        let mut v = vec![
            s("a", 0.05, false),
            s("b", 0.99, true),
            s("c", 0.04, false),
        ];
        apply_floor(&mut v, 2, 0.0);
        // b は既に kept、追加で 1 件だけ昇格する (上位は a)
        assert!(v[1].kept);
        assert!(v[0].kept);
        assert!(!v[2].kept);
    }

    #[test]
    fn apply_floor_uses_ratio_when_larger_than_min_keep() {
        // 10 文中 30% = 3 文を最低保証。min_keep=1 でも ratio が勝つ。
        let mut v: Vec<_> = (0..10)
            .map(|i| s(&format!("s{i}"), i as f32 * 0.01, false))
            .collect();
        apply_floor(&mut v, 1, 0.3);
        let kept_count = v.iter().filter(|x| x.kept).count();
        assert_eq!(kept_count, 3);
    }

    #[test]
    fn apply_floor_caps_at_total_when_target_exceeds() {
        let mut v = vec![s("a", 0.1, false), s("b", 0.2, false)];
        apply_floor(&mut v, 10, 0.0);
        assert!(v.iter().all(|x| x.kept));
    }

    #[test]
    fn apply_floor_noop_on_empty() {
        let mut v: Vec<ScoredSentence> = vec![];
        apply_floor(&mut v, 3, 0.5); // should not panic
    }

    #[test]
    fn split_does_not_break_on_ascii_period() {
        // 数値ピリオドで誤分割しないこと
        let s = split_sentences("値は 3.14 です。次は別の文。");
        assert_eq!(s.len(), 2);
        assert!(s[0].contains("3.14"));
    }

    #[derive(Default)]
    struct DummyComp;
    #[async_trait]
    impl ContextCompressor for DummyComp {
        fn is_active(&self) -> bool {
            true
        }
        async fn compress(&self, _q: &str, text: &str) -> Result<CompressedContext> {
            Ok(CompressedContext {
                kept_text: text.into(),
                original_chars: text.chars().count(),
                kept_chars: text.chars().count(),
                sentences: vec![],
            })
        }
    }

    #[tokio::test]
    async fn default_score_passages_returns_ones_per_input() {
        let c = DummyComp;
        let scores = c
            .score_passages("q", &["a".into(), "b".into(), "c".into()])
            .await
            .unwrap();
        assert_eq!(scores, vec![1.0, 1.0, 1.0]);
    }

    #[tokio::test]
    async fn default_score_passages_empty_input() {
        let c = DummyComp;
        assert!(c.score_passages("q", &[]).await.unwrap().is_empty());
    }
}
