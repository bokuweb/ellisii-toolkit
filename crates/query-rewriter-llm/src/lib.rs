//! `LlmRewriter`: LlmBackend を使ってクエリの言い換えを N 個生成する。
//!
//! 設計メモ:
//! - 元クエリは常に保持し、recall を落とさない安全網にする。
//! - 出力フォーマットは「番号付きリスト」を強制する。小型ローカル LLM でも
//!   JSON より失敗しにくく、本文だけパースしやすい。
//! - LLM が壊れた出力を返しても fallback として元クエリのみを返す
//!   (=Passthrough と同等)。retrieve 全体が止まらないことを最優先。
//!
//! より高度な rewriter として `MultiExpandRewriter` (paraphrase + decompose +
//! HyDE を 1 回の LLM 呼びでまとめて返す) を `multi_expand` サブモジュールに
//! 持つ。診断計測で `LlmRewriter` (paraphrase のみ) よりも recall が伸びる
//! ことが期待される (HyDE が答え側の語彙で検索する効果)。

pub mod multi_expand;
pub use multi_expand::MultiExpandRewriter;

use async_trait::async_trait;
use ellisii_core::Result;
use ellisii_llm_core::{LlmBackend, LlmRequest};
use ellisii_query_rewriter_core::{QueryRewriter, RewrittenQueries};
use std::sync::{Arc, Mutex};

const SYSTEM_PROMPT: &str = "あなたは検索クエリを書き換えるアシスタントです。\n以下の制約を厳守してください:\n- 与えられた質問の意味を保ったまま、検索しやすい言い換えを最大 N 個生成する\n- 同義語・類語・上位/下位概念・別表記 (漢字↔かな↔カナ) を使い、語順や表現を変える\n- 外来語のカタカナ表記 (例: リクワイアド) は、元の英語表記 (例: Required) や代替訳 (例: 必須) を含む別案を必ず 1 件以上生やす\n- 「○○とは」「○○について」など説明を求める質問は、説明対象の語そのもの (例: 「IDリクワイアド」) を残した短いキーワード列も 1 件含める\n- 質問に答えてはならない。検索クエリのみを出力する\n- 出力は \"1. クエリ\\n2. クエリ\" 形式のみ。前置きや解説は書かない";

/// caption hints が設定されているとき、SYSTEM_PROMPT に追記する 1 行制約。
const CAPTION_HINT_SYSTEM_LINE: &str =
    "- 資料中の見出し (caption) と語彙が重なる言い換えを優先する。caption と全く語彙が重ならない言い換えは出さない";

/// caption hints のサンプル上限。多すぎると prompt が膨らんで LLM の指示追従が落ちる
/// (gemma-4-E4B IQ4_XS で経験的に 16〜32 件が上限)。
pub const MAX_CAPTION_HINTS: usize = 24;

pub struct LlmRewriter<L: LlmBackend> {
    llm: L,
    /// 言い換え生成時の温度。多様性を出すため retrieve 本体より高めにする想定。
    pub temperature: f32,
    /// LLM 出力の上限トークン。短いリストで足りるので控えめに。
    pub max_tokens: u32,
    /// 資料 (corpus) に登場する見出し / caption のサンプル。設定すると prompt 中に
    /// "資料中の見出し: ..." として注入され、LLM が **corpus 語彙に寄った** variant を
    /// 出すように誘導する (Run 33 displacement 仮説のフォロー / Run 37)。
    /// 空のときは挙動が従来と同じ。`MAX_CAPTION_HINTS` を超える場合は先頭から truncate。
    caption_hints: Vec<String>,
}

impl<L: LlmBackend> LlmRewriter<L> {
    pub fn new(llm: L) -> Self {
        Self {
            llm,
            temperature: 0.7,
            max_tokens: 256,
            caption_hints: Vec::new(),
        }
    }

    /// corpus 由来の caption / heading サンプルをセットして、prompt 注入を有効化する。
    /// 空 vec を渡すと caption-aware モードを OFF にできる。
    pub fn with_caption_hints(mut self, mut hints: Vec<String>) -> Self {
        hints.retain(|h| !h.trim().is_empty());
        if hints.len() > MAX_CAPTION_HINTS {
            hints.truncate(MAX_CAPTION_HINTS);
        }
        self.caption_hints = hints;
        self
    }

    /// 現在設定されている caption hints の数。テスト / 監査用途。
    pub fn caption_hints_len(&self) -> usize {
        self.caption_hints.len()
    }
}

#[async_trait]
impl<L: LlmBackend> QueryRewriter for LlmRewriter<L> {
    async fn rewrite(&self, query: &str, max_variants: usize) -> Result<RewrittenQueries> {
        if max_variants == 0 || query.trim().is_empty() {
            return Ok(RewrittenQueries::just(query));
        }

        let mut system = SYSTEM_PROMPT.replace('N', &max_variants.to_string());
        let mut user = format!(
            "質問: {q}\n\n上記を最大 {n} 個の検索クエリに書き換えてください。\n出力例:\n1. ...\n2. ...",
            q = query,
            n = max_variants
        );
        if !self.caption_hints.is_empty() {
            system.push('\n');
            system.push_str(CAPTION_HINT_SYSTEM_LINE);
            // user 側にも具体例を出す。LLM が hints を「使ってよい例」と認識しやすい。
            user = format!(
                "資料中の主な見出し:\n{hints}\n\n{rest}",
                hints = self
                    .caption_hints
                    .iter()
                    .map(|h| format!("- {h}"))
                    .collect::<Vec<_>>()
                    .join("\n"),
                rest = user
            );
        }
        let req = LlmRequest {
            system,
            history: Vec::new(),
            user,
            max_tokens: self.max_tokens,
            temperature: self.temperature,
        };

        let buf: Arc<Mutex<String>> = Arc::new(Mutex::new(String::new()));
        let buf2 = buf.clone();
        let cb: Box<dyn FnMut(String) + Send + 'static> =
            Box::new(move |t: String| buf2.lock().unwrap().push_str(&t));
        // LLM 失敗時は元クエリのみで返す (retrieve を止めない)。
        if self.llm.generate_stream(req, cb).await.is_err() {
            return Ok(RewrittenQueries::just(query));
        }

        let raw = buf.lock().unwrap().clone();
        let mut variants = parse_numbered_list(&raw);
        variants.truncate(max_variants);
        Ok(RewrittenQueries {
            original: query.to_string(),
            variants,
        })
    }
}

/// `1. foo\n2) bar\n- baz` のような行を抽出する寛容なパーサ。
///
/// 受理:
/// - `^\s*\d+[.)、:：]\s*` で始まる行
/// - `^\s*[-*・]\s*` で始まる行 (small LLM が混ぜがち)
///
/// 落とす:
/// - 空行 / 上記いずれにもマッチしない行 (前置き・解説とみなす)
pub fn parse_numbered_list(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    for raw_line in text.lines() {
        let line = raw_line.trim();
        if line.is_empty() {
            continue;
        }
        let stripped = strip_list_prefix(line);
        let Some(s) = stripped else { continue };
        let cleaned = s
            .trim()
            .trim_matches(|c: char| c == '"' || c == '「' || c == '」' || c == '『' || c == '』');
        if cleaned.is_empty() {
            continue;
        }
        out.push(cleaned.to_string());
    }
    out
}

fn strip_list_prefix(line: &str) -> Option<&str> {
    let bytes = line.as_bytes();
    // bullet: -, *, ・ (multi-byte)
    if let Some(rest) = line.strip_prefix("- ").or_else(|| line.strip_prefix("* ")) {
        return Some(rest);
    }
    if let Some(rest) = line.strip_prefix("・") {
        return Some(rest.trim_start());
    }
    // numbered: leading digits
    let mut i = 0;
    while i < bytes.len() && bytes[i].is_ascii_digit() {
        i += 1;
    }
    if i == 0 {
        return None;
    }
    let after_digits = &line[i..];
    // separator: . ) : 、 ：
    for sep in [".", ")", ":", "、", "：", "．"] {
        if let Some(rest) = after_digits.strip_prefix(sep) {
            return Some(rest.trim_start());
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use ellisii_core::Result;
    use ellisii_llm_core::{LlmBackend, LlmRequest};

    /// 決め打ちの応答を返すテスト用 LLM。
    struct ScriptedLlm {
        out: String,
    }

    #[async_trait]
    impl LlmBackend for ScriptedLlm {
        async fn generate_stream(
            &self,
            _req: LlmRequest,
            mut on_token: Box<dyn FnMut(String) + Send + 'static>,
        ) -> Result<()> {
            on_token(self.out.clone());
            Ok(())
        }
    }

    /// 常にエラーを返す LLM (failure path 検証用)。
    struct FailingLlm;
    #[async_trait]
    impl LlmBackend for FailingLlm {
        async fn generate_stream(
            &self,
            _req: LlmRequest,
            _on_token: Box<dyn FnMut(String) + Send + 'static>,
        ) -> Result<()> {
            Err(ellisii_core::Error::Llm("boom".into()))
        }
    }

    #[test]
    fn parse_handles_dot_paren_japanese_separators() {
        let s = "前置きです\n1. 猫\n2) ねこ\n3、ネコ\n4：feline";
        assert_eq!(parse_numbered_list(s), vec!["猫", "ねこ", "ネコ", "feline"]);
    }

    #[test]
    fn parse_handles_bullets_and_strips_quotes() {
        let s = "- 「契約解除」\n* 解除権\n・債務不履行";
        assert_eq!(
            parse_numbered_list(s),
            vec!["契約解除", "解除権", "債務不履行"]
        );
    }

    #[test]
    fn parse_drops_blank_and_non_list_lines() {
        let s = "回答します:\n\n1. one\nこの行は捨てる\n2. two\n";
        assert_eq!(parse_numbered_list(s), vec!["one", "two"]);
    }

    /// LlmRequest の system / user を Mutex 越しに記録する LLM。
    struct CapturingLlm {
        out: String,
        captured: Arc<Mutex<Option<LlmRequest>>>,
    }

    #[async_trait]
    impl LlmBackend for CapturingLlm {
        async fn generate_stream(
            &self,
            req: LlmRequest,
            mut on_token: Box<dyn FnMut(String) + Send + 'static>,
        ) -> Result<()> {
            *self.captured.lock().unwrap() = Some(req);
            on_token(self.out.clone());
            Ok(())
        }
    }

    #[tokio::test]
    async fn caption_hints_inject_into_system_and_user_prompt() {
        let captured: Arc<Mutex<Option<LlmRequest>>> = Arc::new(Mutex::new(None));
        let llm = CapturingLlm {
            out: "1. 入湯税\n2. 温泉税".into(),
            captured: captured.clone(),
        };
        let r = LlmRewriter::new(llm)
            .with_caption_hints(vec!["入湯税の税率".into(), "市たばこ税".into()])
            .rewrite("温泉に入ったときに課される税の額", 3)
            .await
            .unwrap();
        assert!(!r.variants.is_empty());

        let req = captured
            .lock()
            .unwrap()
            .clone()
            .expect("LLM was not called");
        assert!(
            req.system.contains("見出し") && req.system.contains("優先"),
            "system prompt missing caption-hint instruction: {}",
            req.system
        );
        assert!(
            req.user.contains("入湯税の税率") && req.user.contains("市たばこ税"),
            "user prompt missing caption hint samples: {}",
            req.user
        );
    }

    #[tokio::test]
    async fn caption_hints_empty_means_no_injection() {
        let captured: Arc<Mutex<Option<LlmRequest>>> = Arc::new(Mutex::new(None));
        let llm = CapturingLlm {
            out: "1. x".into(),
            captured: captured.clone(),
        };
        let _ = LlmRewriter::new(llm)
            .with_caption_hints(vec![])
            .rewrite("foo", 2)
            .await
            .unwrap();
        let req = captured
            .lock()
            .unwrap()
            .clone()
            .expect("LLM was not called");
        assert!(
            !req.system.contains("見出し"),
            "unexpected hint line in system: {}",
            req.system
        );
        assert!(
            !req.user.contains("資料中の主な見出し"),
            "unexpected hint block in user: {}",
            req.user
        );
    }

    #[test]
    fn caption_hints_truncates_at_max() {
        let llm = ScriptedLlm { out: "1. x".into() };
        let many: Vec<String> = (0..(MAX_CAPTION_HINTS + 8))
            .map(|i| format!("見出し{i}"))
            .collect();
        let r = LlmRewriter::new(llm).with_caption_hints(many);
        assert_eq!(r.caption_hints_len(), MAX_CAPTION_HINTS);
    }

    #[test]
    fn caption_hints_drops_blank_entries() {
        let llm = ScriptedLlm { out: "1. x".into() };
        let r = LlmRewriter::new(llm).with_caption_hints(vec![
            "a".into(),
            "  ".into(),
            "".into(),
            "b".into(),
        ]);
        assert_eq!(r.caption_hints_len(), 2);
    }

    #[tokio::test]
    async fn llm_rewriter_returns_truncated_variants() {
        let llm = ScriptedLlm {
            out: "1. 猫\n2. ねこ\n3. ネコ\n4. feline".into(),
        };
        let r = LlmRewriter::new(llm).rewrite("猫", 2).await.unwrap();
        assert_eq!(r.original, "猫");
        assert_eq!(r.variants, vec!["猫", "ねこ"]);
        // all() が原文一致を除去するので "猫" は1回だけ出る
        assert_eq!(r.all(), vec!["猫", "ねこ"]);
    }

    #[tokio::test]
    async fn llm_rewriter_zero_max_returns_passthrough() {
        let llm = ScriptedLlm {
            out: "1. 余計な書き換え".into(),
        };
        let r = LlmRewriter::new(llm).rewrite("q", 0).await.unwrap();
        assert_eq!(r, RewrittenQueries::just("q"));
    }

    #[tokio::test]
    async fn llm_rewriter_empty_query_returns_passthrough() {
        let llm = ScriptedLlm { out: "1. x".into() };
        let r = LlmRewriter::new(llm).rewrite("   ", 3).await.unwrap();
        assert_eq!(r, RewrittenQueries::just("   "));
    }

    #[tokio::test]
    async fn llm_rewriter_falls_back_on_llm_failure() {
        let r = LlmRewriter::new(FailingLlm).rewrite("猫", 3).await.unwrap();
        assert_eq!(r, RewrittenQueries::just("猫"));
    }

    #[tokio::test]
    async fn llm_rewriter_falls_back_on_garbage_output() {
        let llm = ScriptedLlm {
            out: "申し訳ありません、回答できません".into(),
        };
        let r = LlmRewriter::new(llm).rewrite("猫", 3).await.unwrap();
        // 番号リストが見つからない → variants 空
        assert!(r.variants.is_empty());
        assert_eq!(r.all(), vec!["猫"]);
    }
}
