//! クエリ意図 (intent) を LLM で分類してルーティングするための型と関数群。
//!
//! retrieval 戦略は intent ごとに大きく変わる:
//! - [`Intent::Summary`]   → store の `representative_chunks` を直接ロード
//! - [`Intent::Lookup`]    → 通常の hybrid retrieve + CE rerank
//! - [`Intent::Compare`]   → source 別の representative + 比較 prompt
//! - [`Intent::Smalltalk`] → general mode (RAG を発火させない)
//!
//! keyword fast-path で確信度の高いクエリは即決し、それ以外を
//! [`IntentClassifier::classify`] (LLM) に投げる二段構成を想定。

use async_trait::async_trait;
use ellisii_core::Result;
use ellisii_llm_core::{LlmBackend, LlmRequest};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "lowercase")]
pub enum Intent {
    /// 文書全体 / または `topic` に絞った要約・概要・全体像を求めるクエリ。
    Summary {
        /// 「民法の物権を要約して」の `物権` のような主題フィルタ。
        /// `None` は「文書全体」を意味する。
        topic: Option<String>,
    },
    /// 具体的な検索 (条文番号、固有名詞、特定事実)。
    Lookup,
    /// 複数文書の比較。
    Compare {
        /// 比較対象として指定された source 名 / タイトル断片。
        /// 空 = 「scope 内の全 source を対象」とみなす。
        sources: Vec<String>,
    },
    /// 挨拶・短い相槌など RAG 不要。
    Smalltalk,
}

impl Intent {
    pub fn lookup() -> Self {
        Intent::Lookup
    }
    pub fn smalltalk() -> Self {
        Intent::Smalltalk
    }
    pub fn summary_all() -> Self {
        Intent::Summary { topic: None }
    }
    pub fn summary_topic(topic: impl Into<String>) -> Self {
        Intent::Summary {
            topic: Some(topic.into()),
        }
    }
}

/// クエリ → Intent への分類器。
#[async_trait]
pub trait IntentClassifier: Send + Sync {
    async fn classify(&self, query: &str) -> Result<Intent>;
}

/// LLM を呼ぶ前のフィルタ。**確信度が極めて高い** ケースだけ即決し、それ以外は
/// `None` を返して呼び出し元に「LLM で判定してくれ」と委ねる。
///
/// 即決するのは:
/// - 空 / 数文字の挨拶系 → [`Intent::Smalltalk`]
/// - 条文番号 (`第N条` / `Article N`) を含むクエリ → [`Intent::Lookup`]
///   (要約 / 比較ではなく、ピンポイントな参照と確実に言える)
///
/// 要約 / 主題抽出は LLM の方が信頼できるので、ここではフォールスルー。
pub fn fast_path(query: &str) -> Option<Intent> {
    let q = query.trim();
    if q.is_empty() {
        return Some(Intent::Smalltalk);
    }
    let n = q.chars().count();
    let has_kanji_or_digit = q.chars().any(|c| {
        matches!(
            c,
            '\u{4E00}'..='\u{9FFF}' | '0'..='9' | '０'..='９'
        )
    });
    // 漢字 / 数字が無く 8 文字以下 → 高確度で smalltalk (Hello / はい / ok 等)
    if !has_kanji_or_digit && n <= 8 {
        return Some(Intent::Smalltalk);
    }
    if contains_article_id(q) {
        return Some(Intent::Lookup);
    }
    None
}

/// LLM バックエンドを叩いて [`Intent`] を取り出す classifier。
///
/// `fast_path` で確信度の低かったクエリだけを LLM に投げる前提。
/// 出力 parse に失敗したら [`Intent::Lookup`] にフォールバックする
/// (= 安全側: 通常の hybrid retrieve に倒す)。
pub struct LlmIntentClassifier {
    llm: Arc<dyn LlmBackend>,
}

impl LlmIntentClassifier {
    pub fn new(llm: Arc<dyn LlmBackend>) -> Self {
        Self { llm }
    }

    fn system_prompt() -> String {
        "あなたはユーザの質問の意図を分類するアシスタントです。\n\
         入力された日本語または英語のクエリに対し、以下のフォーマットで **3 行だけ** 出力します。\n\
         \n\
         INTENT: <summary | lookup | compare | smalltalk のいずれか 1 つ>\n\
         TOPIC: <主題の名詞句 / 全体なら all / 該当なしなら n/a>\n\
         SOURCES: <比較対象 source 名のカンマ区切り / 該当なしなら n/a>\n\
         \n\
         判定ルール:\n\
         - summary: 文書全体や特定の章 / 編 / 主題を **要約 / 概要 / 全体像 / 主要点** で求めるクエリ\n\
         - lookup: 条文番号 / 用語の定義 / 特定事実など、**ピンポイント** な検索クエリ\n\
         - compare: 複数の文書 / 章 / 条項の **違い・対比** を求めるクエリ\n\
         - smalltalk: 挨拶 / 相槌 / 雑談で RAG 不要のもの\n\
         \n\
         TOPIC の指定方針:\n\
         - TOPIC は文書 **内部** の章節 / 主題 / 観点を指す名詞句のみ。\n\
         - 文書全体が対象なら \"all\" (= 文書名 / 略称 / \"これ\" \"この本\" \"全体\" 等で文書を指しているだけの場合を含む)。\n\
         - lookup や smalltalk のように topic 概念が無い意図では \"n/a\"。\n\
         判定の型 (汎用):\n\
         - 「<文書> の <Y> を要約 / <Y> について教えて」のように **文書内の Y を切り出す** 形 → TOPIC = Y\n\
         - 「<文書> を要約 / 全体を要約 / まとめて」のように **絞り込みが無い** 形 → TOPIC = all\n\
         SOURCES の例: \"契約書AとBの違いは\" → 契約書A, 契約書B、 それ以外 → n/a\n\
         \n\
         前置きや説明を一切含めず、上記 3 行だけを出力すること。"
            .to_string()
    }
}

#[async_trait]
impl IntentClassifier for LlmIntentClassifier {
    async fn classify(&self, query: &str) -> Result<Intent> {
        if let Some(intent) = fast_path(query) {
            return Ok(intent);
        }
        use std::sync::Mutex as StdMutex;
        let req = LlmRequest {
            system: Self::system_prompt(),
            history: Vec::new(),
            user: format!("質問: {query}"),
            max_tokens: 80,
            temperature: 0.0,
        };
        let buf: Arc<StdMutex<String>> = Arc::new(StdMutex::new(String::new()));
        let buf_cb = buf.clone();
        let cb: Box<dyn FnMut(String) + Send + 'static> = Box::new(move |tok: String| {
            if let Ok(mut g) = buf_cb.lock() {
                g.push_str(&tok);
            }
        });
        // LLM 呼び出しが失敗しても致命的ではない。安全側 (Lookup) に倒して
        // 通常の hybrid retrieve に流す。
        if self.llm.generate_stream(req, cb).await.is_err() {
            return Ok(Intent::Lookup);
        }
        let raw = buf.lock().map(|g| g.clone()).unwrap_or_default();
        Ok(parse_intent(&raw))
    }
}

/// 任意の [`IntentClassifier`] を FIFO キャッシュでラップする。
///
/// 同じクエリを再分類すると LLM 呼び出しが二重になり遅延・電力的に勿体ないので
/// 文字列キーで結果を覚える。サイズ上限 (`cap`) を超えたら古い順に追い出す。
///
/// LRU ではなく純粋 FIFO。多くのチャット文脈ではユニークなクエリが線形に増える
/// だけなので、ヒット率は LRU/FIFO で大きく変わらず実装が単純な方を選ぶ。
pub struct CachingClassifier<C: IntentClassifier> {
    inner: C,
    cache: std::sync::Mutex<FifoCache>,
}

struct FifoCache {
    map: std::collections::HashMap<String, Intent>,
    order: std::collections::VecDeque<String>,
    cap: usize,
}

impl<C: IntentClassifier> CachingClassifier<C> {
    pub fn new(inner: C, cap: usize) -> Self {
        Self {
            inner,
            cache: std::sync::Mutex::new(FifoCache {
                map: std::collections::HashMap::new(),
                order: std::collections::VecDeque::new(),
                cap: cap.max(1),
            }),
        }
    }

    /// 現在のキャッシュエントリ数 (test 用)。
    pub fn len(&self) -> usize {
        self.cache.lock().map(|g| g.map.len()).unwrap_or(0)
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

#[async_trait]
impl<C: IntentClassifier> IntentClassifier for CachingClassifier<C> {
    async fn classify(&self, query: &str) -> Result<Intent> {
        if let Ok(g) = self.cache.lock() {
            if let Some(hit) = g.map.get(query) {
                return Ok(hit.clone());
            }
        }
        let intent = self.inner.classify(query).await?;
        if let Ok(mut g) = self.cache.lock() {
            // 二重 insert を避ける (race で別スレッドが先に入れた可能性)
            if !g.map.contains_key(query) {
                while g.map.len() >= g.cap {
                    if let Some(oldest) = g.order.pop_front() {
                        g.map.remove(&oldest);
                    } else {
                        break;
                    }
                }
                g.order.push_back(query.to_string());
                g.map.insert(query.to_string(), intent.clone());
            }
        }
        Ok(intent)
    }
}

/// `第N条` / `第N条の…` / `Article N` パターンを含むかどうかの軽量チェック。
fn contains_article_id(q: &str) -> bool {
    let chars: Vec<char> = q.chars().collect();
    // 「第」+ 数字 (半角/全角) + 「条」
    for i in 0..chars.len() {
        if chars[i] == '第' {
            let mut j = i + 1;
            let mut saw_digit = false;
            while j < chars.len() && (chars[j].is_ascii_digit() || matches!(chars[j], '０'..='９'))
            {
                saw_digit = true;
                j += 1;
            }
            if saw_digit && j < chars.len() && chars[j] == '条' {
                return true;
            }
        }
    }
    // "Article N" (英大文字始まり + 数字)
    let lower = q.to_ascii_lowercase();
    if let Some(pos) = lower.find("article ") {
        let rest = &lower[pos + "article ".len()..];
        if rest
            .chars()
            .next()
            .map(|c| c.is_ascii_digit())
            .unwrap_or(false)
        {
            return true;
        }
    }
    false
}

/// LLM の出力テキストを 3 行タグフォーマットで parse する。
///
/// 期待フォーマット:
/// ```text
/// INTENT: summary | lookup | compare | smalltalk
/// TOPIC: <主題の名詞句 | "all" | "n/a">
/// SOURCES: <比較対象 source のカンマ区切り | "n/a">
/// ```
///
/// - 大文字小文字は区別しない
/// - 行の前後空白は trim
/// - 余計な行 (```fence や preamble) があっても順序通りキーで拾えれば OK
/// - INTENT 行が無い / 未知のラベル → [`Intent::Lookup`] にフォールバック
pub fn parse_intent(text: &str) -> Intent {
    let mut intent_kind: Option<String> = None;
    let mut topic: Option<String> = None;
    let mut sources: Option<String> = None;
    for line in text.lines() {
        let line = line.trim();
        if let Some((key, value)) = line.split_once(':') {
            let k = key.trim().to_ascii_lowercase();
            let v = value.trim().to_string();
            match k.as_str() {
                "intent" => intent_kind = Some(v),
                "topic" => topic = Some(v),
                "sources" => sources = Some(v),
                _ => {}
            }
        }
    }
    let kind = match intent_kind
        .as_deref()
        .map(str::to_ascii_lowercase)
        .as_deref()
    {
        Some("summary") => "summary",
        Some("lookup") => "lookup",
        Some("compare") => "compare",
        Some("smalltalk") => "smalltalk",
        _ => return Intent::Lookup,
    };
    let topic_clean = topic.and_then(normalize_optional_field);
    let sources_clean = sources.and_then(normalize_optional_field);
    match kind {
        "summary" => Intent::Summary { topic: topic_clean },
        "lookup" => Intent::Lookup,
        "compare" => Intent::Compare {
            sources: sources_clean
                .map(|s| {
                    s.split(',')
                        .map(|x| x.trim().to_string())
                        .filter(|x| !x.is_empty())
                        .collect()
                })
                .unwrap_or_default(),
        },
        "smalltalk" => Intent::Smalltalk,
        _ => Intent::Lookup,
    }
}

/// `"all"` / `"n/a"` / 空文字 / 引用符だけ等を `None` に正規化する。
fn normalize_optional_field(raw: String) -> Option<String> {
    let trimmed = raw
        .trim()
        .trim_matches(|c: char| c == '"' || c == '\'' || c == '`' || c.is_whitespace());
    let lower = trimmed.to_ascii_lowercase();
    if trimmed.is_empty()
        || lower == "all"
        || lower == "n/a"
        || lower == "na"
        || lower == "none"
        || lower == "なし"
    {
        None
    } else {
        Some(trimmed.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_summary_with_topic() {
        let out = "INTENT: summary\nTOPIC: 物権\nSOURCES: n/a";
        assert_eq!(parse_intent(out), Intent::summary_topic("物権"));
    }

    #[test]
    fn parses_summary_all_normalizes_to_none() {
        let out = "INTENT: summary\nTOPIC: all\nSOURCES: n/a";
        assert_eq!(parse_intent(out), Intent::summary_all());
    }

    #[test]
    fn parses_lookup() {
        let out = "INTENT: lookup\nTOPIC: n/a\nSOURCES: n/a";
        assert_eq!(parse_intent(out), Intent::Lookup);
    }

    #[test]
    fn parses_smalltalk() {
        let out = "INTENT: smalltalk\nTOPIC: n/a\nSOURCES: n/a";
        assert_eq!(parse_intent(out), Intent::Smalltalk);
    }

    #[test]
    fn parses_compare_with_multiple_sources() {
        let out = "INTENT: compare\nTOPIC: n/a\nSOURCES: 契約書A, 契約書B";
        assert_eq!(
            parse_intent(out),
            Intent::Compare {
                sources: vec!["契約書A".into(), "契約書B".into()]
            }
        );
    }

    #[test]
    fn parses_with_extra_preamble_and_fences() {
        let out = "```\n以下が分類結果です。\nINTENT: Summary\nTOPIC: 相続\nSOURCES: n/a\n```";
        assert_eq!(parse_intent(out), Intent::summary_topic("相続"));
    }

    #[test]
    fn falls_back_to_lookup_on_unknown_label() {
        let out = "INTENT: tangent\nTOPIC: n/a";
        assert_eq!(parse_intent(out), Intent::Lookup);
    }

    #[test]
    fn falls_back_to_lookup_on_garbage() {
        let out = "Sorry I can't help with that.";
        assert_eq!(parse_intent(out), Intent::Lookup);
    }

    #[test]
    fn topic_strips_quotes_and_whitespace() {
        let out = "INTENT: summary\nTOPIC: \"物権\"  \nSOURCES: n/a";
        assert_eq!(parse_intent(out), Intent::summary_topic("物権"));
    }

    #[test]
    fn topic_normalizes_japanese_none() {
        let out = "INTENT: summary\nTOPIC: なし\nSOURCES: n/a";
        assert_eq!(parse_intent(out), Intent::summary_all());
    }

    #[test]
    fn case_insensitive_keys() {
        let out = "intent: summary\ntopic: 物権";
        assert_eq!(parse_intent(out), Intent::summary_topic("物権"));
    }

    #[test]
    fn fast_path_smalltalk_for_short_kana() {
        assert_eq!(fast_path(""), Some(Intent::Smalltalk));
        assert_eq!(fast_path("Hello"), Some(Intent::Smalltalk));
        assert_eq!(fast_path("はい"), Some(Intent::Smalltalk));
        assert_eq!(fast_path("ok"), Some(Intent::Smalltalk));
    }

    #[test]
    fn fast_path_lookup_for_article_id() {
        assert_eq!(fast_path("第94条は？"), Some(Intent::Lookup));
        assert_eq!(fast_path("第１２条"), Some(Intent::Lookup));
        assert_eq!(fast_path("Article 5 indemnification"), Some(Intent::Lookup));
    }

    #[test]
    fn fast_path_delegates_summary_to_llm() {
        // 主題抽出のため LLM に投げる
        assert_eq!(fast_path("民法を要約して"), None);
        assert_eq!(fast_path("民法の物権を要約して"), None);
    }

    #[test]
    fn fast_path_delegates_general_questions() {
        assert_eq!(fast_path("通謀虚偽表示とは何か"), None);
        assert_eq!(fast_path("善意の第三者の保護要件"), None);
    }

    /// 1 つの canned レスポンスをそのまま返すだけの test 用 LLM。
    struct CannedLlm {
        response: String,
    }

    #[async_trait]
    impl LlmBackend for CannedLlm {
        async fn generate_stream(
            &self,
            _req: LlmRequest,
            mut on_token: Box<dyn FnMut(String) + Send + 'static>,
        ) -> Result<()> {
            on_token(self.response.clone());
            Ok(())
        }
    }

    /// 必ず Err を返す壊れた LLM (フォールバック検証用)。
    struct FailingLlm;

    #[async_trait]
    impl LlmBackend for FailingLlm {
        async fn generate_stream(
            &self,
            _req: LlmRequest,
            _on_token: Box<dyn FnMut(String) + Send + 'static>,
        ) -> Result<()> {
            Err(ellisii_core::Error::Llm("test failure".into()))
        }
    }

    #[tokio::test]
    async fn llm_classifier_fast_path_short_circuits_smalltalk() {
        // LLM が壊れていても smalltalk は fast_path で即決される
        let c = LlmIntentClassifier::new(Arc::new(FailingLlm));
        assert_eq!(c.classify("Hello").await.unwrap(), Intent::Smalltalk);
    }

    #[tokio::test]
    async fn llm_classifier_fast_path_short_circuits_article_id() {
        let c = LlmIntentClassifier::new(Arc::new(FailingLlm));
        assert_eq!(c.classify("第94条は？").await.unwrap(), Intent::Lookup);
    }

    #[tokio::test]
    async fn llm_classifier_parses_summary_with_topic() {
        let llm = Arc::new(CannedLlm {
            response: "INTENT: summary\nTOPIC: 物権\nSOURCES: n/a".into(),
        });
        let c = LlmIntentClassifier::new(llm);
        assert_eq!(
            c.classify("民法の物権を要約して").await.unwrap(),
            Intent::summary_topic("物権")
        );
    }

    #[tokio::test]
    async fn llm_classifier_falls_back_to_lookup_on_llm_error() {
        // fast_path にかからない長いクエリで LLM が失敗 → Lookup
        let c = LlmIntentClassifier::new(Arc::new(FailingLlm));
        assert_eq!(
            c.classify("通謀虚偽表示とは何か").await.unwrap(),
            Intent::Lookup
        );
    }

    /// 呼び出し回数を数えるだけの test 用 classifier。
    struct CountingClassifier {
        calls: std::sync::atomic::AtomicUsize,
        next: Intent,
    }

    impl CountingClassifier {
        fn new(next: Intent) -> Self {
            Self {
                calls: std::sync::atomic::AtomicUsize::new(0),
                next,
            }
        }
        fn calls(&self) -> usize {
            self.calls.load(std::sync::atomic::Ordering::Relaxed)
        }
    }

    #[async_trait]
    impl IntentClassifier for CountingClassifier {
        async fn classify(&self, _query: &str) -> Result<Intent> {
            self.calls
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            Ok(self.next.clone())
        }
    }

    #[tokio::test]
    async fn caching_classifier_returns_inner_result_on_miss() {
        let inner = CountingClassifier::new(Intent::summary_topic("物権"));
        let c = CachingClassifier::new(inner, 4);
        assert_eq!(
            c.classify("民法の物権を要約して").await.unwrap(),
            Intent::summary_topic("物権")
        );
        assert_eq!(c.len(), 1);
    }

    #[tokio::test]
    async fn caching_classifier_does_not_call_inner_on_hit() {
        // CountingClassifier::calls を直接見たいので Arc 経由で参照を握る
        struct Wrap(Arc<CountingClassifier>);
        #[async_trait]
        impl IntentClassifier for Wrap {
            async fn classify(&self, q: &str) -> Result<Intent> {
                self.0.classify(q).await
            }
        }
        let counter = Arc::new(CountingClassifier::new(Intent::Lookup));
        let c = CachingClassifier::new(Wrap(counter.clone()), 4);
        let _ = c.classify("通謀虚偽表示とは").await.unwrap();
        let _ = c.classify("通謀虚偽表示とは").await.unwrap();
        let _ = c.classify("通謀虚偽表示とは").await.unwrap();
        assert_eq!(counter.calls(), 1, "後続 2 回はキャッシュから返すべき");
    }

    #[tokio::test]
    async fn caching_classifier_evicts_oldest_when_full() {
        let inner = CountingClassifier::new(Intent::Lookup);
        let c = CachingClassifier::new(inner, 2);
        let _ = c.classify("q1").await.unwrap();
        let _ = c.classify("q2").await.unwrap();
        let _ = c.classify("q3").await.unwrap(); // q1 が追い出される
        assert_eq!(c.len(), 2);
        // q1 が無くなっていることは外から直接確認できないが、再 classify で
        // 内部 classifier の呼び出し回数が増えれば cache miss = 追い出し済みと分かる
    }
}
