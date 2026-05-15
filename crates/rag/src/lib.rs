use ellisii_core::{is_retrieval_noise, HitSource, Result, SearchHit};
use ellisii_store_core::Scope;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct HybridWeights {
    /// 0.0=完全にキーワード, 1.0=完全にベクトル, 0.5=均等。
    /// 既定値は 0.75 で、vector 寄りに重みを置く。
    pub semantic: f32,
}

impl Default for HybridWeights {
    /// 既定値は `semantic = 0.75`。
    ///
    /// 根拠: `embed-static-jp` (1024dim) を用いた sqlite 経路の実測で、
    /// 民法 50 query / CS Wiki 42 query どちらでも `0.75` が MRR / nDCG を
    /// 最大化したため (rag-eval-cli の `tests/real_static_jp.rs`)。
    /// 単純な等重み (0.5) よりも vector 経路を優先しつつ、keyword 経路
    /// (FTS5+BM25+CharBigram) のヒットも 1/4 残すバランス。
    fn default() -> Self {
        Self { semantic: 0.75 }
    }
}

impl HybridWeights {
    pub fn keyword(&self) -> f32 {
        1.0 - self.semantic.clamp(0.0, 1.0)
    }
    pub fn vector(&self) -> f32 {
        self.semantic.clamp(0.0, 1.0)
    }
}

/// クエリの文字種比率を見て semantic / lexical の重みを **±0.2** ほど自動調整する。
///
/// - identifier 系 (漢字 / 数字 / 英数字 / 全角数字) が多いクエリは **lexical 寄り**
///   (条文番号や ID で探すケース。BM25 の方が当たりやすい)
/// - ひらがな / カタカナ主体の抽象クエリは **semantic 寄り**
///   (「効果」「定義」のような短い概念語は dense embedding の方が拾える)
///
/// 戻り値は `[0.05, 0.95]` の semantic 比重。`base_semantic` から ±0.2 の範囲に
/// クランプされる。文字数 4 未満のクエリは `base_semantic` をそのまま返す
/// (識別力が無いので調整しない)。
///
/// 由来: `src-tauri::adjust_hybrid_weight_for_query` を抽出した純粋関数版。
/// テストしやすく、SDK / src-tauri / rag-eval-cli から共通参照できる。
pub fn adjust_hybrid_weight_for_query(base_semantic: f32, query: &str) -> f32 {
    if query.chars().count() < 4 {
        return base_semantic;
    }
    let mut total = 0usize;
    let mut hiragana_kata = 0usize;
    let mut kanji_or_id = 0usize;
    for c in query.chars() {
        if c.is_whitespace() {
            continue;
        }
        total += 1;
        if matches!(c, '\u{3040}'..='\u{30FF}') {
            hiragana_kata += 1;
        } else if matches!(
            c,
            '\u{4E00}'..='\u{9FFF}' | '0'..='9' | 'a'..='z' | 'A'..='Z' | '０'..='９'
        ) {
            kanji_or_id += 1;
        }
    }
    if total == 0 {
        return base_semantic;
    }
    let hk_ratio = hiragana_kata as f32 / total as f32;
    let id_ratio = kanji_or_id as f32 / total as f32;
    let delta = (hk_ratio - id_ratio).clamp(-1.0, 1.0) * 0.2;
    (base_semantic + delta).clamp(0.05, 0.95)
}
use ellisii_embed_core::Embedder;
use ellisii_llm_core::{LlmBackend, LlmRequest};
use ellisii_query_rewriter_core::QueryRewriter;
use ellisii_store_core::VectorStore;

/// クエリが既に十分 specific (= 条文番号 / 引用 / URL / コードスニペット / 50 文字以上の長文) かを
/// 判定する。`true` のとき、上位レイヤは LLM rewriter (paraphrase / HyDE) を skip するのが
/// 安全 (Run 11 / 9 で「specific クエリで rewriter が誤爆 / latency だけ食う」事例を観測)。
///
/// 由来: `src-tauri::looks_specific_query` を rag crate の pub 関数として抽出。
/// `crates/rag-eval-cli/tests/validate_router.rs` で各 corpus の rewriter ROI を見て
/// チューニングされたヒューリスティック。
///
/// 受理 (= specific と判定):
/// - `第N条` / `Article N` / `Section N` / `Sec. N` を含む (数字は半角・全角・漢数字いずれも)
/// - `"…"` / `「…」` / `『…』` / `“…”` で 2 文字以上を引用
/// - URL / メールアドレス
/// - コードスニペット (uppercase 連 / SQL keyword / snake_case / CamelCase / 記号塊)
/// - 50 文字以上の長文
///
/// 受理しない (= rewriter で expand 推奨):
/// - 日本語の自然文 (ひらがな・短い漢字混在)
/// - 概念質問 (「○○とは」「○○の効果」など)
pub fn is_specific_query(query: &str) -> bool {
    use std::sync::OnceLock;
    static ARTICLE_RE: OnceLock<regex::Regex> = OnceLock::new();
    static QUOTE_RE: OnceLock<regex::Regex> = OnceLock::new();
    static URLISH_RE: OnceLock<regex::Regex> = OnceLock::new();

    let article_re = ARTICLE_RE.get_or_init(|| {
        regex::Regex::new(
            r"第[0-9０-９一二三四五六七八九十百千]+条|Article\s+[0-9IVXLCDM]+|Section\s+[0-9]+|Sec\.\s*[0-9]+",
        )
        .expect("article regex")
    });
    if article_re.is_match(query) {
        return true;
    }
    let quote_re = QUOTE_RE.get_or_init(|| {
        regex::Regex::new(r#""[^"]{2,}"|「[^」]{2,}」|『[^』]{2,}』|“[^”]{2,}”"#)
            .expect("quote regex")
    });
    if quote_re.is_match(query) {
        return true;
    }
    let urlish_re = URLISH_RE
        .get_or_init(|| regex::Regex::new(r"https?://|[\w.+-]+@[\w-]+\.[\w.-]+").expect("urlish"));
    if urlish_re.is_match(query) {
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

/// 各クエリについて「char-bigram のうち corpus body のどれかにそのまま出現する割合」
/// (= query-side recall) の最大値を取り、クエリ集合全体で平均する。
///
/// 解釈:
/// - **高い (≥ 0.7)**: クエリ語彙が既に corpus body に literal に出現する
///   → paraphrase rewrite で recall が伸びる余地は小さい (yokohama-style literal lookup)
/// - **低い (≤ 0.4)**: クエリと body の lexical gap が大きい
///   → paraphrase rewrite (LLM rewriter) で bridge する価値あり
///
/// 動機: Run 21 で `specific_query_ratio` だけでは捕捉できない false positive
/// (yokohama「税率はいくら」のような自然文 literal lookup) を query-vs-corpus 軸で
/// 補足する。`docs/eval/recall-evals.md` Run 22 を参照。
///
/// 計算量は O(Q * B * |body bigrams|)。production では body をサンプリングして渡す想定
/// (例: SDK 側で `all_captions` の対応 chunk 上限 256 件を使う)。
///
/// 空配列の取り扱い: queries が空なら `0.0`、bodies が空なら全クエリの max=0 で `0.0`。
pub fn query_body_recall_mean<Q: AsRef<str>, B: AsRef<str>>(queries: &[Q], bodies: &[B]) -> f32 {
    if queries.is_empty() {
        return 0.0;
    }
    let body_bgs: Vec<std::collections::HashSet<String>> =
        bodies.iter().map(|b| char_bigrams(b.as_ref())).collect();
    let mut sum = 0.0f32;
    for q in queries {
        let qb = char_bigrams(q.as_ref());
        if qb.is_empty() {
            continue;
        }
        let mut best = 0.0f32;
        for bb in &body_bgs {
            if bb.is_empty() {
                continue;
            }
            let inter = qb.intersection(bb).count() as f32;
            let recall = inter / qb.len() as f32;
            if recall > best {
                best = recall;
            }
        }
        sum += best;
    }
    sum / queries.len() as f32
}

/// 各クエリの char-bigram のうち、corpus の title (= heading_path[-1] / Markdown H1 など)
/// のいずれかに literal に出現する割合 (= query-side recall against titles) の最大値、
/// クエリ集合全体で平均する。
///
/// 解釈 (`docs/eval/recall-evals.md` Run 26):
/// - **高い (≥ 0.5)**: クエリがタイトル直接マッチ寄り (FAQ / 概念定義 lookup)
///   → `ChunkConfig::synthesize_caption_from_heading=true` が有効化候補。
///   Run 25 で jp-cs-wiki (easy) は MRR +16.7pt の大勝。
/// - **低い (≤ 0.3)**: クエリがタイトルから paraphrase / 概念ジャンプしている
///   → caption synthesis を入れると Run 25 の jp-cs-wiki-hard のように MRR 退行
///   する可能性。OFF のままが安全。
///
/// `query_body_recall_mean` (Run 22) との違い: あちらは body 全体への recall、
/// こちらはタイトル / 見出しに絞った recall。caption synthesis (Run 24) の有効化
/// 判断にはこちらの方が直接的なシグナル。
///
/// 空配列の取り扱い: queries 空 → `0.0`、titles 空 → 全クエリの max=0 で `0.0`。
pub fn query_title_match_mean<Q: AsRef<str>, T: AsRef<str>>(queries: &[Q], titles: &[T]) -> f32 {
    if queries.is_empty() {
        return 0.0;
    }
    let title_bgs: Vec<std::collections::HashSet<String>> =
        titles.iter().map(|t| char_bigrams(t.as_ref())).collect();
    let mut sum = 0.0f32;
    for q in queries {
        let qb = char_bigrams(q.as_ref());
        if qb.is_empty() {
            continue;
        }
        let mut best = 0.0f32;
        for tb in &title_bgs {
            if tb.is_empty() {
                continue;
            }
            let inter = qb.intersection(tb).count() as f32;
            let recall = inter / qb.len() as f32;
            if recall > best {
                best = recall;
            }
        }
        sum += best;
    }
    sum / queries.len() as f32
}

fn char_bigrams(s: &str) -> std::collections::HashSet<String> {
    let chars: Vec<char> = s.chars().filter(|c| !c.is_whitespace()).collect();
    let mut out = std::collections::HashSet::new();
    for w in chars.windows(2) {
        out.insert(w.iter().collect::<String>());
    }
    out
}

/// クエリ集合のうち `is_specific_query` 判定が `true` になる割合 (0.0..=1.0)。
///
/// 動機: corpus signal (`Ellisii::corpus_paraphrase_score`) は body 側の vocab richness
/// で、rewriter ROI を直接予測できないと Run 20 で判明した。代わりに **クエリ分布側の
/// シグナル** として、golden queries (or ユーザクエリログ) が specific = literal lookup
/// に偏っているかどうかを measure する。
///
/// 解釈 (`docs/eval/recall-evals.md` Run 21):
/// - `>= 0.5`: クエリ集合が specific 偏重 → rewriter は `is_specific_query` per-query
///   gate でほぼ skip され、ON にする利得は小さい。`multi_query_max_variants = 0` 既定で OK。
/// - `< 0.3`: クエリ集合が概念質問 / 自然文偏重 → rewriter ON で paraphrase 経由の
///   recall ゲインが期待できる。`multi_query_max_variants >= 1` で試す価値あり。
/// - `0.3..0.5`: ミックス。production では既定 (`skip_rewrite_on_specific=true` で per-query
///   判断) のまま動かしつつ、A/B で確認するのが安全。
///
/// 空配列は `0.0` を返す。
pub fn specific_query_ratio<S: AsRef<str>>(queries: &[S]) -> f32 {
    if queries.is_empty() {
        return 0.0;
    }
    let n = queries.len() as f32;
    let hits = queries
        .iter()
        .filter(|q| is_specific_query(q.as_ref()))
        .count() as f32;
    hits / n
}

/// SQL キーワード / 識別子 / 記号塊などの code-like なマーカーが含まれるか。
/// `is_specific_query` の内部判定で使う独立した heuristic。
pub fn has_code_snippet(query: &str) -> bool {
    let chars: Vec<char> = query.chars().collect();
    // (1) 大文字 ASCII / アンダースコアの連続 ≥4 (SELECT, ORDER, AUTO_INCREMENT 等)
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
    // (2) ASCII 記号 ()/{}/;/=/%/<> を 2 つ以上
    let punct = chars
        .iter()
        .filter(|c| matches!(c, '(' | ')' | '{' | '}' | ';' | '=' | '%' | '<' | '>'))
        .count();
    if punct >= 2 {
        return true;
    }
    // (3) snake_case 識別子 (≥4 chars でかつ _ を含む alphanumeric token)
    let mut tok_start: Option<usize> = None;
    for i in 0..chars.len() {
        let is_id_char = chars[i].is_ascii_alphanumeric() || chars[i] == '_';
        if is_id_char {
            if tok_start.is_none() {
                tok_start = Some(i);
            }
        } else if let Some(s) = tok_start.take() {
            let tok: String = chars[s..i].iter().collect();
            if tok.contains('_') && tok.chars().count() >= 4 {
                return true;
            }
        }
    }
    if let Some(s) = tok_start {
        let tok: String = chars[s..].iter().collect();
        if tok.contains('_') && tok.chars().count() >= 4 {
            return true;
        }
    }
    // (4) CamelCase 識別子 ≥5 chars
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

/// multi-query 検索のオプション。
#[derive(Debug, Clone, Copy)]
pub struct MultiQueryOptions {
    pub weights: HybridWeights,
    /// 元クエリに対する追加 variant 数の上限。0 で完全に Passthrough と等価。
    pub max_variants: usize,
    /// 各 variant ranking に掛ける重み (0.0〜1.0)。元クエリは常に 1.0。
    /// variant をやや低く扱うことで、元の意図を最優先しつつ recall を補完する。
    pub variant_weight: f32,
}

impl Default for MultiQueryOptions {
    fn default() -> Self {
        Self {
            weights: HybridWeights::default(),
            max_variants: 3,
            variant_weight: 0.7,
        }
    }
}

pub struct RagEngine<E: Embedder, S: VectorStore, L: LlmBackend> {
    pub embedder: E,
    pub store: S,
    pub llm: L,
}

impl<E: Embedder, S: VectorStore, L: LlmBackend> RagEngine<E, S, L> {
    pub async fn retrieve(
        &self,
        scope: Scope,
        query: &str,
        top_k: usize,
    ) -> Result<Vec<SearchHit>> {
        self.retrieve_weighted(scope, query, top_k, HybridWeights::default())
            .await
    }

    pub async fn retrieve_weighted(
        &self,
        scope: Scope,
        query: &str,
        top_k: usize,
        weights: HybridWeights,
    ) -> Result<Vec<SearchHit>> {
        let q_emb = self.embedder.embed(&[query.to_string()]).await?;
        let vec_hits = self.store.search(scope, &q_emb[0], top_k * 5).await?;
        let kw_hits = self.store.keyword_search(scope, query, top_k * 5).await?;
        Ok(rrf_weighted(
            &[(vec_hits, weights.vector()), (kw_hits, weights.keyword())],
            top_k,
        ))
    }

    /// QueryRewriter で生成した複数クエリで vec/kw 検索し、RRF で 1 つのランキングに融合する。
    ///
    /// - 元クエリの ranking には重み 1.0、各 variant には `opts.variant_weight` を掛ける。
    /// - vec と kw の比率は `opts.weights` (= 既存の HybridWeights) を流用。
    /// - rewriter が空 / 失敗時は元クエリのみで動くので、Passthrough を渡せば
    ///   `retrieve_weighted` と同等になる (回帰しない設計)。
    pub async fn retrieve_multi(
        &self,
        scope: Scope,
        query: &str,
        top_k: usize,
        rewriter: &dyn QueryRewriter,
        opts: MultiQueryOptions,
    ) -> Result<Vec<SearchHit>> {
        let rewritten = rewriter.rewrite(query, opts.max_variants).await?;
        let queries = rewritten.all();

        let w_vec = opts.weights.vector();
        let w_kw = opts.weights.keyword();
        let mut rankings: Vec<(Vec<SearchHit>, f32)> = Vec::with_capacity(queries.len() * 2);

        for (i, q) in queries.iter().enumerate() {
            let q_weight = if i == 0 { 1.0 } else { opts.variant_weight };
            let q_emb = self.embedder.embed(&[q.clone()]).await?;
            let vec_hits = self.store.search(scope, &q_emb[0], top_k * 5).await?;
            let kw_hits = self.store.keyword_search(scope, q, top_k * 5).await?;
            rankings.push((vec_hits, w_vec * q_weight));
            rankings.push((kw_hits, w_kw * q_weight));
        }

        Ok(rrf_weighted(&rankings, top_k))
    }

    pub async fn answer(
        &self,
        scope: Scope,
        query: &str,
        on_token: Box<dyn FnMut(String) + Send + 'static>,
    ) -> Result<Vec<SearchHit>> {
        let hits = self.retrieve(scope, query, 6).await?;
        let context = hits
            .iter()
            .enumerate()
            .map(|(i, h)| format!("<source id={}>{}</source>", i + 1, h.chunk.text))
            .collect::<Vec<_>>()
            .join("\n");
        let req = LlmRequest {
            system: "あなたは厳密な参考文献付きアシスタントです。<source>に無い情報は答えず、引用を [1][2] の形式で付けてください。".into(),
            history: Vec::new(),
            user: format!("質問: {query}\n\n参考:\n{context}"),
            max_tokens: 1024,
            temperature: 0.2,
        };
        self.llm.generate_stream(req, on_token).await?;
        Ok(hits)
    }
}

pub mod citation;
pub mod eval;
pub mod intent_classifier;
pub mod query_intent;
pub mod rerank;

/// 等重み版 (後方互換)
pub fn rrf(rankings: &[Vec<SearchHit>], top_k: usize) -> Vec<SearchHit> {
    let weighted: Vec<_> = rankings.iter().map(|r| (r.clone(), 1.0_f32)).collect();
    rrf_weighted(&weighted, top_k)
}

/// 重み付き Reciprocal Rank Fusion。同一チャンクが複数 ranking に出現したら `Hybrid` で印を付ける。
///
/// 融合の入口で `is_retrieval_noise` に該当する chunk (目次断片 / 短い OCR
/// 孤児片 / leader 過多) を除外する。これらは ingest 段の `is_low_content`
/// を素通りしても retrieve 上位を汚染しがちで、top_k=6 のような小さい予算
/// では本文チャンクを押し出す原因になる。
pub fn rrf_weighted(rankings: &[(Vec<SearchHit>, f32)], top_k: usize) -> Vec<SearchHit> {
    use std::collections::HashMap;
    let k = 60.0;
    let mut scores: HashMap<uuid::Uuid, (f32, u32, SearchHit)> = HashMap::new();
    for (ranking, weight) in rankings {
        if *weight <= 0.0 {
            continue;
        }
        // ノイズ除外したあとの順位で RRF スコアを付ける。除外しても元 ranking の
        // 残り順位は崩したくないので、フィルタ後に enumerate し直す。
        let cleaned: Vec<&SearchHit> = ranking
            .iter()
            .filter(|h| !is_retrieval_noise(&h.chunk.text))
            .collect();
        for (rank, hit) in cleaned.iter().enumerate() {
            let entry = scores
                .entry(hit.chunk.id)
                .or_insert((0.0, 0, (*hit).clone()));
            entry.0 += weight / (k + rank as f32 + 1.0);
            entry.1 += 1;
        }
    }
    let mut all: Vec<(f32, u32, SearchHit)> = scores.into_values().collect();
    all.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    all.into_iter()
        .take(top_k)
        .map(|(score, hits, mut h)| {
            h.score = score;
            if hits >= 2 {
                h.source = HitSource::Hybrid;
            }
            h
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::{
        adjust_hybrid_weight_for_query, has_code_snippet, is_specific_query, rrf, rrf_weighted,
        HybridWeights,
    };
    use ellisii_core::{Chunk, HitSource as HS, SearchHit};
    use uuid::Uuid;

    #[test]
    fn adjust_hybrid_weight_short_query_is_pass_through() {
        // 4 文字未満は判別不能 → base 値そのまま
        assert!((adjust_hybrid_weight_for_query(0.5, "あ") - 0.5).abs() < 1e-6);
        assert!((adjust_hybrid_weight_for_query(0.5, "abc") - 0.5).abs() < 1e-6);
    }

    #[test]
    fn adjust_hybrid_weight_kanji_heavy_leans_lexical() {
        // 漢字 100% → -0.2 で 0.30
        let w = adjust_hybrid_weight_for_query(0.5, "民法第94条意思表示");
        assert!(w < 0.5, "kanji-heavy should lean lexical: got {w}");
        assert!((w - 0.3).abs() < 0.05, "expected ~0.30, got {w}");
    }

    #[test]
    fn adjust_hybrid_weight_hiragana_heavy_leans_semantic() {
        // ひらがな主体 → +0.2 で 0.70 付近
        let w = adjust_hybrid_weight_for_query(0.5, "あいまいなしつもんですけれども");
        assert!(w > 0.5, "hiragana-heavy should lean semantic: got {w}");
        assert!((w - 0.7).abs() < 0.05, "expected ~0.70, got {w}");
    }

    #[test]
    fn adjust_hybrid_weight_mixed_query_is_close_to_base() {
        // 漢字とひらがなが拮抗 → ほぼ元のまま
        let w = adjust_hybrid_weight_for_query(0.5, "通謀虚偽表示はむこうですか");
        // 漢字 6 / ひらがな 8 → delta ≈ (8/14 - 6/14) * 0.2 ≈ +0.029
        assert!((w - 0.5).abs() < 0.06, "got {w}");
    }

    #[test]
    fn adjust_hybrid_weight_clamps_to_range() {
        // base が極端でもクランプ範囲 [0.05, 0.95] に収まる
        let w = adjust_hybrid_weight_for_query(1.0, "民法第94条意思表示");
        assert!(w <= 0.95);
        let w = adjust_hybrid_weight_for_query(0.0, "あいまいなしつもんですけれども");
        assert!(w >= 0.05);
    }

    #[test]
    fn adjust_hybrid_weight_id_like_query_leans_lexical() {
        // 英数字混在も identifier として扱う
        let w = adjust_hybrid_weight_for_query(0.5, "minpou-94 article-id-required");
        assert!(w < 0.5, "id-like should lean lexical: got {w}");
    }

    /// 既定の重みは vector 寄りに置く。
    /// 根拠: rag-eval-cli の static-jp (1024dim) 実モデル計測で
    /// 民法 / CS Wiki どちらでも `semantic=0.75` が MRR/nDCG を最大化したため
    /// (PR #27 の `tests/real_static_jp.rs` 参照)。
    #[test]
    fn default_semantic_is_zero_point_seventy_five() {
        let w = HybridWeights::default();
        assert!(
            (w.semantic - 0.75).abs() < 1e-6,
            "default semantic should be 0.75 (got {})",
            w.semantic
        );
        assert!((w.vector() - 0.75).abs() < 1e-6);
        assert!((w.keyword() - 0.25).abs() < 1e-6);
    }

    fn hit(label: &str) -> SearchHit {
        // ノイズフィルタが retrieve で stub を弾くので、テスト用 chunk も
        // 本文として通る長さにしておく (内容語 18 文字以上)。
        // ラベル文字列は識別用に末尾に付与する。
        let body = format!(
            "{label} 本文サンプルです。ハイブリッド検索が機能していることを示すための擬似テキストです。"
        );
        SearchHit {
            chunk: Chunk {
                id: Uuid::new_v4(),
                source_id: Uuid::nil(),
                ord: 0,
                text: body,
                heading_path: vec![],
                page: None,
                bbox: None,
                summary: None,
            },
            score: 0.0,
            source: HS::Vector,
        }
    }

    #[test]
    fn rrf_promotes_items_appearing_in_both_lists() {
        let a = hit("A");
        let b = hit("B");
        // A is rank 1 in vector, rank 2 in keyword. B is only in vector.
        let r = rrf(&[vec![a.clone(), b.clone()], vec![hit("C"), a.clone()]], 3);
        assert!(r[0].chunk.text.starts_with("A "));
    }

    #[test]
    fn rrf_weighted_zero_weight_excludes_ranking() {
        let only_in_kw = hit("only-kw");
        let r = rrf_weighted(&[(vec![hit("a")], 1.0), (vec![only_in_kw.clone()], 0.0)], 5);
        assert!(!r.iter().any(|h| h.chunk.text.starts_with("only-kw ")));
    }

    #[test]
    fn rrf_weighted_higher_weight_promotes() {
        // 同じチャンクが両方に出るが、kw 側を重く
        let mut a_vec = hit("A");
        a_vec.source = HS::Vector;
        let a_kw = a_vec.clone();
        let b_vec = hit("B");
        let r1 = rrf_weighted(
            &[
                (vec![a_vec.clone(), b_vec.clone()], 1.0),
                (vec![a_kw.clone()], 0.0),
            ],
            2,
        );
        let r2 = rrf_weighted(
            &[
                (vec![a_vec.clone(), b_vec.clone()], 0.0),
                (vec![a_kw.clone()], 1.0),
            ],
            2,
        );
        // どちらでも A が 1 位だが、片側で B が消えるはず
        assert!(r1[0].chunk.text.starts_with("A "));
        assert!(r2.iter().any(|h| h.chunk.text.starts_with("A ")));
        assert!(!r2.iter().any(|h| h.chunk.text.starts_with("B ")));
    }

    #[test]
    fn rrf_drops_retrieval_noise_hits() {
        // 短い stub と目次断片を混ぜて、本文 chunk だけが top に残ることを確認。
        let body = hit("BODY");
        let stub = SearchHit {
            chunk: Chunk {
                id: Uuid::new_v4(),
                source_id: Uuid::nil(),
                ord: 0,
                text: "もしれません。".into(),
                heading_path: vec![],
                page: None,
                bbox: None,
                summary: None,
            },
            score: 0.0,
            source: HS::Vector,
        };
        let toc = SearchHit {
            chunk: Chunk {
                id: Uuid::new_v4(),
                source_id: Uuid::nil(),
                ord: 0,
                text: "15.6リストの長さの制限\n15.7交差テーブルの他のメリット………11".into(),
                heading_path: vec![],
                page: None,
                bbox: None,
                summary: None,
            },
            score: 0.0,
            source: HS::Vector,
        };
        // stub と toc を上位に置いても、フィルタで弾かれて body だけが残る
        let r = rrf(&[vec![stub.clone(), toc.clone(), body.clone()]], 5);
        assert!(r.iter().all(|h| h.chunk.text.starts_with("BODY ")));
        assert_eq!(r.len(), 1);
    }

    #[test]
    fn rrf_marks_double_hits_as_hybrid() {
        let mut a_vec = hit("A");
        a_vec.source = HS::Vector;
        let mut a_kw = a_vec.clone();
        a_kw.source = HS::Keyword;
        let only_kw = {
            let mut h = hit("B");
            h.source = HS::Keyword;
            h
        };
        let r = rrf(&[vec![a_vec], vec![a_kw, only_kw.clone()]], 3);
        let merged_a = r.iter().find(|h| h.chunk.text.starts_with("A ")).unwrap();
        assert_eq!(merged_a.source, HS::Hybrid);
        let solo_b = r.iter().find(|h| h.chunk.text.starts_with("B ")).unwrap();
        assert_eq!(solo_b.source, HS::Keyword);
    }

    // ─── is_specific_query ──────────────────────────────────────────────

    #[test]
    fn specific_query_recognises_jp_article_id() {
        assert!(is_specific_query("民法第94条の意思表示について教えて"));
        assert!(is_specific_query("特許法第29条"));
        assert!(is_specific_query("第二百二十二条の効力"));
    }

    #[test]
    fn specific_query_recognises_en_article_or_section() {
        assert!(is_specific_query("Article 5 of the agreement"));
        assert!(is_specific_query("Section 1 details"));
        assert!(is_specific_query("Sec. 12 prohibits..."));
        assert!(is_specific_query("Article XII reservation"));
    }

    #[test]
    fn specific_query_recognises_quoted_phrase() {
        assert!(is_specific_query(r#"「通謀虚偽表示」とは何ですか"#));
        assert!(is_specific_query(r#""active record" pattern について"#));
        assert!(is_specific_query("『発明』の定義"));
    }

    #[test]
    fn specific_query_recognises_url_or_email() {
        assert!(is_specific_query("詳細は https://example.com/spec を参照"));
        assert!(is_specific_query("issue@github.com を確認"));
    }

    #[test]
    fn specific_query_recognises_long_query() {
        let long = "あ".repeat(60);
        assert!(is_specific_query(&long));
    }

    #[test]
    fn specific_query_rejects_natural_japanese() {
        // 民法 hard 系の自然文 (PR #65 で過剰判定を直したケース) は specific にならない
        assert!(!is_specific_query(
            "税金を期限までに払えなかったらどうなるか"
        ));
        assert!(!is_specific_query("通謀虚偽表示は無効か"));
        assert!(!is_specific_query("ふるさと納税の控除"));
    }

    #[test]
    fn specific_query_rejects_short_paraphrase() {
        assert!(!is_specific_query("入湯税の税率はいくらですか"));
        assert!(!is_specific_query("徴税吏員とは"));
    }

    // ─── has_code_snippet ───────────────────────────────────────────────

    #[test]
    fn code_snippet_recognises_uppercase_run() {
        assert!(has_code_snippet("SELECT id FROM users"));
        assert!(has_code_snippet("AUTO_INCREMENT を使う"));
    }

    #[test]
    fn code_snippet_recognises_punct_cluster() {
        assert!(has_code_snippet("foo(bar) = baz; に関して"));
        assert!(has_code_snippet("WHERE a < 10 AND b > 5"));
    }

    #[test]
    fn code_snippet_recognises_snake_case() {
        assert!(has_code_snippet("user_id カラムの取り扱い"));
        assert!(has_code_snippet("settings_db に保存される"));
    }

    #[test]
    fn code_snippet_recognises_camel_case() {
        assert!(has_code_snippet("ActiveRecord パターン"));
        assert!(has_code_snippet("MergeRequest を作る"));
    }

    #[test]
    fn code_snippet_rejects_natural_text() {
        assert!(!has_code_snippet("こんにちは"));
        assert!(!has_code_snippet("民法第94条"));
        assert!(!has_code_snippet("入湯税の税率"));
    }

    // ─── specific_query_ratio ──────────────────────────────────────────
    #[test]
    fn specific_query_ratio_empty_is_zero() {
        let qs: Vec<&str> = vec![];
        assert_eq!(super::specific_query_ratio(&qs), 0.0);
    }

    #[test]
    fn specific_query_ratio_all_specific_is_one() {
        let qs = vec![
            "民法第94条",
            "Article 5 of the contract",
            "https://example.com",
        ];
        assert!((super::specific_query_ratio(&qs) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn specific_query_ratio_all_paraphrase_is_zero() {
        let qs = vec!["税金とは", "効果", "ふるさと納税の控除"];
        assert_eq!(super::specific_query_ratio(&qs), 0.0);
    }

    // ─── query_body_recall_mean ────────────────────────────────────────
    #[test]
    fn query_body_recall_mean_empty_inputs() {
        let qs: Vec<&str> = vec![];
        let bs: Vec<&str> = vec!["body"];
        assert_eq!(super::query_body_recall_mean(&qs, &bs), 0.0);
        let qs2 = vec!["query"];
        let bs2: Vec<&str> = vec![];
        assert_eq!(super::query_body_recall_mean(&qs2, &bs2), 0.0);
    }

    #[test]
    fn query_body_recall_mean_perfect_literal_match_is_one() {
        // クエリ全 bigram が body にそのまま出現 → recall=1.0
        let qs = vec!["入湯税の税率はいくら"];
        let bs = vec!["入湯税の税率はいくらかというと、入湯税の税率は100円とする"];
        let r = super::query_body_recall_mean(&qs, &bs);
        assert!((r - 1.0).abs() < 1e-6, "got {r}");
    }

    #[test]
    fn query_body_recall_mean_low_for_paraphrase_corpus() {
        // tokkyo-style: query「発明とは何か」 vs body「自然法則を利用した技術的思想」
        // 「発明」以外の bigram は body に無い → recall は明らかに 1.0 未満
        let qs = vec!["発明とは何か"];
        let bs = vec!["自然法則を利用した技術的思想の創作のうち高度のもの"];
        let r = super::query_body_recall_mean(&qs, &bs);
        assert!(r < 0.5, "expected low recall (paraphrase gap), got {r}");
    }

    #[test]
    fn query_body_recall_mean_takes_max_over_bodies() {
        // 1 つの body にだけ完全一致しても、平均は max を取るので 1.0 にできる
        let qs = vec!["入湯税の税率"];
        let bs = vec!["全く関係のない別の本文です", "入湯税の税率は100円とする"];
        let r = super::query_body_recall_mean(&qs, &bs);
        assert!((r - 1.0).abs() < 1e-6, "got {r}");
    }

    #[test]
    fn query_body_recall_mean_yokohama_higher_than_tokkyo() {
        // 仮想ケースで literal-style corpus が paraphrase corpus より高い recall を示す
        let yokohama_qs = vec!["入湯税の税率はいくら", "たばこ税の税率"];
        let yokohama_bs = vec![
            "入湯税の税率は次の各号に掲げる入湯客の区分に応じる",
            "たばこ税の税率はたばこの本数に応じる",
        ];
        let tokkyo_qs = vec!["発明とは何か", "実施とは"];
        let tokkyo_bs = vec![
            "自然法則を利用した技術的思想の創作のうち高度のもの",
            "物の生産・使用・譲渡・輸入・申出をする行為",
        ];
        let y = super::query_body_recall_mean(&yokohama_qs, &yokohama_bs);
        let t = super::query_body_recall_mean(&tokkyo_qs, &tokkyo_bs);
        assert!(
            y > t + 0.3,
            "literal corpus must score notably higher: yokohama={y}, tokkyo={t}"
        );
    }

    // ─── query_title_match_mean ─────────────────────────────────────────
    #[test]
    fn query_title_match_mean_empty_inputs() {
        let qs: Vec<&str> = vec![];
        let ts: Vec<&str> = vec!["title"];
        assert_eq!(super::query_title_match_mean(&qs, &ts), 0.0);
        let qs2 = vec!["query"];
        let ts2: Vec<&str> = vec![];
        assert_eq!(super::query_title_match_mean(&qs2, &ts2), 0.0);
    }

    #[test]
    fn query_title_match_mean_higher_for_direct_match_than_paraphrase() {
        // タイトル直接マッチ (易) と paraphrase (難) を比較。前者が後者より明確に高いこと。
        let direct_qs = vec!["ACID とは何か"];
        let para_qs = vec!["トランザクション処理の信頼性を保証する性質を 4 つ挙げよ"];
        let ts = vec!["ACID", "B木", "Domain Name System"];
        let r_direct = super::query_title_match_mean(&direct_qs, &ts);
        let r_para = super::query_title_match_mean(&para_qs, &ts);
        assert!(
            r_direct > r_para + 0.2,
            "direct match must score notably higher: direct={r_direct}, paraphrase={r_para}"
        );
        assert!(r_para < 0.3, "paraphrase signal too high: {r_para}");
    }

    #[test]
    fn query_title_match_mean_takes_max_over_titles() {
        let qs = vec!["B木の特性"];
        let ts = vec!["ACID", "B木", "Domain Name System"];
        let r = super::query_title_match_mean(&qs, &ts);
        // "B木" が完全一致するので、bigram "B木" を含む query の半分以上が match
        assert!(r > 0.0, "got {r}");
    }

    #[test]
    fn specific_query_ratio_half_mix() {
        // 4 件: specific (第N条 / Article N) 2 件 + 概念 2 件 → 0.5
        let qs = vec!["民法第94条", "効果について", "Section 12", "税率はいくら"];
        let r = super::specific_query_ratio(&qs);
        assert!((r - 0.5).abs() < 1e-6, "got {r}");
    }
}
