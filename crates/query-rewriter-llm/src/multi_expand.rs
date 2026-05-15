//! `MultiExpandRewriter`: paraphrase / decompose (sub-question) / HyDE を
//! **1 回の LLM 呼び出し** で生成して 1〜6 個の variants を返す rewriter。
//!
//! 設計はもともと src-tauri 内の `expand_all_in_one` ヘルパとして実装されていた
//! ロジックを `QueryRewriter` trait の形に抽出したもの。3 種類の異なる検索改善
//! 戦略を 1 段の decode で取得することで、`LlmRewriter` (paraphrase のみ) より
//! 「答え側の語彙」「複合質問の分解」も同時に拾える:
//!
//! - **paraphrase**: 同義語・言い換えクエリ (1〜2 個、各 ≤30 字)
//! - **subquestion**: 複合質問のときだけ原子的なサブ質問 (0〜2 個、各 ≤40 字)
//! - **HyDE**: 質問にズバリ答える想定回答の 1 段落 (60〜120 字)
//!   → 答え側の語彙でベクトル検索が走るので「脅されて → 強迫」のような
//!   同義語ギャップに強い
//!
//! LLM が壊れた出力を返したり generate が失敗した場合は元クエリのみを返す。

use async_trait::async_trait;
use ellisii_core::Result;
use ellisii_llm_core::{LlmBackend, LlmRequest};
use ellisii_query_rewriter_core::{QueryRewriter, RewrittenQueries};
use std::sync::{Arc, Mutex};

const SYSTEM_PROMPT: &str = "あなたは検索クエリ多様化アシスタントです。日常語で書かれた質問が **どの条文・専門用語に該当するか** を識別し、その用語を含む検索クエリを生成します。\n\
                             \n\
                             重要: 質問の **表層のキーワードを言い換える** のではなく、その質問が **どの法律用語/技術用語の条文** に該当するかを推論してください。\n\
                             - 悪い例: 「税逃れの契約」→「脱税」「租税回避」(これは表層の言い換え)\n\
                             - 良い例: 「税逃れの契約」→「通謀虚偽表示」「相手方と通じた虚偽の意思表示」(対応する民法の用語)\n\
                             - 悪い例: 「畑で取れた野菜」→「採集物」「農作物」(表層)\n\
                             - 良い例: 「畑で取れた野菜」→「天然果実」「物の用法に従い収取する産出物」(民法上の用語)\n\
                             \n\
                             例 1:\n\
                             質問: 中学生がアルバイト代でゲーム機を買ったが、親はキャンセルできるか\n\
                             PARAPHRASE: 未成年者の法律行為の取消\n\
                             PARAPHRASE: 法定代理人の同意\n\
                             SUBQUESTION: 未成年者が単独でした契約は取り消せるか\n\
                             HYDE: 未成年者が法律行為をするにはその法定代理人の同意を要し、同意なくした行為は取り消すことができる。\n\
                             \n\
                             例 2:\n\
                             質問: 知人と組んで虚偽の契約書を作った場合の効力\n\
                             PARAPHRASE: 通謀虚偽表示\n\
                             PARAPHRASE: 相手方と通じてした虚偽の意思表示\n\
                             HYDE: 相手方と通じてした虚偽の意思表示は無効とする。当事者間で外形だけ整えた虚偽の契約は法的効力を持たない。\n\
                             \n\
                             例 3:\n\
                             質問: 違法薬物の売買を約束する取引は有効か\n\
                             PARAPHRASE: 公序良俗違反の法律行為\n\
                             PARAPHRASE: 反社会的契約の効力\n\
                             HYDE: 公の秩序又は善良の風俗に反する法律行為は無効である。麻薬取引のような反社会的な契約は法的に保護されない。\n\
                             \n\
                             例 4:\n\
                             質問: 庭の果樹から落ちた果物の所有権\n\
                             PARAPHRASE: 天然果実の所有権\n\
                             PARAPHRASE: 物の用法に従い収取する産出物\n\
                             HYDE: 物の用法に従い収取する産出物を天然果実とする。土地から得られる作物などは元物の所有者に帰属する。\n\
                             \n\
                             例 5:\n\
                             質問: DBで複数の更新が途中で失敗しないことを保証する性質\n\
                             PARAPHRASE: ACID 不可分性 atomicity\n\
                             PARAPHRASE: トランザクションの原子性\n\
                             HYDE: ACIDはトランザクションの不可分性・一貫性・独立性・永続性を表す頭字語であり、信頼性のあるDBに求められる性質。\n\
                             \n\
                             例 6:\n\
                             質問: 互いにロックを待ち合って永遠に進まなくなる状態\n\
                             PARAPHRASE: デッドロック\n\
                             PARAPHRASE: deadlock 相互排他のループ\n\
                             HYDE: デッドロックとは、複数のプロセスが互いに保持するリソースの解放を待ち合うことで、いずれも進行できなくなる状態のこと。\n\
                             \n\
                             出力フォーマット (各行頭にラベル):\n\
                             PARAPHRASE: <該当する専門用語・条文用語。30 字以内。1〜3 件>\n\
                             SUBQUESTION: <複合質問の原子的な小問。40 字以内。0〜2 件>\n\
                             HYDE: <質問にズバリ答える想定回答。60〜150 字、1 段落>\n\
                             \n\
                             ルール:\n\
                             - 出力は上記 3 ラベルのいずれかで始まる行のみ。前置き・解説・番号禁止\n\
                             - 表層を言い換えるのではなく、対応する **条文の用語** を出す\n\
                             - 関連語・同義語・条文番号・専門用語があれば積極的に含める\n\
                             - SUBQUESTION は無ければ省略する\n\
                             - HYDE は確信が持てなくても自然な日本語の 1 段落として書き切る";

pub struct MultiExpandRewriter<L: LlmBackend> {
    llm: L,
    pub temperature: f32,
    pub max_tokens: u32,
}

impl<L: LlmBackend> MultiExpandRewriter<L> {
    pub fn new(llm: L) -> Self {
        Self {
            llm,
            temperature: 0.3,
            max_tokens: 256,
        }
    }
}

#[async_trait]
impl<L: LlmBackend> QueryRewriter for MultiExpandRewriter<L> {
    async fn rewrite(&self, query: &str, _max_variants: usize) -> Result<RewrittenQueries> {
        // 短すぎる query は拡張しない (元の expand_all_in_one と同じ閾値)
        if query.chars().count() < 5 {
            return Ok(RewrittenQueries::just(query));
        }
        let req = LlmRequest {
            system: SYSTEM_PROMPT.into(),
            history: Vec::new(),
            user: format!("質問: {query}"),
            max_tokens: self.max_tokens,
            temperature: self.temperature,
        };
        let buf: Arc<Mutex<String>> = Arc::new(Mutex::new(String::new()));
        let buf2 = buf.clone();
        let cb: Box<dyn FnMut(String) + Send + 'static> =
            Box::new(move |t: String| buf2.lock().unwrap().push_str(&t));

        if self.llm.generate_stream(req, cb).await.is_err() {
            return Ok(RewrittenQueries::just(query));
        }
        let raw = buf.lock().unwrap().clone();
        let variants = parse_expand_output(&raw, query);
        Ok(RewrittenQueries {
            original: query.to_string(),
            variants,
        })
    }
}

/// `PARAPHRASE: ...` / `SUBQUESTION: ...` / `HYDE: ...` のラベル付き行を抽出。
///
/// - paraphrase は最大 3 件、各 60 文字まで (引用符などをトリム)
/// - subquestion は最大 3 件、各 80 文字まで (先頭の番号付き接頭辞をトリム)
/// - HYDE は 1 件、240 文字まで
/// - 元クエリと完全一致 / 重複 / 短すぎ (paraphrase ≥ 2 字, subq ≥ 4 字, hyde ≥ 30 字) は除外
pub fn parse_expand_output(raw: &str, original: &str) -> Vec<String> {
    let mut paraphrases: Vec<String> = Vec::new();
    let mut subqs: Vec<String> = Vec::new();
    let mut hyde: Option<String> = None;

    for line in raw.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        if let Some(rest) = trimmed
            .strip_prefix("PARAPHRASE:")
            .or_else(|| trimmed.strip_prefix("PARAPHRASE :"))
        {
            let s: String = rest
                .trim()
                .trim_matches(|c: char| {
                    matches!(c, '"' | '「' | '」' | '『' | '』' | '\'' | '-' | '*' | '#')
                })
                .chars()
                .take(60)
                .collect();
            if s.chars().count() >= 2
                && !paraphrases.iter().any(|x| x == &s)
                && s != original
                && paraphrases.len() < 3
            {
                paraphrases.push(s);
            }
        } else if let Some(rest) = trimmed
            .strip_prefix("SUBQUESTION:")
            .or_else(|| trimmed.strip_prefix("SUBQUESTION :"))
        {
            let s: String = rest
                .trim()
                .trim_matches(|c: char| {
                    matches!(c, '"' | '「' | '」' | '『' | '』' | '\'' | '-' | '*' | '#' | '・')
                })
                .trim_start_matches(|c: char| c.is_ascii_digit() || c == '.' || c == ')' || c == '．')
                .trim()
                .chars()
                .take(80)
                .collect();
            if s.chars().count() >= 4
                && !subqs.iter().any(|x| x == &s)
                && s != original
                && subqs.len() < 3
            {
                subqs.push(s);
            }
        } else if let Some(rest) = trimmed
            .strip_prefix("HYDE:")
            .or_else(|| trimmed.strip_prefix("HYDE :"))
        {
            let s = rest.trim().to_string();
            if s.chars().count() >= 30 && hyde.is_none() {
                hyde = Some(s.chars().take(240).collect());
            }
        }
    }

    let mut out = Vec::with_capacity(paraphrases.len() + subqs.len() + 1);
    out.extend(paraphrases);
    out.extend(subqs);
    if let Some(h) = hyde {
        out.push(h);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use ellisii_core::Result;
    use ellisii_llm_core::{LlmBackend, LlmRequest};

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

    #[test]
    fn parse_handles_three_label_kinds() {
        let raw = "PARAPHRASE: 強迫による契約\n\
                   PARAPHRASE: 脅迫を受けた合意\n\
                   SUBQUESTION: 強迫の効力は無効か取消か\n\
                   HYDE: 民法第96条により、詐欺又は強迫による意思表示は取り消すことができる。第三者による詐欺は相手方の善意悪意で結論が変わる。";
        let v = parse_expand_output(raw, "脅されて結ばされた契約");
        assert_eq!(v.len(), 4);
        assert_eq!(v[0], "強迫による契約");
        assert_eq!(v[1], "脅迫を受けた合意");
        assert_eq!(v[2], "強迫の効力は無効か取消か");
        assert!(v[3].starts_with("民法第96条"));
    }

    #[test]
    fn parse_drops_short_paraphrase_and_short_hyde() {
        let raw = "PARAPHRASE: あ\n\
                   PARAPHRASE: 強迫\n\
                   HYDE: 短い";
        let v = parse_expand_output(raw, "q");
        assert_eq!(v, vec!["強迫".to_string()]);
    }

    #[test]
    fn parse_dedups_paraphrase_and_drops_equal_to_original() {
        let raw = "PARAPHRASE: 同じ\n\
                   PARAPHRASE: 同じ\n\
                   PARAPHRASE: q";
        let v = parse_expand_output(raw, "q");
        assert_eq!(v, vec!["同じ".to_string()]);
    }

    #[test]
    fn parse_strips_quotes_and_bullet_chars() {
        let raw = "PARAPHRASE: 「強迫による契約」\n\
                   SUBQUESTION: ・1. 強迫の効力";
        let v = parse_expand_output(raw, "q");
        assert_eq!(v, vec!["強迫による契約", "強迫の効力"]);
    }

    #[tokio::test]
    async fn rewriter_returns_variants_on_well_formed_output() {
        let llm = ScriptedLlm {
            out: "PARAPHRASE: 強迫\n\
                  HYDE: 民法第96条により、詐欺又は強迫による意思表示は取り消すことができる。第三者による詐欺は相手方の善意悪意で結論が変わる。"
                .into(),
        };
        let r = MultiExpandRewriter::new(llm)
            .rewrite("脅されて結ばされた契約はどう扱われるか", 99)
            .await
            .unwrap();
        assert_eq!(r.variants.len(), 2);
        assert_eq!(r.variants[0], "強迫");
        assert!(r.variants[1].contains("強迫"));
    }

    #[tokio::test]
    async fn rewriter_short_query_returns_passthrough() {
        let llm = ScriptedLlm {
            out: "PARAPHRASE: 拡張結果".into(),
        };
        let r = MultiExpandRewriter::new(llm).rewrite("猫", 99).await.unwrap();
        assert_eq!(r, RewrittenQueries::just("猫"));
    }
}
