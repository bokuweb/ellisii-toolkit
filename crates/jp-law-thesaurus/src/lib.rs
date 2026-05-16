//! 日本法令 / 契約書 / 判例 系の **法律ターム ↔ 日常シナリオ表現** を
//! 集めた静的辞書と、それで chunk caption を enrich するためのユーティリティ。
//!
//! 経緯は ellisii の `docs/eval/recall-evals.md` Run 12d-12h を参照。
//! 「シナリオ → 法律ターム」 paraphrase gap を、LLM を使わず **pure string
//! match (<1ms/chunk)** で埋める defensive default。法令系コーパスで
//! 60-97% カバー (Run 12f)、リーガル ecosystem (条文 / 契約書 / 重要事項
//! 説明書 / 特許明細書 / 判例 / 訴訟手続) を v5 で概ね網羅 (Run 12h)。
//!
//! 典型用途:
//!
//! ```no_run
//! use std::sync::Arc;
//! use ellisii_core::Chunk;
//! use ellisii_jp_law_thesaurus::LawThesaurus;
//!
//! // bundled v5 thesaurus を使う (no I/O)。
//! let thes = Arc::new(LawThesaurus::bundled());
//!
//! // chunk 1 つを enrich
//! let mut chunk: Chunk = unimplemented!();
//! thes.enrich_chunk(&mut chunk);
//! ```
//!
//! ローカルにカスタム辞書がある場合は [`LawThesaurus::from_path`] を使う。
//! v5 schema は `____comment_*` キーを section header として埋め込む形式で、
//! serde untagged enum で skip するようになっている。

use ellisii_core::{CaptionEnricher, Chunk};
use serde::Deserialize;
use std::collections::BTreeMap;
use std::path::Path;

/// 1 件の thesaurus entry。
#[derive(Debug, Clone, Deserialize)]
pub struct ThesaurusEntry {
    /// 出典カテゴリ (民法-法律行為, 税法-地方税, 特許明細書 等)。
    pub category: String,
    /// 同義語 / 略語 (例: "通謀虚偽表示, 仮装売買")。
    #[serde(default)]
    pub synonyms: Vec<String>,
    /// 日常シナリオ表現 (例: "税逃れのために売買契約書だけ作る")。
    #[serde(default)]
    pub scenarios: Vec<String>,
}

/// `____comment_*` キーの section header を skip するための untagged enum。
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
enum ThesaurusValue {
    Entry(ThesaurusEntry),
    /// `____comment_*` キーの section header。中身は捨てる。
    Comment(#[allow(dead_code)] String),
}

#[derive(Debug, Deserialize)]
struct ThesaurusFile {
    #[serde(default)]
    name: String,
    #[serde(default)]
    #[allow(dead_code)]
    comment: String,
    entries: BTreeMap<String, ThesaurusValue>,
}

/// 法律ターム → ({synonyms, scenarios, category}) の静的 lookup table。
pub struct LawThesaurus {
    name: String,
    entries: BTreeMap<String, ThesaurusEntry>,
}

/// `with_store_sqlite_with_tokenizer` 系の caption と矛盾しないよう、
/// 拡張部分は識別可能な separator (`｜ シナリオ:`) で付加する。
const CAPTION_SUFFIX_PREFIX: &str = " ｜ シナリオ: ";

/// 1 chunk に対し thesaurus key を最大何個適用するか。caption の肥大化抑止。
const MAX_KEYS_PER_CHUNK: usize = 3;

/// 各 chunk の body 先頭何文字まで走査するか (caption 検索範囲)。
const PROBE_BODY_CHARS: usize = 200;

impl LawThesaurus {
    /// crate 同梱の v5 thesaurus を `include_str!` でロード。
    /// I/O ゼロ、起動コスト一度きり。
    pub fn bundled() -> Self {
        let bytes = include_str!("../data/jp-law-thesaurus.json");
        Self::from_json_str(bytes).expect("bundled thesaurus must parse")
    }

    /// 任意 path から辞書 JSON を読む。schema は同梱版と互換。
    pub fn from_path(p: impl AsRef<Path>) -> Result<Self, String> {
        let txt =
            std::fs::read_to_string(p.as_ref()).map_err(|e| format!("read thesaurus: {e}"))?;
        Self::from_json_str(&txt)
    }

    /// 文字列から parse。
    pub fn from_json_str(s: &str) -> Result<Self, String> {
        let file: ThesaurusFile =
            serde_json::from_str(s).map_err(|e| format!("parse thesaurus json: {e}"))?;
        let entries = file
            .entries
            .into_iter()
            .filter_map(|(k, v)| match v {
                ThesaurusValue::Entry(e) => Some((k, e)),
                ThesaurusValue::Comment(_) => None,
            })
            .collect();
        Ok(Self {
            name: file.name,
            entries,
        })
    }

    /// 辞書名 (例: `"jp-law-thesaurus-v5"`)。
    pub fn name(&self) -> &str {
        &self.name
    }

    /// 有効 entry 数 (section header は除く)。
    pub fn entry_count(&self) -> usize {
        self.entries.len()
    }

    /// 1 chunk を enrich する内部実装。`CaptionEnricher` trait impl から呼ぶ。
    fn do_enrich(&self, chunk: &mut Chunk) -> bool {
        let keys = self.matched_keys(&chunk.text);
        if keys.is_empty() {
            return false;
        }
        let mut additions: Vec<String> = Vec::new();
        for k in &keys {
            if let Some(e) = self.entries.get(*k) {
                additions.extend(e.synonyms.iter().cloned());
                additions.extend(e.scenarios.iter().cloned());
            }
        }
        additions.sort();
        additions.dedup();
        if additions.is_empty() {
            return false;
        }
        let suffix = format!("{}{}", CAPTION_SUFFIX_PREFIX, additions.join(", "));
        if let Some(rest) = chunk.text.strip_prefix('(') {
            if let Some(end) = rest.find(')') {
                let caption = &rest[..end];
                let after = &rest[end..];
                chunk.text = format!("({caption}{suffix}{after}");
                return true;
            }
        }
        // caption 無し chunk: scenarios だけを擬似 caption として prepend
        chunk.text = format!("({})\n{}", additions.join(", "), chunk.text);
        true
    }

    /// chunk.text 先頭 + caption 部分を走査して、辞書 key が部分文字列として
    /// 出現するものを最大 [`MAX_KEYS_PER_CHUNK`] 件、長い key 優先で返す。
    fn matched_keys(&self, text: &str) -> Vec<&str> {
        let probe: String = text.chars().take(PROBE_BODY_CHARS).collect();
        let mut keys: Vec<&str> = self
            .entries
            .keys()
            .map(String::as_str)
            .filter(|k| probe.contains(*k))
            .collect();
        // 長い key を優先 (overlap で短い key が紛れ込まないように)
        keys.sort_by_key(|k| std::cmp::Reverse(k.chars().count()));
        keys.truncate(MAX_KEYS_PER_CHUNK);
        keys
    }

    /// 1 chunk を enrich (caption に synonyms + scenarios を append)。
    /// 何か追記したら true。
    pub fn enrich_chunk(&self, chunk: &mut Chunk) -> bool {
        self.do_enrich(chunk)
    }

    /// 複数 chunk を一括 enrich する便利関数。trait の default 実装と等価。
    pub fn enrich_chunks(&self, chunks: &mut [Chunk]) -> usize {
        <Self as CaptionEnricher>::enrich_chunks(self, chunks)
    }
}

impl CaptionEnricher for LawThesaurus {
    fn enrich_chunk(&self, chunk: &mut Chunk) -> bool {
        self.do_enrich(chunk)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use uuid::Uuid;

    fn mk(text: &str) -> Chunk {
        Chunk {
            id: Uuid::new_v4(),
            source_id: Uuid::new_v4(),
            ord: 0,
            text: text.to_string(),
            heading_path: vec![],
            page: None,
            bbox: None,
            summary: None,
        }
    }

    #[test]
    fn bundled_loads() {
        let t = LawThesaurus::bundled();
        assert!(t.entry_count() >= 400, "v5 should have 400+ entries");
        assert!(t.name().starts_with("jp-law-thesaurus"));
    }

    #[test]
    fn enrich_civil_law_minpou94_pattern() {
        let t = LawThesaurus::bundled();
        let mut chunk =
            mk("(虚偽表示)\n第九十四条 相手方と通じてした虚偽の意思表示は、無効とする。");
        let changed = t.enrich_chunk(&mut chunk);
        assert!(changed);
        // 虚偽表示 key の scenarios が caption に含まれるはず
        assert!(chunk.text.contains("税逃れのために売買契約書だけ作る"));
        // 元の本文は保持される
        assert!(chunk.text.contains("相手方と通じてした"));
    }

    #[test]
    fn no_match_chunk_untouched() {
        let t = LawThesaurus::bundled();
        let mut chunk = mk("ある日とても眠かったので寝た。");
        let changed = t.enrich_chunk(&mut chunk);
        assert!(!changed);
        assert_eq!(chunk.text, "ある日とても眠かったので寝た。");
    }

    #[test]
    fn longer_keys_preferred() {
        let t = LawThesaurus::bundled();
        // "法定相続分" と "相続" の両方が辞書にある。長い方が優先。
        let mut chunk = mk("(法定相続分)\n第九百条 同順位の相続人が…");
        t.enrich_chunk(&mut chunk);
        // 法定相続分 entry が選ばれていることを確認する prosy check:
        // scenario には 「夫が亡くなり子と妻がいる場合の遺産分け方」 等が含まれるはず
        assert!(chunk
            .text
            .contains("夫が亡くなり子と妻がいる場合の遺産分け方"));
    }
}
