use async_trait::async_trait;
use ellisii_core::{Chunk, Result, SearchHit};
use uuid::Uuid;

/// 1 source 内のチャンク列 (ord 昇順) から、`topic` を含む heading のセクションを
/// 取り出す。なければ text に topic を含む chunk、それでもなければ全体 TOC で代替。
///
/// `topic` が空文字なら [`pick_representatives`] と同じ。
pub fn pick_for_topic(chunks: &[Chunk], per_source: usize, topic: &str) -> Vec<Chunk> {
    if per_source == 0 || chunks.is_empty() {
        return vec![];
    }
    let topic = topic.trim();
    if topic.is_empty() {
        return pick_representatives(chunks, per_source);
    }
    let lower = topic.to_lowercase();
    let mut by_heading: Vec<Chunk> = chunks
        .iter()
        .filter(|c| {
            c.heading_path
                .iter()
                .any(|h| h.to_lowercase().contains(&lower))
        })
        .take(per_source)
        .cloned()
        .collect();
    if !by_heading.is_empty() {
        by_heading.sort_by_key(|c| c.ord);
        return by_heading;
    }
    let mut by_text: Vec<Chunk> = chunks
        .iter()
        .filter(|c| c.text.to_lowercase().contains(&lower))
        .take(per_source)
        .cloned()
        .collect();
    if !by_text.is_empty() {
        by_text.sort_by_key(|c| c.ord);
        return by_text;
    }
    pick_representatives(chunks, per_source)
}

/// 1 source 内の chunk 列 (ord 昇順) から TOC 風の代表 chunk を `per_source` 件まで選ぶ。
///
/// 1. `heading_path` の最上位要素が初出となる chunk を ord 順に採る (= 編 / 章 の先頭)
/// 2. 1 で 0 件しか拾えなかった場合のみ、ord 昇順の先頭から `per_source` 個を採る
///
/// 各 [`VectorStore`] 実装で同じ挙動を提供するため store-core に置く。
pub fn pick_representatives(chunks: &[Chunk], per_source: usize) -> Vec<Chunk> {
    if per_source == 0 || chunks.is_empty() {
        return vec![];
    }
    let mut seen: std::collections::HashSet<&str> = std::collections::HashSet::new();
    let mut picked: Vec<usize> = Vec::new();
    for (idx, c) in chunks.iter().enumerate() {
        if picked.len() >= per_source {
            break;
        }
        if let Some(top) = c.heading_path.first() {
            if seen.insert(top.as_str()) {
                picked.push(idx);
            }
        }
    }
    if picked.is_empty() {
        for idx in 0..chunks.len().min(per_source) {
            picked.push(idx);
        }
    }
    picked.sort();
    picked.into_iter().map(|i| chunks[i].clone()).collect()
}

/// 検索 / 削除のスコープ。`Some(notebook_id)` で指定 Notebook に閉じる。
/// `None` は全 Notebook 横断 (export 等の管理操作で使用)。
pub type Scope = Option<Uuid>;

#[async_trait]
pub trait VectorStore: Send + Sync {
    /// `notebook_id` を各 chunk に紐付けて保存する。
    async fn upsert(
        &self,
        notebook_id: Uuid,
        chunks: &[Chunk],
        embeddings: &[Vec<f32>],
    ) -> Result<()>;
    async fn search(
        &self,
        scope: Scope,
        query_embedding: &[f32],
        top_k: usize,
    ) -> Result<Vec<SearchHit>>;
    async fn keyword_search(
        &self,
        scope: Scope,
        query: &str,
        top_k: usize,
    ) -> Result<Vec<SearchHit>>;
    /// 指定 source_id に紐づく全 chunk を物理削除する。戻り値は削除件数。
    async fn delete_by_source(&self, source_id: Uuid) -> Result<usize>;
    /// 指定 notebook 配下のすべての chunk を物理削除する (notebook 削除時など)。
    async fn delete_by_notebook(&self, notebook_id: Uuid) -> Result<usize>;
    async fn count_chunks(&self, source_id: Uuid) -> Result<usize>;
    /// 指定 source_id に紐づく chunk のテキストを ord 順で返す
    /// (画像ソースの OCR テキストをビューアに表示する等に使う)。
    async fn texts_by_source(&self, source_id: Uuid) -> Result<Vec<String>>;

    /// `source_id` 配下の chunk のうち、`ord_center - window ..= ord_center + window`
    /// に該当するものを `(ord, text)` の組で返す (= 前後の隣接 chunk を取り出す)。
    /// 検索でヒットした chunk の周辺をくっつけて LLM に渡すと、条文単位 chunk
    /// などで「答えの周辺文脈」が補完されて精度が上がる。
    async fn neighbor_chunks(
        &self,
        source_id: Uuid,
        ord_center: u32,
        window: u32,
    ) -> Result<Vec<(u32, String)>>;

    /// scope 内の各 source について、文書構造を代表する chunk を `per_source` 個ずつ
    /// 返す (= TOC ライク取り出し)。「要約して」「概要」のような全体クエリで、
    /// ベクトル類似が当てにならないときに使う。
    ///
    /// 選択ルール:
    /// 1. ord 昇順で走査し、`heading_path` の **最上位要素** が初めて現れた chunk を採る
    ///    (= 編 / 章 ごとの先頭 chunk)
    /// 2. 1. で `per_source` に満たなければ、ord 昇順の先頭から不足分を補う
    ///    (= 構造の薄い文書では preamble を返す)
    ///
    /// 戻り順は source ごとに連続し、各 source 内では ord 昇順。
    async fn representative_chunks(
        &self,
        scope: Scope,
        per_source: usize,
    ) -> Result<Vec<Chunk>>;

    /// scope 内の各 source について、`topic` (= 主題語) に関連する代表 chunk を
    /// `per_source` 個ずつ返す。「民法の **物権** を要約して」のような主題付き
    /// 要約クエリ向け。
    ///
    /// 選択ルール (各 source ごと):
    /// 1. `heading_path` のいずれかの要素に `topic` が含まれる chunk を ord 昇順で集め、
    ///    先頭 `per_source` 件を返す (= 該当章節の TOC + 冒頭)
    /// 2. 1 で 0 件しか拾えなければ、`text` に `topic` が含まれる chunk を ord 昇順で
    ///    `per_source` 件まで採る (heading が薄い文書のフォールバック)
    /// 3. それでも 0 件なら [`VectorStore::representative_chunks`] と同じ結果を返す
    ///    (主題が全く外れていた場合は文書全体の TOC に倒す)
    ///
    /// `topic` を空文字で渡した場合は [`VectorStore::representative_chunks`] と等価。
    async fn representative_chunks_for_topic(
        &self,
        scope: Scope,
        per_source: usize,
        topic: &str,
    ) -> Result<Vec<Chunk>>;

    /// scope 内の全 chunk から「先頭の `(...)` 見出し」を抽出して `(chunk_id, caption)` を返す。
    /// 法令や条文など、`(条文タイトル)\n\n第X条 ...` 構造の文書で title-aware rerank を
    /// 行うときに使う。default 実装は空 (= rerank 機能を無効化)。実装側 (sqlite) で
    /// 1 クエリで全件取得して in-process で抽出する想定。
    async fn all_captions(&self, _scope: Scope) -> Result<Vec<(Uuid, String)>> {
        Ok(Vec::new())
    }

    /// 指定 chunk_id 集合に対応する Chunk を返す。順序は不定。caption rerank 等で
    /// pool 外から caption ヒットを引き上げるときに使う。default 実装は空。
    async fn get_chunks_by_ids(&self, _ids: &[Uuid]) -> Result<Vec<Chunk>> {
        Ok(Vec::new())
    }

    /// scope 内の全 chunk について `(chunk_id, heading_path セグメントを '/' 連結した文字列)`
    /// を返す。heading rerank (caption が抽出できない chunk のフォールバック) で使う。
    /// default 実装は空。
    async fn all_headings(&self, _scope: Scope) -> Result<Vec<(Uuid, String)>> {
        Ok(Vec::new())
    }

    /// scope 内の全 chunk から本文中の **定義語** (`「X」という。` / `「X」をいう。`) を抽出して
    /// `(chunk_id, defined_term)` を 1 row/term で返す。同一 chunk 内に定義語が複数あれば
    /// 複数 row になる。Run 42 で導入: caption だけでは捕捉できない「事業所等」のような
    /// 本文内定義語を pool 外から inject するために使う。default 実装は空。
    async fn all_defined_terms(&self, _scope: Scope) -> Result<Vec<(Uuid, String)>> {
        Ok(Vec::new())
    }

    /// scope 内の chunk 総数を返す。caption density / recall 計測の正規化分母などに使う。
    /// default 実装は 0 を返す (= rerank ガイダンス機能が機能しない、で fail-safe)。
    async fn count_chunks_in_scope(&self, _scope: Scope) -> Result<usize> {
        Ok(0)
    }

    /// scope 内の **distinct `source_id`** 数を返す。multi-source notebook の判定や、
    /// `max_chunks_per_source` の自動 routing (Run 56) に使う。default 実装は 0 を
    /// 返す (= multi-source 検出機能が無効化、と fail-safe)。
    async fn count_sources_in_scope(&self, _scope: Scope) -> Result<usize> {
        Ok(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use uuid::Uuid;

    fn chunk(ord: u32, heading: Vec<&str>, text: &str) -> Chunk {
        Chunk {
            id: Uuid::new_v4(),
            source_id: Uuid::nil(),
            ord,
            text: text.into(),
            heading_path: heading.into_iter().map(|s| s.to_string()).collect(),
            page: None,
            bbox: None,
            summary: None,
        }
    }

    #[test]
    fn pick_for_topic_picks_section_by_heading() {
        let chunks = vec![
            chunk(0, vec!["第一編 総則"], "前文"),
            chunk(1, vec!["第二編 物権"], "物権の冒頭"),
            chunk(2, vec!["第二編 物権"], "所有権..."),
            chunk(3, vec!["第三編 債権"], "債権..."),
        ];
        let out = pick_for_topic(&chunks, 5, "物権");
        let texts: Vec<&str> = out.iter().map(|c| c.text.as_str()).collect();
        assert_eq!(texts, vec!["物権の冒頭", "所有権..."]);
    }

    #[test]
    fn pick_for_topic_caps_per_source() {
        let chunks = vec![
            chunk(0, vec!["第二編 物権"], "a"),
            chunk(1, vec!["第二編 物権"], "b"),
            chunk(2, vec!["第二編 物権"], "c"),
        ];
        let out = pick_for_topic(&chunks, 2, "物権");
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].text, "a");
    }

    #[test]
    fn pick_for_topic_falls_back_to_text_match() {
        let chunks = vec![
            chunk(0, vec!["chapter 1"], "introduction"),
            chunk(1, vec!["chapter 2"], "discusses 物権 in depth"),
            chunk(2, vec!["chapter 3"], "irrelevant"),
        ];
        let out = pick_for_topic(&chunks, 5, "物権");
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].text, "discusses 物権 in depth");
    }

    #[test]
    fn pick_for_topic_falls_back_to_global_reps_when_no_match() {
        let chunks = vec![
            chunk(0, vec!["第一編 総則"], "a"),
            chunk(1, vec!["第二編 物権"], "b"),
        ];
        let out = pick_for_topic(&chunks, 5, "完全に外れた主題");
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].text, "a");
        assert_eq!(out[1].text, "b");
    }

    #[test]
    fn pick_for_topic_empty_topic_equals_representative() {
        let chunks = vec![
            chunk(0, vec!["A"], "a"),
            chunk(1, vec!["B"], "b"),
        ];
        let out = pick_for_topic(&chunks, 5, "");
        let reps = pick_representatives(&chunks, 5);
        assert_eq!(out.len(), reps.len());
        for (l, r) in out.iter().zip(reps.iter()) {
            assert_eq!(l.text, r.text);
        }
    }

    #[test]
    fn pick_for_topic_case_insensitive() {
        let chunks = vec![chunk(0, vec!["Privacy Policy"], "...")];
        let out = pick_for_topic(&chunks, 5, "privacy");
        assert_eq!(out.len(), 1);
    }
}
