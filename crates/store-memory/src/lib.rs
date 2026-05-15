use async_trait::async_trait;
use ellisii_core::{Chunk, HitSource, Result, SearchHit};
use ellisii_store_core::{pick_for_topic, pick_representatives, Scope, VectorStore};
use parking_lot::RwLock;
use uuid::Uuid;

#[derive(Default)]
pub struct InMemoryStore {
    inner: RwLock<Inner>,
}

#[derive(Default)]
struct Inner {
    chunks: Vec<Chunk>,
    embeddings: Vec<Vec<f32>>,
    notebook_ids: Vec<Uuid>,
}

impl InMemoryStore {
    pub fn new() -> Self {
        Self::default()
    }
}

#[async_trait]
impl VectorStore for InMemoryStore {
    async fn upsert(
        &self,
        notebook_id: Uuid,
        chunks: &[Chunk],
        embeddings: &[Vec<f32>],
    ) -> Result<()> {
        let mut inner = self.inner.write();
        inner.chunks.extend_from_slice(chunks);
        inner.embeddings.extend_from_slice(embeddings);
        inner
            .notebook_ids
            .extend(std::iter::repeat_n(notebook_id, chunks.len()));
        Ok(())
    }

    async fn search(&self, scope: Scope, query: &[f32], top_k: usize) -> Result<Vec<SearchHit>> {
        let inner = self.inner.read();
        let mut scored: Vec<(f32, usize)> = inner
            .embeddings
            .iter()
            .enumerate()
            .filter(|(i, _)| match scope {
                Some(nb) => inner.notebook_ids.get(*i).copied() == Some(nb),
                None => true,
            })
            .map(|(i, e)| (cosine(query, e), i))
            .collect();
        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        Ok(scored
            .into_iter()
            .take(top_k)
            .map(|(score, i)| SearchHit {
                chunk: inner.chunks[i].clone(),
                score,
                source: HitSource::Vector,
            })
            .collect())
    }

    async fn delete_by_source(&self, source_id: Uuid) -> Result<usize> {
        let mut inner = self.inner.write();
        let mut removed = 0;
        let mut i = 0;
        while i < inner.chunks.len() {
            if inner.chunks[i].source_id == source_id {
                inner.chunks.swap_remove(i);
                inner.embeddings.swap_remove(i);
                inner.notebook_ids.swap_remove(i);
                removed += 1;
            } else {
                i += 1;
            }
        }
        Ok(removed)
    }

    async fn delete_by_notebook(&self, notebook_id: Uuid) -> Result<usize> {
        let mut inner = self.inner.write();
        let mut removed = 0;
        let mut i = 0;
        while i < inner.chunks.len() {
            if inner.notebook_ids.get(i).copied() == Some(notebook_id) {
                inner.chunks.swap_remove(i);
                inner.embeddings.swap_remove(i);
                inner.notebook_ids.swap_remove(i);
                removed += 1;
            } else {
                i += 1;
            }
        }
        Ok(removed)
    }

    async fn count_chunks(&self, source_id: Uuid) -> Result<usize> {
        Ok(self
            .inner
            .read()
            .chunks
            .iter()
            .filter(|c| c.source_id == source_id)
            .count())
    }

    async fn count_chunks_in_scope(&self, scope: Scope) -> Result<usize> {
        let inner = self.inner.read();
        Ok(match scope {
            Some(nb) => inner.notebook_ids.iter().filter(|id| **id == nb).count(),
            None => inner.chunks.len(),
        })
    }

    async fn count_sources_in_scope(&self, scope: Scope) -> Result<usize> {
        let inner = self.inner.read();
        let mut seen: std::collections::HashSet<Uuid> = std::collections::HashSet::new();
        for (i, chunk) in inner.chunks.iter().enumerate() {
            if let Some(nb) = scope {
                if inner.notebook_ids.get(i).copied() != Some(nb) {
                    continue;
                }
            }
            seen.insert(chunk.source_id);
        }
        Ok(seen.len())
    }

    async fn all_captions(&self, scope: Scope) -> Result<Vec<(Uuid, String)>> {
        let inner = self.inner.read();
        let mut out = Vec::new();
        for (i, chunk) in inner.chunks.iter().enumerate() {
            if let Some(nb) = scope {
                if inner.notebook_ids.get(i).copied() != Some(nb) {
                    continue;
                }
            }
            let t = chunk.text.trim_start();
            if let Some(rest) = t.strip_prefix('(') {
                if let Some(end) = rest.find(')') {
                    out.push((chunk.id, rest[..end].to_string()));
                }
            }
        }
        Ok(out)
    }

    async fn all_headings(&self, scope: Scope) -> Result<Vec<(Uuid, String)>> {
        let inner = self.inner.read();
        let mut out = Vec::new();
        for (i, chunk) in inner.chunks.iter().enumerate() {
            if let Some(nb) = scope {
                if inner.notebook_ids.get(i).copied() != Some(nb) {
                    continue;
                }
            }
            if !chunk.heading_path.is_empty() {
                out.push((chunk.id, chunk.heading_path.join("/")));
            }
        }
        Ok(out)
    }

    /// chunk text の先頭 400 char から `extract_defined_terms` で定義語を抽出し、
    /// 1 row/term で返す (Run 43、store-sqlite と同条件)。
    async fn all_defined_terms(&self, scope: Scope) -> Result<Vec<(Uuid, String)>> {
        let inner = self.inner.read();
        let mut out: Vec<(Uuid, String)> = Vec::new();
        for (i, chunk) in inner.chunks.iter().enumerate() {
            if let Some(nb) = scope {
                if inner.notebook_ids.get(i).copied() != Some(nb) {
                    continue;
                }
            }
            // sqlite と同じく substr(text, 1, 400) 相当に揃える。
            let head: String = chunk.text.chars().take(400).collect();
            for term in ellisii_core::caption::extract_defined_terms(&head) {
                out.push((chunk.id, term.to_string()));
            }
        }
        Ok(out)
    }

    async fn get_chunks_by_ids(&self, ids: &[Uuid]) -> Result<Vec<Chunk>> {
        if ids.is_empty() {
            return Ok(Vec::new());
        }
        let inner = self.inner.read();
        let id_set: std::collections::HashSet<Uuid> = ids.iter().copied().collect();
        Ok(inner
            .chunks
            .iter()
            .filter(|c| id_set.contains(&c.id))
            .cloned()
            .collect())
    }

    async fn texts_by_source(&self, source_id: Uuid) -> Result<Vec<String>> {
        let inner = self.inner.read();
        let mut rows: Vec<(u32, String)> = inner
            .chunks
            .iter()
            .filter(|c| c.source_id == source_id)
            .map(|c| (c.ord, c.text.clone()))
            .collect();
        rows.sort_by_key(|(ord, _)| *ord);
        Ok(rows.into_iter().map(|(_, t)| t).collect())
    }

    async fn representative_chunks(&self, scope: Scope, per_source: usize) -> Result<Vec<Chunk>> {
        if per_source == 0 {
            return Ok(vec![]);
        }
        let inner = self.inner.read();
        // scope 内の chunk を source 別にまとめ、各内訳は ord 昇順。
        // 出現順 (= upsert された順) を保つため、source の登場順を記録する。
        let mut order: Vec<Uuid> = Vec::new();
        let mut by_source: std::collections::HashMap<Uuid, Vec<Chunk>> =
            std::collections::HashMap::new();
        for (i, c) in inner.chunks.iter().enumerate() {
            let in_scope = match scope {
                Some(nb) => inner.notebook_ids.get(i).copied() == Some(nb),
                None => true,
            };
            if !in_scope {
                continue;
            }
            if !by_source.contains_key(&c.source_id) {
                order.push(c.source_id);
            }
            by_source.entry(c.source_id).or_default().push(c.clone());
        }
        let mut out: Vec<Chunk> = Vec::new();
        for sid in order {
            let mut chunks = by_source.remove(&sid).unwrap_or_default();
            chunks.sort_by_key(|c| c.ord);
            out.extend(pick_representatives(&chunks, per_source));
        }
        Ok(out)
    }

    async fn representative_chunks_for_topic(
        &self,
        scope: Scope,
        per_source: usize,
        topic: &str,
    ) -> Result<Vec<Chunk>> {
        if per_source == 0 {
            return Ok(vec![]);
        }
        let inner = self.inner.read();
        let mut order: Vec<Uuid> = Vec::new();
        let mut by_source: std::collections::HashMap<Uuid, Vec<Chunk>> =
            std::collections::HashMap::new();
        for (i, c) in inner.chunks.iter().enumerate() {
            let in_scope = match scope {
                Some(nb) => inner.notebook_ids.get(i).copied() == Some(nb),
                None => true,
            };
            if !in_scope {
                continue;
            }
            if !by_source.contains_key(&c.source_id) {
                order.push(c.source_id);
            }
            by_source.entry(c.source_id).or_default().push(c.clone());
        }
        let mut out: Vec<Chunk> = Vec::new();
        for sid in order {
            let mut chunks = by_source.remove(&sid).unwrap_or_default();
            chunks.sort_by_key(|c| c.ord);
            out.extend(pick_for_topic(&chunks, per_source, topic));
        }
        Ok(out)
    }

    async fn neighbor_chunks(
        &self,
        source_id: Uuid,
        ord_center: u32,
        window: u32,
    ) -> Result<Vec<(u32, String)>> {
        let inner = self.inner.read();
        let lo = ord_center.saturating_sub(window);
        let hi = ord_center.saturating_add(window);
        let mut rows: Vec<(u32, String)> = inner
            .chunks
            .iter()
            .filter(|c| c.source_id == source_id && c.ord >= lo && c.ord <= hi)
            .map(|c| (c.ord, c.text.clone()))
            .collect();
        rows.sort_by_key(|(ord, _)| *ord);
        Ok(rows)
    }

    async fn keyword_search(
        &self,
        scope: Scope,
        query: &str,
        top_k: usize,
    ) -> Result<Vec<SearchHit>> {
        let inner = self.inner.read();
        let q = query.to_lowercase();
        Ok(inner
            .chunks
            .iter()
            .enumerate()
            .filter(|(i, _)| match scope {
                Some(nb) => inner.notebook_ids.get(*i).copied() == Some(nb),
                None => true,
            })
            .filter(|(_, c)| c.text.to_lowercase().contains(&q))
            .take(top_k)
            .map(|(_, c)| SearchHit {
                chunk: c.clone(),
                score: 1.0,
                source: HitSource::Keyword,
            })
            .collect())
    }
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len().min(b.len());
    let mut dot = 0.0_f32;
    let mut na = 0.0_f32;
    let mut nb = 0.0_f32;
    for i in 0..len {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    if na == 0.0 || nb == 0.0 {
        0.0
    } else {
        dot / (na.sqrt() * nb.sqrt())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use uuid::Uuid;

    fn chunk(text: &str) -> Chunk {
        Chunk {
            id: Uuid::new_v4(),
            source_id: Uuid::nil(),
            ord: 0,
            text: text.to_string(),
            heading_path: vec![],
            page: None,
            bbox: None,
            summary: None,
        }
    }

    #[tokio::test]
    async fn upsert_and_keyword_search() {
        let s = InMemoryStore::new();
        let nb = Uuid::new_v4();
        s.upsert(
            nb,
            &[chunk("hello world"), chunk("foo bar")],
            &[vec![1.0], vec![0.0]],
        )
        .await
        .unwrap();
        let hits = s.keyword_search(Some(nb), "foo", 5).await.unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].chunk.text, "foo bar");
    }

    #[tokio::test]
    async fn delete_by_source_removes_matching_chunks() {
        let s = InMemoryStore::new();
        let nb = Uuid::new_v4();
        let sid = Uuid::new_v4();
        let mut a = chunk("aaa");
        a.source_id = sid;
        let mut b = chunk("bbb");
        b.source_id = sid;
        let other = chunk("ccc");
        s.upsert(
            nb,
            &[a, b, other.clone()],
            &[vec![1.0], vec![1.0], vec![1.0]],
        )
        .await
        .unwrap();
        let removed = s.delete_by_source(sid).await.unwrap();
        assert_eq!(removed, 2);
        let remaining = s.keyword_search(Some(nb), "ccc", 5).await.unwrap();
        assert_eq!(remaining.len(), 1);
        assert_eq!(remaining[0].chunk.id, other.id);
        assert_eq!(s.count_chunks(sid).await.unwrap(), 0);
    }

    #[tokio::test]
    async fn vector_search_orders_by_cosine() {
        let s = InMemoryStore::new();
        let nb = Uuid::new_v4();
        s.upsert(
            nb,
            &[chunk("a"), chunk("b")],
            &[vec![1.0, 0.0], vec![0.0, 1.0]],
        )
        .await
        .unwrap();
        let hits = s.search(Some(nb), &[1.0, 0.0], 2).await.unwrap();
        assert_eq!(hits[0].chunk.text, "a");
    }

    #[tokio::test]
    async fn scope_isolates_notebooks() {
        let s = InMemoryStore::new();
        let nb_a = Uuid::new_v4();
        let nb_b = Uuid::new_v4();
        s.upsert(nb_a, &[chunk("alpha shared")], &[vec![1.0]])
            .await
            .unwrap();
        s.upsert(nb_b, &[chunk("beta shared")], &[vec![1.0]])
            .await
            .unwrap();

        let in_a = s.keyword_search(Some(nb_a), "shared", 5).await.unwrap();
        assert_eq!(in_a.len(), 1);
        assert!(in_a[0].chunk.text.starts_with("alpha"));

        let in_b = s.keyword_search(Some(nb_b), "shared", 5).await.unwrap();
        assert_eq!(in_b.len(), 1);
        assert!(in_b[0].chunk.text.starts_with("beta"));

        let across = s.keyword_search(None, "shared", 5).await.unwrap();
        assert_eq!(across.len(), 2);
    }

    fn chunk_with(source_id: Uuid, ord: u32, heading: Vec<&str>, text: &str) -> Chunk {
        Chunk {
            id: Uuid::new_v4(),
            source_id,
            ord,
            text: text.to_string(),
            heading_path: heading.into_iter().map(|s| s.to_string()).collect(),
            page: None,
            bbox: None,
            summary: None,
        }
    }

    #[tokio::test]
    async fn representative_chunks_picks_first_per_top_heading() {
        let s = InMemoryStore::new();
        let nb = Uuid::new_v4();
        let sid = Uuid::new_v4();
        let chunks = vec![
            chunk_with(sid, 0, vec!["第一編 総則"], "前文"),
            chunk_with(sid, 1, vec!["第一編 総則"], "..."),
            chunk_with(sid, 2, vec!["第二編 物権"], "物権の冒頭"),
            chunk_with(sid, 3, vec!["第二編 物権"], "..."),
            chunk_with(sid, 4, vec!["第三編 債権"], "債権の冒頭"),
        ];
        let embs = vec![vec![0.0]; chunks.len()];
        s.upsert(nb, &chunks, &embs).await.unwrap();
        let reps = s.representative_chunks(Some(nb), 5).await.unwrap();
        let texts: Vec<&str> = reps.iter().map(|c| c.text.as_str()).collect();
        assert_eq!(texts, vec!["前文", "物権の冒頭", "債権の冒頭"]);
    }

    #[tokio::test]
    async fn representative_chunks_caps_per_source() {
        let s = InMemoryStore::new();
        let nb = Uuid::new_v4();
        let sid = Uuid::new_v4();
        let chunks = vec![
            chunk_with(sid, 0, vec!["A"], "a"),
            chunk_with(sid, 1, vec!["B"], "b"),
            chunk_with(sid, 2, vec!["C"], "c"),
            chunk_with(sid, 3, vec!["D"], "d"),
        ];
        let embs = vec![vec![0.0]; chunks.len()];
        s.upsert(nb, &chunks, &embs).await.unwrap();
        let reps = s.representative_chunks(Some(nb), 2).await.unwrap();
        assert_eq!(reps.len(), 2);
        assert_eq!(reps[0].text, "a");
        assert_eq!(reps[1].text, "b");
    }

    #[tokio::test]
    async fn representative_chunks_falls_back_to_ord_when_no_headings() {
        let s = InMemoryStore::new();
        let nb = Uuid::new_v4();
        let sid = Uuid::new_v4();
        let chunks = vec![
            chunk_with(sid, 0, vec![], "p1"),
            chunk_with(sid, 1, vec![], "p2"),
            chunk_with(sid, 2, vec![], "p3"),
        ];
        let embs = vec![vec![0.0]; chunks.len()];
        s.upsert(nb, &chunks, &embs).await.unwrap();
        let reps = s.representative_chunks(Some(nb), 2).await.unwrap();
        let texts: Vec<&str> = reps.iter().map(|c| c.text.as_str()).collect();
        assert_eq!(texts, vec!["p1", "p2"]);
    }

    #[tokio::test]
    async fn representative_chunks_separates_sources() {
        let s = InMemoryStore::new();
        let nb = Uuid::new_v4();
        let s1 = Uuid::new_v4();
        let s2 = Uuid::new_v4();
        let chunks = vec![
            chunk_with(s1, 0, vec!["X1"], "s1-a"),
            chunk_with(s1, 1, vec!["X2"], "s1-b"),
            chunk_with(s2, 0, vec!["Y1"], "s2-a"),
            chunk_with(s2, 1, vec!["Y2"], "s2-b"),
        ];
        let embs = vec![vec![0.0]; chunks.len()];
        s.upsert(nb, &chunks, &embs).await.unwrap();
        let reps = s.representative_chunks(Some(nb), 2).await.unwrap();
        assert_eq!(reps.len(), 4);
        // 各 source から 2 件ずつ拾えていること
        let s1_count = reps.iter().filter(|c| c.source_id == s1).count();
        let s2_count = reps.iter().filter(|c| c.source_id == s2).count();
        assert_eq!(s1_count, 2);
        assert_eq!(s2_count, 2);
    }

    #[tokio::test]
    async fn representative_chunks_respects_scope() {
        let s = InMemoryStore::new();
        let nb_a = Uuid::new_v4();
        let nb_b = Uuid::new_v4();
        s.upsert(
            nb_a,
            &[chunk_with(Uuid::new_v4(), 0, vec!["A"], "in-a")],
            &[vec![0.0]],
        )
        .await
        .unwrap();
        s.upsert(
            nb_b,
            &[chunk_with(Uuid::new_v4(), 0, vec!["B"], "in-b")],
            &[vec![0.0]],
        )
        .await
        .unwrap();
        let reps_a = s.representative_chunks(Some(nb_a), 3).await.unwrap();
        assert_eq!(reps_a.len(), 1);
        assert_eq!(reps_a[0].text, "in-a");
    }

    #[tokio::test]
    async fn representative_chunks_zero_per_source_returns_empty() {
        let s = InMemoryStore::new();
        let nb = Uuid::new_v4();
        s.upsert(
            nb,
            &[chunk_with(Uuid::new_v4(), 0, vec!["A"], "x")],
            &[vec![0.0]],
        )
        .await
        .unwrap();
        let reps = s.representative_chunks(Some(nb), 0).await.unwrap();
        assert!(reps.is_empty());
    }

    #[tokio::test]
    async fn delete_by_notebook_clears_only_target() {
        let s = InMemoryStore::new();
        let nb_a = Uuid::new_v4();
        let nb_b = Uuid::new_v4();
        s.upsert(nb_a, &[chunk("a1"), chunk("a2")], &[vec![1.0], vec![1.0]])
            .await
            .unwrap();
        s.upsert(nb_b, &[chunk("b1")], &[vec![1.0]]).await.unwrap();
        let removed = s.delete_by_notebook(nb_a).await.unwrap();
        assert_eq!(removed, 2);
        assert_eq!(s.keyword_search(Some(nb_a), "a", 5).await.unwrap().len(), 0);
        assert_eq!(
            s.keyword_search(Some(nb_b), "b1", 5).await.unwrap().len(),
            1
        );
    }

    #[tokio::test]
    async fn all_defined_terms_extracts_quoted_definitions() {
        let s = InMemoryStore::new();
        let nb = Uuid::new_v4();
        // chunk A: 定義句あり (2 つの定義)
        // chunk B: 定義句なし (caption のみ)
        let a_text = "(納税義務者等)\n第129条 事業所税は…事務所又は事業所 (以下本節において「事業所等」という。) において…\n2 「特別徴収義務者」をいう。";
        let b_text = "(都市計画税の税率)\n第133条 都市計画税の税率は…";
        let a = chunk(a_text);
        let b = chunk(b_text);
        let a_id = a.id;
        s.upsert(nb, &[a, b], &[vec![1.0], vec![1.0]])
            .await
            .unwrap();

        let terms = s.all_defined_terms(Some(nb)).await.unwrap();
        let a_terms: Vec<&str> = terms
            .iter()
            .filter(|(id, _)| *id == a_id)
            .map(|(_, t)| t.as_str())
            .collect();
        assert!(a_terms.contains(&"事業所等"), "got {a_terms:?}");
        assert!(a_terms.contains(&"特別徴収義務者"), "got {a_terms:?}");
        // chunk B には定義語なし
        assert_eq!(
            terms
                .iter()
                .filter(|(_, t)| t == "都市計画税の税率")
                .count(),
            0
        );
    }

    #[tokio::test]
    async fn all_defined_terms_respects_notebook_scope() {
        let s = InMemoryStore::new();
        let nb_a = Uuid::new_v4();
        let nb_b = Uuid::new_v4();
        let a = chunk("「Aさん」という。");
        let b = chunk("「Bさん」という。");
        s.upsert(nb_a, &[a], &[vec![1.0]]).await.unwrap();
        s.upsert(nb_b, &[b], &[vec![1.0]]).await.unwrap();
        let scoped_a = s.all_defined_terms(Some(nb_a)).await.unwrap();
        assert_eq!(scoped_a.len(), 1);
        assert_eq!(scoped_a[0].1, "Aさん");
    }

    #[tokio::test]
    async fn count_chunks_in_scope_separates_notebooks() {
        let s = InMemoryStore::new();
        let nb_a = Uuid::new_v4();
        let nb_b = Uuid::new_v4();
        s.upsert(
            nb_a,
            &[chunk("a1"), chunk("a2"), chunk("a3")],
            &[vec![1.0], vec![1.0], vec![1.0]],
        )
        .await
        .unwrap();
        s.upsert(nb_b, &[chunk("b1")], &[vec![1.0]]).await.unwrap();
        assert_eq!(s.count_chunks_in_scope(Some(nb_a)).await.unwrap(), 3);
        assert_eq!(s.count_chunks_in_scope(Some(nb_b)).await.unwrap(), 1);
        assert_eq!(s.count_chunks_in_scope(None).await.unwrap(), 4);
    }
}
