//! SQLite + sqlite-vec + FTS5 によるベクトル + キーワード検索ストア (Notebook scope 対応)。

use async_trait::async_trait;
use ellisii_core::{Chunk, Error, HitSource, Result, SearchHit};
use ellisii_jp_tokenizer_bigram::CharBigramTokenizer;
use ellisii_jp_tokenizer_core::JpTokenizer;
use ellisii_store_core::{pick_for_topic, pick_representatives, Scope, VectorStore};
use parking_lot::Mutex;
use rusqlite::{params, Connection};
use std::path::Path;
use std::sync::Arc;
use std::sync::Once;
use uuid::Uuid;
use zerocopy::AsBytes;

static SQLITE_VEC_INIT: Once = Once::new();

type AutoExtFn = unsafe extern "C" fn(
    *mut rusqlite::ffi::sqlite3,
    *mut *mut std::os::raw::c_char,
    *const rusqlite::ffi::sqlite3_api_routines,
) -> std::os::raw::c_int;

fn ensure_sqlite_vec_loaded() {
    SQLITE_VEC_INIT.call_once(|| unsafe {
        let f: AutoExtFn = std::mem::transmute(sqlite_vec::sqlite3_vec_init as *const ());
        rusqlite::ffi::sqlite3_auto_extension(Some(f));
    });
}

pub struct SqliteStore {
    conn: Arc<Mutex<Connection>>,
    dim: usize,
    tokenizer: Arc<dyn JpTokenizer>,
}

impl SqliteStore {
    pub fn open(path: impl AsRef<Path>, dim: usize) -> Result<Self> {
        Self::open_with_tokenizer(path, dim, Arc::new(CharBigramTokenizer::new()))
    }

    pub fn open_with_tokenizer(
        path: impl AsRef<Path>,
        dim: usize,
        tokenizer: Arc<dyn JpTokenizer>,
    ) -> Result<Self> {
        ensure_sqlite_vec_loaded();
        let conn = Connection::open(&path).map_err(|e| Error::Store(format!("open: {e}")))?;
        Self::init_schema(&conn, dim)?;
        Ok(Self {
            conn: Arc::new(Mutex::new(conn)),
            dim,
            tokenizer,
        })
    }

    pub fn open_in_memory(dim: usize) -> Result<Self> {
        Self::open_in_memory_with_tokenizer(dim, Arc::new(CharBigramTokenizer::new()))
    }

    pub fn open_in_memory_with_tokenizer(
        dim: usize,
        tokenizer: Arc<dyn JpTokenizer>,
    ) -> Result<Self> {
        ensure_sqlite_vec_loaded();
        let conn = Connection::open_in_memory().map_err(|e| Error::Store(format!("mem: {e}")))?;
        Self::init_schema(&conn, dim)?;
        Ok(Self {
            conn: Arc::new(Mutex::new(conn)),
            dim,
            tokenizer,
        })
    }

    fn init_schema(conn: &Connection, dim: usize) -> Result<()> {
        conn.execute_batch(
            r#"
            CREATE TABLE IF NOT EXISTS chunks (
                id TEXT PRIMARY KEY,
                notebook_id TEXT NOT NULL DEFAULT '',
                source_id TEXT NOT NULL,
                ord INTEGER NOT NULL,
                text TEXT NOT NULL,
                heading_path TEXT NOT NULL,
                page INTEGER,
                bbox TEXT,
                summary TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_chunks_source ON chunks(source_id);
            CREATE INDEX IF NOT EXISTS idx_chunks_notebook ON chunks(notebook_id);

            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                tokens,
                tokenize='unicode61 remove_diacritics 2'
            );
            CREATE TABLE IF NOT EXISTS chunks_fts_map (
                fts_rowid INTEGER PRIMARY KEY,
                chunk_id TEXT UNIQUE NOT NULL
            );
            "#,
        )
        .map_err(|e| Error::Store(format!("schema: {e}")))?;

        // 既存 DB へのマイグレーション (notebook_id 列が無ければ追加)
        let has_col = conn
            .prepare("SELECT 1 FROM pragma_table_info('chunks') WHERE name = 'notebook_id'")
            .and_then(|mut s| s.exists([]))
            .unwrap_or(false);
        if !has_col {
            conn.execute_batch(
                "ALTER TABLE chunks ADD COLUMN notebook_id TEXT NOT NULL DEFAULT '';
                 CREATE INDEX IF NOT EXISTS idx_chunks_notebook ON chunks(notebook_id);",
            )
            .map_err(|e| Error::Store(format!("migrate notebook_id: {e}")))?;
        }

        // 既存 chunks_vec があれば次元を確認し、新しい埋め込みモデルの dim と
        // 一致しなければチャンクごと作り直す (dummy → 1024dim 切替などで発生)。
        let existing_dim = sniff_chunks_vec_dim(conn);
        if let Some(prev_dim) = existing_dim {
            if prev_dim != dim {
                tracing::warn!(
                    "vector dim mismatch: stored={}, embedder={}; resetting chunks store \
                     (既存チャンクは消えますが、ソース自体は残るので ↻ で再インデックスしてください)",
                    prev_dim,
                    dim
                );
                conn.execute_batch(
                    "DROP TABLE IF EXISTS chunks_vec;
                     DELETE FROM chunks;
                     DELETE FROM chunks_fts;
                     DELETE FROM chunks_fts_map;",
                )
                .map_err(|e| Error::Store(format!("reset on dim change: {e}")))?;
            }
        }

        let vec_sql = format!(
            "CREATE VIRTUAL TABLE IF NOT EXISTS chunks_vec USING vec0(
                chunk_id TEXT PRIMARY KEY,
                embedding float[{dim}]
            )"
        );
        conn.execute(&vec_sql, [])
            .map_err(|e| Error::Store(format!("vec table: {e}")))?;
        Ok(())
    }
}

/// 既存 `chunks_vec` 仮想テーブルの定義から `embedding` カラムの次元を読む。
/// 存在しない / 解析失敗時は None。
fn sniff_chunks_vec_dim(conn: &Connection) -> Option<usize> {
    let sql: String = conn
        .query_row(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='chunks_vec'",
            [],
            |row| row.get(0),
        )
        .ok()?;
    // 例: "CREATE VIRTUAL TABLE chunks_vec USING vec0(chunk_id TEXT PRIMARY KEY, embedding float[256])"
    let re = regex::Regex::new(r"embedding\s+float\[(\d+)\]").ok()?;
    let cap = re.captures(&sql)?;
    cap.get(1)?.as_str().parse::<usize>().ok()
}

#[async_trait]
impl VectorStore for SqliteStore {
    async fn upsert(
        &self,
        notebook_id: Uuid,
        chunks: &[Chunk],
        embeddings: &[Vec<f32>],
    ) -> Result<()> {
        if chunks.len() != embeddings.len() {
            return Err(Error::Store("chunks/embeddings length mismatch".into()));
        }
        let conn = self.conn.clone();
        let dim = self.dim;
        let nb = notebook_id.to_string();
        // FTS 入力には heading_path を本文の前に折り畳んでから tokenize する。
        // 章タイトルや条文ラベルは BM25 で当たって欲しい (例: 「第3章」「第94条」
        // を query に含むケース)。citation 表示や rerank には影響しない (chunk
        // 本体は `c.text` のまま保存している)。
        let token_strs: Vec<String> = chunks
            .iter()
            .map(|c| {
                if c.heading_path.is_empty() {
                    self.tokenizer.tokenize_for_fts(&c.text)
                } else {
                    let combined = format!("{}\n{}", c.heading_path.join(" "), c.text);
                    self.tokenizer.tokenize_for_fts(&combined)
                }
            })
            .collect();
        let chunks = chunks.to_vec();
        let embeddings = embeddings.to_vec();
        tokio::task::spawn_blocking(move || -> Result<()> {
            let mut c = conn.lock();
            let tx = c.transaction().map_err(|e| Error::Store(format!("tx: {e}")))?;
            for ((chunk, emb), tokens) in chunks.iter().zip(embeddings.iter()).zip(token_strs.iter()) {
                if emb.len() != dim {
                    return Err(Error::Store(format!(
                        "embedding dim mismatch: got {} expect {dim}",
                        emb.len()
                    )));
                }
                let bbox_json = chunk
                    .bbox
                    .as_ref()
                    .map(|b| serde_json::to_string(b).unwrap_or_default());
                let heading = serde_json::to_string(&chunk.heading_path).unwrap_or("[]".into());
                tx.execute(
                    "INSERT OR REPLACE INTO chunks (id, notebook_id, source_id, ord, text, heading_path, page, bbox, summary)
                     VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
                    params![
                        chunk.id.to_string(),
                        nb,
                        chunk.source_id.to_string(),
                        chunk.ord,
                        chunk.text,
                        heading,
                        chunk.page,
                        bbox_json,
                        chunk.summary,
                    ],
                )
                .map_err(|e| Error::Store(format!("insert chunk: {e}")))?;
                let chunk_id_s = chunk.id.to_string();
                let existing_rowid: Option<i64> = tx
                    .query_row(
                        "SELECT fts_rowid FROM chunks_fts_map WHERE chunk_id = ?1",
                        params![chunk_id_s],
                        |r| r.get(0),
                    )
                    .ok();
                if let Some(rowid) = existing_rowid {
                    tx.execute(
                        "DELETE FROM chunks_fts WHERE rowid = ?1",
                        params![rowid],
                    )
                    .map_err(|e| Error::Store(format!("fts del: {e}")))?;
                    tx.execute(
                        "INSERT INTO chunks_fts (rowid, tokens) VALUES (?1, ?2)",
                        params![rowid, tokens],
                    )
                    .map_err(|e| Error::Store(format!("fts ins: {e}")))?;
                } else {
                    tx.execute(
                        "INSERT INTO chunks_fts (tokens) VALUES (?1)",
                        params![tokens],
                    )
                    .map_err(|e| Error::Store(format!("fts ins: {e}")))?;
                    let rowid = tx.last_insert_rowid();
                    tx.execute(
                        "INSERT INTO chunks_fts_map (fts_rowid, chunk_id) VALUES (?1, ?2)",
                        params![rowid, chunk_id_s],
                    )
                    .map_err(|e| Error::Store(format!("fts map: {e}")))?;
                }
                let bytes = floats_to_bytes(emb);
                tx.execute(
                    "INSERT OR REPLACE INTO chunks_vec (chunk_id, embedding) VALUES (?1, ?2)",
                    params![chunk.id.to_string(), bytes],
                )
                .map_err(|e| Error::Store(format!("vec insert: {e}")))?;
            }
            tx.commit().map_err(|e| Error::Store(format!("commit: {e}")))?;
            Ok(())
        })
        .await
        .map_err(|e| Error::Store(format!("join: {e}")))??;
        Ok(())
    }

    async fn search(&self, scope: Scope, query: &[f32], top_k: usize) -> Result<Vec<SearchHit>> {
        if query.len() != self.dim {
            return Err(Error::Store("query dim mismatch".into()));
        }
        let conn = self.conn.clone();
        let q = query.to_vec();
        // sqlite-vec の MATCH には追加 WHERE が不可なので、多めに取って scope で絞る
        let oversample = if scope.is_some() {
            (top_k * 8).max(32)
        } else {
            top_k
        };
        let want_nb = scope.map(|u| u.to_string());
        tokio::task::spawn_blocking(move || -> Result<Vec<SearchHit>> {
            let c = conn.lock();
            let bytes = floats_to_bytes(&q);
            let mut stmt = c
                .prepare(
                    "SELECT chunk_id, distance FROM chunks_vec
                     WHERE embedding MATCH ?1 AND k = ?2
                     ORDER BY distance",
                )
                .map_err(|e| Error::Store(format!("prep vec: {e}")))?;
            let rows = stmt
                .query_map(params![bytes, oversample as i64], |row| {
                    let id: String = row.get(0)?;
                    let dist: f64 = row.get(1)?;
                    Ok((id, dist))
                })
                .map_err(|e| Error::Store(format!("query vec: {e}")))?;
            let mut hits = Vec::new();
            for r in rows {
                let (id, dist) = r.map_err(|e| Error::Store(format!("row: {e}")))?;
                if let Some((chunk, nb)) = load_chunk_with_nb(&c, &id)? {
                    if let Some(want) = &want_nb {
                        if nb.as_deref() != Some(want.as_str()) {
                            continue;
                        }
                    }
                    hits.push(SearchHit {
                        chunk,
                        score: 1.0 - dist as f32,
                        source: HitSource::Vector,
                    });
                    if hits.len() >= top_k {
                        break;
                    }
                }
            }
            Ok(hits)
        })
        .await
        .map_err(|e| Error::Store(format!("join: {e}")))?
    }

    async fn delete_by_source(&self, source_id: Uuid) -> Result<usize> {
        let conn = self.conn.clone();
        let sid = source_id.to_string();
        tokio::task::spawn_blocking(move || -> Result<usize> {
            let mut c = conn.lock();
            let tx = c
                .transaction()
                .map_err(|e| Error::Store(format!("tx: {e}")))?;
            let ids: Vec<String> = {
                let mut stmt = tx
                    .prepare("SELECT id FROM chunks WHERE source_id = ?1")
                    .map_err(|e| Error::Store(format!("prep: {e}")))?;
                let rows = stmt
                    .query_map(params![sid], |r| r.get::<_, String>(0))
                    .map_err(|e| Error::Store(format!("query: {e}")))?;
                rows.filter_map(|r| r.ok()).collect()
            };
            for id in &ids {
                if let Ok(rowid) = tx.query_row::<i64, _, _>(
                    "SELECT fts_rowid FROM chunks_fts_map WHERE chunk_id = ?1",
                    params![id],
                    |r| r.get(0),
                ) {
                    tx.execute("DELETE FROM chunks_fts WHERE rowid = ?1", params![rowid])
                        .ok();
                    tx.execute(
                        "DELETE FROM chunks_fts_map WHERE chunk_id = ?1",
                        params![id],
                    )
                    .ok();
                }
                tx.execute("DELETE FROM chunks_vec WHERE chunk_id = ?1", params![id])
                    .map_err(|e| Error::Store(format!("vec del: {e}")))?;
            }
            let removed = tx
                .execute("DELETE FROM chunks WHERE source_id = ?1", params![sid])
                .map_err(|e| Error::Store(format!("chunk del: {e}")))?;
            tx.commit()
                .map_err(|e| Error::Store(format!("commit: {e}")))?;
            Ok(removed)
        })
        .await
        .map_err(|e| Error::Store(format!("join: {e}")))?
    }

    async fn delete_by_notebook(&self, notebook_id: Uuid) -> Result<usize> {
        let conn = self.conn.clone();
        let nb = notebook_id.to_string();
        tokio::task::spawn_blocking(move || -> Result<usize> {
            let mut c = conn.lock();
            let tx = c
                .transaction()
                .map_err(|e| Error::Store(format!("tx: {e}")))?;
            let ids: Vec<String> = {
                let mut stmt = tx
                    .prepare("SELECT id FROM chunks WHERE notebook_id = ?1")
                    .map_err(|e| Error::Store(format!("prep: {e}")))?;
                let rows = stmt
                    .query_map(params![nb], |r| r.get::<_, String>(0))
                    .map_err(|e| Error::Store(format!("query: {e}")))?;
                rows.filter_map(|r| r.ok()).collect()
            };
            for id in &ids {
                if let Ok(rowid) = tx.query_row::<i64, _, _>(
                    "SELECT fts_rowid FROM chunks_fts_map WHERE chunk_id = ?1",
                    params![id],
                    |r| r.get(0),
                ) {
                    tx.execute("DELETE FROM chunks_fts WHERE rowid = ?1", params![rowid])
                        .ok();
                    tx.execute(
                        "DELETE FROM chunks_fts_map WHERE chunk_id = ?1",
                        params![id],
                    )
                    .ok();
                }
                tx.execute("DELETE FROM chunks_vec WHERE chunk_id = ?1", params![id])
                    .map_err(|e| Error::Store(format!("vec del: {e}")))?;
            }
            let removed = tx
                .execute("DELETE FROM chunks WHERE notebook_id = ?1", params![nb])
                .map_err(|e| Error::Store(format!("chunk del: {e}")))?;
            tx.commit()
                .map_err(|e| Error::Store(format!("commit: {e}")))?;
            Ok(removed)
        })
        .await
        .map_err(|e| Error::Store(format!("join: {e}")))?
    }

    async fn count_chunks(&self, source_id: Uuid) -> Result<usize> {
        let conn = self.conn.clone();
        let sid = source_id.to_string();
        tokio::task::spawn_blocking(move || -> Result<usize> {
            let c = conn.lock();
            let n: i64 = c
                .query_row(
                    "SELECT COUNT(*) FROM chunks WHERE source_id = ?1",
                    params![sid],
                    |r| r.get(0),
                )
                .map_err(|e| Error::Store(format!("count: {e}")))?;
            Ok(n as usize)
        })
        .await
        .map_err(|e| Error::Store(format!("join: {e}")))?
    }

    async fn count_chunks_in_scope(&self, scope: Scope) -> Result<usize> {
        let conn = self.conn.clone();
        tokio::task::spawn_blocking(move || -> Result<usize> {
            let c = conn.lock();
            let n: i64 = match scope {
                Some(nb) => c
                    .query_row(
                        "SELECT COUNT(*) FROM chunks WHERE notebook_id = ?1",
                        params![nb.to_string()],
                        |r| r.get(0),
                    )
                    .map_err(|e| Error::Store(format!("count: {e}")))?,
                None => c
                    .query_row("SELECT COUNT(*) FROM chunks", params![], |r| r.get(0))
                    .map_err(|e| Error::Store(format!("count: {e}")))?,
            };
            Ok(n as usize)
        })
        .await
        .map_err(|e| Error::Store(format!("join: {e}")))?
    }

    async fn count_sources_in_scope(&self, scope: Scope) -> Result<usize> {
        let conn = self.conn.clone();
        tokio::task::spawn_blocking(move || -> Result<usize> {
            let c = conn.lock();
            let n: i64 = match scope {
                Some(nb) => c
                    .query_row(
                        "SELECT COUNT(DISTINCT source_id) FROM chunks WHERE notebook_id = ?1",
                        params![nb.to_string()],
                        |r| r.get(0),
                    )
                    .map_err(|e| Error::Store(format!("count sources: {e}")))?,
                None => c
                    .query_row(
                        "SELECT COUNT(DISTINCT source_id) FROM chunks",
                        params![],
                        |r| r.get(0),
                    )
                    .map_err(|e| Error::Store(format!("count sources: {e}")))?,
            };
            Ok(n as usize)
        })
        .await
        .map_err(|e| Error::Store(format!("join: {e}")))?
    }

    async fn representative_chunks(&self, scope: Scope, per_source: usize) -> Result<Vec<Chunk>> {
        self.representative_chunks_for_topic(scope, per_source, "")
            .await
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
        let conn = self.conn.clone();
        let want_nb = scope.map(|u| u.to_string());
        let topic = topic.to_string();
        tokio::task::spawn_blocking(move || -> Result<Vec<Chunk>> {
            let c = conn.lock();
            // notebook scope を SQL レベルで絞り、source_id, ord 昇順で全 chunk を回収。
            // 件数は notebook 全体に依存するが、要約クエリは頻度が低く scope は
            // 1 notebook に閉じるため許容範囲。
            let (sql, params_vec): (&str, Vec<String>) = if let Some(ref nb) = want_nb {
                (
                    "SELECT id, source_id, ord, text, heading_path, page, bbox, summary
                     FROM chunks
                     WHERE notebook_id = ?1
                     ORDER BY source_id ASC, ord ASC",
                    vec![nb.clone()],
                )
            } else {
                (
                    "SELECT id, source_id, ord, text, heading_path, page, bbox, summary
                     FROM chunks
                     ORDER BY source_id ASC, ord ASC",
                    vec![],
                )
            };
            let mut stmt = c
                .prepare(sql)
                .map_err(|e| Error::Store(format!("prepare: {e}")))?;
            let rows_iter = if params_vec.is_empty() {
                stmt.query_map([], row_to_chunk)
            } else {
                stmt.query_map(params![params_vec[0]], row_to_chunk)
            }
            .map_err(|e| Error::Store(format!("query: {e}")))?;
            // source ごとにバッファして処理 (source_id でソート済み前提)。
            let pick = |buf: &[Chunk]| -> Vec<Chunk> {
                if topic.is_empty() {
                    pick_representatives(buf, per_source)
                } else {
                    pick_for_topic(buf, per_source, &topic)
                }
            };
            let mut out: Vec<Chunk> = Vec::new();
            let mut current_sid: Option<Uuid> = None;
            let mut buf: Vec<Chunk> = Vec::new();
            for row in rows_iter {
                let chunk = row.map_err(|e| Error::Store(format!("row: {e}")))?;
                if Some(chunk.source_id) != current_sid {
                    if !buf.is_empty() {
                        out.extend(pick(&buf));
                        buf.clear();
                    }
                    current_sid = Some(chunk.source_id);
                }
                buf.push(chunk);
            }
            if !buf.is_empty() {
                out.extend(pick(&buf));
            }
            Ok(out)
        })
        .await
        .map_err(|e| Error::Store(format!("join: {e}")))?
    }

    async fn neighbor_chunks(
        &self,
        source_id: Uuid,
        ord_center: u32,
        window: u32,
    ) -> Result<Vec<(u32, String)>> {
        let conn = self.conn.clone();
        let sid = source_id.to_string();
        let lo = ord_center.saturating_sub(window) as i64;
        let hi = (ord_center + window) as i64;
        tokio::task::spawn_blocking(move || -> Result<Vec<(u32, String)>> {
            let c = conn.lock();
            let mut stmt = c
                .prepare(
                    "SELECT ord, text FROM chunks \
                     WHERE source_id = ?1 AND ord BETWEEN ?2 AND ?3 \
                     ORDER BY ord ASC",
                )
                .map_err(|e| Error::Store(format!("prepare: {e}")))?;
            let rows = stmt
                .query_map(params![sid, lo, hi], |r| {
                    Ok((r.get::<_, i64>(0)? as u32, r.get::<_, String>(1)?))
                })
                .map_err(|e| Error::Store(format!("query: {e}")))?;
            let mut out = Vec::new();
            for row in rows {
                out.push(row.map_err(|e| Error::Store(format!("row: {e}")))?);
            }
            Ok(out)
        })
        .await
        .map_err(|e| Error::Store(format!("join: {e}")))?
    }

    async fn all_captions(&self, scope: Scope) -> Result<Vec<(Uuid, String)>> {
        let conn = self.conn.clone();
        tokio::task::spawn_blocking(move || -> Result<Vec<(Uuid, String)>> {
            let c = conn.lock();
            // chunk 先頭の `(...)` を SQL 側で大雑把に取り出す。空白で trim 後 '(' で
            // 始まらないものは弾き、最後の rust 側で再確認する。
            let (sql, mut rows) = match scope {
                Some(nb) => {
                    let nb_s = nb.to_string();
                    let mut stmt = c
                        .prepare(
                            "SELECT id, substr(text, 1, 200) FROM chunks \
                             WHERE notebook_id = ?1",
                        )
                        .map_err(|e| Error::Store(format!("prepare: {e}")))?;
                    let v: Vec<(String, String)> = stmt
                        .query_map(params![nb_s], |r| {
                            Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?))
                        })
                        .map_err(|e| Error::Store(format!("query: {e}")))?
                        .collect::<rusqlite::Result<Vec<_>>>()
                        .map_err(|e| Error::Store(format!("row: {e}")))?;
                    ("scoped", v)
                }
                None => {
                    let mut stmt = c
                        .prepare("SELECT id, substr(text, 1, 200) FROM chunks")
                        .map_err(|e| Error::Store(format!("prepare: {e}")))?;
                    let v: Vec<(String, String)> = stmt
                        .query_map(params![], |r| {
                            Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?))
                        })
                        .map_err(|e| Error::Store(format!("query: {e}")))?
                        .collect::<rusqlite::Result<Vec<_>>>()
                        .map_err(|e| Error::Store(format!("row: {e}")))?;
                    ("all", v)
                }
            };
            let _ = sql;
            let mut out = Vec::with_capacity(rows.len());
            for (id, head) in rows.drain(..) {
                if let Some(cap) = extract_leading_caption(&head) {
                    if let Ok(uid) = Uuid::parse_str(&id) {
                        out.push((uid, cap.to_string()));
                    }
                }
            }
            Ok(out)
        })
        .await
        .map_err(|e| Error::Store(format!("join: {e}")))?
    }

    async fn all_headings(&self, scope: Scope) -> Result<Vec<(Uuid, String)>> {
        let conn = self.conn.clone();
        tokio::task::spawn_blocking(move || -> Result<Vec<(Uuid, String)>> {
            let c = conn.lock();
            let rows: Vec<(String, String)> = match scope {
                Some(nb) => {
                    let nb_s = nb.to_string();
                    let mut stmt = c
                        .prepare("SELECT id, heading_path FROM chunks WHERE notebook_id = ?1")
                        .map_err(|e| Error::Store(format!("prepare: {e}")))?;
                    let iter = stmt
                        .query_map(params![nb_s], |r| {
                            Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?))
                        })
                        .map_err(|e| Error::Store(format!("query: {e}")))?;
                    let v: rusqlite::Result<Vec<(String, String)>> = iter.collect();
                    v.map_err(|e| Error::Store(format!("row: {e}")))?
                }
                None => {
                    let mut stmt = c
                        .prepare("SELECT id, heading_path FROM chunks")
                        .map_err(|e| Error::Store(format!("prepare: {e}")))?;
                    let iter = stmt
                        .query_map(params![], |r| {
                            Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?))
                        })
                        .map_err(|e| Error::Store(format!("query: {e}")))?;
                    let v: rusqlite::Result<Vec<(String, String)>> = iter.collect();
                    v.map_err(|e| Error::Store(format!("row: {e}")))?
                }
            };
            let mut out = Vec::with_capacity(rows.len());
            for (id, heading_json) in rows {
                let segs: Vec<String> = serde_json::from_str(&heading_json).unwrap_or_default();
                if segs.is_empty() {
                    continue;
                }
                let joined = segs.join("/");
                if let Ok(uid) = Uuid::parse_str(&id) {
                    out.push((uid, joined));
                }
            }
            Ok(out)
        })
        .await
        .map_err(|e| Error::Store(format!("join: {e}")))?
    }

    /// 全 chunk の本文先頭 200 char から `extract_defined_terms` で定義語を抽出し、
    /// 1 row/term で `(chunk_id, term)` を返す (Run 42)。
    async fn all_defined_terms(&self, scope: Scope) -> Result<Vec<(Uuid, String)>> {
        let conn = self.conn.clone();
        tokio::task::spawn_blocking(move || -> Result<Vec<(Uuid, String)>> {
            let c = conn.lock();
            let rows: Vec<(String, String)> = match scope {
                Some(nb) => {
                    let nb_s = nb.to_string();
                    let mut stmt = c
                        .prepare(
                            "SELECT id, substr(text, 1, 400) FROM chunks WHERE notebook_id = ?1",
                        )
                        .map_err(|e| Error::Store(format!("prepare: {e}")))?;
                    let iter = stmt
                        .query_map(params![nb_s], |r| {
                            Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?))
                        })
                        .map_err(|e| Error::Store(format!("query: {e}")))?;
                    let v: rusqlite::Result<Vec<(String, String)>> = iter.collect();
                    v.map_err(|e| Error::Store(format!("row: {e}")))?
                }
                None => {
                    let mut stmt = c
                        .prepare("SELECT id, substr(text, 1, 400) FROM chunks")
                        .map_err(|e| Error::Store(format!("prepare: {e}")))?;
                    let iter = stmt
                        .query_map(params![], |r| {
                            Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?))
                        })
                        .map_err(|e| Error::Store(format!("query: {e}")))?;
                    let v: rusqlite::Result<Vec<(String, String)>> = iter.collect();
                    v.map_err(|e| Error::Store(format!("row: {e}")))?
                }
            };
            let mut out: Vec<(Uuid, String)> = Vec::with_capacity(rows.len());
            for (id, head) in rows {
                let Ok(uid) = Uuid::parse_str(&id) else {
                    continue;
                };
                for term in ellisii_core::caption::extract_defined_terms(&head) {
                    out.push((uid, term.to_string()));
                }
            }
            Ok(out)
        })
        .await
        .map_err(|e| Error::Store(format!("join: {e}")))?
    }

    async fn get_chunks_by_ids(&self, ids: &[Uuid]) -> Result<Vec<Chunk>> {
        if ids.is_empty() {
            return Ok(Vec::new());
        }
        let conn = self.conn.clone();
        let id_strs: Vec<String> = ids.iter().map(|u| u.to_string()).collect();
        tokio::task::spawn_blocking(move || -> Result<Vec<Chunk>> {
            let c = conn.lock();
            // IN 句の placeholders を組み立て (id 数は high のときも 1k 程度を想定)。
            let placeholders = (0..id_strs.len())
                .map(|i| format!("?{}", i + 1))
                .collect::<Vec<_>>()
                .join(",");
            let sql = format!(
                "SELECT id, source_id, ord, text, heading_path, page, bbox, summary \
                 FROM chunks WHERE id IN ({placeholders})"
            );
            let mut stmt = c
                .prepare(&sql)
                .map_err(|e| Error::Store(format!("prepare: {e}")))?;
            let params_dyn: Vec<&dyn rusqlite::ToSql> =
                id_strs.iter().map(|s| s as &dyn rusqlite::ToSql).collect();
            let rows = stmt
                .query_map(rusqlite::params_from_iter(params_dyn), row_to_chunk)
                .map_err(|e| Error::Store(format!("query: {e}")))?;
            let mut out = Vec::with_capacity(id_strs.len());
            for r in rows {
                out.push(r.map_err(|e| Error::Store(format!("row: {e}")))?);
            }
            Ok(out)
        })
        .await
        .map_err(|e| Error::Store(format!("join: {e}")))?
    }

    async fn texts_by_source(&self, source_id: Uuid) -> Result<Vec<String>> {
        let conn = self.conn.clone();
        let sid = source_id.to_string();
        tokio::task::spawn_blocking(move || -> Result<Vec<String>> {
            let c = conn.lock();
            let mut stmt = c
                .prepare("SELECT text FROM chunks WHERE source_id = ?1 ORDER BY ord ASC")
                .map_err(|e| Error::Store(format!("prepare: {e}")))?;
            let rows = stmt
                .query_map(params![sid], |r| r.get::<_, String>(0))
                .map_err(|e| Error::Store(format!("query: {e}")))?;
            let mut out = Vec::new();
            for row in rows {
                out.push(row.map_err(|e| Error::Store(format!("row: {e}")))?);
            }
            Ok(out)
        })
        .await
        .map_err(|e| Error::Store(format!("join: {e}")))?
    }

    async fn keyword_search(
        &self,
        scope: Scope,
        query: &str,
        top_k: usize,
    ) -> Result<Vec<SearchHit>> {
        let conn = self.conn.clone();
        let q_tokens = self.tokenizer.tokenize(query);
        if q_tokens.is_empty() {
            return Ok(vec![]);
        }
        let q = q_tokens
            .iter()
            .map(|t| format!("\"{}\"", t.replace('\"', "")))
            .collect::<Vec<_>>()
            .join(" OR ");
        let oversample = if scope.is_some() {
            (top_k * 8).max(32)
        } else {
            top_k
        };
        let want_nb = scope.map(|u| u.to_string());
        tokio::task::spawn_blocking(move || -> Result<Vec<SearchHit>> {
            let c = conn.lock();
            let mut stmt = c
                .prepare(
                    "SELECT m.chunk_id, bm25(chunks_fts) AS score
                     FROM chunks_fts JOIN chunks_fts_map m ON m.fts_rowid = chunks_fts.rowid
                     WHERE chunks_fts MATCH ?1
                     ORDER BY score
                     LIMIT ?2",
                )
                .map_err(|e| Error::Store(format!("prep fts: {e}")))?;
            let rows = stmt
                .query_map(params![q, oversample as i64], |row| {
                    let id: String = row.get(0)?;
                    let score: f64 = row.get(1)?;
                    Ok((id, score))
                })
                .map_err(|e| Error::Store(format!("query fts: {e}")))?;
            let mut hits = Vec::new();
            for r in rows {
                let (id, score) = r.map_err(|e| Error::Store(format!("row: {e}")))?;
                if let Some((chunk, nb)) = load_chunk_with_nb(&c, &id)? {
                    if let Some(want) = &want_nb {
                        if nb.as_deref() != Some(want.as_str()) {
                            continue;
                        }
                    }
                    hits.push(SearchHit {
                        chunk,
                        score: -(score as f32),
                        source: HitSource::Keyword,
                    });
                    if hits.len() >= top_k {
                        break;
                    }
                }
            }
            Ok(hits)
        })
        .await
        .map_err(|e| Error::Store(format!("join: {e}")))?
    }
}

fn load_chunk_with_nb(c: &Connection, id: &str) -> Result<Option<(Chunk, Option<String>)>> {
    let mut stmt = c
        .prepare(
            "SELECT id, source_id, ord, text, heading_path, page, bbox, summary, notebook_id
             FROM chunks WHERE id = ?1",
        )
        .map_err(|e| Error::Store(format!("prep: {e}")))?;
    let mut rows = stmt
        .query(params![id])
        .map_err(|e| Error::Store(format!("query: {e}")))?;
    if let Some(row) = rows.next().map_err(|e| Error::Store(format!("row: {e}")))? {
        let id: String = row.get(0).map_err(|e| Error::Store(e.to_string()))?;
        let source_id: String = row.get(1).map_err(|e| Error::Store(e.to_string()))?;
        let ord: u32 = row.get(2).map_err(|e| Error::Store(e.to_string()))?;
        let text: String = row.get(3).map_err(|e| Error::Store(e.to_string()))?;
        let heading: String = row.get(4).map_err(|e| Error::Store(e.to_string()))?;
        let page: Option<u32> = row.get(5).ok();
        let bbox: Option<String> = row.get(6).ok();
        let summary: Option<String> = row.get(7).ok();
        let nb_raw: Option<String> = row.get(8).ok();
        let nb = nb_raw.filter(|s| !s.is_empty());
        return Ok(Some((
            Chunk {
                id: Uuid::parse_str(&id).map_err(|e| Error::Store(e.to_string()))?,
                source_id: Uuid::parse_str(&source_id).map_err(|e| Error::Store(e.to_string()))?,
                ord,
                text,
                heading_path: serde_json::from_str(&heading).unwrap_or_default(),
                page,
                bbox: bbox.and_then(|s| serde_json::from_str(&s).ok()),
                summary,
            },
            nb,
        )));
    }
    Ok(None)
}

/// `representative_chunks` 用の row → Chunk マッピング (notebook_id は読まない)。
fn row_to_chunk(row: &rusqlite::Row<'_>) -> rusqlite::Result<Chunk> {
    let id: String = row.get(0)?;
    let source_id: String = row.get(1)?;
    let ord: u32 = row.get(2)?;
    let text: String = row.get(3)?;
    let heading: String = row.get(4)?;
    let page: Option<u32> = row.get(5).ok();
    let bbox: Option<String> = row.get(6).ok();
    let summary: Option<String> = row.get(7).ok();
    Ok(Chunk {
        id: Uuid::parse_str(&id).unwrap_or(Uuid::nil()),
        source_id: Uuid::parse_str(&source_id).unwrap_or(Uuid::nil()),
        ord,
        text,
        heading_path: serde_json::from_str(&heading).unwrap_or_default(),
        page,
        bbox: bbox.and_then(|s| serde_json::from_str(&s).ok()),
        summary,
    })
}

// caption 抽出は `ellisii_core::caption` に集約済。`extract_caption_or_lead` を
// そのまま流用する (rag 側の rerank と同一実装)。
use ellisii_core::caption::extract_caption_or_lead as extract_leading_caption;

fn floats_to_bytes(v: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(v.len() * 4);
    for x in v {
        out.extend_from_slice(x.as_bytes());
    }
    out
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
            heading_path: vec!["A".into(), "B".into()],
            page: Some(1),
            bbox: None,
            summary: None,
        }
    }

    #[test]
    fn extract_leading_caption_basic() {
        assert_eq!(
            extract_leading_caption("(入湯税の税率)\n第123条"),
            Some("入湯税の税率")
        );
        assert_eq!(
            extract_leading_caption("  (たばこ税)第85条"),
            Some("たばこ税")
        );
        // `(...)` が無くても、第N条 + 本文があれば本文先頭を fallback として返す。
        assert_eq!(
            extract_leading_caption("第3条 普通税は次のとおり"),
            Some("普通税は次のとおり")
        );
        // 第N条 構造ですらない素の本文は None。
        assert_eq!(extract_leading_caption("ただの本文"), None);
        assert_eq!(extract_leading_caption(""), None);
    }

    #[test]
    fn extract_leading_caption_falls_back_after_revision_note() {
        // 改正注記 + 第N条 本文 のみ (yokohama golden [15] 構造)。
        let t = "(平18条例70・一部改正)\n\n第3条 横浜市が課する普通税は、市民税、固定資産税、軽自動車税、市たばこ税及び事業所税とする。";
        let cap = extract_leading_caption(t).unwrap();
        assert!(cap.starts_with("横浜市が課する普通税"), "got: {cap}");
    }

    #[tokio::test]
    async fn all_captions_returns_caption_or_article_lead() {
        let s = SqliteStore::open_in_memory(4).unwrap();
        let nb = Uuid::new_v4();
        let captioned = chunk("(入湯税の税率)\n第123条 入湯税の税率は100円とする");
        let captioned_id = captioned.id;
        let article_only = chunk("第3条 普通税は次のとおり");
        let article_only_id = article_only.id;
        let plain = chunk("ただの本文です");
        s.upsert(
            nb,
            &[captioned, article_only, plain],
            &[
                vec![1.0, 0.0, 0.0, 0.0],
                vec![0.0, 1.0, 0.0, 0.0],
                vec![0.0, 0.0, 1.0, 0.0],
            ],
        )
        .await
        .unwrap();
        let caps = s.all_captions(Some(nb)).await.unwrap();
        // captioned + article_only の 2 件が拾われる (plain は対象外)。
        assert_eq!(caps.len(), 2, "got: {caps:?}");
        let by_id: std::collections::HashMap<_, _> = caps.into_iter().collect();
        assert_eq!(
            by_id.get(&captioned_id).map(|s| s.as_str()),
            Some("入湯税の税率")
        );
        assert_eq!(
            by_id.get(&article_only_id).map(|s| s.as_str()),
            Some("普通税は次のとおり"),
        );
    }

    #[tokio::test]
    async fn upsert_then_keyword_search() {
        let s = SqliteStore::open_in_memory(4).unwrap();
        let nb = Uuid::new_v4();
        s.upsert(
            nb,
            &[chunk("hello world"), chunk("foo bar baz")],
            &[vec![1.0, 0.0, 0.0, 0.0], vec![0.0, 1.0, 0.0, 0.0]],
        )
        .await
        .unwrap();
        let hits = s.keyword_search(Some(nb), "foo", 5).await.unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].chunk.text, "foo bar baz");
    }

    #[tokio::test]
    async fn vector_search_returns_nearest() {
        let s = SqliteStore::open_in_memory(4).unwrap();
        let nb = Uuid::new_v4();
        let a = chunk("a");
        let b = chunk("b");
        s.upsert(
            nb,
            &[a.clone(), b.clone()],
            &[vec![1.0, 0.0, 0.0, 0.0], vec![0.0, 1.0, 0.0, 0.0]],
        )
        .await
        .unwrap();
        let hits = s.search(Some(nb), &[1.0, 0.0, 0.0, 0.0], 2).await.unwrap();
        assert_eq!(hits[0].chunk.id, a.id);
    }

    #[tokio::test]
    async fn keyword_search_finds_japanese_substring() {
        let s = SqliteStore::open_in_memory(2).unwrap();
        let nb = Uuid::new_v4();
        let mut a = chunk("東京駅前の本屋");
        a.heading_path = vec!["店舗".into()];
        let mut b = chunk("大阪城公園");
        b.heading_path = vec!["観光".into()];
        s.upsert(
            nb,
            &[a.clone(), b.clone()],
            &[vec![1.0, 0.0], vec![0.0, 1.0]],
        )
        .await
        .unwrap();
        let hits = s.keyword_search(Some(nb), "東京駅", 5).await.unwrap();
        assert!(!hits.is_empty(), "expected at least 1 hit for '東京駅'");
        assert_eq!(hits[0].chunk.id, a.id);
        let hits2 = s.keyword_search(Some(nb), "大阪", 5).await.unwrap();
        assert_eq!(hits2[0].chunk.id, b.id);
    }

    #[tokio::test]
    async fn rejects_dim_mismatch() {
        let s = SqliteStore::open_in_memory(4).unwrap();
        let nb = Uuid::new_v4();
        let r = s.upsert(nb, &[chunk("x")], &[vec![1.0, 2.0]]).await;
        assert!(r.is_err());
    }

    #[tokio::test]
    async fn delete_by_source_removes_chunks_and_vectors() {
        let s = SqliteStore::open_in_memory(2).unwrap();
        let nb = Uuid::new_v4();
        let sid = Uuid::new_v4();
        let mut a = chunk("alpha");
        a.source_id = sid;
        let mut b = chunk("beta");
        b.source_id = sid;
        let other = chunk("gamma");
        s.upsert(
            nb,
            &[a, b, other.clone()],
            &[vec![1.0, 0.0], vec![0.0, 1.0], vec![1.0, 1.0]],
        )
        .await
        .unwrap();
        let removed = s.delete_by_source(sid).await.unwrap();
        assert_eq!(removed, 2);
        assert_eq!(s.count_chunks(sid).await.unwrap(), 0);
        let kw = s.keyword_search(Some(nb), "alpha", 5).await.unwrap();
        assert!(kw.is_empty());
        let kw_other = s.keyword_search(Some(nb), "gamma", 5).await.unwrap();
        assert_eq!(kw_other.len(), 1);
        assert_eq!(kw_other[0].chunk.id, other.id);
    }

    #[tokio::test]
    async fn heading_path_roundtrips() {
        let s = SqliteStore::open_in_memory(2).unwrap();
        let nb = Uuid::new_v4();
        let c = chunk("h");
        s.upsert(nb, &[c.clone()], &[vec![1.0, 0.0]]).await.unwrap();
        let hits = s.keyword_search(Some(nb), "h", 1).await.unwrap();
        assert_eq!(hits[0].chunk.heading_path, c.heading_path);
        assert_eq!(hits[0].chunk.page, Some(1));
    }

    #[tokio::test]
    async fn scope_isolates_notebooks_in_keyword_search() {
        let s = SqliteStore::open_in_memory(2).unwrap();
        let nb_a = Uuid::new_v4();
        let nb_b = Uuid::new_v4();
        s.upsert(nb_a, &[chunk("alpha")], &[vec![1.0, 0.0]])
            .await
            .unwrap();
        s.upsert(nb_b, &[chunk("beta alpha")], &[vec![0.0, 1.0]])
            .await
            .unwrap();
        let in_a = s.keyword_search(Some(nb_a), "alpha", 5).await.unwrap();
        assert_eq!(in_a.len(), 1);
        assert_eq!(in_a[0].chunk.text, "alpha");
        let in_b = s.keyword_search(Some(nb_b), "alpha", 5).await.unwrap();
        assert_eq!(in_b.len(), 1);
        assert_eq!(in_b[0].chunk.text, "beta alpha");
    }

    fn chunk_at(source_id: Uuid, ord: u32, heading: Vec<&str>, text: &str) -> Chunk {
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
        let s = SqliteStore::open_in_memory(2).unwrap();
        let nb = Uuid::new_v4();
        let sid = Uuid::new_v4();
        let chunks = vec![
            chunk_at(sid, 0, vec!["第一編 総則"], "前文"),
            chunk_at(sid, 1, vec!["第一編 総則"], "..."),
            chunk_at(sid, 2, vec!["第二編 物権"], "物権の冒頭"),
            chunk_at(sid, 3, vec!["第二編 物権"], "..."),
            chunk_at(sid, 4, vec!["第三編 債権"], "債権の冒頭"),
        ];
        let embs = vec![vec![0.0, 0.0]; chunks.len()];
        s.upsert(nb, &chunks, &embs).await.unwrap();
        let reps = s.representative_chunks(Some(nb), 5).await.unwrap();
        let texts: Vec<&str> = reps.iter().map(|c| c.text.as_str()).collect();
        assert_eq!(texts, vec!["前文", "物権の冒頭", "債権の冒頭"]);
    }

    #[tokio::test]
    async fn representative_chunks_caps_per_source() {
        let s = SqliteStore::open_in_memory(2).unwrap();
        let nb = Uuid::new_v4();
        let sid = Uuid::new_v4();
        let chunks = vec![
            chunk_at(sid, 0, vec!["A"], "a"),
            chunk_at(sid, 1, vec!["B"], "b"),
            chunk_at(sid, 2, vec!["C"], "c"),
        ];
        let embs = vec![vec![0.0, 0.0]; chunks.len()];
        s.upsert(nb, &chunks, &embs).await.unwrap();
        let reps = s.representative_chunks(Some(nb), 2).await.unwrap();
        assert_eq!(reps.len(), 2);
        assert_eq!(reps[0].text, "a");
        assert_eq!(reps[1].text, "b");
    }

    #[tokio::test]
    async fn representative_chunks_falls_back_to_ord_when_no_headings() {
        let s = SqliteStore::open_in_memory(2).unwrap();
        let nb = Uuid::new_v4();
        let sid = Uuid::new_v4();
        let chunks = vec![
            chunk_at(sid, 0, vec![], "p1"),
            chunk_at(sid, 1, vec![], "p2"),
            chunk_at(sid, 2, vec![], "p3"),
        ];
        let embs = vec![vec![0.0, 0.0]; chunks.len()];
        s.upsert(nb, &chunks, &embs).await.unwrap();
        let reps = s.representative_chunks(Some(nb), 2).await.unwrap();
        let texts: Vec<&str> = reps.iter().map(|c| c.text.as_str()).collect();
        assert_eq!(texts, vec!["p1", "p2"]);
    }

    #[tokio::test]
    async fn representative_chunks_separates_sources() {
        let s = SqliteStore::open_in_memory(2).unwrap();
        let nb = Uuid::new_v4();
        let s1 = Uuid::new_v4();
        let s2 = Uuid::new_v4();
        let chunks = vec![
            chunk_at(s1, 0, vec!["X1"], "s1-a"),
            chunk_at(s1, 1, vec!["X2"], "s1-b"),
            chunk_at(s2, 0, vec!["Y1"], "s2-a"),
            chunk_at(s2, 1, vec!["Y2"], "s2-b"),
        ];
        let embs = vec![vec![0.0, 0.0]; chunks.len()];
        s.upsert(nb, &chunks, &embs).await.unwrap();
        let reps = s.representative_chunks(Some(nb), 2).await.unwrap();
        assert_eq!(reps.len(), 4);
        assert_eq!(reps.iter().filter(|c| c.source_id == s1).count(), 2);
        assert_eq!(reps.iter().filter(|c| c.source_id == s2).count(), 2);
    }

    #[tokio::test]
    async fn representative_chunks_respects_scope() {
        let s = SqliteStore::open_in_memory(2).unwrap();
        let nb_a = Uuid::new_v4();
        let nb_b = Uuid::new_v4();
        s.upsert(
            nb_a,
            &[chunk_at(Uuid::new_v4(), 0, vec!["A"], "in-a")],
            &[vec![0.0, 0.0]],
        )
        .await
        .unwrap();
        s.upsert(
            nb_b,
            &[chunk_at(Uuid::new_v4(), 0, vec!["B"], "in-b")],
            &[vec![0.0, 0.0]],
        )
        .await
        .unwrap();
        let reps_a = s.representative_chunks(Some(nb_a), 3).await.unwrap();
        assert_eq!(reps_a.len(), 1);
        assert_eq!(reps_a[0].text, "in-a");
    }

    #[tokio::test]
    async fn delete_by_notebook_clears_only_target() {
        let s = SqliteStore::open_in_memory(2).unwrap();
        let nb_a = Uuid::new_v4();
        let nb_b = Uuid::new_v4();
        s.upsert(
            nb_a,
            &[chunk("a1"), chunk("a2")],
            &[vec![1.0, 0.0], vec![1.0, 0.0]],
        )
        .await
        .unwrap();
        s.upsert(nb_b, &[chunk("b1")], &[vec![0.0, 1.0]])
            .await
            .unwrap();
        let removed = s.delete_by_notebook(nb_a).await.unwrap();
        assert_eq!(removed, 2);
        assert_eq!(s.keyword_search(Some(nb_a), "a", 5).await.unwrap().len(), 0);
        assert_eq!(
            s.keyword_search(Some(nb_b), "b1", 5).await.unwrap().len(),
            1
        );
    }
}
