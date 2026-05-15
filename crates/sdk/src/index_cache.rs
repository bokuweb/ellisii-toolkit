//! `index_dir` / `index_file` の冪等化に使う「path → 指紋 + source_id」キャッシュ。
//!
//! 同じファイルを 2 度 index に投げても、
//! - 内容が変わっていなければ skip
//! - 変わっていれば古い source_id を消してから再 ingest
//! という挙動を取るための土台。
//!
//! 既定実装:
//! - [`MemoryIndexCache`] — process 寿命のみ。永続化なし。テスト向け
//! - [`JsonIndexCache`]   — JSON ファイル 1 本に永続化。実用向け
//!
//! 自前で sqlite テーブル等に保存したい場合は [`IndexCache`] を直接 impl。

use async_trait::async_trait;
use ellisii_core::{Error, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use uuid::Uuid;

/// 1 ファイル分のキャッシュエントリ。
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct IndexEntry {
    /// 取り込み時に発行した source_id。再 ingest で内容が変わっていれば
    /// 上書き、未変更なら ingest 自体を skip する。
    pub source_id: Uuid,
    /// ファイル指紋。既定では `"<size>:<mtime_nanos>"` 形式。
    pub fingerprint: String,
}

/// 冪等 ingest 用キャッシュ。`get` でルックアップ、`put` で保存、`forget` で削除。
#[async_trait]
pub trait IndexCache: Send + Sync {
    async fn get(&self, path: &str) -> Result<Option<IndexEntry>>;
    async fn put(&self, path: &str, entry: IndexEntry) -> Result<()>;
    async fn forget(&self, path: &str) -> Result<()>;
}

/// process 寿命の HashMap キャッシュ。
#[derive(Default)]
pub struct MemoryIndexCache {
    inner: Mutex<HashMap<String, IndexEntry>>,
}

impl MemoryIndexCache {
    pub fn new() -> Self {
        Self::default()
    }
}

#[async_trait]
impl IndexCache for MemoryIndexCache {
    async fn get(&self, path: &str) -> Result<Option<IndexEntry>> {
        Ok(self.inner.lock().unwrap().get(path).cloned())
    }
    async fn put(&self, path: &str, entry: IndexEntry) -> Result<()> {
        self.inner.lock().unwrap().insert(path.to_string(), entry);
        Ok(())
    }
    async fn forget(&self, path: &str) -> Result<()> {
        self.inner.lock().unwrap().remove(path);
        Ok(())
    }
}

/// JSON ファイル 1 本に「path → IndexEntry」を保存する永続キャッシュ。
///
/// 起動時にファイルを読み込み、`put` / `forget` のたびに同期書き込み。
/// 大量ファイル (数万単位) になると I/O コストが効いてくるが、
/// 数百〜数千程度なら実用的。
pub struct JsonIndexCache {
    path: PathBuf,
    inner: Mutex<HashMap<String, IndexEntry>>,
}

impl JsonIndexCache {
    /// 既存ファイルがあれば読み込み、無ければ空でスタート。読み込み失敗は
    /// fatal とせず、空キャッシュ扱いで継続する (= 冪等性は次回からに任せる)。
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let inner = if path.exists() {
            let raw = std::fs::read_to_string(&path)
                .map_err(|e| Error::Other(anyhow::anyhow!("read index cache: {e}")))?;
            serde_json::from_str::<HashMap<String, IndexEntry>>(&raw).unwrap_or_default()
        } else {
            HashMap::new()
        };
        Ok(Self {
            path,
            inner: Mutex::new(inner),
        })
    }

    fn flush_locked(&self, map: &HashMap<String, IndexEntry>) -> Result<()> {
        let raw = serde_json::to_string_pretty(map)
            .map_err(|e| Error::Other(anyhow::anyhow!("serialize cache: {e}")))?;
        if let Some(parent) = self.path.parent() {
            std::fs::create_dir_all(parent).ok();
        }
        std::fs::write(&self.path, raw)
            .map_err(|e| Error::Other(anyhow::anyhow!("write index cache: {e}")))?;
        Ok(())
    }
}

#[async_trait]
impl IndexCache for JsonIndexCache {
    async fn get(&self, path: &str) -> Result<Option<IndexEntry>> {
        Ok(self.inner.lock().unwrap().get(path).cloned())
    }
    async fn put(&self, path: &str, entry: IndexEntry) -> Result<()> {
        let mut g = self.inner.lock().unwrap();
        g.insert(path.to_string(), entry);
        self.flush_locked(&g)
    }
    async fn forget(&self, path: &str) -> Result<()> {
        let mut g = self.inner.lock().unwrap();
        g.remove(path);
        self.flush_locked(&g)
    }
}

/// `<size>:<mtime_nanos>` 形式のファイル指紋を計算する。
///
/// metadata 取得失敗時は `"unknown"` を返し、indexer 側で常に再 ingest する
/// 挙動になる (= 安全側)。
pub fn fingerprint(path: &Path) -> String {
    let meta = match std::fs::metadata(path) {
        Ok(m) => m,
        Err(_) => return "unknown".into(),
    };
    let size = meta.len();
    let mtime_nanos = meta
        .modified()
        .ok()
        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    format!("{size}:{mtime_nanos}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn memory_cache_roundtrip() {
        let c = MemoryIndexCache::new();
        assert!(c.get("a").await.unwrap().is_none());
        let e = IndexEntry {
            source_id: Uuid::new_v4(),
            fingerprint: "1:2".into(),
        };
        c.put("a", e.clone()).await.unwrap();
        assert_eq!(c.get("a").await.unwrap().as_ref(), Some(&e));
        c.forget("a").await.unwrap();
        assert!(c.get("a").await.unwrap().is_none());
    }

    #[tokio::test]
    async fn json_cache_persists_to_disk() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("cache.json");
        let entry = IndexEntry {
            source_id: Uuid::new_v4(),
            fingerprint: "1:2".into(),
        };
        {
            let c = JsonIndexCache::open(&path).unwrap();
            c.put("a", entry.clone()).await.unwrap();
        }
        // 別インスタンスで開き直しても残っている
        let c2 = JsonIndexCache::open(&path).unwrap();
        assert_eq!(c2.get("a").await.unwrap().as_ref(), Some(&entry));
    }

    #[test]
    fn fingerprint_changes_when_content_changes() {
        let tmp = tempfile::tempdir().unwrap();
        let p = tmp.path().join("a.txt");
        std::fs::write(&p, "hello").unwrap();
        let f1 = fingerprint(&p);
        // 書き換え + sleep で mtime が変わるはず
        std::thread::sleep(std::time::Duration::from_millis(10));
        std::fs::write(&p, "hello world!!!").unwrap();
        let f2 = fingerprint(&p);
        assert_ne!(f1, f2);
    }
}
