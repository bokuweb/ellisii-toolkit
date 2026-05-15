//! Per-page OCR result cache.
//!
//! 同じ PDF を何度も再 ingest する開発フロー (chunker / RAG パラメータの
//! チューニング、モデル差し替え検証など) で、毎回 OCR を回し直すのは時間と
//! 電力の無駄。スキャン PDF 1 冊で 5-10 分かかる工程を、**ファイル内容が
//! 変わっていなければ完全 skip** できれば再 ingest が ~30 秒級に縮む。
//!
//! ## キャッシュキー
//!
//! ファイルの (`canonical_path`, `mtime`, `size`) ハッシュを bucket 名にする。
//! mtime / size のどちらかが変われば bucket が別になり、自然に invalidate
//! される。ハッシュは衝突確率の事実上 0 のもので OK (秘密性は要らない) ので
//! `xxhash` 級ではなく標準ライブラリの `DefaultHasher` で十分。
//!
//! ## レイアウト
//!
//! ```text
//! <cache_dir>/ocr/v1/<bucket>/page-<N>.json
//! ```
//!
//! `v1/` は cache スキーマバージョン。OCR 結果の意味が変わるような大規模変更
//! (e.g., bbox 座標系の変更、model 出力フォーマット差し替え) があったら v2 に
//! 上げて全 invalidate する。`OcrBlock` の中身が増えるだけなら serde の
//! 後方互換でカバーされるので bump 不要。
//!
//! ## 失敗のフォールバック
//!
//! cache I/O は **すべて warn して fall through** する設計。ディスク full /
//! 権限不足 / corrupt JSON 等で cache 操作が失敗しても、ingest 本流は OCR を
//! 走らせ続ければ正しい結果が出る。`get` は失敗時 `None` 相当、`put` は失敗
//! 時 silent (warn のみ) を返す。

use ellisii_ocr::OcrBlock;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};
use std::time::SystemTime;

const CACHE_SCHEMA_VERSION: &str = "v1";

/// 1 つの ingest 単位 (= 同一 PDF) に対して bucket directory を提供する。
/// `OcrCache::for_file` で作って、各ページの `get` / `put` を per-page で呼ぶ。
pub struct OcrCache {
    bucket_dir: PathBuf,
}

impl OcrCache {
    /// `cache_root/ocr/<schema>/<file-hash>/` をこのファイル専用 bucket に
    /// 紐付ける。ファイル stat (mtime / size) を hash に含めるので、ファイル
    /// が更新されたら自動的に別 bucket に切り替わって過去キャッシュは
    /// **使われない** (= 旧データが残るが読まれない)。
    ///
    /// ファイルが存在しない / stat 取得失敗時は `None` を返し、cache を完全
    /// 無効化したのと同じ動作 (get/put は no-op) にする。
    pub fn for_file(cache_root: &Path, file_path: &Path) -> Option<Self> {
        let meta = std::fs::metadata(file_path).ok()?;
        let mtime = meta
            .modified()
            .ok()
            .and_then(|t| t.duration_since(SystemTime::UNIX_EPOCH).ok())
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0);
        let size = meta.len();
        let canonical = file_path
            .canonicalize()
            .ok()
            .unwrap_or_else(|| file_path.to_path_buf());
        let mut hasher = DefaultHasher::new();
        canonical.hash(&mut hasher);
        mtime.hash(&mut hasher);
        size.hash(&mut hasher);
        let bucket = format!("{:016x}", hasher.finish());
        let bucket_dir = cache_root
            .join("ocr")
            .join(CACHE_SCHEMA_VERSION)
            .join(bucket);
        Some(Self { bucket_dir })
    }

    /// ページの cache hit を返す。miss / 失敗 / corrupt 時は `None`。
    pub fn get(&self, page: u32) -> Option<Vec<OcrBlock>> {
        let path = self.page_path(page);
        let bytes = std::fs::read(&path).ok()?;
        match serde_json::from_slice::<Vec<OcrBlock>>(&bytes) {
            Ok(blocks) => Some(blocks),
            Err(e) => {
                tracing::warn!(
                    "ocr_cache: corrupt entry {}: {e}; will re-OCR",
                    path.display()
                );
                None
            }
        }
    }

    /// ページの cache を書き込む。失敗時は warn だけして黙って戻る (= cache
    /// 無効化と同じ振る舞いに fall through)。
    pub fn put(&self, page: u32, blocks: &[OcrBlock]) {
        if let Err(e) = std::fs::create_dir_all(&self.bucket_dir) {
            tracing::warn!(
                "ocr_cache: create_dir_all {}: {e}",
                self.bucket_dir.display()
            );
            return;
        }
        let path = self.page_path(page);
        let bytes = match serde_json::to_vec(blocks) {
            Ok(b) => b,
            Err(e) => {
                tracing::warn!("ocr_cache: serialize failed: {e}");
                return;
            }
        };
        // 同じ bucket に他プロセスが同時書き込みしても破損しないよう、tmp →
        // rename の atomic write にする。
        let tmp = path.with_extension("json.tmp");
        if let Err(e) = std::fs::write(&tmp, &bytes) {
            tracing::warn!("ocr_cache: write tmp {}: {e}", tmp.display());
            return;
        }
        if let Err(e) = std::fs::rename(&tmp, &path) {
            tracing::warn!(
                "ocr_cache: rename {} -> {}: {e}",
                tmp.display(),
                path.display()
            );
        }
    }

    fn page_path(&self, page: u32) -> PathBuf {
        self.bucket_dir.join(format!("page-{page}.json"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn block(text: &str, page: u32) -> OcrBlock {
        OcrBlock {
            text: text.into(),
            bbox: [1.0, 2.0, 3.0, 4.0],
            page,
            confidence: 0.9,
        }
    }

    #[test]
    fn round_trips_blocks_per_page() {
        let cache_root = tempfile::tempdir().unwrap();
        let pdf = cache_root.path().join("doc.pdf");
        std::fs::write(&pdf, b"%PDF-1.7 dummy").unwrap();
        let cache = OcrCache::for_file(cache_root.path(), &pdf).unwrap();
        assert!(cache.get(1).is_none());
        cache.put(1, &[block("hello", 1), block("world", 1)]);
        let got = cache.get(1).unwrap();
        assert_eq!(got.len(), 2);
        assert_eq!(got[0].text, "hello");
        assert_eq!(got[1].text, "world");
    }

    #[test]
    fn missing_pages_return_none() {
        let cache_root = tempfile::tempdir().unwrap();
        let pdf = cache_root.path().join("doc.pdf");
        std::fs::write(&pdf, b"%PDF").unwrap();
        let cache = OcrCache::for_file(cache_root.path(), &pdf).unwrap();
        cache.put(2, &[block("two", 2)]);
        assert!(cache.get(1).is_none());
        assert!(cache.get(3).is_none());
        assert!(cache.get(2).is_some());
    }

    #[test]
    fn changing_mtime_or_size_invalidates_implicitly() {
        // 新しい mtime / size のファイルに切り替わると `for_file` が別 bucket
        // を選ぶので、古い cache は読まれない (= 自動 invalidate)。
        let cache_root = tempfile::tempdir().unwrap();
        let pdf = cache_root.path().join("doc.pdf");
        std::fs::write(&pdf, b"%PDF v1").unwrap();
        let cache_a = OcrCache::for_file(cache_root.path(), &pdf).unwrap();
        cache_a.put(1, &[block("v1", 1)]);
        assert!(cache_a.get(1).is_some());

        // overwrite to make mtime/size change.
        std::thread::sleep(std::time::Duration::from_millis(10));
        std::fs::write(&pdf, b"%PDF v2 content longer").unwrap();
        let cache_b = OcrCache::for_file(cache_root.path(), &pdf).unwrap();
        assert!(
            cache_b.get(1).is_none(),
            "new bucket should not see v1 entries"
        );

        // 元のファイルに戻したら… 元の bucket には戻らない (= mtime が更新
        // されているので)。これは想定どおり: 「同じ内容に戻したからキャッシュ
        // ヒットしてほしい」需要は薄く、mtime ベースの invalidate のほうが
        // 安全側 (戻したつもりで微妙に違う、というケースでの fallback)。
    }

    #[test]
    fn corrupt_json_is_treated_as_miss() {
        let cache_root = tempfile::tempdir().unwrap();
        let pdf = cache_root.path().join("doc.pdf");
        std::fs::write(&pdf, b"%PDF").unwrap();
        let cache = OcrCache::for_file(cache_root.path(), &pdf).unwrap();
        // 直接 bucket に壊れた JSON を書く。
        std::fs::create_dir_all(&cache.bucket_dir).unwrap();
        std::fs::write(cache.page_path(1), b"not valid json").unwrap();
        // miss として返り、再 OCR に倒される。
        assert!(cache.get(1).is_none());
    }

    #[test]
    fn for_file_returns_none_for_missing_path() {
        let cache_root = tempfile::tempdir().unwrap();
        let pdf = cache_root.path().join("does-not-exist.pdf");
        assert!(OcrCache::for_file(cache_root.path(), &pdf).is_none());
    }
}
