use async_trait::async_trait;
use ellisii_core::{Error, Result};
use ellisii_ocr::{PdfRasterizer, RasterizedPage};
use ellisii_parsers_core::ParsedBlock;
use pdfium_render::prelude::{PdfDocument, Pdfium};
use std::cell::RefCell;
use std::sync::Arc;
use std::time::SystemTime;

/// libpdfium を bind する。解決順:
///   1. 実行ファイル隣接 (`<exe_dir>/libpdfium.{dylib,so,dll}`)
///   2. macOS .app bundle 配置 (`<exe_dir>/../Frameworks/libpdfium.dylib`)
///   3. `PDFIUM_LIB_PATH` 環境変数 (= 開発時の手動指定)
///   4. `pdfium-auto` の自動 DL (CI 抜け道 / 旧経路フォールバック)
///
/// 1 を最優先にするのは、ビルド時に `vendor/pdfium/<triple>/` から `target/<profile>/`
/// に dylib をコピーする build.rs の出力をそのまま使う運用のため
/// (= pdfium-auto のランタイム DL は廃止寄り)。同じディレクトリに置いておけば
/// dev でも .app でも一貫したロードができ、macOS の library validation でも
/// host と同じ codesign 操作で扱えるので楽。
fn bind_pdfium() -> Result<Pdfium> {
    if let Some(path) = locate_bundled_pdfium() {
        return pdfium_auto::bind_pdfium_from_path(&path)
            .map_err(|e| Error::Parse(format!("pdfium unavailable (bundled {:?}): {e}", path)));
    }
    // 旧経路: pdfium-auto 自動 DL。bundle が無い CI/開発環境向けの保険。
    let path = pdfium_auto::ensure_pdfium_library(None)
        .map_err(|e| Error::Parse(format!("pdfium ensure: {e}")))?;
    pdfium_auto::bind_pdfium_from_path(&path)
        .map_err(|e| Error::Parse(format!("pdfium unavailable: {e}")))
}

#[cfg(target_os = "macos")]
const PDFIUM_LIB_FILENAME: &str = "libpdfium.dylib";
#[cfg(all(unix, not(target_os = "macos")))]
const PDFIUM_LIB_FILENAME: &str = "libpdfium.so";
#[cfg(target_os = "windows")]
const PDFIUM_LIB_FILENAME: &str = "pdfium.dll";

/// 実行ファイル周辺と PDFIUM_LIB_PATH を順に探す。
/// 見つかったパスは存在チェック済み。
fn locate_bundled_pdfium() -> Option<std::path::PathBuf> {
    use std::path::PathBuf;

    if let Ok(env_path) = std::env::var("PDFIUM_LIB_PATH") {
        let p = PathBuf::from(env_path);
        if p.is_file() {
            return Some(p);
        }
    }

    let exe = std::env::current_exe().ok()?;
    let exe_dir = exe.parent()?;

    // 1) <exe_dir>/libpdfium.<ext>  (cargo build / dev / 一部の bundle 形態)
    let candidate = exe_dir.join(PDFIUM_LIB_FILENAME);
    if candidate.is_file() {
        return Some(candidate);
    }

    // 2) macOS .app/Contents/Frameworks/libpdfium.dylib
    //    Tauri bundler は Contents/MacOS/<bin> を実行するので exe_dir は MacOS/。
    //    そこから ../Frameworks/ を覗きにいく。
    #[cfg(target_os = "macos")]
    {
        if let Some(frameworks) = exe_dir.parent().map(|p| p.join("Frameworks")) {
            let candidate = frameworks.join(PDFIUM_LIB_FILENAME);
            if candidate.is_file() {
                return Some(candidate);
            }
        }
    }

    None
}

/// PDF テキストレイヤを抽出する。スキャン PDF (テキストレイヤなし) のときは
/// 空に近い結果が返るため、呼び出し側で OCR にフォールバックする。
///
/// 一次パーサは Google pdfium。日本語の Adobe 既定 CMap (90ms-RKSJ-H など) も
/// ハンドルできる。pdfium バイナリは初回呼出時に `pdfium-auto` 経由で
/// `~/.cache/pdf2md/pdfium-<ver>/` に自動 DL される。
///
/// pdfium が使えない環境 (ネット不通・サポート外プラットフォーム) では
/// `pdf-extract` にフォールバック。`pdf-extract` 0.9 系は Identity-H 以外で
/// `panic!` するため `catch_unwind` で握る。
pub fn parse(path: &str) -> Result<Vec<ParsedBlock>> {
    match parse_with_pdfium(path) {
        Ok(blocks) => {
            tracing::info!("pdf parsed via pdfium: {} blocks ({path})", blocks.len());
            Ok(blocks)
        }
        Err(e) => {
            tracing::warn!("pdfium parse failed ({e}); falling back to pdf-extract");
            parse_with_pdf_extract(path)
        }
    }
}

fn parse_with_pdfium(path: &str) -> Result<Vec<ParsedBlock>> {
    with_cached_document(path, |document| {
        let mut out = Vec::new();
        for (i, page) in document.pages().iter().enumerate() {
            let text = page
                .text()
                .map_err(|e| Error::Parse(format!("pdfium page text {}: {e}", i + 1)))?;
            let body = text.all();
            for para in body.split("\n\n") {
                let trimmed = para.trim();
                if trimmed.is_empty() {
                    continue;
                }
                out.push(ParsedBlock {
                    text: trimmed.to_string(),
                    heading_path: vec![],
                    page: Some(i as u32 + 1),
                    bbox: None,
                });
            }
        }
        Ok(out)
    })
}

// =====================================================================
// Pdfium / Document のスレッドローカルキャッシュ。
//
// 経緯:
// - `Pdfium` の bind 自体は pdfium-auto によりライブラリパスがキャッシュされ、
//   2 回目以降は数 ms / call。ただし内部 dyn binding を毎回作り直すコストは残る。
// - 一方 `load_pdf_from_file` は呼ぶたびに xref を読み直すので、45MB の書籍では
//   ~100-500ms / call。OCR fallback で 1 ページずつ rasterize している現状だと
//   `bind + load` が 465 回走り、最大数分のオーバーヘッドになる。
//
// `pdfium_render::Pdfium` / `PdfDocument` は内部に raw pointer (`!Send + !Sync`)
// を持つため、`Mutex<Pdfium>` を OnceLock に載せようとしても Send/Sync で弾かれる。
// 代わりに `thread_local!` でスレッドごとに 1 個だけ Pdfium を持ち、直近で開いた
// document を `(path, mtime, size)` をキーにキャッシュする。
//
// tokio の blocking スレッドプール (`spawn_blocking`) は同じスレッドを再利用する
// ので、同一 PDF の連続 rasterize はキャッシュにヒットし続ける (1 回 load で 465
// ページ render が回る)。
// =====================================================================

/// `(path, mtime, size)` をキーに、最新の load_pdf_from_file の結果を保持する。
/// - 同一ファイルの連続呼び出しは load を完全にスキップ。
/// - `mtime` か `size` が変わったら invalidate して再 load。
/// - 別ファイルが来たら古い doc を drop して新しい doc を載せ替える
///   (= 1 件だけ持つ LRU 相当)。
struct DocCacheKey {
    path: String,
    mtime: SystemTime,
    size: u64,
}

struct CachedDocument {
    key: DocCacheKey,
    /// SAFETY: `'static` は嘘で、本当の lifetime は `PdfiumState::pdfium` のもの。
    /// この `CachedDocument` は `PdfiumState` 内にしか存在しないので、`pdfium` を
    /// drop する前に必ず `cached` を drop する手順を守れば、外に逃げない限り安全。
    /// `replace_document` / `drop` で `cached = None` にしてから `pdfium` を触る。
    document: PdfDocument<'static>,
}

struct PdfiumState {
    pdfium: Pdfium,
    cached: Option<CachedDocument>,
}

impl PdfiumState {
    fn new(pdfium: Pdfium) -> Self {
        Self {
            pdfium,
            cached: None,
        }
    }

    /// `path` の最新版がキャッシュに無ければ load して載せ替える。
    /// 戻り値は同じ Mutex 内の document への参照。
    // 注意: 戻り値の inner lifetime は `'static` のまま。`PdfDocument<'a>` は
    // `'a` に invariant なので、ここで lifetime を縮めてしまうと caller 側で
    // `&PdfDocument<'_>` の `'_` がエラーになる。`'static` は嘘ではあるが
    // (実体は `PdfiumState::pdfium` を借りている)、`PdfiumState` 自体が static
    // なので「Pdfium が drop される前に document を drop する」 invariant さえ
    // 守れていれば不正アクセスは起きない (`Drop` で `cached = None`)。
    fn ensure_loaded(&mut self, path: &str) -> Result<&PdfDocument<'static>> {
        let meta =
            std::fs::metadata(path).map_err(|e| Error::Parse(format!("stat {path}: {e}")))?;
        let mtime = meta.modified().unwrap_or(SystemTime::UNIX_EPOCH);
        let size = meta.len();

        let needs_reload = match &self.cached {
            None => true,
            Some(c) => c.key.path != path || c.key.mtime != mtime || c.key.size != size,
        };
        if needs_reload {
            // 既存の document を必ず先に drop する (pdfium を借用しているため)。
            self.cached = None;
            let document = self
                .pdfium
                .load_pdf_from_file(path, None)
                .map_err(|e| Error::Parse(format!("pdfium load: {e}")))?;
            // SAFETY: `document` は `self.pdfium` を借りている。`PdfiumState` 内に
            // 同居しており、`PdfiumState` は static (= プロセス終了まで生存) なので
            // 借用は事実上 'static まで延ばせる。`cached` は `pdfium` より先に
            // drop することを `ensure_loaded`/`drop` で保証している。
            let document: PdfDocument<'static> =
                unsafe { std::mem::transmute::<PdfDocument<'_>, PdfDocument<'static>>(document) };
            self.cached = Some(CachedDocument {
                key: DocCacheKey {
                    path: path.to_string(),
                    mtime,
                    size,
                },
                document,
            });
        }
        Ok(&self.cached.as_ref().expect("just inserted").document)
    }
}

impl Drop for PdfiumState {
    fn drop(&mut self) {
        // pdfium より先に document を確実に drop する。
        self.cached = None;
    }
}

thread_local! {
    /// このスレッドの Pdfium + 最新 document キャッシュ。初回 rasterize 時に
    /// lazy bind する。tokio blocking pool の各スレッドがそれぞれ 1 個ずつ持つ。
    static PDFIUM_STATE: RefCell<Option<PdfiumState>> = const { RefCell::new(None) };
}

fn with_cached_document<F, R>(path: &str, f: F) -> Result<R>
where
    F: FnOnce(&PdfDocument<'static>) -> Result<R>,
{
    PDFIUM_STATE.with(|cell| {
        let mut guard = cell.borrow_mut();
        if guard.is_none() {
            *guard = Some(PdfiumState::new(bind_pdfium()?));
        }
        let state = guard.as_mut().expect("just initialized");
        let document = state.ensure_loaded(path)?;
        f(document)
    })
}

fn parse_with_pdf_extract(path: &str) -> Result<Vec<ParsedBlock>> {
    let bytes = std::fs::read(path).map_err(Error::Io)?;
    let text = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        pdf_extract::extract_text_from_mem(&bytes)
    }))
    .map_err(|e| {
        let msg = if let Some(s) = e.downcast_ref::<&str>() {
            (*s).to_string()
        } else if let Some(s) = e.downcast_ref::<String>() {
            s.clone()
        } else {
            "unknown panic".to_string()
        };
        Error::Parse(format!("pdf parser panicked: {msg}"))
    })?
    .map_err(|e| Error::Parse(format!("pdf: {e}")))?;
    Ok(split_pages(&text))
}

pub fn split_pages(text: &str) -> Vec<ParsedBlock> {
    // pdf-extract は form feed (\u{c}) でページ区切り
    let mut out = Vec::new();
    for (i, page) in text.split('\u{c}').enumerate() {
        for para in page.split("\n\n") {
            let trimmed = para.trim();
            if trimmed.is_empty() {
                continue;
            }
            out.push(ParsedBlock {
                text: trimmed.to_string(),
                heading_path: vec![],
                page: Some(i as u32 + 1),
                bbox: None,
            });
        }
    }
    out
}

/// pdfium で各ページを PNG にラスタライズする `PdfRasterizer` 実装。
/// 出力は短命 tempdir に書き、`RasterizedPage._keepalive` でその寿命を延ばす。
///
/// 用途: スキャン PDF など PDF テキスト抽出が空のときに OCR に渡す画像を作る。
pub struct PdfiumRasterizer {
    /// 各ページの最長辺ピクセル。OCR 行検出は ~150-200 dpi 相当が経験的に妥当。
    /// A4 縦 (842pt) を 1600px に展開すると ~136dpi。
    pub target_longest_side_px: u32,
}

impl Default for PdfiumRasterizer {
    fn default() -> Self {
        // 1500px = ~127dpi on A4。経験的に DEIM/PARSeq の認識精度はここから
        // 落ち始めるか落ちないかのライン。1800 (≒152dpi) からの低減で render
        // 自体は ~30% 速くなり、PNG ファイルサイズも縮むので OCR 入力の I/O
        // も軽くなる。スキャン PDF ingest が遅すぎるという課題への低リスク
        // wins の 1 つ。精度劣化が観測されたら 1700 程度まで戻す。
        Self {
            target_longest_side_px: 1500,
        }
    }
}

#[async_trait]
impl PdfRasterizer for PdfiumRasterizer {
    async fn page_count(&self, path: &str) -> Result<u32> {
        let path = path.to_string();
        tokio::task::spawn_blocking(move || page_count_blocking(&path))
            .await
            .map_err(|e| Error::Parse(format!("page_count join: {e}")))?
    }

    async fn rasterize_pages(&self, path: &str, pages: &[u32]) -> Result<Vec<RasterizedPage>> {
        if pages.is_empty() {
            return Ok(vec![]);
        }
        let path = path.to_string();
        let target = self.target_longest_side_px;
        let pages: Vec<u32> = pages.to_vec();
        // pdfium-auto は reqwest::blocking を内部で使うので、tokio コンテキスト
        // から直接呼ぶと nested runtime panic になる。spawn_blocking で逃がす。
        tokio::task::spawn_blocking(move || rasterize_pages_blocking(&path, &pages, target))
            .await
            .map_err(|e| Error::Parse(format!("rasterize join: {e}")))?
    }

    /// streaming 版: 1 spawn_blocking で全ページ render しつつ、1 ページ render
    /// する都度 `tx.blocking_send` で receiver (= OCR ワーカー) に流す。
    ///
    /// 効果: PDF 全体の load は 1 度だけ (`with_cached_document` の thread_local
    /// cache が有効)。一方で receiver 側は最初のページが届いた時点から OCR を
    /// 始められるので、rasterize ループが回っている裏で OCR 並列消費が走る
    /// (= 真の pipelining)。`channel` の容量で先取り量をバウンドし peak disk が
    /// 暴走しないようにする。
    async fn rasterize_pages_streaming(
        &self,
        path: &str,
        pages: &[u32],
        tx: tokio::sync::mpsc::Sender<RasterizedPage>,
    ) -> Result<()> {
        if pages.is_empty() {
            return Ok(());
        }
        let path = path.to_string();
        let target = self.target_longest_side_px;
        let pages: Vec<u32> = pages.to_vec();
        tokio::task::spawn_blocking(move || {
            rasterize_pages_streaming_blocking(&path, &pages, target, tx)
        })
        .await
        .map_err(|e| Error::Parse(format!("rasterize join: {e}")))?
    }
}

fn page_count_blocking(path: &str) -> Result<u32> {
    with_cached_document(path, |document| Ok(document.pages().len() as u32))
}

fn rasterize_pages_blocking(
    path: &str,
    pages: &[u32],
    target_longest_side_px: u32,
) -> Result<Vec<RasterizedPage>> {
    use pdfium_render::prelude::*;

    let dir = tempfile::tempdir().map_err(Error::Io)?;
    let dir_arc: Arc<dyn std::any::Any + Send + Sync> = Arc::new(dir);
    // dir_arc は TempDir を抱えているので、これを clone して各 RasterizedPage に
    // 持たせれば、最後の参照が drop されるまで PNG が消えない。
    let dir_path = dir_arc
        .downcast_ref::<tempfile::TempDir>()
        .expect("Arc holds TempDir")
        .path()
        .to_path_buf();
    let cfg = PdfRenderConfig::new().set_maximum_width(target_longest_side_px as Pixels);

    with_cached_document(path, |document| {
        let total = document.pages().len() as u32;
        let mut out = Vec::new();
        for &p in pages {
            if p == 0 || p > total {
                tracing::warn!("rasterize_pages: requested page {p} out of range (total={total})");
                continue;
            }
            let page = document
                .pages()
                .get(p as u16 - 1)
                .map_err(|e| Error::Parse(format!("get page {p}: {e}")))?;
            let bitmap = page
                .render_with_config(&cfg)
                .map_err(|e| Error::Parse(format!("render page {p}: {e}")))?;
            let img = bitmap.as_image().to_rgba8();
            let png_path = dir_path.join(format!("page-{p}.png"));
            img.save(&png_path)
                .map_err(|e| Error::Parse(format!("save png page {p}: {e}")))?;
            out.push(RasterizedPage {
                page: p,
                path: png_path,
                _keepalive: dir_arc.clone(),
            });
        }
        Ok(out)
    })
}

/// `rasterize_pages_blocking` の streaming 版。各ページ render 完了の都度
/// `tx.blocking_send` で送る (受信側が drop していたら ループを抜ける)。
fn rasterize_pages_streaming_blocking(
    path: &str,
    pages: &[u32],
    target_longest_side_px: u32,
    tx: tokio::sync::mpsc::Sender<RasterizedPage>,
) -> Result<()> {
    use pdfium_render::prelude::*;

    let dir = tempfile::tempdir().map_err(Error::Io)?;
    let dir_arc: Arc<dyn std::any::Any + Send + Sync> = Arc::new(dir);
    let dir_path = dir_arc
        .downcast_ref::<tempfile::TempDir>()
        .expect("Arc holds TempDir")
        .path()
        .to_path_buf();
    let cfg = PdfRenderConfig::new().set_maximum_width(target_longest_side_px as Pixels);

    with_cached_document(path, |document| {
        let total = document.pages().len() as u32;
        for &p in pages {
            if p == 0 || p > total {
                tracing::warn!("rasterize_pages: requested page {p} out of range (total={total})");
                continue;
            }
            let page = document
                .pages()
                .get(p as u16 - 1)
                .map_err(|e| Error::Parse(format!("get page {p}: {e}")))?;
            let bitmap = page
                .render_with_config(&cfg)
                .map_err(|e| Error::Parse(format!("render page {p}: {e}")))?;
            let img = bitmap.as_image().to_rgba8();
            let png_path = dir_path.join(format!("page-{p}.png"));
            img.save(&png_path)
                .map_err(|e| Error::Parse(format!("save png page {p}: {e}")))?;
            let item = RasterizedPage {
                page: p,
                path: png_path,
                _keepalive: dir_arc.clone(),
            };
            // receiver 側 (OCR consumer) が drop されたら以降の render を続ける
            // 意味は無いので break。channel 容量フル時はここで block して back-
            // pressure する (peak disk のバウンドにそのまま効く)。
            if tx.blocking_send(item).is_err() {
                break;
            }
        }
        Ok(())
    })
}

#[cfg(test)]
mod tests {
    use super::{split_pages, PdfiumRasterizer};

    #[test]
    fn splits_pages_on_form_feed() {
        let text = "p1 para1\n\np1 para2\u{c}p2 para1";
        let blocks = split_pages(text);
        assert_eq!(blocks.len(), 3);
        assert_eq!(blocks[0].page, Some(1));
        assert_eq!(blocks[1].page, Some(1));
        assert_eq!(blocks[2].page, Some(2));
        assert_eq!(blocks[2].text, "p2 para1");
    }

    #[test]
    fn empty_or_form_feed_only_yields_no_blocks() {
        assert!(split_pages("").is_empty());
        assert!(split_pages("\u{c}\u{c}\u{c}").is_empty());
    }

    #[test]
    fn skips_empty_paragraphs_in_a_page() {
        let text = "para1\n\n\n\npara2\n\n   \n\npara3";
        let blocks = split_pages(text);
        assert_eq!(blocks.len(), 3);
        for b in &blocks {
            assert_eq!(b.page, Some(1));
            assert!(!b.text.is_empty());
        }
    }

    #[test]
    fn page_numbers_are_1_indexed_and_monotonic() {
        let text = "a\u{c}b\u{c}c\u{c}d";
        let blocks = split_pages(text);
        assert_eq!(blocks.len(), 4);
        for (i, b) in blocks.iter().enumerate() {
            assert_eq!(b.page, Some((i as u32) + 1));
        }
    }

    #[test]
    fn block_metadata_has_no_heading_or_bbox() {
        let blocks = split_pages("foo");
        assert_eq!(blocks.len(), 1);
        assert!(blocks[0].heading_path.is_empty());
        assert!(blocks[0].bbox.is_none());
    }

    #[test]
    fn rasterizer_default_target_is_sane() {
        let r = PdfiumRasterizer::default();
        assert!(r.target_longest_side_px >= 800);
        assert!(r.target_longest_side_px <= 4096);
    }
}
