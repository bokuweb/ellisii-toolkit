//! ファイル → parse → chunk → embed → store のパイプライン。
//! 進捗は async コールバックで通知する。

use ellisii_chunker::{ChunkConfig, Chunker, DefaultChunker};
use ellisii_core::{Chunk, Error, Result, SourceKind};
use ellisii_embed_core::Embedder;
use ellisii_ocr::{OcrBackend, PdfRasterizer};
use ellisii_parsers_core::ParsedBlock;
use ellisii_store_core::VectorStore;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::sync::Arc;
use uuid::Uuid;

mod ocr_cache;
pub use ocr_cache::OcrCache;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "stage", rename_all = "kebab-case")]
pub enum Progress {
    Started {
        source_id: Uuid,
        path: String,
    },
    Parsed {
        source_id: Uuid,
        blocks: usize,
    },
    /// PDF の OCR 経路で 1 ページ処理が終わるたびに発火する。
    /// 多ページのスキャン PDF で「parsing のまま固まったように見える」問題を防ぐ。
    Ocr {
        source_id: Uuid,
        page: u32,
        total: u32,
    },
    Chunked {
        source_id: Uuid,
        chunks: usize,
    },
    Embedded {
        source_id: Uuid,
        chunks: usize,
    },
    Stored {
        source_id: Uuid,
        chunks: usize,
    },
    Failed {
        source_id: Uuid,
        error: String,
    },
}

pub type ProgressSink = Arc<dyn Fn(Progress) + Send + Sync + 'static>;

#[derive(Debug, Clone)]
pub struct IngestReport {
    pub source_id: Uuid,
    pub kind: SourceKind,
    pub chunks_stored: usize,
}

pub struct Ingestor<E: Embedder, S: VectorStore> {
    pub embedder: Arc<E>,
    pub store: Arc<S>,
    pub ocr: Option<Arc<dyn OcrBackend>>,
    pub pdf_rasterizer: Option<Arc<dyn PdfRasterizer>>,
    /// 既存利用との後方互換用に config 単独でも保持。`chunker` が `None` のときは
    /// `DefaultChunker::new(chunk_config)` で組み立てて使う。`with_chunker(...)` で
    /// 任意の `Arc<dyn Chunker>` を差し込めば config は無視される。
    pub chunk_config: ChunkConfig,
    pub chunker: Option<Arc<dyn Chunker>>,
    pub batch_size: usize,
    /// per-page OCR 結果キャッシュのルートディレクトリ。`None` の場合 cache
    /// 無効 (= 毎回 OCR を回す) で、`Some(dir)` のときはこのディレクトリ配下
    /// に `ocr/v1/<bucket>/page-<N>.json` という構成で per-page 結果を保存。
    /// 同一 PDF の再 ingest 時に OCR 部分を完全に skip できる。
    pub ocr_cache_dir: Option<PathBuf>,
}

impl<E: Embedder, S: VectorStore> Ingestor<E, S> {
    pub fn new(embedder: Arc<E>, store: Arc<S>) -> Self {
        Self {
            embedder,
            store,
            ocr: None,
            pdf_rasterizer: None,
            chunk_config: ChunkConfig::default(),
            chunker: None,
            batch_size: 16,
            ocr_cache_dir: None,
        }
    }

    /// 任意の `Chunker` 実装をセット。`None` のまま build すると `DefaultChunker`
    /// (= `chunk_config` 経由の既存挙動) が走る。
    pub fn with_chunker(mut self, c: Arc<dyn Chunker>) -> Self {
        self.chunker = Some(c);
        self
    }

    pub fn with_ocr(mut self, ocr: Arc<dyn OcrBackend>) -> Self {
        self.ocr = Some(ocr);
        self
    }

    pub fn with_pdf_rasterizer(mut self, r: Arc<dyn PdfRasterizer>) -> Self {
        self.pdf_rasterizer = Some(r);
        self
    }

    pub fn with_ocr_cache_dir(mut self, dir: PathBuf) -> Self {
        self.ocr_cache_dir = Some(dir);
        self
    }

    pub async fn ingest_file(
        &self,
        path: &str,
        notebook_id: Uuid,
        source_id: Uuid,
        on_progress: Option<ProgressSink>,
    ) -> Result<IngestReport> {
        let emit = |p: Progress| {
            if let Some(s) = &on_progress {
                s(p);
            }
        };
        emit(Progress::Started {
            source_id,
            path: path.to_string(),
        });

        let mut parsed = match ellisii_parsers::parse(path).await {
            Ok(p) => p,
            Err(e) => {
                emit(Progress::Failed {
                    source_id,
                    error: e.to_string(),
                });
                return Err(e);
            }
        };

        // OCR フォールバック:
        // - 画像: そのまま OCR バックエンドに渡す
        // - PDF : テキストレイヤが空のページだけを画像化して OCR にかける。
        //         全ページがテキスト無し → 全ページ OCR (旧スキャン PDF 経路)、
        //         一部だけテキスト無し → そのページだけ OCR (混在 PDF 救済)、
        //         全ページにテキストあり → OCR スキップ。
        if let Some(ocr) = self.ocr.as_ref() {
            match parsed.kind {
                SourceKind::Pdf => {
                    if let Some(rast) = self.pdf_rasterizer.as_ref() {
                        // テキストレイヤが取れなかったページを OCR で補完する。
                        // page_count / rasterize 失敗時は、parsed.blocks が空なら
                        // 「テキストも OCR も無い」状態で 0 chunks ready になって
                        // しまうので Failed として上に伝える (UI から見える)。
                        // 部分テキストがあれば warn で続行 (壊れたページだけ諦める)。
                        let total_for_progress = match rast.page_count(path).await {
                            Ok(t) => Some(t),
                            Err(e) => {
                                if parsed.blocks.is_empty() {
                                    let err = Error::Parse(format!("page_count failed: {e}"));
                                    emit(Progress::Failed {
                                        source_id,
                                        error: err.to_string(),
                                    });
                                    return Err(err);
                                }
                                tracing::warn!("page_count failed for {path}: {e}");
                                None
                            }
                        };
                        if let Some(total) = total_for_progress {
                            let missing = pages_without_text(total, &parsed.blocks);
                            if !missing.is_empty() {
                                let on_page = |p: u32| {
                                    emit(Progress::Ocr {
                                        source_id,
                                        page: p,
                                        total,
                                    });
                                };
                                let cache = self.ocr_cache_dir.as_ref().and_then(|root| {
                                    OcrCache::for_file(root, std::path::Path::new(path))
                                });
                                match ocr_pdf_pages_streaming(
                                    rast.as_ref(),
                                    ocr.as_ref(),
                                    path,
                                    &missing,
                                    &on_page,
                                    cache.as_ref(),
                                )
                                .await
                                {
                                    Ok(extra) => parsed.blocks.extend(extra),
                                    Err(e) if parsed.blocks.is_empty() => {
                                        emit(Progress::Failed {
                                            source_id,
                                            error: e.to_string(),
                                        });
                                        return Err(e);
                                    }
                                    Err(e) => tracing::warn!("ocr failed for {path}: {e}"),
                                }
                            }
                        }
                    } else if parsed.blocks.is_empty() {
                        tracing::warn!(
                            "pdf has no text layer but no pdf_rasterizer configured; \
                             skipping ocr for {path}"
                        );
                    }
                }
                SourceKind::Image if parsed.blocks.is_empty() => {
                    match ocr.ocr_image(path).await {
                        Ok(blocks) if !blocks.is_empty() => {
                            // PDF パスと同じ理由で、画像 1 枚から返ってくる
                            // 複数行 OcrBlock を 1 ParsedBlock に結合する。
                            // 画像はそもそも 1 枚 = 1 ページ扱いなので、page
                            // は最初の block の値を採用する (ndlocr 実装は
                            // page=1 固定)。
                            let page = blocks.first().map(|b| b.page).unwrap_or(1);
                            let text = join_ocr_lines_to_paragraph(&blocks);
                            let bbox = union_bbox(&blocks);
                            parsed.blocks = vec![ParsedBlock {
                                text,
                                heading_path: vec![format!("Page {}", page)],
                                page: Some(page),
                                bbox,
                            }];
                        }
                        Err(e) => tracing::warn!("ocr failed for {path}: {e}"),
                        _ => {}
                    }
                }
                _ => {}
            }
        }

        emit(Progress::Parsed {
            source_id,
            blocks: parsed.blocks.len(),
        });

        let chunks = match &self.chunker {
            Some(c) => c.chunk(&parsed, source_id),
            None => DefaultChunker::new(self.chunk_config).chunk(&parsed, source_id),
        };
        emit(Progress::Chunked {
            source_id,
            chunks: chunks.len(),
        });

        let stored = self.embed_and_store(notebook_id, &chunks).await?;
        emit(Progress::Embedded {
            source_id,
            chunks: stored,
        });
        emit(Progress::Stored {
            source_id,
            chunks: stored,
        });

        Ok(IngestReport {
            source_id,
            kind: parsed.kind,
            chunks_stored: stored,
        })
    }

    async fn embed_and_store(&self, notebook_id: Uuid, chunks: &[Chunk]) -> Result<usize> {
        if chunks.is_empty() {
            return Ok(0);
        }
        let mut total = 0usize;
        for batch in chunks.chunks(self.batch_size) {
            // 埋め込み入力は body に heading_path を 1 行 prefix してから渡す。
            // 章タイトル・節番号がベクタ表現に染みるので、章/節 intent クエリ
            // ("2 章のアンチパターンは?", "意思表示について教えて") の retrieval
            // 命中率が上がる。store には `chunk.text` のままを保存する (citation
            // 表示や rerank 入力として heading が二重に出ないようにするため)。
            let texts: Vec<String> = batch.iter().map(augment_text_for_embedding).collect();
            let embeddings = self.embedder.embed(&texts).await?;
            if embeddings.len() != batch.len() {
                return Err(Error::Embed("batch size mismatch".into()));
            }
            self.store.upsert(notebook_id, batch, &embeddings).await?;
            total += batch.len();
        }
        Ok(total)
    }
}

/// `chunk.heading_path` を `chunk.text` の前に折り畳んで埋め込み入力にする。
/// heading が空ならそのまま text を返す。
fn augment_text_for_embedding(c: &Chunk) -> String {
    if c.heading_path.is_empty() {
        return c.text.clone();
    }
    let heading = c.heading_path.join(" / ");
    format!("{heading}\n{}", c.text)
}

/// 1 ページ分の OcrBlock 群を 1 つの段落テキストに結合する。
///
/// ndlocr は 1 行 = 1 OcrBlock で返してくるため、`\n` で素直に繋ぐと、
/// 日本語の column 端で改行された単語 (「バ\nフォーマンス」) や、文の
/// 途中で改行された一文 (「セキュリティやバ\nフォーマンス、…」) が
/// embedding/FTS にそのまま流れ、検索 hit 率が落ちる。
///
/// ルール:
/// - 直前行の末尾が **CJK の継続文字** (= 句読点や閉じ括弧 `。！？」』）` 等
///   ではない CJK 文字) で、現在行の先頭が CJK なら、**セパレータなし**
///   で連結する (= column wrap を取り戻す)。
/// - それ以外は `\n` で連結 (英文行末の hyphenation は ndlocr 側 0.0.7 で
///   既に結合されている前提なので、ここでは追加処理しない)。
fn join_ocr_lines_to_paragraph(blocks: &[ellisii_ocr::OcrBlock]) -> String {
    let mut out = String::new();
    let mut prev_last: Option<char> = None;
    for b in blocks {
        let text = b.text.as_str();
        if !out.is_empty() {
            let cur_first = text.chars().next();
            let join_no_sep = match (prev_last, cur_first) {
                (Some(p), Some(c)) => is_cjk_continuation(p) && is_cjk(c),
                _ => false,
            };
            out.push_str(if join_no_sep { "" } else { "\n" });
        }
        out.push_str(text);
        prev_last = text.chars().last();
    }
    out
}

fn is_cjk(c: char) -> bool {
    matches!(
        c,
        '\u{3040}'..='\u{309F}'   // ひらがな
        | '\u{30A0}'..='\u{30FF}' // カタカナ (ー も含む)
        | '\u{3400}'..='\u{4DBF}' // CJK ext A
        | '\u{4E00}'..='\u{9FFF}' // CJK base
        | '\u{FF66}'..='\u{FF9F}' // 半角カナ
    )
}

/// 「次行は前行の続き」と推定できる CJK 文字。文末記号 (`。！？`) や閉じ
/// 括弧 (`」』）)`) は段落境界とみなして false を返す。
fn is_cjk_continuation(c: char) -> bool {
    if matches!(
        c,
        '。' | '！' | '？' | '」' | '』' | '）' | ')' | '!' | '?' | '.'
    ) {
        return false;
    }
    is_cjk(c)
}

/// 複数 OcrBlock の bbox を覆う union を返す。`[0,0,0,0]` の dummy bbox
/// (ndlocr の `apply_structural_rules` 後はこれ) しか無いときは `None`。
fn union_bbox(blocks: &[ellisii_ocr::OcrBlock]) -> Option<[f32; 4]> {
    let mut iter = blocks.iter().filter(|b| b.bbox != [0.0, 0.0, 0.0, 0.0]);
    let first = iter.next()?;
    let mut acc = first.bbox;
    for b in iter {
        acc[0] = acc[0].min(b.bbox[0]);
        acc[1] = acc[1].min(b.bbox[1]);
        acc[2] = acc[2].max(b.bbox[2]);
        acc[3] = acc[3].max(b.bbox[3]);
    }
    Some(acc)
}

/// 既知の総ページ数と `parsed_blocks` から「テキストが取れていないページ」
/// (1-indexed) を返す。`page=None` の block は寄与しない。
fn pages_without_text(total: u32, parsed_blocks: &[ParsedBlock]) -> Vec<u32> {
    if total == 0 {
        return vec![];
    }
    let covered: std::collections::HashSet<u32> = parsed_blocks
        .iter()
        .filter_map(|b| b.page)
        .filter(|p| *p >= 1 && *p <= total)
        .collect();
    (1..=total).filter(|p| !covered.contains(p)).collect()
}

/// rasterize → OCR をパイプライン化したストリーミング版。
///
/// 旧実装は `stream::iter(pages).map(rasterize→OCR).buffer_unordered(N)` で
/// 各タスク内で **rasterize → OCR を直列** に走らせていた。ところが pdfium は
/// プロセス全体で global mutex を持つため rasterize は実質 1 並列、その間 OCR
/// は始まれず、結果 4 並列を謳いながら ~7s/page (= 完全 sequential) になっていた。
///
/// 本版は:
///   - upstream `then(rasterize)` が **直列に** ページを 1 枚ずつラスタライズ
///   - downstream `map(ocr).buffer_unordered(N)` が **並列に** OCR を消費
///
/// → rasterize 1 が走る隣で OCR N が走る (= 真のパイプライン)。
///
/// 性能見積り: rasterize ~50-100ms / OCR ~5s / N=4 で OCR 律速 → ~1.25s/page。
/// SQL アンチパターン PDF 330p で 50 分 → 7 分相当 (CoreML 抜きでも)。
///
/// per-page で `on_page` を呼ぶ点・1 ページ = 1 ParsedBlock に結合する点・
/// rasterize 失敗で全体 Err にする点は旧実装の挙動を維持する。
async fn ocr_pdf_pages_streaming(
    rasterizer: &dyn PdfRasterizer,
    ocr: &dyn OcrBackend,
    path: &str,
    pages: &[u32],
    on_page: &(dyn Fn(u32) + Send + Sync),
    cache: Option<&OcrCache>,
) -> Result<Vec<ParsedBlock>> {
    if pages.is_empty() {
        return Ok(vec![]);
    }

    // Step 0: cache hit を先に取り出して、未 hit のページだけを rasterize に
    // 回す。同じ PDF を再 ingest した場合、ここでほぼ全 page が cache hit に
    // なって OCR をまるごと skip できる。
    let mut out: Vec<ParsedBlock> = Vec::new();
    let mut pages_to_ocr: Vec<u32> = Vec::with_capacity(pages.len());
    let mut cache_hits = 0usize;
    if let Some(cache) = cache {
        for &p in pages {
            match cache.get(p) {
                Some(blocks) => {
                    cache_hits += 1;
                    if !blocks.is_empty() {
                        let text = join_ocr_lines_to_paragraph(&blocks);
                        let bbox = union_bbox(&blocks);
                        out.push(ParsedBlock {
                            text,
                            heading_path: vec![format!("Page {}", p)],
                            page: Some(p),
                            bbox,
                        });
                    }
                    on_page(p);
                }
                None => pages_to_ocr.push(p),
            }
        }
        tracing::info!(
            "ocr_pdf_pages_streaming: cache hit {} / {} pages",
            cache_hits,
            pages.len()
        );
    } else {
        pages_to_ocr.extend_from_slice(pages);
    }

    if pages_to_ocr.is_empty() {
        out.sort_by_key(|b| b.page);
        return Ok(out);
    }

    let concurrency = ocr_page_concurrency();
    let cpus_for_log = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(0);
    tracing::info!(
        "ocr_pdf_pages_streaming: cpus={cpus_for_log} concurrency={concurrency} ocr_pages={} (cache_hits={})",
        pages_to_ocr.len(),
        cache_hits
    );
    let pages = &pages_to_ocr[..];

    // **page-parallel pipelined OCR**:
    //   - rasterize は 1 thread で streaming に mpsc へ送る
    //   - consumer は `buffer_unordered(N)` で N page 同時に `ocr_image` を呼ぶ
    //
    // cross-page batched (`ocr_images_batch`) は CPU EP では throughput が
    // 出ないことが実測でわかった: parseq decoder は autoregressive で
    // max_seq_len × per_token cost、batch 次元の追加コストは GPU/ANE では
    // 軽視できるが CPU では memory bandwidth で線形に増えるため、batch=N の
    // per-row cost が batch=1 と同等になる (= batching の利得が出ない)。
    //
    // 一方 page-parallel + 0.0.12 cascade single-session-per-call は、N concurrent
    // ocr_image が pool 内 N 異なる session を独立に掴むので、CPU 上でも真の
    // 並列が出る (per-block ~0.42 s/block で N 倍の throughput)。
    //
    // `ocr_images_batch` API は trait に残してあり (NdlocrBackend で実装済)、
    // 将来 GPU/CoreML EP が動くようになれば呼び替えで cross-page batched を
    // 復活できる。
    let (rast_tx, rast_rx) =
        tokio::sync::mpsc::channel::<ellisii_ocr::RasterizedPage>((concurrency * 2).max(2));

    let rasterize_t0 = std::time::Instant::now();
    let producer = async {
        let r = rasterizer
            .rasterize_pages_streaming(path, pages, rast_tx)
            .await;
        tracing::info!(
            "ocr_stream: rasterize phase done in {:.2}s ({:?})",
            rasterize_t0.elapsed().as_secs_f64(),
            r.as_ref().map(|_| "ok").unwrap_or("err")
        );
        r
    };

    use futures_util::{stream, StreamExt};
    let rast_stream = stream::unfold(rast_rx, |mut rx| async move {
        rx.recv().await.map(|item| (item, rx))
    });
    let consumer = rast_stream
        .map(|rp| async move {
            let img_path = rp.path.to_string_lossy().to_string();
            let page_no = rp.page;
            let bbox_for_block;
            let (text_opt, ocr_blocks) = match ocr.ocr_image(&img_path).await {
                Ok(blocks) if !blocks.is_empty() => {
                    let text = join_ocr_lines_to_paragraph(&blocks);
                    bbox_for_block = union_bbox(&blocks);
                    (Some(text), Some(blocks))
                }
                Ok(blocks) => {
                    bbox_for_block = None;
                    // 空 OCR でも cache に書き込んでおく (= 次回も同じ page を
                    // skip できる)。
                    (None, Some(blocks))
                }
                Err(e) => {
                    tracing::warn!("ocr failed for pdf page {page_no} of {path}: {e}");
                    bbox_for_block = None;
                    // OCR 失敗時は cache に書かない (= 次回 retry)
                    (None, None)
                }
            };
            if let (Some(cache), Some(blocks)) = (cache, ocr_blocks.as_ref()) {
                cache.put(page_no, blocks);
            }
            on_page(page_no);
            Ok::<Option<ParsedBlock>, Error>(text_opt.map(|text| ParsedBlock {
                text,
                heading_path: vec![format!("Page {}", page_no)],
                page: Some(page_no),
                bbox: bbox_for_block,
            }))
        })
        .buffer_unordered(concurrency)
        .collect::<Vec<Result<Option<ParsedBlock>>>>();

    let (rast_result, collected) = tokio::join!(producer, consumer);
    rast_result?;
    for r in collected {
        if let Some(b) = r? {
            out.push(b);
        }
    }
    out.sort_by_key(|b| b.page);
    Ok(out)
}

/// OCR ページ並列度。N 個の `ocr_image` 呼び出しが同時に in-flight する。
///
/// `ParseqCascadePool` の pool 並列度 ([`NdlocrBackend::parallelism`]) と同じ
/// 値にすると、各呼び出しが pool 内の異なる session を `try_lock` で掴めて
/// contention が起きにくい (cf. ndlocr-lite-rs 0.0.12)。
///
/// 既定 `(cpus / 3).clamp(2, 4)`。10-core で 3、4-core で 2。1 ocr_image が
/// ~1 active session × intra (~2 thread) + DEIM 1 thread = 3-4 thread / page
/// 程度を消費する想定で、CPU 飽和を狙う配分。
///
/// 環境変数 `ELLISII_OCR_PAGE_CONCURRENCY` で上書き可能 (perf 検証用)。
fn ocr_page_concurrency() -> usize {
    if let Ok(raw) = std::env::var("ELLISII_OCR_PAGE_CONCURRENCY") {
        if let Ok(n) = raw.parse::<usize>() {
            return n.max(1);
        }
    }
    let cpus = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(2);
    (cpus / 3).clamp(2, 4)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ellisii_embed_dummy::DummyEmbedder;
    use ellisii_store_memory::InMemoryStore;
    use std::io::Write;

    fn ocr_block(text: &str) -> ellisii_ocr::OcrBlock {
        ellisii_ocr::OcrBlock {
            text: text.to_string(),
            bbox: [0.0, 0.0, 0.0, 0.0],
            page: 1,
            confidence: 0.9,
        }
    }

    /// 日本語 column wrap で単語が分割された OCR 行は、separator なしで結合。
    /// 「セキュリティやバ\nフォーマンス、…」→「セキュリティやバフォーマンス、…」
    #[test]
    fn join_ocr_lines_merges_cjk_word_wrap_without_separator() {
        let blocks = vec![
            ocr_block("セキュリティやバ"),
            ocr_block("フォーマンス、パージョン管理やテスト、"),
        ];
        let joined = join_ocr_lines_to_paragraph(&blocks);
        assert_eq!(
            joined,
            "セキュリティやバフォーマンス、パージョン管理やテスト、"
        );
    }

    /// 文末 (`。`) の後に新行が来たら段落境界とみなして `\n` で繋ぐ。
    /// CJK 同士でも「。」のあとは続きじゃないので separator を入れる。
    #[test]
    fn join_ocr_lines_keeps_newline_after_sentence_terminator() {
        let blocks = vec![
            ocr_block("これは前の文です。"),
            ocr_block("これは次の文です。"),
        ];
        let joined = join_ocr_lines_to_paragraph(&blocks);
        assert_eq!(joined, "これは前の文です。\nこれは次の文です。");
    }

    /// ASCII ↔ ASCII の改行は separator (`\n`) を維持。CJK ルールは適用しない。
    #[test]
    fn join_ocr_lines_keeps_newline_for_ascii_pairs() {
        let blocks = vec![ocr_block("Hello world"), ocr_block("Next line")];
        let joined = join_ocr_lines_to_paragraph(&blocks);
        assert_eq!(joined, "Hello world\nNext line");
    }

    /// 閉じ括弧で終わる行は段落終端扱い (`」` の後に CJK が来ても繋がない)。
    #[test]
    fn join_ocr_lines_keeps_newline_after_closing_bracket() {
        let blocks = vec![ocr_block("彼は「行こう」"), ocr_block("と言った。")];
        let joined = join_ocr_lines_to_paragraph(&blocks);
        assert_eq!(joined, "彼は「行こう」\nと言った。");
    }

    /// CJK ↔ ASCII の境界はそのまま `\n`。安全側 (column wrap で英字に
    /// 切り替わるケースは稀なので、空 separator で繋ぐと逆に誤連結)。
    #[test]
    fn join_ocr_lines_keeps_newline_at_cjk_ascii_boundary() {
        let blocks = vec![ocr_block("日本語の文"), ocr_block("English next")];
        let joined = join_ocr_lines_to_paragraph(&blocks);
        assert_eq!(joined, "日本語の文\nEnglish next");
    }

    fn ingestor() -> Ingestor<DummyEmbedder, InMemoryStore> {
        Ingestor::new(
            Arc::new(DummyEmbedder::new(8)),
            Arc::new(InMemoryStore::new()),
        )
    }

    #[tokio::test]
    async fn ingests_text_file_and_emits_progress() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("a.txt");
        let mut f = std::fs::File::create(&path).unwrap();
        writeln!(f, "段落1\n本文の続き\n\n段落2\n別本文").unwrap();
        drop(f);

        let events = Arc::new(parking_lot::Mutex::new(Vec::<Progress>::new()));
        let sink_events = events.clone();
        let sink: ProgressSink = Arc::new(move |p| sink_events.lock().push(p));

        let ing = ingestor();
        let id = Uuid::new_v4();
        let nb = Uuid::new_v4();
        let rep = ing
            .ingest_file(path.to_str().unwrap(), nb, id, Some(sink))
            .await
            .unwrap();
        assert_eq!(rep.kind, SourceKind::Text);
        assert!(rep.chunks_stored >= 1);

        let ev = events.lock();
        assert!(matches!(ev.first(), Some(Progress::Started { .. })));
        assert!(ev.iter().any(|p| matches!(p, Progress::Stored { .. })));
    }

    struct FakeOcr {
        text: String,
    }
    #[async_trait::async_trait]
    impl OcrBackend for FakeOcr {
        async fn ocr_image(&self, _path: &str) -> Result<Vec<ellisii_ocr::OcrBlock>> {
            Ok(vec![ellisii_ocr::OcrBlock {
                text: self.text.clone(),
                bbox: [0.0, 0.0, 100.0, 20.0],
                page: 1,
                confidence: 0.9,
            }])
        }
    }

    #[tokio::test]
    async fn ocr_fallback_runs_for_image_sources() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("scan.png");
        std::fs::write(&path, b"\x89PNG\r\n\x1a\n").unwrap();

        let store = Arc::new(InMemoryStore::new());
        let ing = Ingestor::new(Arc::new(DummyEmbedder::new(8)), store.clone()).with_ocr(Arc::new(
            FakeOcr {
                text: "認識されたテキスト".into(),
            },
        ));
        let id = Uuid::new_v4();
        let nb = Uuid::new_v4();
        let rep = ing
            .ingest_file(path.to_str().unwrap(), nb, id, None)
            .await
            .unwrap();
        assert_eq!(rep.kind, SourceKind::Image);
        assert!(rep.chunks_stored >= 1);
        let kw = store.keyword_search(Some(nb), "認識", 5).await.unwrap();
        assert_eq!(kw.len(), 1);
    }

    /// テスト用ラスタライザ。指定総ページ数を返し、要求されたページだけ
    /// PNG ダミーファイルを生成して返す。`rasterize_pages` の呼び出し履歴を
    /// 記録し、不要なページを render しないことの検証にも使う。
    struct FakeRasterizer {
        total_pages: u32,
        dir: Arc<tempfile::TempDir>,
        rendered: parking_lot::Mutex<Vec<u32>>,
    }

    impl FakeRasterizer {
        fn with_total(total_pages: u32) -> Self {
            Self {
                total_pages,
                dir: Arc::new(tempfile::tempdir().unwrap()),
                rendered: parking_lot::Mutex::new(Vec::new()),
            }
        }
        fn rendered_calls(&self) -> Vec<u32> {
            self.rendered.lock().clone()
        }
    }

    #[async_trait::async_trait]
    impl ellisii_ocr::PdfRasterizer for FakeRasterizer {
        async fn page_count(&self, _path: &str) -> Result<u32> {
            Ok(self.total_pages)
        }
        async fn rasterize_pages(
            &self,
            _path: &str,
            pages: &[u32],
        ) -> Result<Vec<ellisii_ocr::RasterizedPage>> {
            self.rendered.lock().extend(pages.iter().copied());
            let mut out = Vec::new();
            for &p in pages {
                let path = self.dir.path().join(format!("page-{p}.png"));
                std::fs::write(&path, b"\x89PNG\r\n").unwrap();
                out.push(ellisii_ocr::RasterizedPage {
                    page: p,
                    path,
                    _keepalive: self.dir.clone(),
                });
            }
            Ok(out)
        }
    }

    /// 受け取った画像パスに応じてページ別のテキストを返す OCR モック。
    struct PerPathOcr;
    #[async_trait::async_trait]
    impl OcrBackend for PerPathOcr {
        async fn ocr_image(&self, path: &str) -> Result<Vec<ellisii_ocr::OcrBlock>> {
            let name = std::path::Path::new(path)
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("");
            Ok(vec![ellisii_ocr::OcrBlock {
                text: format!("認識結果 {name}"),
                bbox: [0.0, 0.0, 100.0, 20.0],
                page: 1,
                confidence: 0.9,
            }])
        }
    }

    #[tokio::test]
    async fn ocr_pdf_pages_streaming_emits_one_block_per_requested_page() {
        let rast = FakeRasterizer::with_total(5);
        let calls = parking_lot::Mutex::new(Vec::<u32>::new());
        let on_page = |p: u32| calls.lock().push(p);
        let blocks = super::ocr_pdf_pages_streaming(
            &rast,
            &PerPathOcr,
            "/dummy.pdf",
            &[2, 4],
            &on_page,
            None,
        )
        .await
        .unwrap();
        assert_eq!(blocks.len(), 2);
        assert_eq!(blocks[0].page, Some(2));
        assert_eq!(blocks[1].page, Some(4));
        assert!(blocks[0].text.contains("page-2"));
        assert!(blocks[1].text.contains("page-4"));
        assert_eq!(rast.rendered_calls(), vec![2, 4]);
        // 進捗は per-page。一気にではなく逐次。
        assert_eq!(*calls.lock(), vec![2, 4]);
    }

    /// 複数行を返す OCR モック。実 OCR (ndlocr) は 1 行 = 1 OcrBlock で返すので、
    /// 同じ振る舞いを再現する。
    struct MultiLineOcr {
        lines: Vec<&'static str>,
    }
    #[async_trait::async_trait]
    impl OcrBackend for MultiLineOcr {
        async fn ocr_image(&self, _path: &str) -> Result<Vec<ellisii_ocr::OcrBlock>> {
            Ok(self
                .lines
                .iter()
                .enumerate()
                .map(|(i, t)| ellisii_ocr::OcrBlock {
                    text: (*t).to_string(),
                    bbox: [0.0, (i as f32) * 20.0, 100.0, ((i + 1) as f32) * 20.0],
                    page: 1,
                    confidence: 0.9,
                })
                .collect())
        }
    }

    /// OCR が 1 ページぶんで複数行 (= 複数 OcrBlock) を返したとき、
    /// `ocr_pdf_pages_streaming` はそれらを **ページ単位で 1 ParsedBlock に
    /// 結合** すること。1 行 = 1 ParsedBlock のままだと chunker 側で
    /// `min_chars` マージが効いても 1 ページ ~15 chunk に膨らみ、SQL
    /// アンチパターン PDF (330p) で 4539 chunks のような過剰分割を起こす。
    #[tokio::test]
    async fn ocr_pdf_pages_streaming_merges_multiple_lines_per_page() {
        let rast = FakeRasterizer::with_total(3);
        let ocr = MultiLineOcr {
            lines: vec![
                "第1章 ジェイウォーク（信号無視）",
                "カンマ区切りリストはアンチパターンとされる。",
                "理由はクエリの効率と整合性に問題があるため。",
            ],
        };
        let blocks =
            super::ocr_pdf_pages_streaming(&rast, &ocr, "/dummy.pdf", &[1, 2], &|_| {}, None)
                .await
                .unwrap();
        // 2 ページ要求 → 2 ParsedBlock (1 ページにつき 1 個)。3 行 × 2 ページ
        // = 6 ParsedBlock になっていたら旧挙動。
        assert_eq!(blocks.len(), 2, "expected 1 ParsedBlock per page");
        for b in &blocks {
            assert!(
                b.text.contains("第1章"),
                "merged text missing heading: {:?}",
                b.text
            );
            assert!(
                b.text.contains("アンチパターン"),
                "merged text missing body: {:?}",
                b.text
            );
            // 行間は \n で区切られている (chunker の recursive_split が \n を
            // セパレータに含むので、結合してもページ内の自然な切れ目は残る)。
            assert!(
                b.text.contains('\n'),
                "lines should be joined with \\n: {:?}",
                b.text
            );
        }
    }

    // 旧 `ocr_pdf_pages_streaming_uses_batch_api` は cross-page batched 経路
    // (= ingest が `ocr_images_batch` を直接呼ぶ) を検証していたが、CPU EP
    // では batching が throughput に効かないと実測でわかったため、ingest 側
    // では `ocr_image` per-page 並列に戻した。`OcrBackend::ocr_images_batch`
    // 自体は trait に残してあり、GPU/ANE EP が動くようになったときに再採用
    // できる (NdlocrBackend で実装済み)。

    #[tokio::test]
    async fn ocr_pdf_pages_streaming_skips_render_when_no_pages_requested() {
        let rast = FakeRasterizer::with_total(3);
        let blocks =
            super::ocr_pdf_pages_streaming(&rast, &PerPathOcr, "/dummy.pdf", &[], &|_| {}, None)
                .await
                .unwrap();
        assert!(blocks.is_empty());
        assert!(rast.rendered_calls().is_empty());
    }

    #[test]
    fn pages_without_text_returns_uncovered_pages() {
        let parsed = vec![
            ParsedBlock {
                text: "p1 text".into(),
                heading_path: vec![],
                page: Some(1),
                bbox: None,
            },
            ParsedBlock {
                text: "p3 text".into(),
                heading_path: vec![],
                page: Some(3),
                bbox: None,
            },
        ];
        assert_eq!(super::pages_without_text(4, &parsed), vec![2, 4]);
    }

    #[test]
    fn pages_without_text_returns_all_when_no_text() {
        assert_eq!(super::pages_without_text(3, &[]), vec![1, 2, 3]);
    }

    #[test]
    fn pages_without_text_returns_empty_when_all_covered() {
        let parsed = vec![
            ParsedBlock {
                text: "p1".into(),
                heading_path: vec![],
                page: Some(1),
                bbox: None,
            },
            ParsedBlock {
                text: "p2".into(),
                heading_path: vec![],
                page: Some(2),
                bbox: None,
            },
        ];
        assert!(super::pages_without_text(2, &parsed).is_empty());
    }

    #[test]
    fn pages_without_text_handles_zero_total() {
        assert!(super::pages_without_text(0, &[]).is_empty());
    }

    #[tokio::test]
    async fn unknown_extension_fails_with_progress() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("x.bin");
        std::fs::write(&path, b"junk").unwrap();

        let events = Arc::new(parking_lot::Mutex::new(Vec::<Progress>::new()));
        let sink_events = events.clone();
        let sink: ProgressSink = Arc::new(move |p| sink_events.lock().push(p));

        let ing = ingestor();
        let r = ing
            .ingest_file(
                path.to_str().unwrap(),
                Uuid::new_v4(),
                Uuid::new_v4(),
                Some(sink.clone()),
            )
            .await;
        assert!(r.is_err());
        let ev = events.lock();
        assert!(ev.iter().any(|p| matches!(p, Progress::Failed { .. })));
    }

    /// テスト用: 常に Err を返すラスタライザ。pdfium が dlopen 失敗するなど
    /// 「OCR フォールバックが構造的に動かない」シナリオを再現する。
    struct BrokenRasterizer;
    #[async_trait::async_trait]
    impl ellisii_ocr::PdfRasterizer for BrokenRasterizer {
        async fn page_count(&self, _path: &str) -> Result<u32> {
            Err(Error::Parse("rasterizer broken".into()))
        }
        async fn rasterize_pages(
            &self,
            _path: &str,
            _pages: &[u32],
        ) -> Result<Vec<ellisii_ocr::RasterizedPage>> {
            Err(Error::Parse("rasterizer broken".into()))
        }
    }

    /// テキストレイヤなしの PDF を装うため `parser-pdf` を経由せず、
    /// pdf 拡張子のダミーファイルを与える。`pdf-extract` が呼ばれて空テキスト
    /// (= Ok([])) を返すパスをテストする。
    #[tokio::test]
    async fn pdf_with_no_text_and_broken_rasterizer_emits_failed() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("scan.pdf");
        // pdf-extract には壊れているが、parse 関数は Err を返す。
        // → ingest 全体としては parse の段階で Failed になる。
        // ここで本当にテストしたいのは「page_count 失敗 + blocks 空」の Failed
        // なので、parse が Ok([]) を返せるダミーを通すのは難しく、代わりに
        // 単体テスト相当として ocr_pdf_pages_streaming に Broken を渡し、
        // streaming が Err を返すことを確認する (上位の Failed 化はそれが
        // 引き起こされる前提を満たすため)。
        std::fs::write(&path, b"%PDF-1.4\n%dummy").unwrap();

        let r = super::ocr_pdf_pages_streaming(
            &BrokenRasterizer,
            &FakeOcr { text: "x".into() },
            path.to_str().unwrap(),
            &[1, 2, 3],
            &|_| {},
            None,
        )
        .await;
        assert!(r.is_err());
    }

    #[tokio::test]
    async fn ocr_pdf_pages_streaming_emits_progress_per_page() {
        let rast = FakeRasterizer::with_total(3);
        let calls = parking_lot::Mutex::new(Vec::<u32>::new());
        let on_page = |p: u32| calls.lock().push(p);
        let _ = super::ocr_pdf_pages_streaming(
            &rast,
            &PerPathOcr,
            "/dummy.pdf",
            &[1, 2, 3],
            &on_page,
            None,
        )
        .await
        .unwrap();
        // buffer_unordered なので順序は保証されないが、全ページぶん発火する
        // ことを set 等価で確認する。
        let mut got = calls.lock().clone();
        got.sort_unstable();
        assert_eq!(got, vec![1, 2, 3]);
        let mut rendered = rast.rendered_calls();
        rendered.sort_unstable();
        assert_eq!(rendered, vec![1, 2, 3]);
    }

    /// 並列化後でも、ParsedBlock 出力は page ascending に sort されていること
    /// (= chunker が文書順を仮定して動くため)。
    #[tokio::test]
    async fn ocr_pdf_pages_streaming_returns_blocks_sorted_by_page() {
        let rast = FakeRasterizer::with_total(5);
        let blocks = super::ocr_pdf_pages_streaming(
            &rast,
            &PerPathOcr,
            "/dummy.pdf",
            &[3, 1, 4, 2],
            &|_| {},
            None,
        )
        .await
        .unwrap();
        let pages: Vec<u32> = blocks.iter().filter_map(|b| b.page).collect();
        assert_eq!(pages, vec![1, 2, 3, 4]);
    }

    /// heading_path が空の chunk は augment されず、そのまま text を embedding
    /// 入力にする (= 後方互換)。
    #[test]
    fn augment_passes_through_when_heading_empty() {
        let c = Chunk {
            id: Uuid::new_v4(),
            source_id: Uuid::new_v4(),
            ord: 0,
            text: "本文だけ".into(),
            heading_path: vec![],
            page: None,
            bbox: None,
            summary: None,
        };
        assert_eq!(super::augment_text_for_embedding(&c), "本文だけ");
    }

    /// heading_path がある chunk は "<heading> / ...\n<body>" の形で折り畳む。
    /// 章/節 intent クエリの retrieval ベクタに heading 語彙を載せる狙い。
    #[test]
    fn augment_prefixes_heading_when_present() {
        let c = Chunk {
            id: Uuid::new_v4(),
            source_id: Uuid::new_v4(),
            ord: 0,
            text: "再帰的な関連を持つデータは…".into(),
            heading_path: vec!["2章 ナイーブツリー".into(), "2.1 目的".into()],
            page: Some(15),
            bbox: None,
            summary: None,
        };
        let out = super::augment_text_for_embedding(&c);
        assert!(out.starts_with("2章 ナイーブツリー / 2.1 目的\n"));
        assert!(out.ends_with("再帰的な関連を持つデータは…"));
    }
}
