use crate::{OcrBackend, OcrBlock, OcrConfig};
use async_trait::async_trait;
use ellisii_core::{Error, Result};
use ndlocr_lite_rs::cascade::{
    estimate_pred_char_bucket, DEFAULT_CASCADE_THRESHOLD_30_TO_50,
    DEFAULT_CASCADE_THRESHOLD_50_TO_100,
};
use ndlocr_lite_rs::infer::cached::ParseqCascadePool;
use ndlocr_lite_rs::infer::deim_cached::DeimPool;
use ndlocr_lite_rs::io as nd_io;
use ndlocr_lite_rs::pipeline::crop::{crop_rgb_u8, BBox, CroppedImage};
use ndlocr_lite_rs::pipeline::reading_order::sort_bboxes_in_reading_order;
use ndlocr_lite_rs::postprocess::page_rules::apply_structural_rules;
use std::sync::{Arc, OnceLock};

/// `(x0, y0, x1, y1, cropped_image)` — 1 行の bbox とその切り出し画像。
type LineCrop = (usize, usize, usize, usize, CroppedImage);

pub struct NdlocrBackend {
    pub config: OcrConfig,
    /// parseq の Session を保持する。OCR を呼ぶたびにモデルを load し直すと
    /// 行数 N に対し parse コストが N 倍掛かるため、ここで 1 度だけ作る。
    /// `OnceLock` で初回 OCR 時に lazy 初期化する (起動時の memory pressure 回避)。
    ///
    /// **cascade**: 30 / 50 / 100 char モデルを 3 つ抱える。1 行の bbox の
    /// アスペクト比から推定文字数を出して適切なモデルに振り分けるので、典型的な
    /// 日本語ページ (短い行が多い) で **大きい model100 を回避** でき、parseq
    /// 推論が ~2-3x 速くなる。各モデルが個別の `ParseqPool` を持ち、
    /// `parallelism` 個の Session を抱える。
    parseq_pool: OnceLock<Arc<ParseqCascadePool>>,
    /// DEIM 行検出 Session のプール。stateless `deim::detect_rgb_u8` は
    /// 1 ページ毎に Session::commit_from_file で ONNX を再 load しており、
    /// M-series で ~2 sec/page の固定オーバーヘッドが乗っていた。
    /// 0.0.4 で導入された `DeimPool` を使えば、N 個の Session を抱えて
    /// ページ並列でも try_lock で空きを奪い合えるようになり、ロード再発を
    /// 起こさず buffer_unordered によるページ並列も可能になる。
    deim_pool: OnceLock<Arc<DeimPool>>,
}

impl NdlocrBackend {
    pub fn new(config: OcrConfig) -> Self {
        Self {
            config,
            parseq_pool: OnceLock::new(),
            deim_pool: OnceLock::new(),
        }
    }

    /// `ParseqCascadePool` 内の 1 バケットあたりの session 数。
    ///
    /// **page-level 並列度に合わせる** ([`ocr_page_concurrency`])。
    /// page-level concurrency = N で同じ瞬間に N 個の `ocr_image` 呼び出しが
    /// 走る。各 page は内部で `recognize_batch_with_buckets_rgb_u8` を呼び、
    /// 30-bucket session の lock を取りに行く (典型的な日本語ページは全行が
    /// 30-bucket)。session が 1 つしかないと N 個の page 呼び出しが pool30 の
    /// Mutex で serialize されてしまい、外側の page 並列度が無効化される
    /// (page 1 が parseq 中、page 2/3 は session 待ち)。
    ///
    /// session を N 個用意すれば、N 個の page が pool30 の異なる session を
    /// `try_lock` で別々に掴めるようになり、本当の並列に近づく。50/100 bucket
    /// も同じ N で揃える (典型ページではほぼ idle なのでメモリだけのコスト)。
    ///
    /// 旧 `(cpus / 2).clamp(1, 4)` は cascade fan-out が常時走る前提で session
    /// を多く抱え過ぎ、thread 爆発の原因だった。0.0.10 / 0.0.11 で fan-out
    /// 自体が消えたので、この値は **page 並列度と完全に揃える** のが正しい。
    fn parallelism() -> usize {
        let cpus = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(2);
        (cpus / 3).clamp(1, 4)
    }

    /// 初回呼び出し時のみ ParseqCascadePool を構築し、以降は同じプールを共有する。
    /// 30/50/100 の 3 モデルを `thread::scope` で並列ロード (cold start 短縮)。
    fn get_or_init_pool(&self) -> Result<Arc<ParseqCascadePool>> {
        if let Some(s) = self.parseq_pool.get() {
            return Ok(s.clone());
        }
        let n = Self::parallelism();
        let pool = ParseqCascadePool::load(
            &self.config.model30,
            &self.config.model50,
            &self.config.model100,
            &self.config.charset,
            n,
        )
        .map_err(|e| Error::Ocr(format!("parseq cascade pool load (n={n}): {e}")))?;
        let arc = Arc::new(pool);
        let _ = self.parseq_pool.set(arc.clone());
        Ok(self.parseq_pool.get().cloned().unwrap_or(arc))
    }

    /// DEIM Pool を 1 度だけロードして使い回す。並列度は ParseqPool と揃える
    /// (= ページ並列 N 個を同じ N で動かす設計)。
    fn get_or_init_deim(&self) -> Result<Arc<DeimPool>> {
        if let Some(s) = self.deim_pool.get() {
            return Ok(s.clone());
        }
        let n = Self::parallelism();
        let pool = DeimPool::load(&self.config.det_model, n)
            .map_err(|e| Error::Ocr(format!("deim pool load (n={n}): {e}")))?;
        let arc = Arc::new(pool);
        let _ = self.deim_pool.set(arc.clone());
        Ok(self.deim_pool.get().cloned().unwrap_or(arc))
    }
}

#[async_trait]
impl OcrBackend for NdlocrBackend {
    async fn ocr_image(&self, path: &str) -> Result<Vec<OcrBlock>> {
        let path = path.to_string();
        let cfg = self.config.clone();
        let pool = self.get_or_init_pool()?;
        let deim = self.get_or_init_deim()?;
        let started = std::time::Instant::now();
        tracing::info!(
            "ocr started: {path} (parseq parallel={})",
            pool.parallelism()
        );
        let path_for_blocking = path.clone();
        let result = tokio::task::spawn_blocking(move || {
            ocr_image_blocking(&path_for_blocking, &cfg, &pool, &deim)
        })
        .await
        .map_err(|e| Error::Ocr(format!("join: {e}")))?;
        let elapsed = started.elapsed().as_secs_f64();
        match &result {
            Ok(blocks) => tracing::info!(
                "ocr done in {elapsed:.1}s: {} blocks ({path})",
                blocks.len()
            ),
            Err(e) => tracing::warn!("ocr failed in {elapsed:.1}s: {e} ({path})"),
        }
        result
    }

    /// 複数画像をまとめて 1 度の parseq cascade 推論で処理する batch API。
    ///
    /// **狙い**: parseq decoder は autoregressive (max_seq_len × per-token cost)
    /// で、batch 次元の追加コストはほぼ無視できる。1 ページ ~25 行 / 各行
    /// ~10 token で 1 推論 ~100ms。3 ページぶん 75 行を 1 batch にしても
    /// 100-150ms 程度しかかからない (per-row では 1.5-2ms)。1 ページずつ
    /// 別の OCR 呼び出しで処理すると ~25-30ms/row 相当の overhead が掛かって
    /// いた (= 行数線形に近い). cross-page batched で **per-row の parseq
    /// コスト 5-10x 短縮** を狙う。
    ///
    /// 流れ:
    ///   1. 各 page を `prepare_page_crops` で load + DEIM 検出 + 行 crop
    ///      (rayon で並列 — DEIM は session 単位 lock なので pool 並列度ぶん
    ///      実並列が得られる)
    ///   2. 全 page の crops を 1 つの batch_inputs に concat (各行の
    ///      origin (page_idx, line_idx) を覚えておく)
    ///   3. ParseqCascadePool で **1 度だけ** recognize 呼ぶ
    ///   4. 結果を origin に従って per-page に振り分け
    ///   5. apply_structural_rules を per-page で適用
    ///
    /// 1 ページの load/DEIM 失敗は warn で握り潰して空 Vec 扱い (= 既定実装と
    /// 同じ挙動)。構造的な失敗 (model 未ロード等) のみ Err で返す。
    async fn ocr_images_batch(&self, paths: &[&str]) -> Result<Vec<Vec<OcrBlock>>> {
        if paths.is_empty() {
            return Ok(vec![]);
        }
        let cfg = self.config.clone();
        let pool = self.get_or_init_pool()?;
        let deim = self.get_or_init_deim()?;
        let owned_paths: Vec<String> = paths.iter().map(|s| s.to_string()).collect();
        let started = std::time::Instant::now();
        tracing::info!(
            "ocr batch started: {} pages (parseq parallel={})",
            paths.len(),
            pool.parallelism()
        );
        let result = tokio::task::spawn_blocking(move || {
            ocr_images_batch_blocking(&owned_paths, &cfg, &pool, &deim)
        })
        .await
        .map_err(|e| Error::Ocr(format!("join: {e}")))?;
        let elapsed = started.elapsed().as_secs_f64();
        match &result {
            Ok(per_page) => {
                let total_blocks: usize = per_page.iter().map(|p| p.len()).sum();
                tracing::info!(
                    "ocr batch done in {elapsed:.1}s: {} pages / {} blocks total",
                    per_page.len(),
                    total_blocks
                );
            }
            Err(e) => tracing::warn!("ocr batch failed in {elapsed:.1}s: {e}"),
        }
        result
    }
}

/// 1 page ぶん load + DEIM + 行 crop。失敗は Err として上に返す (呼び出し側で
/// warn + 空 Vec 扱いにできる)。crop は (x0, y0, x1, y1, CroppedImage) の Vec で
/// 読み順 sort 済み。
fn prepare_page_crops(path: &str, cfg: &OcrConfig, deim: &DeimPool) -> Result<Vec<LineCrop>> {
    let img = nd_io::load_rgb_u8(std::path::Path::new(path))
        .map_err(|e| Error::Ocr(format!("load: {e}")))?;
    let dets = deim
        .detect_rgb_u8(&img.data, img.width, img.height, cfg.det_conf_threshold)
        .map_err(|e| Error::Ocr(format!("deim: {e}")))?;
    let mut bboxes: Vec<[i32; 4]> = dets
        .into_iter()
        .filter(|d| d.class_name.starts_with("line_"))
        .filter_map(|d| {
            let x0 = d.box_xyxy[0];
            let y0 = d.box_xyxy[1];
            let x1 = d.box_xyxy[2];
            let y1 = d.box_xyxy[3];
            if x0 < 0 || y0 < 0 || x0 >= x1 || y0 >= y1 {
                return None;
            }
            if (x1 as usize) > img.width || (y1 as usize) > img.height {
                return None;
            }
            Some([x0, y0, x1, y1])
        })
        .collect();
    sort_bboxes_in_reading_order(&mut bboxes);
    let img_data: &[u8] = &img.data;
    let img_w = img.width;
    let img_h = img.height;
    bboxes
        .iter()
        .map(|[x0i, y0i, x1i, y1i]| {
            let (x0, y0, x1, y1) = (*x0i as usize, *y0i as usize, *x1i as usize, *y1i as usize);
            let crop = crop_rgb_u8(img_data, img_w, img_h, BBox::new(x0, y0, x1, y1))
                .map_err(|e| Error::Ocr(format!("crop: {e}")))?;
            Ok((x0, y0, x1, y1, crop))
        })
        .collect()
}

fn ocr_images_batch_blocking(
    paths: &[String],
    cfg: &OcrConfig,
    pool: &ParseqCascadePool,
    deim: &DeimPool,
) -> Result<Vec<Vec<OcrBlock>>> {
    use rayon::prelude::*;

    // Step 1: 全ページぶん load + DEIM + crop。DEIM は pool 内 N session を
    // try_lock で奪い合うので、rayon 並列で投げて pool 並列度ぶん実並列に
    // 進む。load/DEIM 失敗は warn にして空 crop で続行 (per-page failure)。
    let per_page_crops: Vec<Vec<LineCrop>> = paths
        .par_iter()
        .map(|path| match prepare_page_crops(path, cfg, deim) {
            Ok(c) => c,
            Err(e) => {
                tracing::warn!("ocr batch: prepare_page_crops failed for {path}: {e}");
                Vec::new()
            }
        })
        .collect();

    // Step 2: 全 page の crops を 1 つの batch_inputs に concat。
    // origins[i] = (page_idx, line_idx_within_page) で結果を逆引きする。
    let total_lines: usize = per_page_crops.iter().map(|c| c.len()).sum();
    let mut batch_inputs: Vec<(&[u8], usize, usize, Option<f32>)> = Vec::with_capacity(total_lines);
    let mut origins: Vec<(usize, usize)> = Vec::with_capacity(total_lines);
    for (page_idx, crops) in per_page_crops.iter().enumerate() {
        for (line_idx, (_, _, _, _, c)) in crops.iter().enumerate() {
            let bucket = estimate_pred_char_bucket(
                c.width,
                c.height,
                DEFAULT_CASCADE_THRESHOLD_30_TO_50,
                DEFAULT_CASCADE_THRESHOLD_50_TO_100,
            );
            batch_inputs.push((c.data.as_slice(), c.width, c.height, Some(bucket)));
            origins.push((page_idx, line_idx));
        }
    }

    // Step 3: 1 度だけ parseq cascade 呼び出し。空 batch なら skip。
    let recs = if batch_inputs.is_empty() {
        Vec::new()
    } else {
        pool.recognize_batch_with_buckets_rgb_u8(&batch_inputs)
            .map_err(|e| Error::Ocr(format!("parseq batch: {e}")))?
    };
    if recs.len() != batch_inputs.len() {
        return Err(Error::Ocr(format!(
            "parseq batch returned {} results for {} crops",
            recs.len(),
            batch_inputs.len()
        )));
    }

    // Step 4: origin で per-page に振り分け、min_line_confidence で足切り。
    let mut per_page_blocks: Vec<Vec<OcrBlock>> = (0..paths.len()).map(|_| Vec::new()).collect();
    for ((page_idx, line_idx), rec) in origins.into_iter().zip(recs) {
        if rec.mean_confidence < cfg.min_line_confidence || rec.text.trim().is_empty() {
            continue;
        }
        let (x0, y0, x1, y1, _) = &per_page_crops[page_idx][line_idx];
        per_page_blocks[page_idx].push(OcrBlock {
            text: rec.text,
            bbox: [*x0 as f32, *y0 as f32, *x1 as f32, *y1 as f32],
            page: 1,
            confidence: rec.mean_confidence,
        });
    }

    // Step 5: apply_structural_rules を per-page。bbox は 0 にする
    // (旧 ocr_image 実装と同じ。下流の chunk テキスト連結で使うだけ)。
    let out: Vec<Vec<OcrBlock>> = per_page_blocks
        .into_iter()
        .map(|blocks| {
            let lines: Vec<String> = blocks.iter().map(|b| b.text.clone()).collect();
            apply_structural_rules(&lines)
                .into_iter()
                .map(|text| OcrBlock {
                    text,
                    bbox: [0.0, 0.0, 0.0, 0.0],
                    page: 1,
                    confidence: 0.0,
                })
                .collect()
        })
        .collect();
    Ok(out)
}

fn ocr_image_blocking(
    path: &str,
    cfg: &OcrConfig,
    pool: &ParseqCascadePool,
    deim: &DeimPool,
) -> Result<Vec<OcrBlock>> {
    let img = nd_io::load_rgb_u8(std::path::Path::new(path))
        .map_err(|e| Error::Ocr(format!("load: {e}")))?;
    let dets = deim
        .detect_rgb_u8(&img.data, img.width, img.height, cfg.det_conf_threshold)
        .map_err(|e| Error::Ocr(format!("deim: {e}")))?;

    // (1) 行検出を「読み順」に並べ替える。
    //     DEIM はモデル出力順 (信頼度等) で来るので、そのままだと文章として
    //     めちゃくちゃになる。横/縦/混在ページに対応した sort を
    //     ndlocr-lite-rs 側 (`pipeline::reading_order::sort_bboxes_in_reading_order`)
    //     に集約してあるのでそれを呼ぶ。
    let mut bboxes: Vec<[i32; 4]> = dets
        .into_iter()
        .filter(|d| d.class_name.starts_with("line_"))
        .filter_map(|d| {
            let x0 = d.box_xyxy[0];
            let y0 = d.box_xyxy[1];
            let x1 = d.box_xyxy[2];
            let y1 = d.box_xyxy[3];
            if x0 < 0 || y0 < 0 || x0 >= x1 || y0 >= y1 {
                return None;
            }
            if (x1 as usize) > img.width || (y1 as usize) > img.height {
                return None;
            }
            Some([x0, y0, x1, y1])
        })
        .collect();
    sort_bboxes_in_reading_order(&mut bboxes);

    // すべての行を crop してから ParseqPool で一気に推論する。
    //
    // 重要: ここでは `recognize_batch_single_session_rgb_u8` を使う。
    // 通常の `recognize_batch_rgb_u8` は pool 内部で **全 Session を並列に
    // 掴んで** チャンクを分配する設計なので、ingest 側で `buffer_unordered(N)`
    // により N 本のページが同時に `ocr_image` を呼んだ場合、各呼び出しが pool
    // を完全占有してしまい、結果的に「同時に 1 本しか parseq を流せない」
    // 状態になっていた。
    // single-session 版は 1 ページぶん全行を 1 Session でまとめて推論する
    // ので、他の Session は別ワーカーが同時に使える (= ページ間並列が崩れない)。
    let img_data: &[u8] = &img.data;
    let img_w = img.width;
    let img_h = img.height;
    let crops: Vec<(
        usize,
        usize,
        usize,
        usize,
        ndlocr_lite_rs::pipeline::crop::CroppedImage,
    )> = bboxes
        .iter()
        .map(|[x0i, y0i, x1i, y1i]| {
            let (x0, y0, x1, y1) = (*x0i as usize, *y0i as usize, *x1i as usize, *y1i as usize);
            let crop = crop_rgb_u8(img_data, img_w, img_h, BBox::new(x0, y0, x1, y1))
                .map_err(|e| Error::Ocr(format!("crop: {e}")))?;
            Ok::<_, Error>((x0, y0, x1, y1, crop))
        })
        .collect::<Result<Vec<_>>>()?;

    // cascade に渡す入力: 各行の crop に **推定文字数バケット** を添える。
    // 32-char 以下 → 30 モデル、〜45 → 50 モデル、それ以上 → 100 モデル。
    // 短い行 (見出し / 1 文 / 列番号など) ほど小さなモデルで decode できるので
    // 全体スループットが ~2-3x 上がる (典型的な日本語ページ: 80% が 30 モデル
    // で十分)。
    //
    // 推定式: 行 crop の幅 / 高さ (アスペクト比) × 2.5 を期待文字数とみなす。
    // 1 行 = 高さ ~h px、文字幅 ~h × 0.4px と見積もると ratio ≒ chars / 0.4
    // = chars × 2.5 が概ね合う。閾値 25 / 45 は ndlocr-lite-rs CLI 既定値と
    // 揃えている。
    let batch_inputs: Vec<(&[u8], usize, usize, Option<f32>)> = crops
        .iter()
        .map(|(_, _, _, _, c)| {
            let bucket = estimate_pred_char_bucket(
                c.width,
                c.height,
                DEFAULT_CASCADE_THRESHOLD_30_TO_50,
                DEFAULT_CASCADE_THRESHOLD_50_TO_100,
            );
            (c.data.as_slice(), c.width, c.height, Some(bucket))
        })
        .collect();
    let recs = pool
        .recognize_batch_with_buckets_rgb_u8(&batch_inputs)
        .map_err(|e| Error::Ocr(format!("parseq batch: {e}")))?;
    if recs.len() != crops.len() {
        return Err(Error::Ocr(format!(
            "parseq batch returned {} results for {} crops",
            recs.len(),
            crops.len()
        )));
    }

    let mut out: Vec<OcrBlock> = Vec::with_capacity(recs.len());
    for ((x0, y0, x1, y1, _), rec) in crops.into_iter().zip(recs) {
        if rec.mean_confidence < cfg.min_line_confidence || rec.text.trim().is_empty() {
            continue;
        }
        out.push(OcrBlock {
            text: rec.text,
            bbox: [x0 as f32, y0 as f32, x1 as f32, y1 as f32],
            page: 1,
            confidence: rec.mean_confidence,
        });
    }

    // 後処理:
    //   - 隣接重複行の削除 (見出しがマージン+本文で 2 重抽出される等)
    //   - ページ装飾 (`-`, `1,` 等) の除去
    //   - 条文番号の正規化、短い見出しの整形、ノイズ除去
    //   - apply_structural_rules は ndlocr-lite-rs の OCR パイプライン本流と
    //     同じものを使う。Decoration 除去 / dedup も内部で行う。
    let lines: Vec<String> = out.iter().map(|b| b.text.clone()).collect();
    let cleaned = apply_structural_rules(&lines);

    // 後処理で行数が変わるので、bbox は失われる (今のところ下流で
    // chunk テキスト連結にしか使っていないので問題なし)。
    let cleaned_blocks: Vec<OcrBlock> = cleaned
        .into_iter()
        .map(|text| OcrBlock {
            text,
            bbox: [0.0, 0.0, 0.0, 0.0],
            page: 1,
            confidence: 0.0,
        })
        .collect();
    Ok(cleaned_blocks)
}
