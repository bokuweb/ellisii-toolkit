//! OCR throughput benchmark.
//!
//! 実モデル + 実 PNG 入力で `NdlocrBackend` のスループットを測る。`cargo bench`
//! は **依存していない fixture (model + image) が無い場合は早期 return** する
//! ので、unconditional に走らせて事故が起きない。
//!
//! 計測対象:
//!   - `ocr_image` を 1 ページずつ N 並列で呼ぶ (= 本番の page-parallel 経路)
//!   - `ocr_images_batch(paths)` を batch_size = N で呼ぶ (cross-page batched
//!     経路。CPU では現在 ingest からは使われていないが、将来 GPU/ANE 復活時の
//!     ベースラインとして残す)
//!
//! 環境変数:
//!   - `ELLISII_BENCH_MODEL_DIR`  ndlocr モデル一式が置かれたディレクトリ
//!     (既定: `$HOME/Library/Application Support/ellisii/models/ndlocr/` —
//!     アプリの初回起動で自動配置される場所と同じ)
//!   - `ELLISII_BENCH_IMAGES`     bench に使う PNG ディレクトリ。中身の `.png`
//!     ファイルを順に使う。少なくとも 4 枚あると batch_size=4 まで測れる。
//!     (簡単な作り方: ellisii ingest を一度走らせると `/var/folders/.../page-N.png`
//!      に rasterize 済みの PNG が一時的に出るので、それを別の場所にコピーして
//!      指定する。)
//!   - `NDLOCR_INTRA_THREADS`     ndlocr-lite-rs 側 intra_threads 上書き
//!     (既定 `cpus / pool` クランプ 1..=8、ellisii の起動時設定と同じ)
//!
//! `scripts/bench-ocr-configs.sh` から各設定を env で振り分けて再実行する想定。

#![cfg(feature = "onnx")]

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use ellisii_ocr::{NdlocrBackend, OcrBackend, OcrConfig};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

fn default_model_dir() -> PathBuf {
    if let Ok(d) = std::env::var("ELLISII_BENCH_MODEL_DIR") {
        return PathBuf::from(d);
    }
    let home = std::env::var("HOME").unwrap_or_default();
    PathBuf::from(home).join("Library/Application Support/ellisii/models/ndlocr")
}

/// 実モデル一式が揃っていれば `OcrConfig` を返す。1 ファイルでも欠けていたら
/// `None` (= bench を skip する)。
fn try_build_config(dir: &Path) -> Option<OcrConfig> {
    let det = dir.join("deim-s-1024x1024.onnx");
    let m100 = dir.join("parseq-ndl-24x768-100-tiny-153epoch-tegaki3-r8data-202604.onnx");
    let m50 = dir.join("parseq-ndl-24x384-50-tiny-300epoch-tegaki3-r8data-202604.onnx");
    let m30 = dir.join("parseq-ndl-24x256-30-tiny-189epoch-tegaki3-r8data-202604.onnx");
    let charset = dir.join("NDLmoji.yaml");
    for p in [&det, &m100, &m50, &m30, &charset] {
        if !p.is_file() {
            return None;
        }
    }
    Some(OcrConfig {
        det_model: det,
        model100: m100,
        model50: m50,
        model30: m30,
        charset,
        det_conf_threshold: 0.3,
        min_line_confidence: 0.3,
    })
}

fn collect_pngs(dir: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    if let Ok(rd) = std::fs::read_dir(dir) {
        for entry in rd.flatten() {
            let p = entry.path();
            if p.extension().and_then(|e| e.to_str()) == Some("png") {
                out.push(p);
            }
        }
    }
    out.sort();
    out
}

/// criterion bench main. fixture がそろっていない場合は warning して退場 (=
/// CI で skip 扱い)。
fn ocr_benches(c: &mut Criterion) {
    let model_dir = default_model_dir();
    let cfg = match try_build_config(&model_dir) {
        Some(c) => c,
        None => {
            eprintln!(
                "[ocr_bench] ndlocr models not found under {}; skipping. \
                 Set ELLISII_BENCH_MODEL_DIR to override.",
                model_dir.display()
            );
            return;
        }
    };
    let images_dir = match std::env::var("ELLISII_BENCH_IMAGES") {
        Ok(d) => PathBuf::from(d),
        Err(_) => {
            eprintln!(
                "[ocr_bench] ELLISII_BENCH_IMAGES not set; skipping. \
                 Point it at a directory of rasterized .png files."
            );
            return;
        }
    };
    let pngs = collect_pngs(&images_dir);
    if pngs.is_empty() {
        eprintln!(
            "[ocr_bench] no .png files in {}; skipping.",
            images_dir.display()
        );
        return;
    }
    eprintln!(
        "[ocr_bench] using {} PNG fixture(s) from {}",
        pngs.len(),
        images_dir.display()
    );

    // tokio runtime + backend は全 bench で共有 (cold start を 1 度に押し込む)。
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .worker_threads(4)
        .build()
        .expect("tokio runtime");
    let backend: Arc<NdlocrBackend> = Arc::new(NdlocrBackend::new(cfg));
    // 1 度先に呼んで pool init / model load を済ませてから時計を回す
    // (criterion の warmup でも吸収されるが、明示しておくと出力ログが
    // 読みやすい)。
    if let Some(first) = pngs.first() {
        let _ = rt.block_on(backend.ocr_image(first.to_string_lossy().as_ref()));
    }

    // bench 1: ocr_image を batch_size = 1, 2, 4 で並列に呼ぶ。
    // ingest 本番経路 (`buffer_unordered(N)`) と同じ振る舞いを criterion 上で
    // 再現するため、tokio JoinSet で N futures を spawn して全部 await する。
    let mut g = c.benchmark_group("ocr_image_concurrent");
    // OCR は per-iter 数秒かかる重い計測なので、sample 数を default 100 から
    // 減らし、warmup / measurement_time も短めに固定する (criterion の
    // 「100 samples 収集できない」warning が出続けるのを防ぐ)。
    g.sample_size(10)
        .warm_up_time(Duration::from_secs(3))
        .measurement_time(Duration::from_secs(20));
    for &n in &[1usize, 2, 3, 4] {
        if pngs.len() < n {
            continue;
        }
        g.bench_function(BenchmarkId::from_parameter(n), |b| {
            let pngs = pngs.clone();
            let backend = backend.clone();
            b.iter(|| {
                rt.block_on(async {
                    let mut joins = Vec::with_capacity(n);
                    for p in pngs.iter().take(n) {
                        let backend = backend.clone();
                        let p = p.to_string_lossy().to_string();
                        joins.push(tokio::spawn(async move { backend.ocr_image(&p).await }));
                    }
                    for j in joins {
                        let _ = j.await;
                    }
                })
            });
        });
    }
    g.finish();

    // bench 2: ocr_images_batch を batch_size = 1, 2, 4 で 1 度ずつ呼ぶ。
    // 同じ N ページ処理だが API は別 (parseq cascade を 1 度の呼び出しで束ねる)。
    let mut g = c.benchmark_group("ocr_images_batch");
    g.sample_size(10)
        .warm_up_time(Duration::from_secs(3))
        .measurement_time(Duration::from_secs(20));
    for &n in &[1usize, 2, 3, 4] {
        if pngs.len() < n {
            continue;
        }
        g.bench_function(BenchmarkId::from_parameter(n), |b| {
            let paths: Vec<String> = pngs
                .iter()
                .take(n)
                .map(|p| p.to_string_lossy().to_string())
                .collect();
            let backend = backend.clone();
            b.iter(|| {
                rt.block_on(async {
                    let refs: Vec<&str> = paths.iter().map(|s| s.as_str()).collect();
                    let _ = backend.ocr_images_batch(&refs).await;
                })
            });
        });
    }
    g.finish();
}

criterion_group!(benches, ocr_benches);
criterion_main!(benches);
