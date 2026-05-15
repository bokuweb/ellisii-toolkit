//! `render_pdf` — small CLI to dump N pages of a PDF as PNGs.
//!
//! Used by `scripts/bench-ocr-configs.sh` to materialize fixture images for
//! the OCR criterion bench without needing a manual ingest run.
//!
//! Usage:
//!     cargo run --release -p ellisii-parser-pdf --example render_pdf -- \
//!         --pdf /path/to/scan.pdf --pages 4 --out /tmp/bench-pngs
//!
//! Args:
//!     --pdf   path to source PDF (required)
//!     --pages number of pages to render starting from `--start` (default 4)
//!     --start 1-indexed page to start from (default 1)
//!     --out   destination directory (created if missing). PNGs written as
//!             `page-<N>.png`.
//!     --dpi   target longest-side pixels (default 1500, matches ingest)
//!
//! 早期 panic & プロセス exit code が一義的なので、bench script から
//! `set -e` で安全に呼べる。

use ellisii_ocr::PdfRasterizer;
use ellisii_parser_pdf::PdfiumRasterizer;
use std::path::PathBuf;
use std::process::ExitCode;

#[derive(Debug)]
struct Args {
    pdf: PathBuf,
    pages: u32,
    start: u32,
    out: PathBuf,
    dpi: u32,
}

fn parse_args() -> Result<Args, String> {
    let mut pdf: Option<PathBuf> = None;
    let mut pages: u32 = 4;
    let mut start: u32 = 1;
    let mut out: Option<PathBuf> = None;
    let mut dpi: u32 = 1500;
    let mut iter = std::env::args().skip(1);
    while let Some(a) = iter.next() {
        match a.as_str() {
            "--pdf" => pdf = Some(PathBuf::from(iter.next().ok_or("--pdf needs a value")?)),
            "--pages" => {
                pages = iter
                    .next()
                    .ok_or("--pages needs a value")?
                    .parse()
                    .map_err(|e| format!("--pages: {e}"))?
            }
            "--start" => {
                start = iter
                    .next()
                    .ok_or("--start needs a value")?
                    .parse()
                    .map_err(|e| format!("--start: {e}"))?
            }
            "--out" => out = Some(PathBuf::from(iter.next().ok_or("--out needs a value")?)),
            "--dpi" => {
                dpi = iter
                    .next()
                    .ok_or("--dpi needs a value")?
                    .parse()
                    .map_err(|e| format!("--dpi: {e}"))?
            }
            other => return Err(format!("unknown arg: {other}")),
        }
    }
    let pdf = pdf.ok_or("--pdf is required")?;
    let out = out.ok_or("--out is required")?;
    if pages == 0 || start == 0 {
        return Err("--pages and --start must be >= 1".into());
    }
    Ok(Args {
        pdf,
        pages,
        start,
        out,
        dpi,
    })
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> ExitCode {
    let args = match parse_args() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("error: {e}");
            eprintln!();
            eprintln!(
                "usage: render_pdf --pdf <path> --pages <N> --out <dir> [--start 1] [--dpi 1500]"
            );
            return ExitCode::from(2);
        }
    };
    if let Err(e) = std::fs::create_dir_all(&args.out) {
        eprintln!("create_dir_all {}: {e}", args.out.display());
        return ExitCode::from(1);
    }

    let rast = PdfiumRasterizer {
        target_longest_side_px: args.dpi,
    };
    let pdf_str = args.pdf.to_string_lossy().to_string();
    // 先に総ページ数を確認して範囲外 page を弾く (rasterize_pages は warn で
    // skip するが、CLI として明示的なエラーにしておくと bench script の
    // failure mode がはっきりする)。
    let total = match rast.page_count(&pdf_str).await {
        Ok(t) => t,
        Err(e) => {
            eprintln!("page_count {pdf_str}: {e}");
            return ExitCode::from(1);
        }
    };
    if args.start > total {
        eprintln!("--start {} > total pages {total}", args.start);
        return ExitCode::from(1);
    }
    let last = (args.start + args.pages - 1).min(total);
    let pages: Vec<u32> = (args.start..=last).collect();
    eprintln!(
        "render_pdf: {} → {} (pages {}..={}, dpi={})",
        args.pdf.display(),
        args.out.display(),
        args.start,
        last,
        args.dpi
    );

    let rendered = match rast.rasterize_pages(&pdf_str, &pages).await {
        Ok(r) => r,
        Err(e) => {
            eprintln!("rasterize_pages: {e}");
            return ExitCode::from(1);
        }
    };

    // RasterizedPage の `path` は tempdir 内に書かれている。`copy` で出力先に
    // 移して、tempdir の `_keepalive` が drop されたあとも残るようにする。
    for rp in &rendered {
        let dest = args.out.join(format!("page-{}.png", rp.page));
        if let Err(e) = std::fs::copy(&rp.path, &dest) {
            eprintln!("copy {} -> {}: {e}", rp.path.display(), dest.display());
            return ExitCode::from(1);
        }
        eprintln!("  wrote {}", dest.display());
    }
    eprintln!("render_pdf: done ({} pages)", rendered.len());
    ExitCode::SUCCESS
}
