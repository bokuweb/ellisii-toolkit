// vendor/pdfium/<triple>/lib/libpdfium.{dylib,so,dll} を
// target/<profile>/ (= 実行ファイルと同じディレクトリ) にコピーする。
//
// 目的:
//   - pdfium-auto のランタイム DL を回避し、決定論的な pdfium をビルド時に固定
//   - macOS の library validation 周りの不安定要素を消す (= host と同じ
//     ディレクトリ・同じ codesign 操作で扱える状態にする)
//   - オフライン環境でも初回起動から動く
//
// vendor が無い場合 (= `scripts/fetch-pdfium.sh` 未実行 / CI 抜け道) は
// 何もしない。ランタイムは pdfium-auto に fallback する。

use std::path::PathBuf;

fn main() {
    let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    let target_arch = std::env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();

    let (arch_tag, lib_rel, lib_name) = match (target_os.as_str(), target_arch.as_str()) {
        ("macos", "aarch64") => ("mac-arm64", "lib/libpdfium.dylib", "libpdfium.dylib"),
        ("macos", "x86_64") => ("mac-x64", "lib/libpdfium.dylib", "libpdfium.dylib"),
        ("linux", "x86_64") => ("linux-x64", "lib/libpdfium.so", "libpdfium.so"),
        ("linux", "aarch64") => ("linux-arm64", "lib/libpdfium.so", "libpdfium.so"),
        ("windows", "x86_64") => ("win-x64", "bin/pdfium.dll", "pdfium.dll"),
        _ => {
            // 未知のターゲットは fetch スクリプトも当てられないので何もしない。
            return;
        }
    };

    // workspace root を CARGO_MANIFEST_DIR から推定 (crates/parser-pdf からの ../..)
    let manifest_dir = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    let workspace_root = manifest_dir
        .ancestors()
        .nth(2)
        .expect("crates/parser-pdf is two levels under workspace")
        .to_path_buf();

    // vendor は version pin を含むディレクトリ名なので glob で見る
    let vendor_root = workspace_root.join("vendor").join("pdfium");
    let mut vendor_lib: Option<PathBuf> = None;
    if let Ok(entries) = std::fs::read_dir(&vendor_root) {
        for ent in entries.flatten() {
            let name = ent.file_name();
            let name_str = name.to_string_lossy();
            if name_str.starts_with(&format!("{arch_tag}-")) {
                let p = ent.path().join(lib_rel);
                if p.exists() {
                    vendor_lib = Some(p);
                    break;
                }
            }
        }
    }

    let Some(src) = vendor_lib else {
        println!(
            "cargo:warning=vendor pdfium not found under {}; runtime will fall back to pdfium-auto. \
             run scripts/fetch-pdfium.sh to vendor it.",
            vendor_root.display()
        );
        return;
    };

    println!("cargo:rerun-if-changed={}", src.display());

    // OUT_DIR は `target/<profile>/build/<crate>-<hash>/out`。
    // ここから `../../..` で `target/<profile>/` (= 実行ファイル directory) に到達する。
    // この crate を含むビルドだけを対象にしているので、他クレート向けの
    // OUT_DIR を汚すことはない。
    let out_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    let target_profile_dir = out_dir
        .ancestors()
        .nth(3)
        .expect("OUT_DIR has at least 3 ancestors above target/<profile>/")
        .to_path_buf();
    let dst = target_profile_dir.join(lib_name);

    // ファイルが既に存在し、source と mtime が同じならスキップ (= cargo の再ビルドで
    // 毎回 I/O しない)。
    let needs_copy = match (std::fs::metadata(&src), std::fs::metadata(&dst)) {
        (Ok(s), Ok(d)) => s.modified().ok() != d.modified().ok() || s.len() != d.len(),
        _ => true,
    };
    if needs_copy {
        if let Err(e) = std::fs::copy(&src, &dst) {
            println!(
                "cargo:warning=failed to copy {} -> {}: {}; pdfium-auto fallback will be used at runtime",
                src.display(),
                dst.display(),
                e
            );
            return;
        }
    }

    // 実行時に "exe と同じ dir" を指して bind するためのヒントとして env も baked。
    // (binary の位置で resolve する場合は実行時 logic を使うので必須ではないが、
    //  PDFIUM_LIB_PATH を set すると pdfium-auto の fast-path にも乗る。)
    println!("cargo:rustc-env=ELLISII_VENDORED_PDFIUM_NAME={lib_name}");
}
