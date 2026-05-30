//! `litert` feature 有効 *かつ* CLiteRTLM 共有ライブラリの在処が分かるときだけ
//! 実体をリンクし、`litert_linked` cfg を立てる。
//!
//! ライブラリの解決順:
//!   1. `LITERT_LM_LIB_DIR` が指すディレクトリ (手動上書き、最優先)。
//!   2. `LITERT_LM_PREBUILT` が truthy なら、ホスト platform 向け prebuilt を
//!      ellisii-toolkit の GitHub Release から自動 DL + SHA-256 検証してリンク。
//!   3. いずれも無ければ警告のみ出して no-op スタブにフォールバック。
//!
//! 公式 prebuilt の mac dylib はファイル名に `lib` prefix が無く通常の `-l` 解決に
//! 乗らないため、unix ではフルパスをそのままリンカ引数として渡す。Windows は
//! import lib (`CLiteRTLM.lib`) を `-l` 解決し、`.dll` を実行時に解決させる。
//!
//! 自動 DL は「明示的な opt-in (env)」のときだけ走る。これにより `--all-features`
//! CI 等はネットワークに触れず (hermetic)、従来どおりスタブにフォールバックする。
//! DL/検証に失敗しても panic せず警告 + スタブに退避する。

use std::path::{Path, PathBuf};
use std::process::Command;

/// Release アセットを取りに行く既定の base URL (`/{tag}/{asset}` を後置)。
const DEFAULT_BASE_URL: &str = "https://github.com/bokuweb/ellisii-toolkit/releases/download";
/// 既定の prebuilt Release タグ。LiteRT-LM 本体の version に追従させる。
const DEFAULT_PREBUILT_TAG: &str = "litert-prebuilt-v0.12.0";

fn main() {
    // 独自 cfg を常に宣言しておく (unexpected_cfgs lint 回避)。
    println!("cargo:rustc-check-cfg=cfg(litert_linked)");
    for var in [
        "LITERT_LM_LIB_DIR",
        "LITERT_LM_LIB_NAME",
        "LITERT_LM_PREBUILT",
        "LITERT_LM_PREBUILT_TAG",
        "LITERT_LM_PREBUILT_BASE_URL",
    ] {
        println!("cargo:rerun-if-env-changed={var}");
    }

    if std::env::var_os("CARGO_FEATURE_LITERT").is_none() {
        return;
    }

    // 1. 手動上書き: ディレクトリ指定があればそれを最優先。
    if let Ok(dir) = std::env::var("LITERT_LM_LIB_DIR") {
        let lib_name = std::env::var("LITERT_LM_LIB_NAME")
            .unwrap_or_else(|_| default_lib_name(&host_target()));
        let lib_path = Path::new(&dir).join(&lib_name);
        if lib_path.exists() {
            emit_link(&lib_path);
            return;
        }
        println!(
            "cargo:warning=CLiteRTLM library not found at {}; falling back to the no-op stub.",
            lib_path.display()
        );
        return;
    }

    // 2. prebuilt 自動 DL (明示 opt-in のときだけ)。
    if prebuilt_opt_in() {
        match download_prebuilt(&host_target()) {
            Ok(lib_path) => {
                emit_link(&lib_path);
                return;
            }
            Err(e) => {
                println!(
                    "cargo:warning=failed to fetch CLiteRTLM prebuilt ({e}); \
                     falling back to the no-op stub. Set LITERT_LM_LIB_DIR to a locally \
                     built dylib to override."
                );
                return;
            }
        }
    }

    // 3. スタブ。
    println!(
        "cargo:warning=`litert` feature is enabled but neither LITERT_LM_LIB_DIR nor \
         LITERT_LM_PREBUILT is set; falling back to the no-op stub. Set LITERT_LM_PREBUILT=1 \
         to auto-download the host prebuilt, or LITERT_LM_LIB_DIR to point at a local dylib."
    );
}

/// `LITERT_LM_PREBUILT` が truthy、または `LITERT_LM_PREBUILT_TAG` 明示で opt-in。
fn prebuilt_opt_in() -> bool {
    if std::env::var_os("LITERT_LM_PREBUILT_TAG").is_some() {
        return true;
    }
    match std::env::var("LITERT_LM_PREBUILT") {
        Ok(v) => {
            let v = v.trim().to_ascii_lowercase();
            !matches!(v.as_str(), "" | "0" | "false" | "no" | "off")
        }
        Err(_) => false,
    }
}

/// build.rs に渡る `CARGO_CFG_TARGET_OS` / `CARGO_CFG_TARGET_ARCH` からホスト triple を読む。
struct HostTarget {
    os: String,
    arch: String,
}

fn host_target() -> HostTarget {
    HostTarget {
        os: std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default(),
        arch: std::env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default(),
    }
}

/// `LITERT_LM_LIB_DIR` 利用時の既定ライブラリ名 (ホスト platform 依存)。
fn default_lib_name(t: &HostTarget) -> String {
    match t.os.as_str() {
        "windows" => "CLiteRTLM.dll".to_string(),
        "macos" => "CLiteRTLM_mac.dylib".to_string(),
        _ => "libCLiteRTLM.so".to_string(),
    }
}

/// ホスト向けに DL すべき (Release アセット名, キャッシュ保存名) の並び。
///
/// アセット名は platform/arch を含む人間可読名 (Stage 2 CI のアップロード名と一致)。
/// 保存名は **ローダが参照する名前 (install_name / SONAME / DLL 名)** に揃える:
///   - macOS dylib の install_name は `@rpath/CLiteRTLM_mac.dylib` なので保存名も同じ。
///   - linux .so の SONAME は `libCLiteRTLM.so`。
///   - windows import lib は `CLiteRTLM.dll` を名指しするので両者をその名前で保存。
///
/// 並びの **末尾がリンク対象** (unix: 共有ライブラリ / windows: import lib)。
fn asset_plan(t: &HostTarget) -> Result<Vec<(String, String)>, String> {
    let plan = match (t.os.as_str(), t.arch.as_str()) {
        ("macos", _) => vec![(
            "CLiteRTLM-macos-universal.dylib".to_string(),
            "CLiteRTLM_mac.dylib".to_string(),
        )],
        ("linux", "x86_64") => vec![(
            "libCLiteRTLM-linux-x86_64.so".to_string(),
            "libCLiteRTLM.so".to_string(),
        )],
        ("linux", "aarch64") => vec![(
            "libCLiteRTLM-linux-aarch64.so".to_string(),
            "libCLiteRTLM.so".to_string(),
        )],
        ("windows", "x86_64") => vec![
            (
                "CLiteRTLM-windows-x86_64.dll".to_string(),
                "CLiteRTLM.dll".to_string(),
            ),
            (
                "CLiteRTLM-windows-x86_64.lib".to_string(),
                "CLiteRTLM.lib".to_string(),
            ),
        ],
        (os, arch) => {
            return Err(format!(
                "no CLiteRTLM prebuilt available for {os}/{arch}; \
                 build from source and set LITERT_LM_LIB_DIR"
            ))
        }
    };
    Ok(plan)
}

/// prebuilt を DL + 検証し、リンク対象 (unix: 共有ライブラリ / windows: import lib)
/// のフルパスを返す。キャッシュ済みなら DL をスキップする。
fn download_prebuilt(t: &HostTarget) -> Result<PathBuf, String> {
    let plan = asset_plan(t)?;
    let base_url =
        std::env::var("LITERT_LM_PREBUILT_BASE_URL").unwrap_or_else(|_| DEFAULT_BASE_URL.into());
    let tag =
        std::env::var("LITERT_LM_PREBUILT_TAG").unwrap_or_else(|_| DEFAULT_PREBUILT_TAG.into());
    let out_dir = PathBuf::from(std::env::var("OUT_DIR").map_err(|_| "OUT_DIR unset".to_string())?);
    let cache = out_dir.join("litert-prebuilt").join(&tag);
    std::fs::create_dir_all(&cache).map_err(|e| format!("mkdir cache: {e}"))?;

    let mut link_target = cache.clone();
    for (asset, cached) in &plan {
        let dest = cache.join(cached);
        fetch_verified(&base_url, &tag, asset, &dest)?;
        link_target = dest;
    }
    Ok(link_target)
}

/// `{base}/{tag}/{asset}` を DL し、隣の `.sha256` で検証して `dest` に確定保存する。
/// `dest` が既に存在し sha256 が一致するなら DL をスキップ。
fn fetch_verified(base: &str, tag: &str, asset: &str, dest: &Path) -> Result<(), String> {
    let url = format!("{base}/{tag}/{asset}");
    let sha_url = format!("{url}.sha256");
    let expected = curl_text(&sha_url)
        .map_err(|e| format!("download {sha_url}: {e}"))?
        .split_whitespace()
        .next()
        .map(|s| s.to_ascii_lowercase())
        .ok_or_else(|| format!("empty checksum file {sha_url}"))?;

    if dest.exists() {
        if let Ok(actual) = sha256_hex(dest) {
            if actual == expected {
                println!("cargo:rerun-if-changed={}", dest.display());
                return Ok(());
            }
        }
        let _ = std::fs::remove_file(dest);
    }

    // 一旦 tmp に落としてから検証 → atomic rename で確定。
    let tmp = dest.with_extension("download.tmp");
    curl_to_file(&url, &tmp).map_err(|e| format!("download {url}: {e}"))?;
    let actual = sha256_hex(&tmp)?;
    if actual != expected {
        let _ = std::fs::remove_file(&tmp);
        return Err(format!(
            "checksum mismatch for {asset}: expected {expected}, got {actual}"
        ));
    }
    std::fs::rename(&tmp, dest).map_err(|e| format!("finalize {}: {e}", dest.display()))?;
    println!("cargo:rerun-if-changed={}", dest.display());
    Ok(())
}

/// curl で URL の本文を文字列として取得 (.sha256 用)。
fn curl_text(url: &str) -> Result<String, String> {
    let out = Command::new("curl")
        .args(["-fsSL", url])
        .output()
        .map_err(|e| format!("spawn curl: {e}"))?;
    if !out.status.success() {
        return Err(format!("curl exited {}", out.status));
    }
    String::from_utf8(out.stdout).map_err(|e| format!("non-utf8 body: {e}"))
}

/// curl で URL を `dest` に保存。
fn curl_to_file(url: &str, dest: &Path) -> Result<(), String> {
    let status = Command::new("curl")
        .args(["-fsSL", "-o"])
        .arg(dest)
        .arg(url)
        .status()
        .map_err(|e| format!("spawn curl: {e}"))?;
    if !status.success() {
        return Err(format!("curl exited {status}"));
    }
    Ok(())
}

/// ファイルの SHA-256 を小文字 hex で返す。
fn sha256_hex(path: &Path) -> Result<String, String> {
    use sha2::{Digest, Sha256};
    let bytes = std::fs::read(path).map_err(|e| format!("read {}: {e}", path.display()))?;
    let digest = Sha256::digest(&bytes);
    Ok(digest.iter().map(|b| format!("{b:02x}")).collect())
}

/// リンク対象を rustc に伝える。unix はフルパス + rpath、windows は import lib。
fn emit_link(lib_path: &Path) {
    let dir = lib_path
        .parent()
        .map(|p| p.to_path_buf())
        .unwrap_or_default();

    let is_windows = std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("windows");
    if is_windows {
        // import lib (`CLiteRTLM.lib`) を `-l` 解決。`.dll` は同ディレクトリにある前提で
        // 実行時 PATH に乗せる必要がある (consumer 側 or 下の copy で面倒を見る)。
        println!("cargo:rustc-link-search=native={}", dir.display());
        println!("cargo:rustc-link-lib=dylib=CLiteRTLM");
        copy_dll_next_to_binary(&dir);
    } else {
        // dylib/.so をフルパスでリンク + 実行時に同ディレクトリを rpath に追加。
        println!("cargo:rustc-link-arg={}", lib_path.display());
        println!("cargo:rustc-link-arg=-Wl,-rpath,{}", dir.display());
    }
    println!("cargo:rustc-cfg=litert_linked");
    println!("cargo:rerun-if-changed={}", lib_path.display());
}

/// Windows: `CLiteRTLM.dll` を出力バイナリと同じ profile ディレクトリにコピーして
/// 実行時に解決できるようにする (best-effort)。OUT_DIR から profile dir を逆算する。
fn copy_dll_next_to_binary(cache_dir: &Path) {
    let dll = cache_dir.join("CLiteRTLM.dll");
    if !dll.exists() {
        return;
    }
    // OUT_DIR = target/<profile>/build/<pkg>-<hash>/out → profile dir は 3 つ上。
    if let Ok(out_dir) = std::env::var("OUT_DIR") {
        let profile_dir = Path::new(&out_dir)
            .ancestors()
            .nth(3)
            .map(Path::to_path_buf);
        if let Some(dest_dir) = profile_dir {
            let dest = dest_dir.join("CLiteRTLM.dll");
            let _ = std::fs::copy(&dll, &dest);
        }
    }
}
