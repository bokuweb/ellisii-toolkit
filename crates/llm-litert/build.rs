//! `litert` feature 有効 *かつ* CLiteRTLM 共有ライブラリの在処が分かるときだけ
//! 実体をリンクし、`litert_linked` cfg を立てる。
//!
//! 公式 prebuilt の dylib はファイル名に `lib` prefix が無く (`CLiteRTLM_mac.dylib`)
//! 通常の `-l` 解決に乗らないため、フルパスをそのままリンカ引数として渡す。
//! ライブラリの在処は `LITERT_LM_LIB_DIR` で指定する。
//!
//! feature が有効でもライブラリが見つからない場合 (CI の `--all-features` 等) は
//! panic せず警告のみ出し、crate は no-op スタブにフォールバックする。

fn main() {
    // 独自 cfg を常に宣言しておく (unexpected_cfgs lint 回避)。
    println!("cargo:rustc-check-cfg=cfg(litert_linked)");
    println!("cargo:rerun-if-env-changed=LITERT_LM_LIB_DIR");
    println!("cargo:rerun-if-env-changed=LITERT_LM_LIB_NAME");

    if std::env::var_os("CARGO_FEATURE_LITERT").is_none() {
        return;
    }

    let Ok(dir) = std::env::var("LITERT_LM_LIB_DIR") else {
        println!(
            "cargo:warning=`litert` feature is enabled but LITERT_LM_LIB_DIR is not set; \
             falling back to the no-op stub. Set it to the directory containing the \
             CLiteRTLM dylib to enable real inference."
        );
        return;
    };

    // 既定は macOS prebuilt の名前。Linux/Windows ビルドでは LITERT_LM_LIB_NAME で上書き。
    let lib_name =
        std::env::var("LITERT_LM_LIB_NAME").unwrap_or_else(|_| "CLiteRTLM_mac.dylib".to_string());
    let lib_path = std::path::Path::new(&dir).join(&lib_name);

    if !lib_path.exists() {
        println!(
            "cargo:warning=CLiteRTLM library not found at {}; falling back to the no-op stub.",
            lib_path.display()
        );
        return;
    }

    // dylib をフルパスでリンク + 実行時に同ディレクトリを rpath に追加。
    println!("cargo:rustc-link-arg={}", lib_path.display());
    println!("cargo:rustc-link-arg=-Wl,-rpath,{dir}");
    println!("cargo:rustc-cfg=litert_linked");
    println!("cargo:rerun-if-changed={}", lib_path.display());
}
