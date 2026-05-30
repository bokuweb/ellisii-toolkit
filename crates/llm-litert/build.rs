//! `litert` feature 有効時に CLiteRTLM 共有ライブラリをリンクする。
//!
//! 公式 prebuilt の dylib はファイル名に `lib` prefix が無く (`CLiteRTLM_mac.dylib`)
//! 通常の `-l` 解決に乗らないため、フルパスをそのままリンカ引数として渡す。
//! ライブラリの在処は `LITERT_LM_LIB_DIR` で指定する。

fn main() {
    println!("cargo:rerun-if-env-changed=LITERT_LM_LIB_DIR");
    println!("cargo:rerun-if-env-changed=LITERT_LM_LIB_NAME");

    if std::env::var_os("CARGO_FEATURE_LITERT").is_none() {
        return;
    }

    let dir = match std::env::var("LITERT_LM_LIB_DIR") {
        Ok(d) => d,
        Err(_) => panic!(
            "`litert` feature is enabled but LITERT_LM_LIB_DIR is not set.\n\
             Point it at the directory containing the CLiteRTLM dylib, e.g.\n\
             export LITERT_LM_LIB_DIR=/path/to/CLiteRTLM_mac.xcframework/macos-arm64_x86_64"
        ),
    };

    // 既定は macOS prebuilt の名前。Linux/Windows ビルドでは LITERT_LM_LIB_NAME で上書き。
    let lib_name =
        std::env::var("LITERT_LM_LIB_NAME").unwrap_or_else(|_| "CLiteRTLM_mac.dylib".to_string());
    let lib_path = std::path::Path::new(&dir).join(&lib_name);

    assert!(
        lib_path.exists(),
        "CLiteRTLM library not found at {}",
        lib_path.display()
    );

    // dylib をフルパスでリンク + 実行時に同ディレクトリを rpath に追加。
    println!("cargo:rustc-link-arg={}", lib_path.display());
    println!("cargo:rustc-link-arg=-Wl,-rpath,{dir}");
    println!("cargo:rerun-if-changed={}", lib_path.display());
}
