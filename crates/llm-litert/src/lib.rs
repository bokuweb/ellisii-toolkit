//! LiteRT-LM (Google AI Edge) を使った `.litertlm` / `.task` 推論バックエンド。
//!
//! `litert` feature 有効時のみ実体が組み込まれる。デフォルトでは空 (Err) を返す。
//! Gemma 4 (E2B / E4B) を Google 公式ランタイムで動かす経路。

use ellisii_llm_core::ModelFamily;

// 実体は `litert` feature 有効 *かつ* build.rs が dylib を見つけたとき (litert_linked) のみ。
// feature だけ有効でライブラリが無い環境 (CI の --all-features 等) はスタブにフォールバック。
#[cfg(litert_linked)]
mod ffi;

/// LiteRT-LM バックエンドの構成。
#[derive(Debug, Clone)]
pub struct LiteRtConfig {
    pub model_path: std::path::PathBuf,
    pub family: ModelFamily,
    /// 推論バックエンド文字列 (`"cpu"` / `"gpu"`)。
    pub backend: String,
    /// 重み prefetch 等のキャッシュディレクトリ。`None` で無効。
    pub cache_dir: Option<std::path::PathBuf>,
}

impl LiteRtConfig {
    /// モデルパスと family から既定構成を作る。
    /// `ELLISII_LITERT_BACKEND` で `cpu` / `gpu` を上書きできる (既定 `cpu`)。
    pub fn new(model_path: impl Into<std::path::PathBuf>, family: ModelFamily) -> Self {
        let backend = std::env::var("ELLISII_LITERT_BACKEND").unwrap_or_else(|_| "cpu".to_string());
        Self {
            model_path: model_path.into(),
            family,
            backend,
            cache_dir: None,
        }
    }
}

#[cfg(litert_linked)]
mod backend {
    use super::*;
    use async_trait::async_trait;
    use ellisii_core::{Error, Result};
    use ellisii_llm_core::{LlmBackend, LlmRequest};
    use std::ffi::{CStr, CString};
    use std::os::raw::{c_char, c_void};

    /// engine ポインタは LiteRT-LM 側で共有可能。session 生成〜消費を Mutex で
    /// 直列化して扱うため Send + Sync を unsafe に付与する。
    struct EngineHandle(*mut ffi::LiteRtLmEngine);
    unsafe impl Send for EngineHandle {}
    unsafe impl Sync for EngineHandle {}

    impl Drop for EngineHandle {
        fn drop(&mut self) {
            if !self.0.is_null() {
                unsafe { ffi::litert_lm_engine_delete(self.0) };
            }
        }
    }

    pub struct LiteRtBackend {
        engine: std::sync::Arc<std::sync::Mutex<EngineHandle>>,
        family: ModelFamily,
    }

    impl LiteRtBackend {
        /// モデルを読み込み engine を構築する。失敗時は `Error::Llm`。
        pub fn load(cfg: LiteRtConfig) -> Result<Self> {
            let model_path = cfg
                .model_path
                .to_str()
                .ok_or_else(|| Error::Llm("model path is not valid UTF-8".into()))?;
            let model_c = CString::new(model_path)
                .map_err(|e| Error::Llm(format!("model path has NUL: {e}")))?;
            let backend_c = CString::new(cfg.backend.as_str())
                .map_err(|e| Error::Llm(format!("backend has NUL: {e}")))?;

            let engine = unsafe {
                let settings = ffi::litert_lm_engine_settings_create(
                    model_c.as_ptr(),
                    backend_c.as_ptr(),
                    std::ptr::null(),
                    std::ptr::null(),
                );
                if settings.is_null() {
                    return Err(Error::Llm(
                        "failed to create LiteRT-LM engine settings".into(),
                    ));
                }
                if let Some(dir) = cfg.cache_dir.as_ref().and_then(|d| d.to_str()) {
                    if let Ok(dir_c) = CString::new(dir) {
                        ffi::litert_lm_engine_settings_set_cache_dir(settings, dir_c.as_ptr());
                    }
                }
                let engine = ffi::litert_lm_engine_create(settings);
                ffi::litert_lm_engine_settings_delete(settings);
                engine
            };
            if engine.is_null() {
                return Err(Error::Llm(format!(
                    "failed to create LiteRT-LM engine for model {}",
                    cfg.model_path.display()
                )));
            }
            Ok(Self {
                engine: std::sync::Arc::new(std::sync::Mutex::new(EngineHandle(engine))),
                family: cfg.family,
            })
        }
    }

    /// C コールバックに渡すコンテキスト。トークンごとに `on_token` を呼び、
    /// 終端 / エラーで `done` に結果を送る。
    struct CbCtx {
        on_token: Box<dyn FnMut(String) + Send + 'static>,
        done: std::sync::mpsc::Sender<std::result::Result<(), String>>,
    }

    extern "C" fn trampoline(
        data: *mut c_void,
        chunk: *const c_char,
        is_final: bool,
        error_msg: *const c_char,
    ) {
        if data.is_null() {
            return;
        }
        let ctx = unsafe { &mut *(data as *mut CbCtx) };
        if !error_msg.is_null() {
            let msg = unsafe { CStr::from_ptr(error_msg) }
                .to_string_lossy()
                .into_owned();
            let _ = ctx.done.send(Err(msg));
            return;
        }
        if !chunk.is_null() {
            let s = unsafe { CStr::from_ptr(chunk) }
                .to_string_lossy()
                .into_owned();
            if !s.is_empty() {
                (ctx.on_token)(s);
            }
        }
        if is_final {
            let _ = ctx.done.send(Ok(()));
        }
    }

    #[async_trait]
    impl LlmBackend for LiteRtBackend {
        async fn generate_stream(
            &self,
            req: LlmRequest,
            on_token: Box<dyn FnMut(String) + Send + 'static>,
        ) -> Result<()> {
            let engine = self.engine.clone();
            let family = self.family;
            tokio::task::spawn_blocking(move || -> Result<()> {
                let prompt = ellisii_llm_prompt::format(family, &req);

                // engine への session 生成〜消費を直列化。
                let guard = engine
                    .lock()
                    .map_err(|_| Error::Llm("LiteRT-LM engine mutex poisoned".into()))?;
                let engine_ptr = guard.0;

                let session = unsafe {
                    let config = ffi::litert_lm_session_config_create();
                    if config.is_null() {
                        return Err(Error::Llm("failed to create session config".into()));
                    }
                    ffi::litert_lm_session_config_set_max_output_tokens(
                        config,
                        req.max_tokens as i32,
                    );
                    // プロンプト整形は ellisii-llm-prompt 側で済ませるので二重適用を防ぐ。
                    ffi::litert_lm_session_config_set_apply_prompt_template(config, false);

                    let sampler = sampler_params(req.temperature);
                    ffi::litert_lm_session_config_set_sampler_params(config, &sampler);

                    let session = ffi::litert_lm_engine_create_session(engine_ptr, config);
                    ffi::litert_lm_session_config_delete(config);
                    session
                };
                if session.is_null() {
                    return Err(Error::Llm("failed to create LiteRT-LM session".into()));
                }

                let (tx, rx) = std::sync::mpsc::channel();
                let mut ctx = CbCtx { on_token, done: tx };
                let prompt_bytes = prompt.text.as_bytes();
                let input = ffi::LiteRtLmInputData {
                    type_: ffi::LiteRtLmInputDataType::Text,
                    data: prompt_bytes.as_ptr() as *const c_void,
                    size: prompt_bytes.len(),
                };

                let rc = unsafe {
                    ffi::litert_lm_session_generate_content_stream(
                        session,
                        &input,
                        1,
                        trampoline,
                        &mut ctx as *mut CbCtx as *mut c_void,
                    )
                };

                let result = if rc != 0 {
                    Err(Error::Llm(format!(
                        "litert_lm_session_generate_content_stream failed (rc={rc})"
                    )))
                } else {
                    // コールバックは別スレッドから呼ばれる。終端 / エラーまでブロック。
                    match rx.recv() {
                        Ok(Ok(())) => Ok(()),
                        Ok(Err(msg)) => Err(Error::Llm(format!("LiteRT-LM generation: {msg}"))),
                        Err(_) => Err(Error::Llm(
                            "LiteRT-LM stream ended without completion".into(),
                        )),
                    }
                };

                unsafe { ffi::litert_lm_session_delete(session) };
                drop(guard);
                result
            })
            .await
            .map_err(|e| Error::Llm(format!("LiteRT-LM blocking task panicked: {e}")))?
        }
    }

    /// temperature から sampler を決める。0 以下は greedy (決定的)。
    fn sampler_params(temperature: f32) -> ffi::LiteRtLmSamplerParams {
        if temperature <= 0.0 {
            ffi::LiteRtLmSamplerParams {
                type_: ffi::LiteRtLmSamplerType::Greedy,
                top_k: 1,
                top_p: 1.0,
                temperature: 1.0,
                seed: 0,
            }
        } else {
            ffi::LiteRtLmSamplerParams {
                type_: ffi::LiteRtLmSamplerType::TopP,
                top_k: 64,
                top_p: 0.95,
                temperature,
                seed: 0,
            }
        }
    }
}

#[cfg(not(litert_linked))]
mod backend {
    use super::*;
    use async_trait::async_trait;
    use ellisii_core::{Error, Result};
    use ellisii_llm_core::{LlmBackend, LlmRequest};

    pub struct LiteRtBackend;

    impl LiteRtBackend {
        pub fn load(_cfg: LiteRtConfig) -> Result<Self> {
            Err(Error::Llm(
                "llm-litert not linked against CLiteRTLM; rebuild with `--features litert` and \
                 LITERT_LM_LIB_DIR pointing at the dylib"
                    .into(),
            ))
        }
    }

    #[async_trait]
    impl LlmBackend for LiteRtBackend {
        async fn generate_stream(
            &self,
            _req: LlmRequest,
            _on_token: Box<dyn FnMut(String) + Send + 'static>,
        ) -> Result<()> {
            Err(Error::Llm("litert feature disabled".into()))
        }
    }
}

pub use backend::LiteRtBackend;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_defaults_to_cpu_backend() {
        std::env::remove_var("ELLISII_LITERT_BACKEND");
        let cfg = LiteRtConfig::new("/models/gemma-4-E2B-it.litertlm", ModelFamily::Gemma4);
        assert_eq!(cfg.backend, "cpu");
        assert_eq!(cfg.family, ModelFamily::Gemma4);
        assert!(cfg.cache_dir.is_none());
    }
}
