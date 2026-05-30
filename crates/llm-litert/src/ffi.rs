//! LiteRT-LM C ABI (`c/engine.h`) のうち本バックエンドが使う最小サブセットの宣言。
//!
//! 生成シンボルではなく手書き extern "C"。対象 dylib は build.rs でリンクする。

#![allow(non_camel_case_types)]
// ABI を完全に写すため、現状未使用の variant / 宣言も残す。
#![allow(dead_code)]

use std::os::raw::{c_char, c_int, c_void};

#[repr(C)]
pub struct LiteRtLmEngine {
    _private: [u8; 0],
}
#[repr(C)]
pub struct LiteRtLmEngineSettings {
    _private: [u8; 0],
}
#[repr(C)]
pub struct LiteRtLmSession {
    _private: [u8; 0],
}
#[repr(C)]
pub struct LiteRtLmSessionConfig {
    _private: [u8; 0],
}

/// `LiteRtLmSamplerType` (engine.h と一致させること)。
#[repr(C)]
#[derive(Clone, Copy)]
pub enum LiteRtLmSamplerType {
    Unspecified = 0,
    TopK = 1,
    TopP = 2,
    Greedy = 3,
}

#[repr(C)]
pub struct LiteRtLmSamplerParams {
    pub type_: LiteRtLmSamplerType,
    pub top_k: i32,
    pub top_p: f32,
    pub temperature: f32,
    pub seed: i32,
}

/// `LiteRtLmInputDataType`。テキスト入力のみ使う。
#[repr(C)]
#[derive(Clone, Copy)]
pub enum LiteRtLmInputDataType {
    Text = 0,
    Image = 1,
    ImageEnd = 2,
    Audio = 3,
    AudioEnd = 4,
}

#[repr(C)]
pub struct LiteRtLmInputData {
    pub type_: LiteRtLmInputDataType,
    pub data: *const c_void,
    pub size: usize,
}

/// `void (*)(void* callback_data, const char* chunk, bool is_final, const char* error_msg)`
pub type LiteRtLmStreamCallback = extern "C" fn(*mut c_void, *const c_char, bool, *const c_char);

extern "C" {
    pub fn litert_lm_set_min_log_level(level: c_int);

    pub fn litert_lm_engine_settings_create(
        model_path: *const c_char,
        backend_str: *const c_char,
        vision_backend_str: *const c_char,
        audio_backend_str: *const c_char,
    ) -> *mut LiteRtLmEngineSettings;
    pub fn litert_lm_engine_settings_set_cache_dir(
        settings: *mut LiteRtLmEngineSettings,
        cache_dir: *const c_char,
    );
    pub fn litert_lm_engine_settings_delete(settings: *mut LiteRtLmEngineSettings);

    pub fn litert_lm_engine_create(settings: *const LiteRtLmEngineSettings) -> *mut LiteRtLmEngine;
    pub fn litert_lm_engine_delete(engine: *mut LiteRtLmEngine);

    pub fn litert_lm_session_config_create() -> *mut LiteRtLmSessionConfig;
    pub fn litert_lm_session_config_set_max_output_tokens(
        config: *mut LiteRtLmSessionConfig,
        max_output_tokens: c_int,
    );
    pub fn litert_lm_session_config_set_apply_prompt_template(
        config: *mut LiteRtLmSessionConfig,
        apply_prompt_template: bool,
    );
    pub fn litert_lm_session_config_set_sampler_params(
        config: *mut LiteRtLmSessionConfig,
        sampler_params: *const LiteRtLmSamplerParams,
    );
    pub fn litert_lm_session_config_delete(config: *mut LiteRtLmSessionConfig);

    pub fn litert_lm_engine_create_session(
        engine: *mut LiteRtLmEngine,
        config: *mut LiteRtLmSessionConfig,
    ) -> *mut LiteRtLmSession;
    pub fn litert_lm_session_delete(session: *mut LiteRtLmSession);

    pub fn litert_lm_session_generate_content_stream(
        session: *mut LiteRtLmSession,
        inputs: *const LiteRtLmInputData,
        num_inputs: usize,
        callback: LiteRtLmStreamCallback,
        callback_data: *mut c_void,
    ) -> c_int;
}
