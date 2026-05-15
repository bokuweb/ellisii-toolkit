//! llama.cpp (`llama-cpp-2`) を使った GGUF 推論バックエンド。
//!
//! `gguf` feature 有効時のみ実体が組み込まれる。デフォルトでは空 (Err) を返す。

use async_trait::async_trait;
use ellisii_core::{Error, Result};
use ellisii_llm_core::{LlmBackend, LlmRequest, ModelFamily};

#[derive(Debug, Clone)]
pub struct LlamaConfig {
    pub model_path: std::path::PathBuf,
    pub family: ModelFamily,
    pub n_ctx: u32,
    pub n_batch: u32,
    pub n_gpu_layers: i32,
    /// per-token decode に使うスレッド数。`None` で llama.cpp の既定 (= 物理コア数)。
    /// CPU 推論で Intel ハイブリッドコア (P + E) のとき、E コアに撒くと
    /// 遅くなるので P コア数に絞ると速い (`ELLISII_LLAMA_N_THREADS=8` 等)。
    pub n_threads: Option<i32>,
    /// prompt processing (PP) に使うスレッド数。`None` で llama.cpp の既定。
    /// PP は matmul-heavy なのでコア種別を問わず多いほど速い。ハイブリッド
    /// CPU では P + E 全物理コアを投入すると TTFT (= 最初のトークンまでの
    /// 時間) が縮む (`ELLISII_LLAMA_N_THREADS_BATCH=12` で上書き可能)。
    pub n_threads_batch: Option<i32>,
    pub seed: u32,
    /// 永続化 KV cache の保存ディレクトリ。指定があれば、初回 generate 時に
    /// このディレクトリ配下のスナップショットを load し、generate 完了時に
    /// 上書き保存する。`None` ならプロセス内のみのキャッシュ (再起動で消える)。
    pub cache_dir: Option<std::path::PathBuf>,
    /// KV cache 型のヒント (`"q4_0"` / `"q8_0"` / `"f16"` / `"bf16"`)。
    /// マシンの KV 用メモリ予算から tier ベースで自動決定する:
    /// 大容量 → F16 (品質寄せ), 中 → Q8_0, 低 → Q4_0 (メモリ救済)。
    /// `ELLISII_LLAMA_KV_TYPE` env で上書き可能。
    pub kv_type_hint: String,
}

impl LlamaConfig {
    /// マシンの実装メモリ + モデルサイズから n_ctx / n_batch を動的に決める。
    /// 環境変数 `ELLISII_LLAMA_N_CTX` / `ELLISII_LLAMA_N_BATCH` / `ELLISII_LLAMA_N_GPU_LAYERS`
    /// が指定されていればそれを優先する。
    pub fn new(model_path: impl Into<std::path::PathBuf>, family: ModelFamily) -> Self {
        let model_path: std::path::PathBuf = model_path.into();
        let model_bytes = std::fs::metadata(&model_path).map(|m| m.len()).unwrap_or(0);
        let total_ram_bytes = detect_total_ram_bytes();
        let model_gib = bytes_to_gib(model_bytes);
        let total_gib = bytes_to_gib(total_ram_bytes);

        // 実行モードを判定:
        //   - Apple Silicon (Metal, unified memory): GPU offload, RAM = GPU メモリ
        //   - macOS Intel: Metal は使えるが旧 Mac は VRAM 控えめ → 半分まで offload
        //   - Linux/Windows: 別 GPU の VRAM を信頼できないので CPU フォールバック既定
        //     (CUDA でフル offload したい場合は ELLISII_LLAMA_N_GPU_LAYERS=999 で上書き)
        let mode = detect_runtime_mode();

        // KV cache に使える budget を mode で算出する。
        // unified: 全 RAM を共有プールとみなす
        // cpu: モデル + OS 予約後の RAM 全部
        // discrete-gpu: nvidia-smi で取れた free VRAM を主、不足分は RAM
        // OS + 他アプリ + llama.cpp 自身の compute buffer (PP 用, 数百 MB〜)。
        // 3 GiB だと実利用環境 (ブラウザ・IDE 等) で残量を読み違え、
        // KV 上限を攻めすぎて Decode Error -3 (KV alloc 失敗) を踏むことが
        // あったため 4 GiB に引き上げる。
        let reserve_gib = 4.0_f64;
        let detected_gpu = detect_nvidia_gpu();
        let available_for_kv_gib = match mode {
            RuntimeMode::Unified => (total_gib - model_gib - reserve_gib).max(0.5),
            RuntimeMode::Cpu => (total_gib - model_gib - reserve_gib).max(0.5),
            RuntimeMode::DiscreteGpu => {
                if let Some(g) = &detected_gpu {
                    let vram_gib = (g.free_vram_mb as f64) / 1024.0;
                    // モデルが VRAM に丸ごと載れば余りを KV に。載らなければ
                    // CPU offload と組み合わせて RAM の余りを足す。
                    if vram_gib >= model_gib {
                        (vram_gib - model_gib).max(0.5)
                    } else {
                        ((vram_gib - model_gib).max(0.0) + (total_gib - reserve_gib).max(0.5))
                            .max(0.5)
                    }
                } else {
                    // VRAM 値が取れなかった = nvidia-smi 失敗 / AMD GPU 等。
                    // 保守的に 4 GiB を残量とみなし、ユーザに env override を推奨。
                    4.0
                }
            }
        };

        // tier 階層で n_ctx / n_batch を決める。CPU は KV だけでなく
        // prompt eval も RAM/CPU で動くため n_batch は控えめにする。
        // n_ctx: 上限 8192 にキャップ。
        // - Gemma 4 E2B/E4B は実用的にはほぼ 4–8K で十分 (RAG コンテキスト +
        //   履歴 + 質問)。
        // - n_ctx を増やすと KV cache のメモリと per-token decode コストが
        //   両方増える。16K まで広げても embedding 後段が線形に伸びるだけで
        //   メリットが薄い。
        // - 中間 truncation を入れているので、長い入力は壊さず縮められる。
        // しきい値は安全マージンを取って高めに設定する。
        // 以前は available 6/3/1.5 GiB で切っていたが、KV 量子化を使っても
        // compute buffer + activations が積まれて実 RAM 消費が予測値を
        // 上回ることがあるため、tier 境界を 8/4/2 GiB に引き上げて
        // ギリギリの環境では小さい n_ctx に倒す。
        let auto_n_ctx: u32 = if available_for_kv_gib >= 8.0 {
            8192
        } else if available_for_kv_gib >= 4.0 {
            4096
        } else if available_for_kv_gib >= 2.0 {
            2048
        } else if available_for_kv_gib >= 1.0 {
            1024
        } else {
            // <1 GiB しか余っていない: 履歴 + 質問 + 生成で精一杯のサイズ。
            // ここまで詰まっていると Decode Error -3 の可能性が高いので
            // n_ctx/n_batch ともに最小化して、せめてクラッシュを避ける。
            512
        };
        // n_batch: prompt processing スループットに直結。Apple Silicon は
        // 512 あたりが sweet spot だが、ここを攻めると compute buffer が
        // 線形に膨らむ (n_batch * d_model * f16 + 中間 tensor)。Tight な
        // 環境では小さく倒して安全側に。
        let auto_n_batch: u32 = match mode {
            RuntimeMode::Cpu => {
                if available_for_kv_gib >= 8.0 {
                    256
                } else if available_for_kv_gib >= 2.0 {
                    128
                } else {
                    64
                }
            }
            _ => {
                if available_for_kv_gib >= 4.0 {
                    512
                } else if available_for_kv_gib >= 2.0 {
                    256
                } else if available_for_kv_gib >= 1.0 {
                    128
                } else {
                    64
                }
            }
        };
        let auto_n_gpu_layers: i32 = match mode {
            RuntimeMode::Unified => 999, // 全層 offload
            RuntimeMode::Cpu => 0,       // CPU 専用
            RuntimeMode::DiscreteGpu => {
                // VRAM が取れていればモデルが載るかどうかで判定:
                //   free_vram >= model + 1GB バッファ → 全層 offload (999)
                //   free_vram >= model * 0.6 → 部分 offload (経験則で 32 層)
                //   それ以下 → CPU 専用
                // VRAM 値が無ければ 0 (旧挙動、ユーザに env 上書きを促す)。
                if let Some(g) = &detected_gpu {
                    let vram_gib = (g.free_vram_mb as f64) / 1024.0;
                    if vram_gib >= model_gib + 1.0 {
                        999
                    } else if vram_gib >= model_gib * 0.6 {
                        32
                    } else {
                        0
                    }
                } else {
                    0
                }
            }
        };

        let n_ctx = std::env::var("ELLISII_LLAMA_N_CTX")
            .ok()
            .and_then(|v| v.parse::<u32>().ok())
            .unwrap_or(auto_n_ctx);
        let n_batch = std::env::var("ELLISII_LLAMA_N_BATCH")
            .ok()
            .and_then(|v| v.parse::<u32>().ok())
            .unwrap_or(auto_n_batch);
        let n_gpu_layers = std::env::var("ELLISII_LLAMA_N_GPU_LAYERS")
            .ok()
            .and_then(|v| v.parse::<i32>().ok())
            .unwrap_or(auto_n_gpu_layers);
        // n_threads (decode) / n_threads_batch (prompt processing) を分離:
        //   - decode は per-token で KV cache bandwidth 律速 → P コアのみ
        //   - PP は matmul-heavy → P + E 全物理コア
        //   ハイブリッド CPU を検出できたときだけ自動 pin する。
        //   ホモジニアスや Linux/Mac では None (= llama.cpp 既定) に倒す。
        let env_threads = std::env::var("ELLISII_LLAMA_N_THREADS").ok();
        let env_threads_batch = std::env::var("ELLISII_LLAMA_N_THREADS_BATCH").ok();
        let detected_pcores = cpu_topology::detect_performance_core_count();
        let detected_total = cpu_topology::detect_total_physical_core_count();
        let is_cpu = matches!(mode, RuntimeMode::Cpu);
        let n_threads =
            cpu_topology::resolve_n_threads(env_threads.as_deref(), is_cpu, detected_pcores);
        let n_threads_batch = cpu_topology::resolve_n_threads_batch(
            env_threads_batch.as_deref(),
            is_cpu,
            detected_total,
        );
        if let (None, Some(pcores), true) = (&env_threads, detected_pcores, is_cpu) {
            tracing::info!("llama n_threads auto-pinned to {pcores} P-cores (hybrid CPU detected)",);
        }
        if let (None, Some(total), true) = (&env_threads_batch, detected_total, is_cpu) {
            tracing::info!("llama n_threads_batch auto-pinned to {total} physical cores (P+E)",);
        }

        // KV cache 型の tier-based 自動既定:
        //   高 (≥6 GiB 予算 = 24+ GB Mac 等) → F16 (品質寄せ、メモリ余裕)
        //   中 (≥3 GiB)                       → Q8_0 (バランス)
        //   低 (<3 GiB)                       → Q4_0 (メモリ救済)
        // ELLISII_LLAMA_KV_TYPE で上書きできる。LOW_SPEC=1 のときは tier に
        // 関わらず Q4_0 (imp 側で env 解釈と一緒に処理)。
        // tier しきい値は n_ctx 側と揃える。境界の精度差より、
        // ギリギリ環境で q8_0 → q4_0 に早めに落として OOM を避ける方が大事。
        let auto_kv_type: &str = if available_for_kv_gib >= 8.0 {
            "f16"
        } else if available_for_kv_gib >= 4.0 {
            "q8_0"
        } else {
            "q4_0"
        };

        tracing::info!(
            "llama auto-tune: mode={:?}, gpu={:?}, total_ram={:.1}GiB, model={:.1}GiB, available_for_kv={:.1}GiB → n_ctx={}, n_batch={}, n_gpu_layers={}, kv_type={}",
            mode,
            detected_gpu.as_ref().map(|g| format!("{} {}MiB free", g.name, g.free_vram_mb)),
            total_gib,
            model_gib,
            available_for_kv_gib,
            n_ctx,
            n_batch,
            n_gpu_layers,
            auto_kv_type,
        );

        Self {
            model_path,
            family,
            n_ctx,
            n_batch,
            n_gpu_layers,
            n_threads,
            n_threads_batch,
            seed: 0xE11151,
            cache_dir: None,
            kv_type_hint: auto_kv_type.to_string(),
        }
    }
}

fn bytes_to_gib(b: u64) -> f64 {
    (b as f64) / 1024.0 / 1024.0 / 1024.0
}

fn detect_total_ram_bytes() -> u64 {
    let mut sys = sysinfo::System::new();
    sys.refresh_memory();
    sys.total_memory()
}

/// 検出された GPU 情報。`nvidia-smi` のパースで取れたものだけ返す。
/// AMD / Intel GPU は対象外 (ROCm / Level Zero の検出は将来の拡張ポイント)。
#[derive(Debug, Clone)]
pub struct GpuInfo {
    pub name: String,
    pub total_vram_mb: u32,
    pub free_vram_mb: u32,
}

/// `nvidia-smi` を叩いて 1 つ目の NVIDIA GPU の名前と総 / 空き VRAM を取り出す。
/// バイナリが無い・実行失敗・出力が空のいずれでも `None` を返す (=「NVIDIA 無し」扱い)。
///
/// 依存ゼロ (process spawn のみ) で動かすため、NVML は使わない。複数 GPU がある
/// 場合は先頭の値だけ返す (実用上 dev マシンで複数 GPU は稀)。
pub fn detect_nvidia_gpu() -> Option<GpuInfo> {
    let output = std::process::Command::new("nvidia-smi")
        .args([
            "--query-gpu=name,memory.total,memory.free",
            "--format=csv,noheader,nounits",
        ])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    let line = stdout.lines().next()?.trim();
    if line.is_empty() {
        return None;
    }
    let parts: Vec<&str> = line.split(',').map(str::trim).collect();
    if parts.len() < 3 {
        return None;
    }
    let total_vram_mb: u32 = parts[1].parse().ok()?;
    let free_vram_mb: u32 = parts[2].parse().ok()?;
    Some(GpuInfo {
        name: parts[0].to_string(),
        total_vram_mb,
        free_vram_mb,
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum RuntimeMode {
    /// Apple Silicon (Metal + unified memory). GPU と CPU で RAM を共有。
    Unified,
    /// 別 GPU (NVIDIA / AMD)。VRAM が別管理で精密推定が困難。
    DiscreteGpu,
    /// CPU 専用。Metal/CUDA バックエンドが無いか、利用不可。
    Cpu,
}

/// 自動 / 環境変数 / NVIDIA 検出から実行モードを決める。
///
/// 優先順:
/// 1. `ELLISII_LLAMA_MODE=cpu|unified|discrete` 強制上書き
/// 2. macOS (aarch64) → Unified (Apple Silicon)
/// 3. それ以外 + NVIDIA GPU 検出 → DiscreteGpu
/// 4. それ以外 → Cpu
pub fn detect_runtime_mode() -> RuntimeMode {
    if let Ok(force) = std::env::var("ELLISII_LLAMA_MODE") {
        match force.to_lowercase().as_str() {
            "cpu" => return RuntimeMode::Cpu,
            "unified" | "metal" | "apple" => return RuntimeMode::Unified,
            "discrete" | "cuda" | "rocm" | "vulkan" => return RuntimeMode::DiscreteGpu,
            _ => {}
        }
    }
    // macOS は基本 Apple Silicon (Metal Unified)。
    // Intel Mac でも Metal は使えるが、unified ではないので保守的に CPU 扱い。
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    {
        RuntimeMode::Unified
    }
    // それ以外 (Linux / Windows / Intel Mac):
    //   nvidia-smi が叩けて GPU が返るなら DiscreteGpu に倒す。
    //   llama.cpp 側が CUDA / Vulkan ビルドでなければ実際の offload は失敗するが、
    //   モード判定は最大限有利な側に倒しておき、`available_for_kv_gib` を
    //   VRAM 基準で見積もる。バックエンド feature が無いと load 時にエラーで
    //   落ちるので、ここでの誤検出は致命的にはならない。
    #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
    {
        if detect_nvidia_gpu().is_some() {
            RuntimeMode::DiscreteGpu
        } else {
            RuntimeMode::Cpu
        }
    }
}

#[cfg(feature = "gguf")]
mod backend {
    pub use super::imp::LlamaCppBackend;
}

#[cfg(not(feature = "gguf"))]
mod backend {
    use super::*;

    pub struct LlamaCppBackend;

    impl LlamaCppBackend {
        pub fn load(_cfg: LlamaConfig) -> Result<Self> {
            Err(Error::Llm(
                "llm-llamacpp built without `gguf` feature; rebuild with --features gguf".into(),
            ))
        }
    }

    #[async_trait]
    impl LlmBackend for LlamaCppBackend {
        async fn generate_stream(
            &self,
            _req: LlmRequest,
            _on_token: Box<dyn FnMut(String) + Send + 'static>,
        ) -> Result<()> {
            Err(Error::Llm("gguf feature disabled".into()))
        }
    }
}

pub use backend::LlamaCppBackend;

pub mod cpu_topology;
pub mod feasibility;

/// KV snapshot のファイル名指紋。互換でない設定 (model 違い・量子化違い・GPU
/// offload 量違い・family 違い) で誤 load が起きないよう、関係するパラメータと
/// モデルファイルのメタデータを混ぜ込んで hex 16 文字に丸める。
#[allow(dead_code)] // gguf feature 無効時は呼び出し元が無いがテストで使う
fn compute_kv_fingerprint(cfg: &LlamaConfig) -> String {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    let mut h = DefaultHasher::new();
    cfg.model_path.to_string_lossy().hash(&mut h);
    cfg.n_ctx.hash(&mut h);
    cfg.n_batch.hash(&mut h);
    // n_gpu_layers が違うと KV のレイアウト (per-layer 配置) が変わる可能性が
    // あるので snapshot を分ける。
    cfg.n_gpu_layers.hash(&mut h);
    // family はトークナイザを決め、cached_prompt のトークン ID 列が変わる。
    // ModelFamily 自体は Hash 派生していないので識別子文字列で混ぜる。
    family_tag(cfg.family).hash(&mut h);
    // モデルファイルの実体 (サイズ + mtime) を含めて、同一パスで再ダウンロード
    // / 別 quant への入れ替えが起きたときに自動で snapshot を捨てる。
    let (file_size, file_mtime_ns) = model_file_signature(&cfg.model_path);
    file_size.hash(&mut h);
    file_mtime_ns.hash(&mut h);
    // KV cache の量子化を変えても指紋が変わるよう、固定文字列で q8_0 を含める。
    // (この backend は KV を Q8_0 で固定しているため、その事実を埋めて
    //  将来 KV type を変えたとき自動的に snapshot が無効になるようにする)
    "kv-q8_0".hash(&mut h);
    format!("{:016x}", h.finish())
}

/// `ELLISII_LLAMA_KV_TYPE` (任意) + low_spec + tier-based 既定から KV 型を決める。
///
/// 優先順:
/// 1. `env_value` で明示指定があればそれを採用 (`q4_0` / `q8_0` / `f16` / `bf16` 等)
/// 2. `low_spec = true` なら Q4_0 (メモリ最優先)
/// 3. それ以外は `tier_default` (LlamaConfig::new で算出した tier-based 既定)
///
/// 不正な env 値は警告ログを出し、tier_default に倒す。
#[cfg(feature = "gguf")]
fn parse_kv_type(
    env_value: Option<&str>,
    low_spec: bool,
    tier_default: &str,
) -> llama_cpp_2::context::params::KvCacheType {
    use llama_cpp_2::context::params::KvCacheType;
    fn lookup(s: &str) -> Option<KvCacheType> {
        match s {
            "q4_0" | "q4" => Some(KvCacheType::Q4_0),
            "q4_1" => Some(KvCacheType::Q4_1),
            "q5_0" => Some(KvCacheType::Q5_0),
            "q5_1" => Some(KvCacheType::Q5_1),
            "q8_0" | "q8" => Some(KvCacheType::Q8_0),
            "f16" | "fp16" => Some(KvCacheType::F16),
            "bf16" => Some(KvCacheType::BF16),
            _ => None,
        }
    }
    if let Some(s) = env_value.map(|s| s.trim().to_ascii_lowercase()) {
        if !s.is_empty() {
            if let Some(t) = lookup(&s) {
                return t;
            }
            tracing::warn!("unknown ELLISII_LLAMA_KV_TYPE={s:?}; falling back to default");
        }
    }
    if low_spec {
        return KvCacheType::Q4_0;
    }
    lookup(&tier_default.to_ascii_lowercase()).unwrap_or(KvCacheType::Q8_0)
}

/// `ELLISII_LOW_SPEC` env と実機スペックから low_spec 判定を解決する純関数。
///
/// 優先順:
/// 1. `env_value` が `1` / `true` → 強制 ON (= Q4_0 KV へ倒す)
/// 2. `env_value` が `0` / `false` → 強制 OFF (= 自動判定を抑止)
/// 3. それ以外 (env 未指定 or 不正値): 自動判定
///    - CPU モード AND `total_ram_gib <= 8.0` → ON
///    - それ以外 → OFF
///
/// 自動判定の境界 (8GiB) の根拠:
///   低 spec ノート PC の典型 RAM 8GiB では、モデル (2-5GiB) + KV q8_0 +
///   compute buffer + OS/他アプリ予約で常時タイト。Q4_0 に落とすことで
///   KV 半減 → swap 回避 + 残 RAM を増やせる。
///   16GiB 以上ある CPU マシンや GPU offload マシンでは、Q8_0 のまま
///   品質を取った方がトータルで得 (eval で確認済み)。
pub fn resolve_low_spec(env_value: Option<&str>, is_cpu_mode: bool, total_ram_gib: f64) -> bool {
    if let Some(s) = env_value.map(|s| s.trim().to_ascii_lowercase()) {
        if !s.is_empty() {
            match s.as_str() {
                "1" | "true" | "yes" | "on" => return true,
                "0" | "false" | "no" | "off" => return false,
                _ => {} // fall through to auto
            }
        }
    }
    is_cpu_mode && total_ram_gib <= 8.0
}

/// 実環境を読み取って `ELLISII_LOW_SPEC` 解決値を返すコンビニエンス関数。
///
/// 内部は [`resolve_low_spec`] への純粋な委譲で、`env::var(env_key)` /
/// [`detect_runtime_mode`] / [`detect_total_ram_bytes`] を読みに行く部分だけ
/// を吸収する。Tauri 等から `crates/llm-llamacpp` の private 関数を直接呼べない
/// ため、この単一エントリポイントから auto 判定を流用できるようにする。
pub fn auto_low_spec_env(env_key: &str) -> bool {
    let env = std::env::var(env_key).ok();
    let is_cpu = matches!(detect_runtime_mode(), RuntimeMode::Cpu);
    let ram_gib = bytes_to_gib(detect_total_ram_bytes());
    resolve_low_spec(env.as_deref(), is_cpu, ram_gib)
}

#[allow(dead_code)]
fn family_tag(f: ModelFamily) -> &'static str {
    match f {
        ModelFamily::Gemma4 => "gemma4",
        ModelFamily::Qwen => "qwen",
    }
}

/// モデルファイルの (サイズ, mtime ナノ秒) を best-effort で取得。失敗時は
/// (0, 0) を返し、fingerprint は他のフィールドだけで決まる挙動になる。
#[allow(dead_code)]
fn model_file_signature(path: &std::path::Path) -> (u64, i128) {
    let Ok(meta) = std::fs::metadata(path) else {
        return (0, 0);
    };
    let size = meta.len();
    let mtime_ns = meta
        .modified()
        .ok()
        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
        .map(|d| d.as_nanos() as i128)
        .unwrap_or(0);
    (size, mtime_ns)
}

#[cfg(feature = "gguf")]
mod imp {
    use super::*;
    use llama_cpp_2::context::params::LlamaContextParams;
    use llama_cpp_2::context::LlamaContext;
    use llama_cpp_2::llama_backend::LlamaBackend;
    use llama_cpp_2::llama_batch::LlamaBatch;
    use llama_cpp_2::model::params::LlamaModelParams;
    #[allow(deprecated)]
    use llama_cpp_2::model::Special;
    use llama_cpp_2::model::{AddBos, LlamaModel};
    use llama_cpp_2::sampling::LlamaSampler;
    use llama_cpp_2::token::LlamaToken;
    use std::num::NonZeroU32;
    use std::sync::Arc;
    use std::sync::Mutex;
    use std::sync::OnceLock;

    static BACKEND: OnceLock<Arc<LlamaBackend>> = OnceLock::new();

    fn backend_static() -> Result<&'static LlamaBackend> {
        if let Some(b) = BACKEND.get() {
            return Ok(b.as_ref());
        }
        let b = LlamaBackend::init().map_err(|e| Error::Llm(format!("backend: {e}")))?;
        let _ = BACKEND.set(Arc::new(b));
        Ok(BACKEND
            .get()
            .map(|a| a.as_ref())
            .expect("backend just inserted"))
    }

    /// 1 つの長寿命 `LlamaContext` と、そこに既に decode 済みのトークン列を
    /// 保持する。ターン間で KV cache を使い回し、共通プレフィックス分の
    /// re-decode を省略するためのもの。
    ///
    /// `LlamaContext` 内部に raw pointer を持つため自動 derive では `Send` に
    /// ならないが、`Mutex` で必ず単一スレッドからのみ触る前提なので
    /// `unsafe impl Send` する。
    struct SessionCell {
        ctx: LlamaContext<'static>,
        /// 直近のターンで KV cache に書き込まれているプロンプトトークン列。
        /// 生成 (= assistant) 部分の token は含めない: 次ターンのプロンプト
        /// (= prompt formatter で再構成されるテキスト) が tokenize 結果と
        /// 必ずしも 1:1 に一致しないため、prompt 部分だけを比較対象にして
        /// 「prompt の共通 prefix」だけを再利用する方が安全。
        cached_prompt: Vec<LlamaToken>,
    }
    unsafe impl Send for SessionCell {}

    pub struct LlamaCppBackend {
        backend: &'static LlamaBackend,
        model: &'static LlamaModel,
        family: ModelFamily,
        n_ctx: u32,
        n_batch: u32,
        kv_type_hint: String,
        n_threads: Option<i32>,
        n_threads_batch: Option<i32>,
        /// 長寿命 LlamaContext。初回 generate 時に作って以降使い回す。
        /// `Option` なのは、エラーで context が壊れた場合に drop して作り直す
        /// 余地を残しているため。
        session: Arc<Mutex<Option<SessionCell>>>,
        /// 永続化 KV cache のスナップショットファイルパス
        /// (`<cache_dir>/<fingerprint>.bin`)。`None` なら永続化しない。
        kv_snapshot: Option<std::path::PathBuf>,
    }

    impl LlamaCppBackend {
        pub fn load(cfg: LlamaConfig) -> Result<Self> {
            let backend = backend_static()?;
            // mmap=true / mlock=false を明示する。
            //   - mmap: モデルファイルを必要なページだけ読み込む。低 RAM の
            //     Windows では特に重要 (= モデル全体の常駐を避ける)。
            //   - mlock: 物理メモリへの常駐を強制する OS API。低 spec マシン
            //     で他アプリが OOM する原因になるので必ず off。
            //   llama-cpp-2 0.1.145 の既定はそれぞれ true/false で同じだが、
            //   将来の default 変更に対する防御として明示しておく。
            let mp = LlamaModelParams::default()
                .with_n_gpu_layers(cfg.n_gpu_layers as u32)
                .with_use_mmap(true)
                .with_use_mlock(false);
            let model = LlamaModel::load_from_file(backend, &cfg.model_path, &mp)
                .map_err(|e| Error::Llm(format!("load model: {e}")))?;
            // モデルはプロセス終了まで生かしっぱなしにする (LlamaContext<'static>
            // を作るためには &'static LlamaModel が必要)。Box::leak はここで
            // 1 回だけ呼ばれるので実質リークではない (= シングルトン的扱い)。
            let model_static: &'static LlamaModel = Box::leak(Box::new(model));
            // KV snapshot の指紋: モデルファイルパス + n_ctx + n_batch + KV q8_0
            // を反映。設定が変わると別ファイルになるので、互換性のないスナップ
            // ショットを誤って load するのを防ぐ。
            //
            // `ELLISII_DISABLE_KV_SNAPSHOT=1` で完全 OFF にできる (各生成終了時の
            // disk write を抑制したいとき / SSD 寿命を気にする環境向け)。
            let snapshot_disabled = std::env::var("ELLISII_DISABLE_KV_SNAPSHOT")
                .ok()
                .map(|v| {
                    let s = v.trim().to_ascii_lowercase();
                    matches!(s.as_str(), "1" | "true" | "yes" | "on")
                })
                .unwrap_or(false);
            let kv_snapshot = if snapshot_disabled {
                tracing::info!("kv snapshot disabled via ELLISII_DISABLE_KV_SNAPSHOT");
                None
            } else {
                cfg.cache_dir.as_ref().map(|dir| {
                    let fingerprint = super::compute_kv_fingerprint(&cfg);
                    let _ = std::fs::create_dir_all(dir);
                    dir.join(format!("{fingerprint}.bin"))
                })
            };

            tracing::info!(
                "llama backend loaded: n_ctx={}, n_batch={}, n_gpu_layers={}, kv_snapshot={:?}",
                cfg.n_ctx,
                cfg.n_batch,
                cfg.n_gpu_layers,
                kv_snapshot,
            );
            Ok(Self {
                backend,
                model: model_static,
                family: cfg.family,
                n_ctx: cfg.n_ctx,
                n_batch: cfg.n_batch,
                kv_type_hint: cfg.kv_type_hint.clone(),
                n_threads: cfg.n_threads,
                n_threads_batch: cfg.n_threads_batch,
                session: Arc::new(Mutex::new(None)),
                kv_snapshot,
            })
        }
    }

    #[async_trait]
    impl LlmBackend for LlamaCppBackend {
        async fn generate_stream(
            &self,
            req: LlmRequest,
            on_token: Box<dyn FnMut(String) + Send + 'static>,
        ) -> Result<()> {
            let backend = self.backend;
            let model = self.model;
            let family = self.family;
            let n_ctx = self.n_ctx;
            let n_batch = self.n_batch;
            let kv_type_hint = self.kv_type_hint.clone();
            let n_threads = self.n_threads;
            let n_threads_batch = self.n_threads_batch;
            let session_arc = self.session.clone();
            let kv_snapshot = self.kv_snapshot.clone();
            // FnMut を blocking タスクへ送るため Mutex でラップ
            let cb = std::sync::Arc::new(std::sync::Mutex::new(on_token));
            tokio::task::spawn_blocking(move || -> Result<()> {
                let prompt = ellisii_llm_prompt::format(family, &req);
                // n_batch を超えるプロンプトは後段で分割 decode する。
                // Metal の unified memory が小さい環境で OOM を防ぐため小さめの既定値。
                let n_batch_eff = n_batch.max(64);

                // 長寿命セッションを取得 or 初期化。
                // session_arc.lock() を保持したまま生成を進めるので、複数スレッドから
                // 同時 generate 要求が来てもシリアル化される (= 旧 gen_lock の役割
                // も兼ねる)。
                let mut session_guard = session_arc
                    .lock()
                    .map_err(|_| Error::Llm("session mutex poisoned".to_string()))?;
                let mut just_initialized = false;
                if session_guard.is_none() {
                    just_initialized = true;
                    // Flash Attention: AUTO ポリシーで llama.cpp に判断させる
                    //   (対応モデルなら ENABLED、非対応なら DISABLED に落ちる)。
                    // KV cache q8_0: fp16 → q8_0 に量子化することでメモリが半減
                    //   (= n_ctx を倍取れる) し、Apple Silicon Metal では速度も
                    //   ほぼ同等。q4_0 まで落とすと精度劣化が出やすいので q8_0 を採用。
                    // n_ubatch: prompt processing 時の物理バッチ。n_batch と
                    //   揃えておけば PP スループットが最大化する。
                    let flash_auto = llama_cpp_sys_2::LLAMA_FLASH_ATTN_TYPE_AUTO;
                    // KV cache type decision order:
                    //   1. ELLISII_LLAMA_KV_TYPE explicit
                    //   2. low_spec=true → Q4_0 (env=1/true で強制 OR
                    //      auto: CPU mode + RAM ≤ 8GiB)
                    //   3. tier-based default (kv_type_hint, computed in
                    //      LlamaConfig::new from RAM headroom: large→F16 / mid→Q8_0 / low→Q4_0)
                    let env_kv = std::env::var("ELLISII_LLAMA_KV_TYPE").ok();
                    let env_kv_ref = env_kv.as_deref();
                    let env_low = std::env::var("ELLISII_LOW_SPEC").ok();
                    let is_cpu = matches!(super::detect_runtime_mode(), super::RuntimeMode::Cpu);
                    let total_ram_gib =
                        super::bytes_to_gib(super::detect_total_ram_bytes());
                    let low_spec = super::resolve_low_spec(
                        env_low.as_deref(),
                        is_cpu,
                        total_ram_gib,
                    );
                    let kv_type = parse_kv_type(env_kv_ref, low_spec, &kv_type_hint);
                    tracing::info!(
                        "llama kv cache type: {:?} (hint={}, env={:?}, low_spec={})",
                        kv_type, kv_type_hint, env_kv, low_spec
                    );
                    let mut cp = LlamaContextParams::default()
                        .with_n_ctx(NonZeroU32::new(n_ctx))
                        .with_n_batch(n_batch_eff)
                        .with_n_ubatch(n_batch_eff)
                        .with_flash_attention_policy(flash_auto)
                        .with_type_k(kv_type)
                        .with_type_v(kv_type);
                    // n_threads / n_threads_batch:
                    //   - unset の側は llama.cpp 既定 (= 物理コア数自動検出) が効く
                    //   - n_threads      = decode 用。ハイブリッド CPU では P コアのみ
                    //     (E コアに撒くと per-token decode が遅くなる)
                    //   - n_threads_batch = prompt processing 用。P + E 全物理コア
                    //     (matmul-heavy で多いほど速い → TTFT 短縮)
                    //   両者は LlamaConfig::new で cpu_topology から自動決定される。
                    //   値が違う場合は別 API で個別指定する。
                    if let Some(t) = n_threads {
                        cp = cp.with_n_threads(t);
                        tracing::info!("llama n_threads (decode) set to {t}");
                    }
                    if let Some(tb) = n_threads_batch {
                        cp = cp.with_n_threads_batch(tb);
                        tracing::info!("llama n_threads_batch (PP) set to {tb}");
                    }
                    let ctx: LlamaContext<'static> = model
                        .new_context(backend, cp)
                        .map_err(|e| Error::Llm(format!("context: {e}")))?;
                    *session_guard = Some(SessionCell {
                        ctx,
                        cached_prompt: Vec::new(),
                    });
                }
                let session: &mut SessionCell = session_guard.as_mut().expect("just inserted");
                let ctx: &mut LlamaContext<'static> = &mut session.ctx;

                // 永続化 KV snapshot の load:
                //   - 初回 generate (just_initialized) かつ snapshot ファイルが
                //     ある場合のみ試す
                //   - 失敗時はサイレントにスキップ (= 通常の re-decode に倒す)
                //   - 成功時は cached_prompt にロード済みトークン列をセットして
                //     prefix-diff の対象にする
                if just_initialized {
                    if let Some(snap) = kv_snapshot.as_ref() {
                        if snap.exists() {
                            let load_started = std::time::Instant::now();
                            match ctx.state_load_file(snap, n_ctx as usize) {
                                Ok(loaded) => {
                                    tracing::info!(
                                        "kv snapshot loaded: {} tokens in {:.2}s ({:?})",
                                        loaded.len(),
                                        load_started.elapsed().as_secs_f64(),
                                        snap,
                                    );
                                    session.cached_prompt = loaded;
                                }
                                Err(e) => {
                                    tracing::warn!(
                                        "kv snapshot load failed ({:?}): {e}; falling back to fresh context",
                                        snap
                                    );
                                    // 壊れた snapshot は削除しておく
                                    let _ = std::fs::remove_file(snap);
                                }
                            }
                        }
                    }
                }

                let mut tokens: Vec<LlamaToken> = model
                    .str_to_token(&prompt.text, AddBos::Always)
                    .map_err(|e| Error::Llm(format!("tokenize: {e}")))?;

                // (A) コンテキスト切り詰め:
                //   出力トークン (req.max_tokens) と少しの余裕を確保した上で、
                //   プロンプトが n_ctx を超えていたら**中央**を削る。
                //   先頭の chat template + system prompt と、末尾の質問 + アシスタント
                //   開始マーカーは必ず残す (両端を切ると構造が壊れて空応答になりがち)。
                let reserve = (req.max_tokens as usize).saturating_add(64);
                let prompt_budget = (n_ctx as usize).saturating_sub(reserve).max(64);
                if tokens.len() > prompt_budget {
                    // 先頭 96 / 末尾 256 を必ず残す。残りの予算を中央で詰める。
                    let head_keep = 96.min(tokens.len() / 4);
                    let tail_keep = 256.min(tokens.len() / 2);
                    let total_keep = head_keep + tail_keep;
                    if total_keep < prompt_budget {
                        let drop_start = head_keep;
                        let drop_end = tokens.len() - tail_keep;
                        let drop_count = drop_end - drop_start;
                        let need_drop = tokens.len() - prompt_budget;
                        let actual_drop = drop_count.min(need_drop);
                        tracing::warn!(
                            "prompt {} tokens > budget {} (n_ctx={}, reserve={}); truncating middle: head={} tail={} drop={}",
                            tokens.len(),
                            prompt_budget,
                            n_ctx,
                            reserve,
                            head_keep,
                            tail_keep,
                            actual_drop
                        );
                        tokens.drain(drop_start..drop_start + actual_drop);
                    }
                    // それでもまだ超えていれば末尾優先で切る (構造は守る)
                    if tokens.len() > prompt_budget {
                        let drop = tokens.len() - prompt_budget;
                        tracing::warn!(
                            "still over budget after middle-drop; dropping {} more from head",
                            drop
                        );
                        tokens.drain(0..drop);
                    }
                }

                // (B) Prompt prefix KV reuse:
                //   直前ターンで decode 済みのトークン列 (= session.cached_prompt)
                //   と新プロンプト (tokens) の共通先頭を求め、共通でない部分から
                //   だけ decode する。これによりマルチターンで:
                //     - system prompt
                //     - RAG コンテキスト (前ターンと同じソース集合なら同じ tokens)
                //     - 過去の user/assistant 履歴
                //   が再エンコードされなくなる (= 数百〜数千トークン分の re-decode
                //   をスキップ)。
                let common_prefix_len = session
                    .cached_prompt
                    .iter()
                    .zip(tokens.iter())
                    .take_while(|(a, b)| *a == *b)
                    .count();
                // KV cache は (cached_prompt の長さ + 前回生成分) が乗っているので、
                // common_prefix_len 以降をまるごと捨てる。clear_kv_cache_seq は
                // [p0, p1) の半開区間で消すので p1=None で末尾までクリア。
                if !session.cached_prompt.is_empty() {
                    if common_prefix_len == 0 {
                        // 完全に別系統のプロンプト (system / RAG どちらも新しい)。
                        // `clear_kv_cache_seq(Some(0), Some(0), None)` (= 区間 [0, end) を
                        // seq 0 から削除) を期待していたが、llama-cpp-2 0.1.145 の
                        // `llama_memory_seq_rm` 経由ではこの呼び出しが完全な reset に
                        // ならないケースがある (= 直後の decode で
                        // `inconsistent sequence positions ... X = N, Y = 0` を踏む)。
                        // SDK 経由で `LlmRewriter::rewrite` を回すと第 2 呼び出し以降
                        // 全部この経路に落ちて Err になり、rewriter が passthrough に
                        // フォールバックして multi-query が事実上無効化される
                        // (`docs/eval/recall-evals.md` Run 7 で計測済み)。
                        //
                        // 安全寄りに `clear_kv_cache` で全 sequence を一括消去する。
                        // prefix が共有されている (multi-turn) 場合は selective clear に
                        // 落ちるので、prefix reuse の最適化はそのまま生きる。
                        ctx.clear_kv_cache();
                    } else if common_prefix_len <= session.cached_prompt.len() {
                        // 「前回 prompt 全長」または「分岐ポイント」より後ろをクリア。
                        // 共通 prefix 長 == cached.len() のときも、前回の生成分 KV が
                        // 末尾に残っているのでクリアする必要がある。
                        let _ = ctx.clear_kv_cache_seq(
                            Some(0),
                            Some(common_prefix_len as u32),
                            None,
                        );
                    }
                }
                let to_decode = &tokens[common_prefix_len..];
                tracing::debug!(
                    "prompt cache: total={} common_prefix={} new={} (skipped {} tokens of decode)",
                    tokens.len(),
                    common_prefix_len,
                    to_decode.len(),
                    common_prefix_len,
                );

                // (C) チャンク decode:
                //   プロンプト (の差分) を n_batch 単位で分けて順に投入する。
                //   これにより 1 バッチ > n_batch で起きる ggml_abort を防げる。
                let chunk_size = n_batch_eff as usize;
                let total = tokens.len();
                let new_total = to_decode.len();
                let mut batch = LlamaBatch::new(chunk_size.max(512), 1);
                // common_prefix_len から数え始める。pos は KV cache 上の絶対位置。
                let mut pos: i32 = common_prefix_len as i32;
                if new_total == 0 {
                    // 新規 token が無い (新プロンプトが前回と完全一致): 直前 sample
                    // 用に末尾 1 トークンを「再 evaluation」する必要がある。
                    // KV から最後の 1 つを剥がして再投入。
                    if total > 0 {
                        let last_pos = (total - 1) as u32;
                        let _ =
                            ctx.clear_kv_cache_seq(Some(0), Some(last_pos), None);
                        batch.clear();
                        batch
                            .add(tokens[total - 1], last_pos as i32, &[0], true)
                            .map_err(|e| Error::Llm(format!("batch.add (resample): {e}")))?;
                        ctx.decode(&mut batch)
                            .map_err(|e| Error::Llm(format!("decode (resample): {e}")))?;
                        pos = total as i32;
                    }
                } else {
                    let mut chunk_start = 0usize;
                    while chunk_start < new_total {
                        let chunk_end = (chunk_start + chunk_size).min(new_total);
                        batch.clear();
                        for (i, tok) in to_decode[chunk_start..chunk_end].iter().enumerate() {
                            let global_i = chunk_start + i;
                            let is_last_overall = global_i == new_total - 1;
                            batch
                                .add(*tok, pos + i as i32, &[0], is_last_overall)
                                .map_err(|e| Error::Llm(format!("batch.add: {e}")))?;
                        }
                        ctx.decode(&mut batch)
                            .map_err(|e| Error::Llm(format!("decode prompt chunk: {e}")))?;
                        pos += (chunk_end - chunk_start) as i32;
                        chunk_start = chunk_end;
                    }
                }

                // Sampler:
                //   - temperature == 0 (greedy) のときは temp/top_p を入れず
                //     greedy だけで良い (毎トークン O(vocab) のスケール/ソート
                //     を完全スキップでき体感で 5–10% 速い)。
                //   - temperature > 0 のときだけ温度 + top_p + greedy の
                //     チェーンを組む。
                let mut sampler = if req.temperature <= 0.0 {
                    LlamaSampler::chain_simple([LlamaSampler::greedy()])
                } else {
                    LlamaSampler::chain_simple([
                        LlamaSampler::temp(req.temperature),
                        LlamaSampler::top_p(0.95, 1),
                        LlamaSampler::greedy(),
                    ])
                };
                let mut produced = String::new();
                // produced の何バイトまで cb に渡したかを追跡する。
                // ストップシーケンス先漏れ防止で「保留」した分は次の token 受信時に
                // emit するため、prev_len ではなくこの累積カーソルを使う。
                let mut emitted_len: usize = 0;
                // emit を細かく per-token 呼ぶと Tauri IPC が毎回走るので
                // 「ある程度まとまるか時間が経つまで」バッファして一括 cb する。
                // しきい値は文字数 (バイト) と経過時間。stop / EOG / 終了時は強制 flush。
                const EMIT_FLUSH_BYTES: usize = 24;
                const EMIT_FLUSH_INTERVAL: std::time::Duration =
                    std::time::Duration::from_millis(25);
                let mut last_flush = std::time::Instant::now();
                let mut n_cur: i32 = pos;
                let max_n_cur = (n_ctx as i32).saturating_sub(1);
                for _ in 0..req.max_tokens {
                    // KV cache が満杯になる手前で停止 (これを越えると ggml_abort)
                    if n_cur >= max_n_cur {
                        tracing::warn!(
                            "n_cur ({}) reached n_ctx limit ({}); stopping generation early",
                            n_cur,
                            n_ctx
                        );
                        break;
                    }
                    let next = sampler.sample(&*ctx, batch.n_tokens() - 1);
                    sampler.accept(next);
                    if model.is_eog_token(next) {
                        break;
                    }
                    // token_to_str は deprecated だが、新 API (token_to_piece) は
                    // encoding_rs::Decoder 必須でストリーミング用途には過剰。
                    // 単純なテキスト用途なので deprecated を許容する。
                    #[allow(deprecated)]
                    let piece = model
                        .token_to_str(next, Special::Tokenize)
                        .unwrap_or_default();
                    produced.push_str(&piece);

                    // 停止シーケンス (例 "<end_of_turn>") を含んだら、その手前まで
                    // だけを UI に流して打ち切る。停止トークン自体を画面に出さない。
                    //
                    // 重要: トークナイザがストップシーケンスを複数トークンに分割する
                    // (例: `<` `end` `_` `of` `_` `turn` `>`) ことがあるため、produced の
                    // 末尾が任意のストップシーケンスの **prefix** に一致している間は
                    // その分だけ emit を保留する (= 確定するまで UI に出さない)。
                    let stop_at = prompt
                        .stop_sequences
                        .iter()
                        .filter_map(|s| produced.find(s))
                        .min();
                    let safe_until = match stop_at {
                        Some(idx) => idx,
                        None => {
                            let mut max_partial = 0usize;
                            for s in &prompt.stop_sequences {
                                let max_check = s.len().min(produced.len());
                                for n in (1..=max_check).rev() {
                                    if produced.len() < n {
                                        continue;
                                    }
                                    let tail_start = produced.len() - n;
                                    if !produced.is_char_boundary(tail_start) {
                                        continue;
                                    }
                                    let tail = &produced[tail_start..];
                                    if s.starts_with(tail) {
                                        if n > max_partial {
                                            max_partial = n;
                                        }
                                        break;
                                    }
                                }
                            }
                            produced.len() - max_partial
                        }
                    };
                    let emit_until = safe_until;
                    // バッファ: しきい値 (24 バイト or 25 ms) に達したか、stop_at で
                    // 強制 flush するときのみ cb を呼ぶ。
                    let stop_hit = stop_at.is_some();
                    let pending_bytes = emit_until.saturating_sub(emitted_len);
                    let elapsed = last_flush.elapsed();
                    let should_flush = stop_hit
                        || pending_bytes >= EMIT_FLUSH_BYTES
                        || (pending_bytes > 0 && elapsed >= EMIT_FLUSH_INTERVAL);
                    if should_flush && emit_until > emitted_len {
                        let emit = &produced[emitted_len..emit_until];
                        if !emit.is_empty() {
                            if let Ok(mut g) = cb.lock() {
                                (g)(emit.to_string());
                            }
                        }
                        emitted_len = emit_until;
                        last_flush = std::time::Instant::now();
                    }
                    if stop_hit {
                        break;
                    }
                    batch.clear();
                    batch
                        .add(next, n_cur, &[0], true)
                        .map_err(|e| Error::Llm(format!("batch.add token: {e}")))?;
                    n_cur += 1;
                    ctx.decode(&mut batch)
                        .map_err(|e| Error::Llm(format!("decode tok: {e}")))?;
                }
                // 生成ループが EOG / max_tokens で抜けたとき、emit バッファに
                // 取り残された分を最終 flush する。
                if produced.len() > emitted_len {
                    let tail = &produced[emitted_len..];
                    if !tail.is_empty() {
                        if let Ok(mut g) = cb.lock() {
                            (g)(tail.to_string());
                        }
                    }
                }

                // 次ターンとの prefix 比較用に「今回 decode した prompt」を保存する。
                // 生成 (assistant) 部分は含めない (次ターンの prompt は formatter で
                // 再構成されるため、tokenize 結果と必ずしも一致しない)。
                session.cached_prompt = tokens;

                // 永続化 KV snapshot の save:
                //   - prompt が一定長以上 (= system + RAG prefix が乗っている)
                //     のときだけ保存。短すぎる場合は load コスト > 再 decode に
                //     なって逆効果。
                //   - generate のクリティカルパスから外すために OS スレッドへ
                //     fire-and-forget。spawn された側で session Mutex を再取得
                //     するので、次ターンの開始は save 完了まで待つ可能性があるが、
                //     ユーザに「Done」イベントは即座に返せる (体感の応答時間
                //     が大きく改善)。
                let should_save = kv_snapshot.is_some() && session.cached_prompt.len() >= 200;
                let save_payload: Option<(std::path::PathBuf, Vec<LlamaToken>)> =
                    if should_save {
                        Some((
                            kv_snapshot.clone().expect("checked above"),
                            session.cached_prompt.clone(),
                        ))
                    } else {
                        None
                    };
                drop(session_guard);
                if let Some((snap_path, toks)) = save_payload {
                    let session_arc_save = session_arc.clone();
                    std::thread::Builder::new()
                        .name("ellisii-kv-save".into())
                        .spawn(move || {
                            let started = std::time::Instant::now();
                            let g = match session_arc_save.lock() {
                                Ok(g) => g,
                                Err(_) => return,
                            };
                            if let Some(s) = g.as_ref() {
                                match s.ctx.state_save_file(&snap_path, &toks) {
                                    Ok(()) => tracing::debug!(
                                        "kv snapshot saved (bg): {} tokens in {:.2}s",
                                        toks.len(),
                                        started.elapsed().as_secs_f64()
                                    ),
                                    Err(e) => {
                                        tracing::warn!("kv snapshot save failed: {e}");
                                    }
                                }
                            }
                        })
                        .ok();
                }
                Ok(())
            })
            .await
            .map_err(|e| Error::Llm(format!("join: {e}")))??;
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "gguf")]
    #[test]
    fn parse_kv_type_uses_tier_default() {
        use llama_cpp_2::context::params::KvCacheType;
        assert_eq!(parse_kv_type(None, false, "f16"), KvCacheType::F16);
        assert_eq!(parse_kv_type(None, false, "q8_0"), KvCacheType::Q8_0);
        assert_eq!(parse_kv_type(None, false, "q4_0"), KvCacheType::Q4_0);
    }

    #[cfg(feature = "gguf")]
    #[test]
    fn parse_kv_type_low_spec_forces_q4_0_regardless_of_tier() {
        use llama_cpp_2::context::params::KvCacheType;
        assert_eq!(parse_kv_type(None, true, "f16"), KvCacheType::Q4_0);
        assert_eq!(parse_kv_type(None, true, "q8_0"), KvCacheType::Q4_0);
    }

    #[cfg(feature = "gguf")]
    #[test]
    fn parse_kv_type_env_overrides_tier_and_low_spec() {
        use llama_cpp_2::context::params::KvCacheType;
        // env=f16 は tier 低 + low_spec を override
        assert_eq!(parse_kv_type(Some("f16"), true, "q4_0"), KvCacheType::F16);
        assert_eq!(parse_kv_type(Some("Q8_0"), false, "f16"), KvCacheType::Q8_0);
    }

    #[cfg(feature = "gguf")]
    #[test]
    fn parse_kv_type_invalid_env_falls_through() {
        use llama_cpp_2::context::params::KvCacheType;
        // 不正な env → tier_default に倒れる (low_spec=false なら tier そのまま)
        assert_eq!(
            parse_kv_type(Some("garbage"), false, "q8_0"),
            KvCacheType::Q8_0
        );
        // 不正な env + low_spec=true → low_spec で Q4_0
        assert_eq!(
            parse_kv_type(Some("garbage"), true, "f16"),
            KvCacheType::Q4_0
        );
    }

    #[test]
    fn resolve_low_spec_env_explicit_on_wins() {
        // env=1/true/yes/on → ON、ハードウェアに関わらず採用
        assert!(resolve_low_spec(Some("1"), false, 64.0));
        assert!(resolve_low_spec(Some("true"), false, 64.0));
        assert!(resolve_low_spec(Some("YES"), true, 32.0));
        assert!(resolve_low_spec(Some(" on "), false, 16.0));
    }

    #[test]
    fn resolve_low_spec_env_explicit_off_wins() {
        // env=0/false/no/off → OFF、低 RAM CPU でも採用しない
        assert!(!resolve_low_spec(Some("0"), true, 4.0));
        assert!(!resolve_low_spec(Some("false"), true, 8.0));
        assert!(!resolve_low_spec(Some("no"), true, 6.0));
    }

    #[test]
    fn resolve_low_spec_invalid_env_falls_through_to_auto() {
        // 不正値は無視 → 自動判定
        assert!(resolve_low_spec(Some("garbage"), true, 4.0));
        assert!(!resolve_low_spec(Some("garbage"), true, 16.0));
        assert!(!resolve_low_spec(Some(""), false, 4.0));
    }

    #[test]
    fn resolve_low_spec_auto_triggers_on_low_ram_cpu_only() {
        // CPU かつ 8GiB 以下 → ON
        assert!(resolve_low_spec(None, true, 4.0));
        assert!(resolve_low_spec(None, true, 8.0));
        // 8GiB 超 → OFF
        assert!(!resolve_low_spec(None, true, 8.1));
        assert!(!resolve_low_spec(None, true, 16.0));
        // GPU offload や Apple Silicon Unified では auto しない
        assert!(!resolve_low_spec(None, false, 4.0));
        assert!(!resolve_low_spec(None, false, 8.0));
    }

    fn cfg(model_path: &str, family: ModelFamily) -> LlamaConfig {
        LlamaConfig {
            model_path: std::path::PathBuf::from(model_path),
            family,
            n_ctx: 4096,
            n_batch: 512,
            n_gpu_layers: 0,
            n_threads: None,
            n_threads_batch: None,
            seed: 0,
            cache_dir: None,
            kv_type_hint: "q8_0".into(),
        }
    }

    #[test]
    fn fingerprint_is_stable_for_same_config() {
        let a = compute_kv_fingerprint(&cfg("/tmp/x.gguf", ModelFamily::Gemma4));
        let b = compute_kv_fingerprint(&cfg("/tmp/x.gguf", ModelFamily::Gemma4));
        assert_eq!(a, b);
    }

    #[test]
    fn fingerprint_changes_with_model_path() {
        let a = compute_kv_fingerprint(&cfg("/tmp/a.gguf", ModelFamily::Gemma4));
        let b = compute_kv_fingerprint(&cfg("/tmp/b.gguf", ModelFamily::Gemma4));
        assert_ne!(a, b);
    }

    #[test]
    fn fingerprint_changes_with_family() {
        let a = compute_kv_fingerprint(&cfg("/tmp/x.gguf", ModelFamily::Gemma4));
        let b = compute_kv_fingerprint(&cfg("/tmp/x.gguf", ModelFamily::Qwen));
        assert_ne!(
            a, b,
            "different tokenizer family must invalidate the snapshot"
        );
    }

    #[test]
    fn fingerprint_changes_with_n_gpu_layers() {
        let mut a = cfg("/tmp/x.gguf", ModelFamily::Gemma4);
        let mut b = cfg("/tmp/x.gguf", ModelFamily::Gemma4);
        a.n_gpu_layers = 0;
        b.n_gpu_layers = 99;
        assert_ne!(compute_kv_fingerprint(&a), compute_kv_fingerprint(&b));
    }

    #[test]
    fn fingerprint_changes_with_n_ctx_and_n_batch() {
        let a = cfg("/tmp/x.gguf", ModelFamily::Gemma4);
        let mut b = cfg("/tmp/x.gguf", ModelFamily::Gemma4);
        b.n_ctx = a.n_ctx * 2;
        assert_ne!(compute_kv_fingerprint(&a), compute_kv_fingerprint(&b));
        b.n_ctx = a.n_ctx;
        b.n_batch = a.n_batch * 2;
        assert_ne!(compute_kv_fingerprint(&a), compute_kv_fingerprint(&b));
    }

    #[test]
    fn fingerprint_reflects_model_file_changes() {
        let dir = std::env::temp_dir().join("ellisii_kv_fp_test");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("model.gguf");
        std::fs::write(&path, b"v1").expect("write v1");
        let mut c = cfg("", ModelFamily::Gemma4);
        c.model_path = path.clone();
        let fp1 = compute_kv_fingerprint(&c);
        // ファイル内容が変われば size か mtime のどちらかが変わる前提。
        // mtime の解像度を考慮し短い間で別サイズに書き換える。
        std::fs::write(&path, b"v2-different-size").expect("write v2");
        let fp2 = compute_kv_fingerprint(&c);
        assert_ne!(fp1, fp2, "model file content change must invalidate fp");
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn fingerprint_falls_back_when_model_missing() {
        // ファイルが無くてもクラッシュしない (size=0, mtime=0 で計算継続)。
        let c = cfg("/nonexistent/path/never.gguf", ModelFamily::Gemma4);
        let fp = compute_kv_fingerprint(&c);
        assert_eq!(fp.len(), 16);
    }
}
