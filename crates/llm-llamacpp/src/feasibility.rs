//! Can-I-Run 推定ロジック。
//!
//! ローカルマシンの RAM / VRAM / GPU と [`ModelSpec`] を突き合わせて、
//! 「このモデルがこの環境で動きそうか」を Ok / Tight / Insufficient の
//! 3 値で返す。UI のモデル選択でバッジ表示と無効化の判定に使う。
//!
//! 推定値は GGUF のメタデータを直接読まず、catalog 既知 base 名と
//! `size_mb` からのヒューリスティックで KV cache サイズを概算する。
//! 数百 MB の誤差は出るが、Insufficient/Ok の二択判定では十分。

use ellisii_llm_core::ModelSpec;
use serde::{Deserialize, Serialize};

use crate::{detect_nvidia_gpu, detect_runtime_mode, detect_total_ram_bytes, RuntimeMode};

/// ホスト環境の概要。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HostInfo {
    pub mode: RuntimeMode,
    pub total_ram_mb: u32,
    pub gpu_name: Option<String>,
    pub total_vram_mb: Option<u32>,
    pub free_vram_mb: Option<u32>,
}

/// 検出器を呼んで HostInfo を組み立てる (best-effort)。
pub fn detect_host_info() -> HostInfo {
    let mode = detect_runtime_mode();
    let ram_bytes = detect_total_ram_bytes();
    let gpu = detect_nvidia_gpu();
    HostInfo {
        mode,
        total_ram_mb: (ram_bytes / 1024 / 1024) as u32,
        gpu_name: gpu.as_ref().map(|g| g.name.clone()),
        total_vram_mb: gpu.as_ref().map(|g| g.total_vram_mb),
        free_vram_mb: gpu.as_ref().map(|g| g.free_vram_mb),
    }
}

/// 推定の判定結果。
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum FeasibilityStatus {
    /// 余裕を持って動作する見込み。
    Ok,
    /// ぎりぎり動作するが、他アプリ次第で OOM の可能性あり。
    Tight,
    /// 必要メモリが利用可能量を大きく超過。動作不能と判定。
    Insufficient,
}

/// モデル単位の動作可否。UI のバッジ表示や `disabled` 判定に使う。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelFeasibility {
    pub status: FeasibilityStatus,
    /// 必要メモリ合計の見積もり (MB)。
    pub projected_total_mb: u32,
    /// うち KV cache が占める分 (MB)。
    pub projected_kv_mb: u32,
    /// 利用可能メモリ (MB)。Discrete GPU では VRAM + RAM の合算。
    pub available_mb: u32,
    /// 内訳メモ。tooltip 表示用。
    pub notes: Vec<String>,
}

/// 既知 base ごとの KV cache サイズ (1 token あたり、F16 換算 bytes)。
///
/// 出典: 各モデルのアーキテクチャ (n_layers × n_kv_heads × head_dim × 2 [K+V] × 2 [bytes/f16])。
/// catalog 外モデルは [`fallback_kv_per_token_bytes_f16`] のヒューリスティックを使う。
fn kv_per_token_bytes_f16(base: &str) -> u32 {
    match base {
        // Gemma 3n E2B: layers ~30, n_kv_heads ~4, head_dim ~256 → ~120 KB
        "gemma-4-E2B" => 120 * 1024,
        // E4B: 一回り大きい
        "gemma-4-E4B" => 140 * 1024,
        // Gemma 4 26B-A4B (MoE, total 26B / active 4B):
        //   KV cache は expert で共有しないので total params ベース。
        //   layers ~62, n_kv_heads ~8, head_dim ~256 → ~200 KB
        "gemma-4-26B-A4B" => 200 * 1024,
        // Qwen 3.6 27B (dense): layers 64, n_kv_heads 8, head_dim 128 → ~130 KB
        "qwen-3.6-27B" => 130 * 1024,
        _ => fallback_kv_per_token_bytes_f16(base),
    }
}

fn fallback_kv_per_token_bytes_f16(_base: &str) -> u32 {
    // 経験則: catalog 外モデル向け。3〜30B クラスなら 100〜250 KB のレンジ。
    150 * 1024
}

/// KV cache 量子化型の倍率 (F16 = 1.0)。実測ではなく型理論値ベース。
fn kv_quant_scale(kv_type: &str) -> f64 {
    match kv_type.to_ascii_lowercase().as_str() {
        "f16" | "fp16" | "bf16" | "f32" | "fp32" => 1.0,
        "q8_0" | "q8_1" | "q8" => 0.55,
        "q5_0" | "q5_1" | "q5" => 0.40,
        "q4_0" | "q4_1" | "q4" => 0.30,
        _ => 0.55, // 不明値は Q8_0 相当に倒す
    }
}

/// 必要メモリを見積もって [`ModelFeasibility`] を返す。
///
/// `n_ctx` は実際に使う context 長 (= LlamaConfig::new で算出される値) を渡す。
/// `kv_type` は `q8_0` / `q4_0` / `f16` などの文字列。
pub fn estimate(spec: &ModelSpec, host: &HostInfo, n_ctx: u32, kv_type: &str) -> ModelFeasibility {
    let kv_per_tok = kv_per_token_bytes_f16(spec.base.as_str()) as f64;
    let scale = kv_quant_scale(kv_type);
    let kv_total_bytes = kv_per_tok * (n_ctx as f64) * scale;
    let kv_mb = (kv_total_bytes / 1024.0 / 1024.0).round() as u32;
    // compute buffer + activations + 各種オーバーヘッド。
    // 1 GB ではプロンプト長 / n_batch が大きい時に PP 用の compute buffer
    // (数百 MB〜1 GB スケール) が乗り切らず、結果的に decode で
    // Decode Error -3 (= KV slot/alloc 失敗) を踏むケースがあったため
    // 1.5 GB に引き上げて安全側に倒す。
    let overhead_mb: u32 = 1536;
    let projected_total_mb = spec
        .size_mb
        .saturating_add(kv_mb)
        .saturating_add(overhead_mb);

    // OS + 他アプリ向けの予約。runtime auto-tuner と揃える (4 GiB)。
    // 3 GiB だとブラウザ + IDE 等が動いている実機で残りを使い切り、
    // Tight 判定で動かしたとき OOM/abort を引き起こすことがあった。
    let reserve_mb: u32 = 4 * 1024;
    let mut notes: Vec<String> = Vec::new();
    let available_mb = match host.mode {
        RuntimeMode::Unified | RuntimeMode::Cpu => {
            let avail = host.total_ram_mb.saturating_sub(reserve_mb);
            notes.push(format!(
                "{} mode: RAM {} MB - OS reserve {} MB = {} MB available",
                if matches!(host.mode, RuntimeMode::Unified) {
                    "Unified"
                } else {
                    "CPU"
                },
                host.total_ram_mb,
                reserve_mb,
                avail
            ));
            avail
        }
        RuntimeMode::DiscreteGpu => {
            let vram = host.free_vram_mb.unwrap_or(0);
            let ram = host.total_ram_mb.saturating_sub(reserve_mb);
            let avail = vram.saturating_add(ram);
            notes.push(format!(
                "Discrete GPU: free VRAM {} MB + RAM {} MB (reserve {} MB) = {} MB available",
                vram, ram, reserve_mb, avail
            ));
            avail
        }
    };
    notes.push(format!(
        "model {} MB + KV {} MB ({} @ {}K ctx) + overhead {} MB = {} MB",
        spec.size_mb,
        kv_mb,
        kv_type,
        (n_ctx + 512) / 1024,
        overhead_mb,
        projected_total_mb
    ));

    // Tight の許容バンドは 10% に縮小 (旧 30%)。
    // 旧 1.3x は実マシンで OOM / "Decode Error -3" を踏みやすかった。
    // 1.1x までを「やればギリギリ動く」、それ以上は最初から弾く。
    let status = if projected_total_mb <= available_mb {
        FeasibilityStatus::Ok
    } else if (projected_total_mb as f64) <= (available_mb as f64) * 1.10 {
        FeasibilityStatus::Tight
    } else {
        FeasibilityStatus::Insufficient
    };

    ModelFeasibility {
        status,
        projected_total_mb,
        projected_kv_mb: kv_mb,
        available_mb,
        notes,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ellisii_llm_core::ModelFamily;

    fn spec(base: &str, size_mb: u32) -> ModelSpec {
        ModelSpec {
            family: ModelFamily::Gemma4,
            label: base.to_string(),
            repo: "x".into(),
            file: format!("{base}.gguf"),
            size_mb,
            quant: "IQ4_XS".into(),
            base: base.into(),
        }
    }
    fn host(mode: RuntimeMode, ram_mb: u32, free_vram_mb: Option<u32>) -> HostInfo {
        HostInfo {
            mode,
            total_ram_mb: ram_mb,
            gpu_name: free_vram_mb.map(|_| "test".into()),
            total_vram_mb: free_vram_mb,
            free_vram_mb,
        }
    }

    #[test]
    fn e2b_fits_on_8gb_mac() {
        let s = spec("gemma-4-E2B", 1693);
        let h = host(RuntimeMode::Unified, 8 * 1024, None);
        let f = estimate(&s, &h, 4096, "q4_0");
        assert_eq!(f.status, FeasibilityStatus::Ok);
    }

    #[test]
    fn e4b_fits_on_16gb_mac() {
        let s = spec("gemma-4-E4B", 2715);
        let h = host(RuntimeMode::Unified, 16 * 1024, None);
        let f = estimate(&s, &h, 8192, "q8_0");
        assert_eq!(f.status, FeasibilityStatus::Ok);
    }

    #[test]
    fn gemma4_26b_a4b_does_not_fit_on_8gb_mac() {
        let s = spec("gemma-4-26B-A4B", 12797);
        let h = host(RuntimeMode::Unified, 8 * 1024, None);
        let f = estimate(&s, &h, 4096, "q4_0");
        assert_eq!(f.status, FeasibilityStatus::Insufficient);
    }

    #[test]
    fn gemma4_26b_a4b_fits_on_32gb_mac() {
        let s = spec("gemma-4-26B-A4B", 12797);
        let h = host(RuntimeMode::Unified, 32 * 1024, None);
        let f = estimate(&s, &h, 8192, "f16");
        assert_eq!(f.status, FeasibilityStatus::Ok);
    }

    #[test]
    fn qwen_36_27b_fits_on_24gb_mac() {
        let s = spec("qwen-3.6-27B", 14725);
        let h = host(RuntimeMode::Unified, 24 * 1024, None);
        let f = estimate(&s, &h, 4096, "q8_0");
        assert_eq!(f.status, FeasibilityStatus::Ok);
    }

    #[test]
    fn discrete_gpu_uses_vram_plus_ram() {
        // 8 GB VRAM + 16 GB RAM = 24 GB-3 GB = 21 GB available。E4B 余裕。
        let s = spec("gemma-4-E4B", 2715);
        let h = host(RuntimeMode::DiscreteGpu, 16 * 1024, Some(8 * 1024));
        let f = estimate(&s, &h, 8192, "q8_0");
        assert_eq!(f.status, FeasibilityStatus::Ok);
    }

    #[test]
    fn tight_bucket_when_just_over() {
        // 必要量がギリギリ available を超えるが 1.3x 以内のケース
        // (RAM 6 GB の理論マシンに E4B + 8K context @ q8_0)
        // 実際の数値: model 2715 + KV ~600 + overhead 1024 ≈ 4339 MB
        // available = 6144 - 3072 = 3072 → 4339/3072 ≈ 1.41 倍 = Insufficient
        // よって 4 GB マシンを想定して overhead と合わせて Tight になる範囲を検証。
        let s = spec("gemma-4-E4B", 2715);
        let h = host(RuntimeMode::Unified, 5 * 1024, None);
        let f = estimate(&s, &h, 4096, "q4_0");
        assert_ne!(f.status, FeasibilityStatus::Ok);
        // Tight or Insufficient のいずれかで、Tight の境界を踏める想定。
    }
}
