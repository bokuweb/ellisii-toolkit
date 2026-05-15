//! CPU トポロジ検出 (主に Intel ハイブリッドコア対策)。
//!
//! 12〜14 世代 Intel Core / Meteor Lake 以降は P コア (Performance) と
//! E コア (Efficient) が混在しており、llama.cpp の既定スレッド数で
//! E コアにスレッドが撒かれると per-token decode が大きく遅くなる
//! (実測で 20〜50% 遅化)。
//!
//! このモジュールは「P コア数」を best-effort で返す。検出できない
//! プラットフォームや、同一効率クラスのみのマシン (= ホモジニアス、
//! P/E の区別が無い) では `None` を返し、呼び出し側は llama.cpp の
//! 既定 (= 物理コア数) に倒す。

/// 1 物理コアの効率クラス情報。
///
/// `efficiency_class` は Windows の `GetLogicalProcessorInformationEx`
/// (`PROCESSOR_RELATIONSHIP::EfficiencyClass`) と同じ意味で、値が
/// 大きいほど高性能 (= Pコア)。同値しか無いマシンはホモジニアスと
/// 判定する。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CoreInfo {
    pub efficiency_class: u8,
}

/// `cores` のうち最大効率クラスを持つ物理コア数を返す。
///
/// - 空入力 → `None`
/// - 全コアが同一効率クラス (ホモジニアス CPU) → `None`
///   (P/E 区別が無いので呼び出し側は既存ロジックに倒す)
/// - それ以外 → 最大効率クラス = P コア数を返す
pub fn count_performance_cores(cores: &[CoreInfo]) -> Option<u32> {
    if cores.is_empty() {
        return None;
    }
    let max_class = cores.iter().map(|c| c.efficiency_class).max()?;
    let min_class = cores.iter().map(|c| c.efficiency_class).min()?;
    if max_class == min_class {
        return None;
    }
    Some(
        cores
            .iter()
            .filter(|c| c.efficiency_class == max_class)
            .count() as u32,
    )
}

/// `n_threads` の最終値を解決する純関数。`LlamaConfig::new` から呼ばれる。
///
/// 優先順:
/// 1. `env_value` を i32 にパースできればそれを採用 (= ユーザ明示の override)
/// 2. それ以外で `is_cpu_mode == true` かつ `detected_pcores = Some(n)` なら
///    n を採用 (Intel ハイブリッドの E コア撒き散らしを防ぐ)
/// 3. それ以外は `None` (llama.cpp の既定 = 物理コア数に倒す)
///
/// `is_cpu_mode = false` (= GPU offload) のときは P-core 検出値を使わない。
/// GPU 推論ではホスト側 CPU スレッドはオーケストレーション用なので、E コアに
/// 撒かれても per-token decode への影響は CPU 専用時に比べて小さい。
pub fn resolve_n_threads(
    env_value: Option<&str>,
    is_cpu_mode: bool,
    detected_pcores: Option<u32>,
) -> Option<i32> {
    if let Some(s) = env_value {
        if let Ok(v) = s.trim().parse::<i32>() {
            if v > 0 {
                return Some(v);
            }
        }
    }
    if is_cpu_mode {
        if let Some(n) = detected_pcores {
            if n > 0 {
                return Some(n as i32);
            }
        }
    }
    None
}

/// `cores` の総物理コア数を返す (= P コア + E コアの合計)。
///
/// ハイブリッド CPU (P/E 混在) でのみ `Some` を返す。ホモジニアスな
/// CPU では `None` を返し、呼び出し側は llama.cpp 既定 (= 物理コア数
/// 自動検出) に倒す。
///
/// 用途は `n_threads_batch` (= prompt processing 時のスレッド数)。
/// matmul-heavy な PP 段ではコア種別を問わず多いほどスループットが
/// 出るので、E コアも含めた全物理コアを使うのが速い。一方で per-token
/// decode は KV cache の bandwidth 律速で、E コアを足すと逆効果。
pub fn count_total_physical_cores(cores: &[CoreInfo]) -> Option<u32> {
    if cores.is_empty() {
        return None;
    }
    let max_class = cores.iter().map(|c| c.efficiency_class).max()?;
    let min_class = cores.iter().map(|c| c.efficiency_class).min()?;
    if max_class == min_class {
        return None;
    }
    Some(cores.len() as u32)
}

/// `n_threads_batch` (prompt processing 用) を解決する純関数。
///
/// 優先順:
/// 1. `env_value` を i32 にパースできればそれを採用
/// 2. CPU モードかつ `detected_total = Some(n)` なら n
///    (ハイブリッド CPU で E コアも含めた全物理コアを PP に投入)
/// 3. それ以外は `None` (llama.cpp の既定に倒す)
pub fn resolve_n_threads_batch(
    env_value: Option<&str>,
    is_cpu_mode: bool,
    detected_total: Option<u32>,
) -> Option<i32> {
    if let Some(s) = env_value {
        if let Ok(v) = s.trim().parse::<i32>() {
            if v > 0 {
                return Some(v);
            }
        }
    }
    if is_cpu_mode {
        if let Some(n) = detected_total {
            if n > 0 {
                return Some(n as i32);
            }
        }
    }
    None
}

/// 実環境の総物理コア数 (P + E) を best-effort で取得する。
///
/// ハイブリッド CPU 限定で値を返す (= ホモジニアスでは `None`)。
pub fn detect_total_physical_core_count() -> Option<u32> {
    let cores = platform::detect_cores()?;
    count_total_physical_cores(&cores)
}

/// 実環境の P コア数を best-effort で取得する。
///
/// - Windows: `GetLogicalProcessorInformationEx` でコア毎の
///   EfficiencyClass を取得し、最大値を持つコア数を返す
/// - それ以外: `None` (Linux/Mac の hybrid 検出は将来の拡張)
pub fn detect_performance_core_count() -> Option<u32> {
    let cores = platform::detect_cores()?;
    count_performance_cores(&cores)
}

#[cfg(target_os = "windows")]
mod platform {
    use super::CoreInfo;
    use windows_sys::Win32::System::SystemInformation::{
        GetLogicalProcessorInformationEx, RelationProcessorCore,
        SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX,
    };

    /// `GetLogicalProcessorInformationEx(RelationProcessorCore)` を呼んで、
    /// 各物理コアの EfficiencyClass を集めて返す。
    ///
    /// API は可変長レコード列を返すので、まず長さ取得の失敗呼び出しで
    /// バッファサイズを得て、その後本呼び出しを行う典型パターン。
    /// 失敗時は `None` を返し、呼び出し側は llama.cpp 既定に倒す。
    pub fn detect_cores() -> Option<Vec<CoreInfo>> {
        unsafe {
            let mut len: u32 = 0;
            // 1 回目: バッファサイズ取得 (Buffer=null で必ず失敗するので戻り値は無視)
            let _ = GetLogicalProcessorInformationEx(
                RelationProcessorCore,
                std::ptr::null_mut(),
                &mut len,
            );
            if len == 0 {
                return None;
            }
            // OS が要求する長さでバッファ確保
            let mut buf: Vec<u8> = vec![0u8; len as usize];
            let ok = GetLogicalProcessorInformationEx(
                RelationProcessorCore,
                buf.as_mut_ptr() as *mut SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX,
                &mut len,
            );
            if ok == 0 {
                return None;
            }

            // 可変長レコードを `Size` フィールドで進めながら走査する。
            let mut cores: Vec<CoreInfo> = Vec::new();
            let mut offset: usize = 0;
            while offset + std::mem::size_of::<SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>()
                <= buf.len()
            {
                let info_ptr =
                    buf.as_ptr().add(offset) as *const SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX;
                let info = &*info_ptr;
                if info.Relationship == RelationProcessorCore {
                    // union の Processor バリアント (PROCESSOR_RELATIONSHIP) を読む。
                    // RelationProcessorCore で取得しているので Processor が有効。
                    let proc_rel = info.Anonymous.Processor;
                    cores.push(CoreInfo {
                        efficiency_class: proc_rel.EfficiencyClass,
                    });
                }
                let size = info.Size as usize;
                if size == 0 {
                    // 0 進行は無限ループになるので保険で打ち切る
                    break;
                }
                offset += size;
            }
            if cores.is_empty() {
                None
            } else {
                Some(cores)
            }
        }
    }
}

#[cfg(not(target_os = "windows"))]
mod platform {
    use super::CoreInfo;
    pub fn detect_cores() -> Option<Vec<CoreInfo>> {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cores(classes: &[u8]) -> Vec<CoreInfo> {
        classes
            .iter()
            .map(|&c| CoreInfo {
                efficiency_class: c,
            })
            .collect()
    }

    #[test]
    fn empty_input_returns_none() {
        assert_eq!(count_performance_cores(&[]), None);
    }

    #[test]
    fn homogeneous_cpu_returns_none() {
        // 全コア class=0 (例: 旧世代 Intel / AMD Ryzen): P/E 区別が無い
        assert_eq!(count_performance_cores(&cores(&[0, 0, 0, 0, 0, 0, 0, 0])), None);
        // 全コア class=1 でも同様
        assert_eq!(count_performance_cores(&cores(&[1, 1, 1, 1])), None);
    }

    #[test]
    fn alder_lake_12700k_topology_returns_8_pcores() {
        // 12700K: P コア 8 (class=1) + E コア 4 (class=0)
        let topology = cores(&[1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0]);
        assert_eq!(count_performance_cores(&topology), Some(8));
    }

    #[test]
    fn raptor_lake_13900k_topology_returns_8_pcores() {
        // 13900K: P コア 8 (class=1) + E コア 16 (class=0)
        let mut topology = vec![CoreInfo { efficiency_class: 1 }; 8];
        topology.extend(vec![CoreInfo { efficiency_class: 0 }; 16]);
        assert_eq!(count_performance_cores(&topology), Some(8));
    }

    #[test]
    fn meteor_lake_three_tier_returns_top_class_count() {
        // Meteor Lake は P (class=2) + E (class=1) + LP-E (class=0) の 3 段。
        // 最大クラス = P コアのみをカウントする。
        let topology = cores(&[2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0]);
        assert_eq!(count_performance_cores(&topology), Some(6));
    }

    #[test]
    fn single_core_pcpu_returns_none() {
        // 1 コアだけならホモジニアス扱い → None
        assert_eq!(count_performance_cores(&cores(&[1])), None);
    }

    #[test]
    fn resolve_env_override_always_wins() {
        // env=8 はモード/検出値に関わらず採用される
        assert_eq!(resolve_n_threads(Some("8"), true, Some(4)), Some(8));
        assert_eq!(resolve_n_threads(Some("8"), false, None), Some(8));
        assert_eq!(resolve_n_threads(Some(" 12 "), true, Some(8)), Some(12));
    }

    #[test]
    fn resolve_invalid_env_falls_through() {
        // パース不能 / 0 / 負値の env は無視 → 後段ロジックへ
        assert_eq!(resolve_n_threads(Some("garbage"), true, Some(8)), Some(8));
        assert_eq!(resolve_n_threads(Some("0"), true, Some(8)), Some(8));
        assert_eq!(resolve_n_threads(Some("-1"), true, Some(8)), Some(8));
        assert_eq!(resolve_n_threads(Some(""), true, Some(8)), Some(8));
    }

    #[test]
    fn resolve_uses_pcore_count_in_cpu_mode() {
        // CPU mode + 検出値あり → P-core 数で pin
        assert_eq!(resolve_n_threads(None, true, Some(8)), Some(8));
        assert_eq!(resolve_n_threads(None, true, Some(6)), Some(6));
    }

    #[test]
    fn resolve_skips_pcore_pinning_in_gpu_mode() {
        // GPU mode では検出値があっても採用しない (None で llama.cpp 既定)
        assert_eq!(resolve_n_threads(None, false, Some(8)), None);
    }

    #[test]
    fn resolve_falls_back_to_none_when_no_signal() {
        // 何の手がかりも無ければ None (= llama.cpp 既定 = 物理コア数)
        assert_eq!(resolve_n_threads(None, true, None), None);
        assert_eq!(resolve_n_threads(None, false, None), None);
    }

    #[test]
    fn total_physical_returns_full_count_on_hybrid() {
        // 12700K: P 8 + E 4 = 12 物理コア
        let topology = cores(&[1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0]);
        assert_eq!(count_total_physical_cores(&topology), Some(12));
    }

    #[test]
    fn total_physical_returns_none_on_homogeneous() {
        // 旧 Intel / AMD Ryzen のホモジニアス CPU は None
        // (llama.cpp 既定に倒す)
        assert_eq!(count_total_physical_cores(&cores(&[0, 0, 0, 0, 0, 0, 0, 0])), None);
        assert_eq!(count_total_physical_cores(&cores(&[1, 1, 1, 1])), None);
    }

    #[test]
    fn total_physical_returns_none_on_empty() {
        assert_eq!(count_total_physical_cores(&[]), None);
    }

    #[test]
    fn total_physical_includes_all_tiers_on_three_class_cpu() {
        // Meteor Lake (P + E + LP-E) でも全コアをカウント (PP は全部投入)
        let topology = cores(&[2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0]);
        assert_eq!(count_total_physical_cores(&topology), Some(16));
    }

    #[test]
    fn resolve_batch_env_override_wins() {
        assert_eq!(resolve_n_threads_batch(Some("16"), true, Some(12)), Some(16));
        assert_eq!(resolve_n_threads_batch(Some("16"), false, None), Some(16));
    }

    #[test]
    fn resolve_batch_invalid_env_falls_through() {
        assert_eq!(resolve_n_threads_batch(Some("garbage"), true, Some(12)), Some(12));
        assert_eq!(resolve_n_threads_batch(Some("0"), true, Some(12)), Some(12));
    }

    #[test]
    fn resolve_batch_uses_total_physical_in_cpu_mode() {
        assert_eq!(resolve_n_threads_batch(None, true, Some(12)), Some(12));
    }

    #[test]
    fn resolve_batch_skips_in_gpu_mode() {
        assert_eq!(resolve_n_threads_batch(None, false, Some(12)), None);
    }

    #[test]
    fn resolve_batch_falls_back_to_none_when_no_signal() {
        assert_eq!(resolve_n_threads_batch(None, true, None), None);
    }

    #[test]
    fn detect_returns_none_on_non_windows_or_unsupported() {
        // ビルドプラットフォームに依存するスモークテスト。
        // 現状の platform::detect_cores 実装は常に None を返すので、
        // この関数も None。CI で固定 (Windows 実装が入ったら更新)。
        assert_eq!(detect_performance_core_count(), None);
    }
}
