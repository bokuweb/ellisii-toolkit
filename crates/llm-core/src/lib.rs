use async_trait::async_trait;
use ellisii_core::Result;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub enum ModelFamily {
    /// Google Gemma 4 (`<start_of_turn>` 形式 — Gemma 3 と互換)
    Gemma4,
    /// ChatML (`<|im_start|>`) 系。Qwen 2.5 / Qwen 3 共通
    Qwen,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelSpec {
    pub family: ModelFamily,
    pub label: String,
    pub repo: String,
    pub file: String,
    pub size_mb: u32,
    /// 例: "Q4_K_M"
    pub quant: String,
    /// 同 family + 同サイズで quant 違いをまとめるためのキー (例 "gemma-4-E2B")
    pub base: String,
}

pub fn default_catalog() -> Vec<ModelSpec> {
    // すべて IQ4_XS で統一。Q4_K_M とほぼ同等品質のまま:
    //   - サイズ ~40% 小さい (例: E2B 2.96GB → 1.69GB)
    //   - Apple Silicon Metal で per-token 生成が 1.2-1.3× 速い
    //   - Q3 系より精度劣化が小さく、KV cache の余裕を取りやすい
    //
    // E2B → E4B → 27B の順 (軽い → 重い)。27B は要 ~16GB+ メモリ。
    vec![
        ModelSpec {
            family: ModelFamily::Gemma4,
            label: "Gemma 4 E2B (IQ4_XS)".into(),
            repo: "unsloth/gemma-4-E2B-it-GGUF".into(),
            file: "gemma-4-E2B-it-IQ4_XS.gguf".into(),
            size_mb: 1693,
            quant: "IQ4_XS".into(),
            base: "gemma-4-E2B".into(),
        },
        ModelSpec {
            family: ModelFamily::Gemma4,
            label: "Gemma 4 E4B (IQ4_XS)".into(),
            repo: "unsloth/gemma-4-E4B-it-GGUF".into(),
            file: "gemma-4-E4B-it-IQ4_XS.gguf".into(),
            size_mb: 2715,
            quant: "IQ4_XS".into(),
            base: "gemma-4-E4B".into(),
        },
        // 大型モデル枠。E2B/E4B と違い「効率的アーキテクチャ」ではないので
        // ストレートに 27B〜30B 規模。IQ4_XS で 14〜16GB、推論時 KV cache を
        // 含めると 24GB+ メモリが現実的ライン。
        //
        // Gemma 4 26B-A4B: MoE 構造で active 4B。dense 31B より per-token
        // decode が速く Apple Silicon に向く。chat テンプレは <start_of_turn>。
        // Qwen 3.6 27B: dense 27B。日本語含む多言語推論で安定。
        ModelSpec {
            family: ModelFamily::Gemma4,
            label: "Gemma 4 26B-A4B (IQ4_XS, MoE)".into(),
            repo: "unsloth/gemma-4-26B-A4B-it-GGUF".into(),
            file: "gemma-4-26B-A4B-it-UD-IQ4_XS.gguf".into(),
            size_mb: 12797,
            quant: "IQ4_XS".into(),
            base: "gemma-4-26B-A4B".into(),
        },
        ModelSpec {
            family: ModelFamily::Qwen,
            label: "Qwen 3.6 27B (IQ4_XS)".into(),
            repo: "unsloth/Qwen3.6-27B-GGUF".into(),
            file: "Qwen3.6-27B-IQ4_XS.gguf".into(),
            size_mb: 14725,
            quant: "IQ4_XS".into(),
            base: "qwen-3.6-27B".into(),
        },
    ]
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ChatRole {
    User,
    Assistant,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatTurn {
    pub role: ChatRole,
    pub content: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmRequest {
    pub system: String,
    /// 過去ターン (古い順)。`user` と交互の対が想定。空なら単発質問。
    #[serde(default)]
    pub history: Vec<ChatTurn>,
    pub user: String,
    pub max_tokens: u32,
    pub temperature: f32,
}

#[async_trait]
pub trait LlmBackend: Send + Sync {
    async fn generate_stream(
        &self,
        req: LlmRequest,
        on_token: Box<dyn FnMut(String) + Send + 'static>,
    ) -> Result<()>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_catalog_is_non_empty_and_well_formed() {
        let catalog = default_catalog();
        assert!(!catalog.is_empty());
        for spec in &catalog {
            assert!(!spec.label.is_empty(), "label empty: {:?}", spec);
            assert!(spec.repo.contains('/'), "repo not HF style: {}", spec.repo);
            assert!(spec.file.ends_with(".gguf"), "file not gguf: {}", spec.file);
            assert!(spec.size_mb > 0);
            assert!(!spec.quant.is_empty());
            assert!(!spec.base.is_empty());
        }
    }

    #[test]
    fn default_catalog_bases_are_unique() {
        let catalog = default_catalog();
        let mut bases: Vec<&str> = catalog.iter().map(|s| s.base.as_str()).collect();
        bases.sort();
        let dedup_len = {
            let mut b = bases.clone();
            b.dedup();
            b.len()
        };
        assert_eq!(bases.len(), dedup_len, "duplicate base in catalog");
    }

    #[test]
    fn model_family_serde_kebab() {
        assert_eq!(
            serde_json::to_string(&ModelFamily::Gemma4).unwrap(),
            "\"gemma4\""
        );
        assert_eq!(
            serde_json::to_string(&ModelFamily::Qwen).unwrap(),
            "\"qwen\""
        );
    }

    #[test]
    fn chat_role_serde_lowercase() {
        assert_eq!(serde_json::to_string(&ChatRole::User).unwrap(), "\"user\"");
        assert_eq!(
            serde_json::to_string(&ChatRole::Assistant).unwrap(),
            "\"assistant\""
        );
    }

    #[test]
    fn llm_request_history_defaults_to_empty() {
        let json = serde_json::json!({
            "system": "sys",
            "user": "hi",
            "max_tokens": 8u32,
            "temperature": 0.5_f32,
        });
        let req: LlmRequest = serde_json::from_value(json).unwrap();
        assert!(req.history.is_empty());
        assert_eq!(req.user, "hi");
    }
}
