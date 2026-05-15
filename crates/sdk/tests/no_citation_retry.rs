//! `AskOptions::no_citation_retry` の挙動検証 (Run 64)。
//!
//! 初回応答に `[N]` が無ければ厳格 prompt で 1 度だけ retry、2 度目に citation が
//! 出ればそれを最終応答として on_token に流す。retry-once policy のため 2 度目も
//! 無 citation なら諦める (3 度目以降は呼ばない)。

use async_trait::async_trait;
use ellisii_core::{Chunk, Result};
use ellisii_llm_core::{LlmBackend, LlmRequest};
use ellisii_sdk::{AskOptions, Ellisii};
use ellisii_store_core::VectorStore as _;
use std::sync::{Arc, Mutex};
use uuid::Uuid;

/// 呼び出し回数を記録し、それに応じて違う応答を返す mock LLM。
/// scripted_responses: 1 周目 / 2 周目 / 3 周目... の応答を順番に返す。
struct ScriptedLlm {
    scripted: Vec<&'static str>,
    call_count: Mutex<usize>,
    last_system: Mutex<String>,
}

impl ScriptedLlm {
    fn new(responses: Vec<&'static str>) -> Self {
        Self {
            scripted: responses,
            call_count: Mutex::new(0),
            last_system: Mutex::new(String::new()),
        }
    }
    fn calls(&self) -> usize {
        *self.call_count.lock().unwrap()
    }
    fn last_system(&self) -> String {
        self.last_system.lock().unwrap().clone()
    }
}

#[async_trait]
impl LlmBackend for ScriptedLlm {
    async fn generate_stream(
        &self,
        req: LlmRequest,
        mut on_token: Box<dyn FnMut(String) + Send + 'static>,
    ) -> Result<()> {
        let idx = {
            let mut c = self.call_count.lock().unwrap();
            let i = *c;
            *c += 1;
            i
        };
        *self.last_system.lock().unwrap() = req.system.clone();
        let response = self
            .scripted
            .get(idx)
            .copied()
            .unwrap_or("[fallback no scripted response]");
        // 1 トークン = 1 文字ぐらいで分割してストリーミングっぽくする (任意)。
        on_token(response.to_string());
        Ok(())
    }
}

async fn setup_ellisii(llm: Arc<ScriptedLlm>) -> Ellisii {
    let nb = Uuid::new_v4();
    let ellisii = Ellisii::builder()
        .with_embedder_dummy(32)
        .with_store_memory()
        .with_llm(llm as Arc<dyn LlmBackend>)
        .with_notebook_id(nb)
        .build()
        .unwrap();
    // 1 chunk だけ ingest して hits > 0 を保証する。
    let chunk = Chunk {
        id: Uuid::new_v4(),
        source_id: Uuid::new_v4(),
        ord: 0,
        text: "民法第94条 相手方と通謀してした虚偽の意思表示は、無効とする。".to_string(),
        heading_path: vec![],
        page: None,
        bbox: None,
        summary: None,
    };
    let embs = ellisii.embedder().embed(&["dummy".to_string()]).await.unwrap();
    ellisii.store().upsert(nb, &[chunk], &embs).await.unwrap();
    ellisii
}

/// 1 周目 citation 無し → retry が走り 2 周目 citation あり。
#[tokio::test]
async fn retries_when_first_attempt_has_no_citation() {
    let llm = Arc::new(ScriptedLlm::new(vec![
        "通謀虚偽表示は無効です。", // 1 周目: [N] 無し
        "通謀虚偽表示は無効です [1]。", // 2 周目: 厳格 prompt で [1] が出た
    ]));
    let ellisii = setup_ellisii(llm.clone()).await;

    let received = Arc::new(Mutex::new(String::new()));
    let r2 = received.clone();
    let _hits = ellisii
        .ask(
            "通謀虚偽表示について",
            AskOptions {
                no_citation_retry: true,
                route_by_intent: false, // 自動 LlmIntentClassifier 呼び出しを抑止
                ..Default::default()
            },
            move |tok| {
                r2.lock().unwrap().push_str(&tok);
            },
        )
        .await
        .unwrap();

    assert_eq!(llm.calls(), 2, "exactly 2 LLM calls (first + retry)");
    let out = received.lock().unwrap().clone();
    assert!(out.contains("通謀虚偽表示は無効です。"), "1st attempt streamed (out={out:?})");
    assert!(out.contains("[出典付きで再生成]"), "retry divider streamed");
    assert!(out.contains("[1]"), "2nd attempt with citation streamed");
    // 2 周目は厳格 prompt のはず
    assert!(
        llm.last_system().contains("[N]") || llm.last_system().contains("引用"),
        "retry system prompt should include strict citation directive, got: {}",
        llm.last_system()
    );
}

/// 1 周目 citation あり → retry しない (1 call のみ)。
#[tokio::test]
async fn no_retry_when_first_attempt_has_citation() {
    let llm = Arc::new(ScriptedLlm::new(vec![
        "通謀虚偽表示は無効です [1]。", // 1 周目で既に [1] あり
        "should not be called",
    ]));
    let ellisii = setup_ellisii(llm.clone()).await;

    let received = Arc::new(Mutex::new(String::new()));
    let r2 = received.clone();
    let _ = ellisii
        .ask(
            "通謀虚偽表示について",
            AskOptions {
                no_citation_retry: true,
                route_by_intent: false, // 自動 LlmIntentClassifier 呼び出しを抑止
                ..Default::default()
            },
            move |tok| r2.lock().unwrap().push_str(&tok),
        )
        .await
        .unwrap();
    assert_eq!(llm.calls(), 1, "only initial LLM call when citation present");
    let out = received.lock().unwrap().clone();
    assert!(!out.contains("[出典付きで再生成]"), "no retry divider");
}

/// `no_citation_retry=false` (default) では citation 無しでも retry しない。
#[tokio::test]
async fn retry_disabled_by_default_flag() {
    let llm = Arc::new(ScriptedLlm::new(vec![
        "通謀虚偽表示は無効です。", // [N] 無し
        "should not be called",
    ]));
    let ellisii = setup_ellisii(llm.clone()).await;
    let _ = ellisii
        .ask(
            "通謀虚偽表示について",
            AskOptions {
                route_by_intent: false,
                ..Default::default() // no_citation_retry: false
            },
            |_| {},
        )
        .await
        .unwrap();
    assert_eq!(llm.calls(), 1, "no retry when flag is off");
}
