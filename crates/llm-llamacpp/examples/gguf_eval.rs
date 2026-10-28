//! JSONL inference runner: gguf_eval MODEL FAMILY REQUESTS.jsonl
//! Each input line contains {"id": "...", "request": LlmRequest}.
//! Responses/timing go to stdout; native performance events go to stderr.
use ellisii_llm_core::{LlmBackend, LlmRequest, ModelFamily};
use ellisii_llm_llamacpp::{LlamaConfig, LlamaCppBackend};
use serde::Deserialize;
use std::io::{BufRead, Write};
use std::sync::{Arc, Mutex};
use std::time::Instant;

#[derive(Deserialize)]
struct Input {
    id: String,
    request: LlmRequest,
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt()
        .with_env_filter("ellisii_generation=info")
        .json()
        .with_writer(std::io::stderr)
        .init();
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 4 {
        return Err("usage: gguf_eval MODEL {lfm3|gemma4|qwen} REQUESTS.jsonl".into());
    }
    let family: ModelFamily = serde_json::from_value(serde_json::Value::String(args[2].clone()))?;
    let cfg = LlamaConfig::new(&args[1], family);
    let backend = LlamaCppBackend::load(cfg)?;
    for line in std::io::BufReader::new(std::fs::File::open(&args[3])?).lines() {
        let input: Input = serde_json::from_str(&line?)?;
        let started = Instant::now();
        let output = Arc::new(Mutex::new((String::new(), None)));
        let collected = output.clone();
        backend
            .generate_stream(
                input.request.clone(),
                Box::new(move |chunk| {
                    let mut result = collected.lock().unwrap();
                    result
                        .1
                        .get_or_insert_with(|| started.elapsed().as_secs_f64() * 1000.0);
                    result.0.push_str(&chunk);
                }),
            )
            .await?;
        let result = output.lock().unwrap();
        println!(
            "{}",
            serde_json::json!({
                "id": input.id, "request": input.request, "response": result.0,
                "elapsed_ms": started.elapsed().as_secs_f64() * 1000.0,
                "first_chunk_ms": result.1,
            })
        );
        std::io::stdout().flush()?;
    }
    Ok(())
}
