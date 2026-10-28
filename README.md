# ellisii-toolkit

[![CI](https://github.com/bokuweb/ellisii-toolkit/actions/workflows/ci.yml/badge.svg)](https://github.com/bokuweb/ellisii-toolkit/actions/workflows/ci.yml)
[![License: PolyForm-NC-1.0.0](https://img.shields.io/badge/license-PolyForm--NC--1.0.0-blue.svg)](./LICENSE)

Reusable Rust crates for building local-first RAG / NotebookLM-style
applications. Extracted from [ellisii](https://github.com/bokuweb/ellisii)
so the lower-level building blocks can be consumed from other projects
without bringing in the Tauri app or the notebook domain layer.

## Status

Pre-1.0. APIs may change. Not published to crates.io — consume via
`git` dependency:

```toml
[dependencies]
ellisii-rag = { git = "https://github.com/bokuweb/ellisii-toolkit", rev = "<commit>" }
```

## What's inside

- **Parsers** — PDF / DOCX / XLSX / PPTX / Markdown / text / audio
  (`parsers`, `parser-*`, `parsers-core`)
- **OCR** — wrapper around `ndlocr-lite-rs`
- **Chunking** (`chunker`)
- **Embedders** — trait + Japanese static embedding implementation
  (`embed-core`, `embed-static-jp`, `embed-dummy`)
- **Vector stores** — trait + in-memory and SQLite (`sqlite-vec` + FTS5)
  backends (`store-core`, `store-memory`, `store-sqlite`)
- **LLM backends** — trait + stub and `llama.cpp` implementations
  (`llm-core`, `llm-stub`, `llm-llamacpp`, `llm-prompt`)
- **RAG pipeline** — retrieval, reranking, prompting, streaming, and
  recall / answer evaluation harnesses (`rag`, `rag-answer-eval`,
  `rag-eval-cli`)
- **Japanese tokenizers** (`jp-tokenizer-*`)
- **Provence reranker** (`provence-*`)
- **Query rewriter** (`query-rewriter-*`)
- **Ingest pipeline** — parse → chunk → embed → store orchestration
  (`ingest`)
- **SDK** — facade crate that re-exports the common surface (`sdk`)

## LFM3 text inference

`ellisii-llm-llamacpp` supports the official
[LiquidAI/LFM3-1.7B-GGUF](https://huggingface.co/LiquidAI/LFM3-1.7B-GGUF)
with `ModelFamily::Lfm3` and the `gguf` feature (`llamacpp` for `ellisii-sdk`). Download
`LFM3-1.7B-Q4_K_M.gguf`, then use the same SDK API as Gemma:

```rust,ignore
use ellisii_sdk::ModelFamily;

let builder = ellisii_sdk::Ellisii::builder()
    .with_llm_llamacpp("/models/LFM3-1.7B-Q4_K_M.gguf", ModelFamily::Lfm3)?;
```

The ChatML template disables thinking. LFM3's recurrent state is cleared
between requests; prompt prefix reuse and disk snapshots are disabled for
this family. Full conversation history is still supported. Context size
keeps the existing memory-based defaults; 256K context is not allocated
automatically.

The bindings are pinned to a small fork updating bundled llama.cpp to
**b11960 / v0.8.0**, the minimum required by the official model card.
Gemma and Qwen use the same updated runtime. Gemma4 E2B/E4B use the
canonical `<|turn>` / `<turn|>` template with a dedicated system turn
(thinking disabled), rather than Gemma3 turn markers.

For repeatable JSONL inference with native token/timing events:

```sh
ELLISII_DISABLE_PROMPT_CACHE=1 ELLISII_LLAMA_KV_TYPE=f16 \
cargo run --release -p ellisii-llm-llamacpp --features gguf \
  --example gguf_eval -- /models/LFM3-1.7B-Q4_K_M.gguf lfm3 requests.jsonl
```

Input lines have `{"id":"1","request":{"system":"日本語で答えてください。",
"history":[],"user":"日本の首都は？","max_tokens":32,"temperature":0.0}}`.
Responses and wall time are written to stdout. Native prompt/decode token
counts and milliseconds are emitted as JSON tracing events to stderr.
`first_chunk_ms` measures the first buffered callback, not raw first-token latency.

## License

[PolyForm Noncommercial 1.0.0](./LICENSE). Free for personal, research,
educational, and noncommercial use. Commercial use requires a separate
license — contact the author.

Third-party dependency licenses: see
[`THIRD_PARTY_LICENSES.html`](./THIRD_PARTY_LICENSES.html) (regenerate
with `cargo about generate about.hbs --all-features`).
