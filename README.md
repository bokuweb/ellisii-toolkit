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

## License

[PolyForm Noncommercial 1.0.0](./LICENSE). Free for personal, research,
educational, and noncommercial use. Commercial use requires a separate
license — contact the author.

Third-party dependency licenses: see
[`THIRD_PARTY_LICENSES.html`](./THIRD_PARTY_LICENSES.html) (regenerate
with `cargo about generate about.hbs --all-features`).
