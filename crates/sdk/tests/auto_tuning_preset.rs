//! `SearchOptions::auto_tuning()` / `AskOptions::auto_tuning()` (Run 57) の
//! 設計検証。preset が想定通りのフィールドを on/off にしているか、
//! Default を継承しているかをチェックする。

use ellisii_sdk::{AskOptions, SearchOptions};

#[test]
fn search_options_auto_tuning_enables_proven_signals() {
    let opts = SearchOptions::auto_tuning();
    // Run 47-56 で実証された auto signals
    assert!(opts.auto_heading_rerank, "auto_heading_rerank should be on");
    assert!(
        opts.auto_max_chunks_per_source,
        "auto_max_chunks_per_source should be on"
    );
    // 既定の caption_rerank は維持
    assert!(opts.caption_rerank, "caption_rerank stays on (Default)");
}

#[test]
fn search_options_auto_tuning_inherits_defaults_for_unrelated_fields() {
    let preset = SearchOptions::auto_tuning();
    let default = SearchOptions::default();
    // auto_tuning() で変更しないフィールドは Default と一致
    assert_eq!(preset.top_k, default.top_k);
    assert_eq!(preset.semantic_weight, default.semantic_weight);
    assert_eq!(preset.ce_rerank_top_n, default.ce_rerank_top_n);
    assert_eq!(preset.ce_rerank_weight, default.ce_rerank_weight);
    assert_eq!(preset.heading_rerank, default.heading_rerank); // explicit は OFF のまま
    assert_eq!(preset.max_chunks_per_source, default.max_chunks_per_source); // explicit は 0 のまま
    assert_eq!(preset.auto_adjust_weight, default.auto_adjust_weight);
    assert_eq!(
        preset.skip_rewrite_on_specific,
        default.skip_rewrite_on_specific
    );
}

#[test]
fn search_options_auto_tuning_supports_spread() {
    // README/ドキュメントで紹介する spread パターンが compile + 期待通りに動くか
    let opts = SearchOptions {
        top_k: 10,
        ..SearchOptions::auto_tuning()
    };
    assert_eq!(opts.top_k, 10);
    assert!(opts.auto_heading_rerank);
    assert!(opts.auto_max_chunks_per_source);
}

#[test]
fn ask_options_auto_tuning_enables_proven_signals() {
    let opts = AskOptions::auto_tuning();
    assert!(opts.auto_heading_rerank);
    assert!(opts.auto_max_chunks_per_source);
    assert!(opts.caption_rerank);
}

#[test]
fn ask_options_auto_tuning_inherits_defaults_for_unrelated_fields() {
    let preset = AskOptions::auto_tuning();
    let default = AskOptions::default();
    assert_eq!(preset.top_k, default.top_k);
    assert_eq!(preset.max_tokens, default.max_tokens);
    assert_eq!(preset.temperature, default.temperature);
    assert_eq!(preset.route_by_intent, default.route_by_intent);
}

#[test]
fn ask_options_auto_tuning_supports_spread() {
    let opts = AskOptions {
        max_tokens: 1024,
        ..AskOptions::auto_tuning()
    };
    assert_eq!(opts.max_tokens, 1024);
    assert!(opts.auto_heading_rerank);
    assert!(opts.auto_max_chunks_per_source);
}
