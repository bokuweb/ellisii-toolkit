//! eval ハーネスの最小スモーク。fixture が JSON として正しくパースでき、
//! 想定キーが揃っていることを検査する。
//!
//! 実 embedder + store を回した end-to-end の retrieve 計測は別ファイル
//! (将来 `#[ignore]` で `cargo test -- --ignored` 経路に追加予定)。

use ellisii_rag::eval::{ndcg_at_k, recall_at_k, summarize, GoldenSet};

#[test]
fn fixture_parses_and_has_items() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/eval/golden.example.json");
    let raw = std::fs::read_to_string(&path).expect("read fixture");
    let set = GoldenSet::from_json_str(&raw).expect("parse fixture");
    assert_eq!(set.name, "smoke-jp-law");
    assert!(set.items.len() >= 3);
    for item in &set.items {
        assert!(!item.query.is_empty());
        assert!(
            !item.relevant.is_empty(),
            "every golden must list ≥1 relevant id"
        );
    }
}

#[test]
fn metrics_run_against_fixture_with_synthetic_predictions() {
    // 検索結果の代わりに合成データで metric が動くことを保証する。
    // 1 問目 (query="民法第94条..."): 完全一致で先頭ヒット → recall=1, ndcg=1
    // 2 問目: 3 位にヒット → recall=1, ndcg<1
    // 3 問目 (multi-relevant): 1 件だけ recall → 0.5
    let pairs = vec![
        (
            vec!["minpou-94".to_string(), "x".to_string(), "y".to_string()],
            vec!["minpou-94".to_string()],
        ),
        (
            vec!["x".to_string(), "y".to_string(), "minpou-94".to_string()],
            vec!["minpou-94".to_string()],
        ),
        (
            vec!["minpou-15".to_string(), "z".to_string()],
            vec!["minpou-15".to_string(), "minpou-16".to_string()],
        ),
    ];
    let s = summarize(&pairs, 3);
    assert_eq!(s.queries, 3);
    assert!(s.recall_at_k > 0.0 && s.recall_at_k <= 1.0);
    assert!(s.ndcg_at_k > 0.0 && s.ndcg_at_k <= 1.0);
    assert!(s.mrr > 0.0);
}

#[test]
fn metrics_individual_sanity() {
    let pred = vec!["a".to_string(), "b".to_string(), "c".to_string()];
    let rel = vec!["b".to_string()];
    let r = recall_at_k(&pred, &rel, 3);
    let n = ndcg_at_k(&pred, &rel, 3);
    assert!((r - 1.0).abs() < 1e-6);
    // 2 位ヒットなので NDCG は 1/log2(3) / 1 = 約 0.6309
    assert!((n - (1.0 / (3.0_f32.log2()))).abs() < 1e-3);
}
