//! Multi-domain eval — 民法 (法務) と Wikipedia CS の 2 fixture を回し、
//! `HybridWeights::default() = 0.5` が両ドメインで最適かを検証する。
//!
//! 単一ドメインで「keyword-only がベスト」と主張するのは早計なので、
//! 異質なコーパス (語彙・文体が大きく違う) を 2 つ以上で計測してから
//! default 変更の判断材料にする。
//!
//! このテストは "両ドメインで sqlite backend は recall@10 ≥ 0.7 出る" という
//! 最低限の品質ガードを兼ねる。

use ellisii_rag::eval::GoldenSet;
use ellisii_rag_eval_cli::{run_eval_with_backend, Backend, CorpusEntry};
use std::path::PathBuf;

fn load_fixture(domain: &str) -> (Vec<CorpusEntry>, GoldenSet) {
    let base = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .join("rag/tests/fixtures/eval")
        .join(domain);
    let corpus: Vec<CorpusEntry> =
        serde_json::from_str(&std::fs::read_to_string(base.join("corpus.json")).unwrap()).unwrap();
    let golden: GoldenSet =
        serde_json::from_str(&std::fs::read_to_string(base.join("golden.json")).unwrap()).unwrap();
    (corpus, golden)
}

#[tokio::test]
async fn cs_wiki_fixture_loads_and_validates() {
    let (corpus, golden) = load_fixture("jp-cs-wiki");
    assert!(corpus.len() >= 30, "CS corpus too small: {}", corpus.len());
    assert!(
        golden.items.len() >= 30,
        "CS golden too small: {}",
        golden.items.len()
    );
    let ids: std::collections::HashSet<&str> = corpus.iter().map(|c| c.doc_id.as_str()).collect();
    for it in &golden.items {
        for r in &it.relevant {
            assert!(ids.contains(r.as_str()), "{r} missing in CS corpus");
        }
    }
}

#[tokio::test]
async fn sqlite_backend_meets_quality_floor_on_both_domains() {
    for domain in ["jp-civil-law", "jp-cs-wiki"] {
        let (corpus, golden) = load_fixture(domain);
        let rows = run_eval_with_backend(Backend::Sqlite, &corpus, &golden, &[0.0, 0.5], 10)
            .await
            .unwrap();
        let kw = &rows[0].summary;
        let hyb = &rows[1].summary;
        println!(
            "{domain}: keyword recall@10={:.3} MRR={:.3} | hybrid 0.5 recall@10={:.3} MRR={:.3}",
            kw.recall_at_k, kw.mrr, hyb.recall_at_k, hyb.mrr
        );
        assert!(
            kw.recall_at_k >= 0.7,
            "{domain} sqlite keyword recall@10 below 0.7: {}",
            kw.recall_at_k
        );
        assert!(
            hyb.recall_at_k >= 0.7,
            "{domain} sqlite hybrid recall@10 below 0.7: {}",
            hyb.recall_at_k
        );
    }
}

#[tokio::test]
async fn keyword_only_at_least_matches_default_hybrid_in_both_domains() {
    // 「sqlite keyword (s=0) が default hybrid (s=0.5) を recall@10 で下回らない」を
    // 両ドメインでロックする。これが true なら HybridWeights::default の引き下げ
    // (or キーワード寄りへの shift) を提案できる。逆に false に転んだドメインが出たら、
    // ドメイン別に切替える方針 (B3) が正解という根拠になる。
    for domain in ["jp-civil-law", "jp-cs-wiki"] {
        let (corpus, golden) = load_fixture(domain);
        let rows = run_eval_with_backend(Backend::Sqlite, &corpus, &golden, &[0.0, 0.5], 10)
            .await
            .unwrap();
        let kw = rows[0].summary.recall_at_k;
        let hyb = rows[1].summary.recall_at_k;
        assert!(
            kw + 1e-6 >= hyb,
            "{domain}: keyword recall {kw} should be >= hybrid 0.5 recall {hyb} \
             (if this fails for some domain, default weight should stay at 0.5 \
             and the choice should be domain-aware instead)",
        );
    }
}
