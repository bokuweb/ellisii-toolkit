//! Memory vs Sqlite backend の keyword 経路を比較する統合テスト。
//!
//! 目的:
//! `InMemoryStore::keyword_search` はクエリ全体の部分文字列マッチで
//! 日本語複数トークンクエリでほぼ 0 になる。`SqliteStore` (FTS5 + BM25 + 文字バイグラム
//! トークナイザ) ならトークン化が効くので、同じ corpus/golden で keyword-only
//! (semantic=0.0) でも実用的な hit/recall が出るはず。
//!
//! このテストは "sqlite backend は keyword で意味のある検索が出来る" を
//! 数値的にロックする回帰ガード。

use ellisii_rag::eval::{GoldenItem, GoldenSet};
use ellisii_rag_eval_cli::{run_eval_with_backend, Backend, CorpusEntry};

fn corpus() -> Vec<CorpusEntry> {
    // 民法から虚偽表示・錯誤・詐欺の 3 条 + ノイズ。
    vec![
        CorpusEntry {
            doc_id: "minpou-93".into(),
            title: "第九十三条".into(),
            caption: "心裡留保".into(),
            text: "意思表示は、表意者がその真意ではないことを知ってしたときであっても、\
                   そのためにその効力を妨げられない。"
                .into(),
        },
        CorpusEntry {
            doc_id: "minpou-94".into(),
            title: "第九十四条".into(),
            caption: "虚偽表示".into(),
            text: "相手方と通じてした虚偽の意思表示は、無効とする。\
                   前項の規定による意思表示の無効は、善意の第三者に対抗することができない。"
                .into(),
        },
        CorpusEntry {
            doc_id: "minpou-95".into(),
            title: "第九十五条".into(),
            caption: "錯誤".into(),
            text: "意思表示は、次に掲げる錯誤に基づくものであって、\
                   その錯誤が法律行為の目的及び取引上の社会通念に照らして重要なものであるときは、\
                   取り消すことができる。"
                .into(),
        },
        CorpusEntry {
            doc_id: "minpou-96".into(),
            title: "第九十六条".into(),
            caption: "詐欺又は強迫".into(),
            text: "詐欺又は強迫による意思表示は、取り消すことができる。"
                .into(),
        },
        CorpusEntry {
            doc_id: "noise-1".into(),
            title: "商法第501条".into(),
            caption: "絶対的商行為".into(),
            text: "次に掲げる行為は商行為とする。".into(),
        },
        CorpusEntry {
            doc_id: "noise-2".into(),
            title: "刑法第199条".into(),
            caption: "殺人".into(),
            text: "人を殺した者は、死刑又は無期若しくは5年以上の懲役に処する。".into(),
        },
    ]
}

fn golden() -> GoldenSet {
    GoldenSet {
        name: "backend-cmp".into(),
        items: vec![
            GoldenItem {
                query: "通謀虚偽表示は無効か".into(),
                relevant: vec!["minpou-94".into()],
                tags: vec![],
            },
            GoldenItem {
                query: "錯誤による意思表示は取り消せるか".into(),
                relevant: vec!["minpou-95".into()],
                tags: vec![],
            },
            GoldenItem {
                query: "詐欺による意思表示の効力".into(),
                relevant: vec!["minpou-96".into()],
                tags: vec![],
            },
            GoldenItem {
                query: "心裡留保の規定".into(),
                relevant: vec!["minpou-93".into()],
                tags: vec![],
            },
        ],
    }
}

#[tokio::test]
async fn sqlite_backend_keyword_only_finds_relevant_chunks() {
    let rows = run_eval_with_backend(Backend::Sqlite, &corpus(), &golden(), &[0.0], 5)
        .await
        .expect("eval succeeds");
    assert_eq!(rows.len(), 1);
    let s = &rows[0].summary;
    println!(
        "sqlite keyword-only: recall@5={:.3} hit@5={:.3} nDCG@5={:.3} MRR={:.3}",
        s.recall_at_k, s.hit_at_k, s.ndcg_at_k, s.mrr
    );
    // FTS5 + BM25 + 文字バイグラムトークナイザがあれば、4 問中 3 問以上は
    // 上位 5 件で正解にヒットする (≥ 75%) ことを期待する。
    assert!(
        s.hit_at_k >= 0.75,
        "expected sqlite keyword hit@5 ≥ 0.75, got {}",
        s.hit_at_k
    );
    assert!(
        s.mrr > 0.5,
        "expected sqlite keyword MRR > 0.5, got {}",
        s.mrr
    );
}

#[tokio::test]
async fn memory_backend_keyword_only_struggles_on_multi_token_queries() {
    // この振る舞いは「悪い」ものだが、現状の InMemoryStore::keyword_search が
    // 部分文字列マッチであることを明示的にテストでロックしておく。
    // sqlite に切替えれば改善する、という比較の片側を担保する。
    let rows = run_eval_with_backend(Backend::Memory, &corpus(), &golden(), &[0.0], 5)
        .await
        .expect("eval succeeds");
    let s = &rows[0].summary;
    println!(
        "memory keyword-only: recall@5={:.3} hit@5={:.3}",
        s.recall_at_k, s.hit_at_k
    );
    // クエリ全体が text の部分文字列にならないので、ほぼ全滅するはず。
    assert!(
        s.hit_at_k < 0.5,
        "memory backend unexpectedly performed well on keyword-only: hit@5={}",
        s.hit_at_k
    );
}

#[tokio::test]
async fn sqlite_keyword_outperforms_memory_keyword() {
    let mem = run_eval_with_backend(Backend::Memory, &corpus(), &golden(), &[0.0], 5)
        .await
        .unwrap();
    let sql = run_eval_with_backend(Backend::Sqlite, &corpus(), &golden(), &[0.0], 5)
        .await
        .unwrap();
    assert!(
        sql[0].summary.recall_at_k > mem[0].summary.recall_at_k + 0.3,
        "sqlite recall {} should clearly beat memory recall {}",
        sql[0].summary.recall_at_k,
        mem[0].summary.recall_at_k,
    );
}
