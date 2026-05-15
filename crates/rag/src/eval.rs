//! RAG 検索品質の決定的メトリクス。golden Q&A セットに対して回帰検知するための
//! 関数群。embedder / store には依存しないので、単体テストや eval ハーネスから
//! 純粋関数として呼び出せる。
//!
//! 用語:
//! - `predicted`: 検索結果の chunk-id 列 (上位順、長さは任意)
//! - `relevant`:  golden で「正解」とラベル付けされた chunk-id の集合
//!
//! 提供メトリクス:
//! - `recall_at_k`: 上位 K 件に正解 chunk-id がいくつ入っているか / 正解総数
//! - `hit_at_k`:    上位 K 件に少なくとも 1 件正解が入っていれば 1.0
//! - `ndcg_at_k`:   binary relevance の Normalized DCG (順位を考慮した精度)
//! - `mrr`:         先頭から見て最初に正解が出た順位の逆数
//!
//! golden セット形式は [`GoldenItem`] / [`GoldenSet`] を JSON 直列化することで
//! 共有する。ファイル例は `tests/fixtures/eval/golden.example.json`。

use serde::{Deserialize, Serialize};

/// golden Q&A 1 件分。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GoldenItem {
    /// 質問文 (検索クエリ)
    pub query: String,
    /// この質問に対して「正解」とみなす chunk-id の集合。
    /// id は store-sqlite 上の `chunks.id` (Uuid 文字列) でもよいし、
    /// fixture document 上で安定する別の合成 id でもよい (eval 側の責務)。
    pub relevant: Vec<String>,
    /// 任意の難易度 / カテゴリタグ。レポートでの集計に使う。
    #[serde(default)]
    pub tags: Vec<String>,
}

/// 複数の golden Q&A をまとめた eval セット。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GoldenSet {
    pub name: String,
    pub items: Vec<GoldenItem>,
}

impl GoldenSet {
    pub fn from_json_str(s: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(s)
    }
}

/// Recall@K = (上位 K 件に含まれる relevant 数) / (relevant 総数)。
/// relevant が空なら 1.0 を返す (= 評価不能だが「逃した」とは扱わない)。
pub fn recall_at_k(predicted: &[String], relevant: &[String], k: usize) -> f32 {
    if relevant.is_empty() {
        return 1.0;
    }
    let head: &[String] = if predicted.len() > k {
        &predicted[..k]
    } else {
        predicted
    };
    let hit = relevant
        .iter()
        .filter(|r| head.iter().any(|p| p == *r))
        .count();
    hit as f32 / relevant.len() as f32
}

/// Hit@K = 上位 K 件に少なくとも 1 件 relevant があれば 1.0。
pub fn hit_at_k(predicted: &[String], relevant: &[String], k: usize) -> f32 {
    if relevant.is_empty() {
        return 1.0;
    }
    let head: &[String] = if predicted.len() > k {
        &predicted[..k]
    } else {
        predicted
    };
    if head.iter().any(|p| relevant.iter().any(|r| r == p)) {
        1.0
    } else {
        0.0
    }
}

/// NDCG@K (binary relevance)。`gain_i = 1 if predicted[i] in relevant else 0`、
/// `DCG = Σ gain_i / log2(i + 2)`、`IDCG = Σ_{i=0..min(K, |relevant|)} 1/log2(i+2)`、
/// `NDCG = DCG / IDCG` (IDCG=0 のときは 1.0)。
pub fn ndcg_at_k(predicted: &[String], relevant: &[String], k: usize) -> f32 {
    if relevant.is_empty() || k == 0 {
        return 1.0;
    }
    let limit = predicted.len().min(k);
    let mut dcg = 0.0_f32;
    for i in 0..limit {
        if relevant.iter().any(|r| r == &predicted[i]) {
            dcg += 1.0 / ((i as f32 + 2.0).log2());
        }
    }
    let ideal_n = relevant.len().min(k);
    let mut idcg = 0.0_f32;
    for i in 0..ideal_n {
        idcg += 1.0 / ((i as f32 + 2.0).log2());
    }
    if idcg == 0.0 {
        1.0
    } else {
        dcg / idcg
    }
}

/// Mean Reciprocal Rank — 単一クエリ版 (= Reciprocal Rank)。
/// 最初に出てきた relevant chunk の位置 r (1-indexed) から `1/r`。
/// 該当無しなら 0.0。
pub fn reciprocal_rank(predicted: &[String], relevant: &[String]) -> f32 {
    if relevant.is_empty() {
        return 1.0;
    }
    for (i, p) in predicted.iter().enumerate() {
        if relevant.iter().any(|r| r == p) {
            return 1.0 / (i as f32 + 1.0);
        }
    }
    0.0
}

/// 複数 query を集計したサマリ。
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EvalSummary {
    pub queries: usize,
    pub recall_at_k: f32,
    pub hit_at_k: f32,
    pub ndcg_at_k: f32,
    pub mrr: f32,
}

/// 各 query の (predicted, golden) ペアからサマリを計算する。
pub fn summarize(pairs: &[(Vec<String>, Vec<String>)], k: usize) -> EvalSummary {
    if pairs.is_empty() {
        return EvalSummary::default();
    }
    let n = pairs.len() as f32;
    let mut s = EvalSummary {
        queries: pairs.len(),
        ..Default::default()
    };
    for (pred, rel) in pairs {
        s.recall_at_k += recall_at_k(pred, rel, k);
        s.hit_at_k += hit_at_k(pred, rel, k);
        s.ndcg_at_k += ndcg_at_k(pred, rel, k);
        s.mrr += reciprocal_rank(pred, rel);
    }
    s.recall_at_k /= n;
    s.hit_at_k /= n;
    s.ndcg_at_k /= n;
    s.mrr /= n;
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ids(xs: &[&str]) -> Vec<String> {
        xs.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn recall_full_match_is_one() {
        let pred = ids(&["a", "b", "c"]);
        let rel = ids(&["a", "c"]);
        assert!((recall_at_k(&pred, &rel, 3) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn recall_partial_match() {
        let pred = ids(&["a", "x", "y"]);
        let rel = ids(&["a", "b"]);
        assert!((recall_at_k(&pred, &rel, 3) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn recall_truncated_to_k() {
        let pred = ids(&["x", "y", "a"]);
        let rel = ids(&["a"]);
        assert!((recall_at_k(&pred, &rel, 2) - 0.0).abs() < 1e-6);
        assert!((recall_at_k(&pred, &rel, 3) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn recall_empty_relevant_is_one() {
        let pred = ids(&["a"]);
        assert!((recall_at_k(&pred, &[], 5) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn hit_basic() {
        assert_eq!(hit_at_k(&ids(&["x", "a"]), &ids(&["a"]), 5), 1.0);
        assert_eq!(hit_at_k(&ids(&["x", "y"]), &ids(&["a"]), 5), 0.0);
    }

    #[test]
    fn ndcg_perfect_ranking() {
        // 正解 [a, b]、予測 [a, b, c] → 完全に上位に並ぶ → 1.0
        let pred = ids(&["a", "b", "c"]);
        let rel = ids(&["a", "b"]);
        assert!((ndcg_at_k(&pred, &rel, 3) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn ndcg_lower_when_relevant_is_demoted() {
        // 正解は 1 件だが予測の 3 位に居る → 1.0 未満
        let pred = ids(&["x", "y", "a"]);
        let rel = ids(&["a"]);
        let ndcg = ndcg_at_k(&pred, &rel, 3);
        assert!(ndcg < 1.0 && ndcg > 0.0);
    }

    #[test]
    fn ndcg_zero_when_nothing_in_top_k() {
        let pred = ids(&["x", "y", "z"]);
        let rel = ids(&["a"]);
        assert!((ndcg_at_k(&pred, &rel, 3) - 0.0).abs() < 1e-6);
    }

    #[test]
    fn rr_basic() {
        assert!((reciprocal_rank(&ids(&["a", "b"]), &ids(&["a"])) - 1.0).abs() < 1e-6);
        assert!((reciprocal_rank(&ids(&["x", "a"]), &ids(&["a"])) - 0.5).abs() < 1e-6);
        assert!((reciprocal_rank(&ids(&["x", "y"]), &ids(&["a"])) - 0.0).abs() < 1e-6);
    }

    #[test]
    fn summarize_averages_across_queries() {
        let pairs = vec![
            (ids(&["a", "x"]), ids(&["a"])), // perfect for both
            (ids(&["x", "a"]), ids(&["a"])), // a at rank 2
        ];
        let s = summarize(&pairs, 5);
        assert_eq!(s.queries, 2);
        assert!((s.recall_at_k - 1.0).abs() < 1e-6); // both recalled
        assert!((s.hit_at_k - 1.0).abs() < 1e-6);
        // mrr = (1.0 + 0.5) / 2 = 0.75
        assert!((s.mrr - 0.75).abs() < 1e-6);
    }

    #[test]
    fn golden_set_roundtrip_json() {
        let set = GoldenSet {
            name: "smoke".into(),
            items: vec![GoldenItem {
                query: "民法第94条の意思表示について教えて".into(),
                relevant: vec!["民法-94".into()],
                tags: vec!["jp-law".into()],
            }],
        };
        let s = serde_json::to_string(&set).expect("serialize");
        let back = GoldenSet::from_json_str(&s).expect("parse");
        assert_eq!(back.name, "smoke");
        assert_eq!(back.items.len(), 1);
        assert_eq!(back.items[0].relevant, vec!["民法-94"]);
    }
}
