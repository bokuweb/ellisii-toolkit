//! Run 12m: enrichment ON/OFF を auto-route するための corpus signal を
//! calibrate する measurement-first ハーネス。
//!
//! Run 12k-12l で `with_caption_enrichment_default()` の net 効果が corpus
//! 性質に強く依存することが分かったが、ON/OFF を recommend するための
//! しきい値を出すには、各 fixture の corpus signal と Run 12l の outcome
//! を並べて見る必要がある。本ハーネスはそれを 1 表に出す。
//!
//! 計算する signal は [`ellisii_core::caption::body_vocab_novelty`]
//! (= `Ellisii::corpus_paraphrase_score` と同じ式)。各 chunk について
//! `(caption, body)` を取り出して novelty を測り、上限 256 件で平均する。
//!
//! Run 12l outcome (cap=off の Δ) は hard-coded で添える (5 fixtures に閉じる)。
//!
//! 使い方:
//! ```sh
//! cargo run -p ellisii-sdk --example eval_enrichment_signal --release
//! ```

use std::path::PathBuf;

use ellisii_core::caption::body_vocab_novelty;
use ellisii_rag::{is_specific_query, query_body_recall_mean, query_title_match_mean};
use serde::Deserialize;

#[derive(Debug, Deserialize, Clone)]
struct CorpusEntry {
    #[serde(default)]
    caption: String,
    text: String,
}

#[derive(Debug, Deserialize)]
struct GoldenItem {
    query: String,
}

#[derive(Debug, Deserialize)]
struct GoldenFile {
    items: Vec<GoldenItem>,
}

fn fixture_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .join("rag/tests/fixtures/eval")
}

/// Run 12l (cap=off) / Run 12n (拡張) の outcome — 推奨判定の reference。
/// None なら未測定。
fn run12l_outcome(name: &str) -> Option<(f32, f32, &'static str)> {
    match name {
        // Run 12l 5 fixtures
        "jp-civil-law-hard" => Some((0.143, 0.295, "win")),
        "jp-labor-law" => Some((0.000, -0.043, "天井 (neutral)")),
        "jp-cs-wiki-hard" => Some((0.026, 0.010, "mild win")),
        "jp-workplace-regs" => Some((-0.025, -0.053, "退行")),
        "jp-tokkyo-hou" => Some((-0.016, -0.054, "退行")),
        _ => None,
    }
}

/// 推奨判定 (Run 12m provisional 閾値 = q-cap match 0.20)。
fn predict(q_cap_match: f32) -> &'static str {
    if q_cap_match < 0.15 {
        "ON 推奨"
    } else if q_cap_match >= 0.25 {
        "OFF 推奨"
    } else {
        "uncertain"
    }
}

/// caption 長 (chars) の平均。Run 12r 仮説: OFF 域でも短い caption
/// (= article-title-style) は dilute に強い、長い caption (= 規程パラグラフ)
/// は dilute に弱い。
fn caption_len_mean(corpus: &[CorpusEntry]) -> f32 {
    let lens: Vec<f32> = corpus
        .iter()
        .filter(|e| !e.caption.is_empty())
        .map(|e| e.caption.chars().count() as f32)
        .collect();
    if lens.is_empty() {
        0.0
    } else {
        lens.iter().sum::<f32>() / lens.len() as f32
    }
}

fn corpus_paraphrase_score(corpus: &[CorpusEntry]) -> f32 {
    const SAMPLE_LIMIT: usize = 256;
    let take = corpus.len().min(SAMPLE_LIMIT);
    let mut sum = 0.0f32;
    let mut n = 0usize;
    for e in corpus.iter().take(take) {
        if e.caption.is_empty() {
            continue;
        }
        let body = e.text.replacen(&e.caption, "", 1);
        sum += body_vocab_novelty(&e.caption, &body);
        n += 1;
    }
    if n == 0 {
        0.0
    } else {
        sum / n as f32
    }
}

fn main() -> anyhow::Result<()> {
    // 指定があればそれ、無ければ fixture root を scan して有効な fixture を全部出す。
    let env_fixtures = std::env::var("ELLISII_EVAL_FIXTURES").ok();
    let fixtures: Vec<String> = if let Some(s) = env_fixtures {
        s.split(',').map(|s| s.trim().to_string()).collect()
    } else {
        let mut out = Vec::new();
        for ent in std::fs::read_dir(fixture_root())? {
            let p = ent?.path();
            if p.is_dir() && p.join("corpus.json").exists() && p.join("golden.json").exists() {
                if let Some(name) = p.file_name().and_then(|s| s.to_str()) {
                    out.push(name.to_string());
                }
            }
        }
        out.sort();
        out
    };

    println!(
        "| fixture                | n    | cap_len | paraphrase | q_specific% | q-cap match | q-body recall | MRR Δ  | 12l 判定       | 12m 予測   |"
    );
    println!(
        "|------------------------|-----:|--------:|-----------:|------------:|------------:|--------------:|-------:|----------------|------------|"
    );

    for name in &fixtures {
        let path = fixture_root().join(name).join("corpus.json");
        let corpus: Vec<CorpusEntry> = serde_json::from_str(&std::fs::read_to_string(&path)?)?;
        let golden_path = fixture_root().join(name).join("golden.json");
        let golden: GoldenFile = serde_json::from_str(&std::fs::read_to_string(&golden_path)?)?;
        let queries: Vec<&str> = golden.items.iter().map(|i| i.query.as_str()).collect();

        let cap_len = caption_len_mean(&corpus);
        let paraphrase = corpus_paraphrase_score(&corpus);
        let q_specific_pct = if queries.is_empty() {
            0.0
        } else {
            100.0 * queries.iter().filter(|q| is_specific_query(q)).count() as f32
                / queries.len() as f32
        };
        let captions: Vec<&str> = corpus
            .iter()
            .map(|e| e.caption.as_str())
            .filter(|s| !s.is_empty())
            .collect();
        let bodies: Vec<&str> = corpus.iter().map(|e| e.text.as_str()).collect();
        let q_cap_match = if captions.is_empty() {
            0.0
        } else {
            query_title_match_mean(&queries, &captions)
        };
        let q_body_recall = query_body_recall_mean(&queries, &bodies);

        let outcome = run12l_outcome(name);
        let (mrr_str, verdict) = match outcome {
            Some((_, mrr_d, v)) => (format!("{:+.3}", mrr_d), v),
            None => ("  ?  ".to_string(), "?"),
        };
        let pred = predict(q_cap_match);
        println!(
            "| {:<22} | {:>4} | {:>7.1} | {:>10.3} | {:>10.1}% | {:>11.3} | {:>13.3} | {:>6} | {:<14} | {:<10} |",
            name,
            corpus.len(),
            cap_len,
            paraphrase,
            q_specific_pct,
            q_cap_match,
            q_body_recall,
            mrr_str,
            verdict,
            pred,
        );
    }

    println!();
    println!("# 推奨判定 (Run 12l outcome を真値、paraphrase_score を予測子として閾値探索)");
    println!("# 高 score (paraphrase-heavy) → enrichment ON, cap rerank OFF");
    println!("# 低 score (literal lookup)   → enrichment OFF, cap rerank の判断は別軸");

    Ok(())
}
