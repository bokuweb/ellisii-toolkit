//! ellisii-rag-eval CLI shim — 実装は `lib.rs` 側に置く。

use anyhow::{anyhow, bail, Context, Result};
use ellisii_rag::eval::GoldenSet;
use ellisii_rag_answer_eval::{heuristic::TokenOverlapJudge, AnswerJudge};
use ellisii_rag_eval_cli::{
    run_eval_with_options, validate_golden_against_corpus, Backend, CorpusEntry, EmbedderKind,
    EvalOptions, EvalReport, DEFAULT_DIM,
};
use std::path::PathBuf;
use std::sync::Arc;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum JudgeKind {
    None,
    TokenOverlap,
}

struct Args {
    corpus: PathBuf,
    golden: PathBuf,
    k: usize,
    weights: Vec<f32>,
    json: bool,
    backend: Backend,
    embedder: EmbedderKind,
    judge: JudgeKind,
}

fn parse_args() -> Result<Args> {
    let mut corpus = None;
    let mut golden = None;
    let mut k: usize = 10;
    let mut weights: Vec<f32> = vec![0.0, 0.5, 1.0];
    let mut json = false;
    let mut backend = Backend::Memory;
    let mut embedder_name: String = "bigram".into();
    let mut model_dir: Option<PathBuf> = None;
    let mut judge = JudgeKind::None;

    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        match a.as_str() {
            "--corpus" => {
                corpus = Some(PathBuf::from(
                    it.next().ok_or_else(|| anyhow!("--corpus needs a value"))?,
                ))
            }
            "--golden" => {
                golden = Some(PathBuf::from(
                    it.next().ok_or_else(|| anyhow!("--golden needs a value"))?,
                ))
            }
            "--k" => {
                k = it
                    .next()
                    .ok_or_else(|| anyhow!("--k needs a value"))?
                    .parse()?
            }
            "--weights" => {
                let raw = it.next().ok_or_else(|| anyhow!("--weights needs csv"))?;
                weights = raw
                    .split(',')
                    .map(|s| s.trim().parse::<f32>())
                    .collect::<std::result::Result<_, _>>()?;
            }
            "--backend" => {
                backend = Backend::parse(
                    &it.next()
                        .ok_or_else(|| anyhow!("--backend needs a value"))?,
                )?;
            }
            "--embedder" => {
                embedder_name = it
                    .next()
                    .ok_or_else(|| anyhow!("--embedder needs a value"))?;
            }
            "--model-dir" => {
                model_dir = Some(PathBuf::from(
                    it.next()
                        .ok_or_else(|| anyhow!("--model-dir needs a value"))?,
                ));
            }
            "--judge" => {
                judge = match it.next().as_deref() {
                    Some("none") => JudgeKind::None,
                    Some("token-overlap") => JudgeKind::TokenOverlap,
                    Some(other) => bail!("unknown judge {other:?} (expected none | token-overlap)"),
                    None => bail!("--judge needs a value"),
                };
            }
            "--json" => json = true,
            "-h" | "--help" => {
                print_usage();
                std::process::exit(0);
            }
            other => bail!("unknown arg: {other}"),
        }
    }

    let embedder = match embedder_name.as_str() {
        "bigram" => EmbedderKind::Bigram { dim: DEFAULT_DIM },
        "static-jp" => EmbedderKind::StaticJp {
            model_dir: model_dir
                .ok_or_else(|| anyhow!("--embedder static-jp requires --model-dir"))?,
        },
        other => bail!("unknown embedder {other:?} (expected bigram | static-jp)"),
    };

    Ok(Args {
        corpus: corpus.ok_or_else(|| anyhow!("missing --corpus"))?,
        golden: golden.ok_or_else(|| anyhow!("missing --golden"))?,
        k,
        weights,
        json,
        backend,
        embedder,
        judge,
    })
}

fn print_usage() {
    eprintln!(
        "Usage: ellisii-rag-eval --corpus <path> --golden <path>\n\
        \x20  [--k 10] [--weights 0.0,0.5,1.0] [--backend memory|sqlite]\n\
        \x20  [--embedder bigram|static-jp] [--model-dir <path>]\n\
        \x20  [--judge none|token-overlap] [--json]"
    );
}

#[tokio::main(flavor = "multi_thread")]
async fn main() -> Result<()> {
    let args = parse_args()?;
    let corpus_raw = std::fs::read_to_string(&args.corpus)
        .with_context(|| format!("read corpus {:?}", args.corpus))?;
    let corpus: Vec<CorpusEntry> =
        serde_json::from_str(&corpus_raw).context("parse corpus json")?;
    let golden_raw = std::fs::read_to_string(&args.golden)
        .with_context(|| format!("read golden {:?}", args.golden))?;
    let golden: GoldenSet = serde_json::from_str(&golden_raw).context("parse golden json")?;

    if corpus.is_empty() {
        bail!("corpus is empty");
    }
    if golden.items.is_empty() {
        bail!("golden has no items");
    }
    let missing = validate_golden_against_corpus(&corpus, &golden);
    if !missing.is_empty() {
        eprintln!("warning: {} relevant ids not in corpus:", missing.len());
        for (q, r) in missing.iter().take(5) {
            eprintln!("  - {r} (from query {q:?})");
        }
    }

    let judge: Option<Arc<dyn AnswerJudge>> = match args.judge {
        JudgeKind::None => None,
        JudgeKind::TokenOverlap => Some(Arc::new(TokenOverlapJudge::default())),
    };
    let opts = EvalOptions {
        backend: args.backend,
        embedder: args.embedder.build()?,
        weights: args.weights.clone(),
        k: args.k,
        judge,
        rewriter: None,
        multi: ellisii_rag::MultiQueryOptions::default(),
    };
    let rows = run_eval_with_options(&opts, &corpus, &golden).await?;

    println!(
        "\n=== ellisii-rag headless eval ===\n\
         backend: {}\n\
         golden : {} ({} queries)\n\
         corpus : {} entries\n\
         k      : {}\n",
        args.backend.as_str(),
        golden.name,
        golden.items.len(),
        corpus.len(),
        args.k,
    );
    let show_faith = rows.iter().any(|r| r.faithfulness.is_some());
    if show_faith {
        println!(
            "{:>8}  {:>10}  {:>8}  {:>10}  {:>8}  {:>10}",
            "semantic", "recall@k", "hit@k", "nDCG@k", "MRR", "faith.mean"
        );
    } else {
        println!(
            "{:>8}  {:>10}  {:>8}  {:>10}  {:>8}",
            "semantic", "recall@k", "hit@k", "nDCG@k", "MRR"
        );
    }
    println!("{}", "-".repeat(if show_faith { 62 } else { 50 }));
    for r in &rows {
        if show_faith {
            let f = r.faithfulness.as_ref().map(|s| s.mean).unwrap_or(f32::NAN);
            println!(
                "{:>8.2}  {:>10.3}  {:>8.3}  {:>10.3}  {:>8.3}  {:>10.3}",
                r.semantic,
                r.summary.recall_at_k,
                r.summary.hit_at_k,
                r.summary.ndcg_at_k,
                r.summary.mrr,
                f,
            );
        } else {
            println!(
                "{:>8.2}  {:>10.3}  {:>8.3}  {:>10.3}  {:>8.3}",
                r.semantic,
                r.summary.recall_at_k,
                r.summary.hit_at_k,
                r.summary.ndcg_at_k,
                r.summary.mrr,
            );
        }
    }
    println!();

    if args.json {
        let report = EvalReport {
            corpus_path: args.corpus.display().to_string(),
            golden_path: args.golden.display().to_string(),
            golden_name: golden.name.clone(),
            backend: args.backend.as_str().to_string(),
            corpus_size: corpus.len(),
            queries: golden.items.len(),
            k: args.k,
            rows,
        };
        println!(
            "{}",
            serde_json::to_string(&report).context("serialize report")?
        );
    }

    Ok(())
}
