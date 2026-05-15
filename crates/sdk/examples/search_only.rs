//! 最小例: ディレクトリを index して検索する (LLM 不要 / モデル不要)。
//!
//! 実行:
//! ```sh
//! cargo run -p ellisii-sdk --example search_only -- ./docs "クエリ文字列"
//! ```
//!
//! 注意: DummyEmbedder はランダム vector を返すだけなので、ベクトル検索の
//! 精度は出ません。「動かし方の確認」用です。実用には `static-jp` feature
//! などを使ってください (`docs/sdk.md` 参照)。

use ellisii_sdk::{Ellisii, IndexEvent, IndexOptions, SearchOptions};

#[tokio::main(flavor = "current_thread")]
async fn main() -> anyhow::Result<()> {
    let mut args = std::env::args().skip(1);
    let dir = args
        .next()
        .ok_or_else(|| anyhow::anyhow!("usage: search_only <dir> <query>"))?;
    let query = args.next().unwrap_or_else(|| "demo".to_string());

    let ellisii = Ellisii::builder()
        .with_embedder_dummy(64)
        .with_store_memory()
        .build()?;

    let report = ellisii
        .index_dir(
            &dir,
            IndexOptions {
                on_progress: Some(Box::new(|ev: IndexEvent| match ev {
                    IndexEvent::Started { path } => {
                        eprintln!(" → ingesting {}", path.display())
                    }
                    IndexEvent::Ingested { path, chunks } => {
                        eprintln!(" ✓ {} ({} chunks)", path.display(), chunks)
                    }
                    IndexEvent::Unchanged { path } => {
                        eprintln!(" = unchanged {}", path.display())
                    }
                    IndexEvent::Skipped { path, reason } => {
                        eprintln!(" - skipped {}: {}", path.display(), reason)
                    }
                    IndexEvent::Failed { path, error } => {
                        eprintln!(" ✗ {}: {}", path.display(), error)
                    }
                })),
                ..Default::default()
            },
        )
        .await?;
    eprintln!(
        "indexed {} files ({} chunks); skipped {}, failed {}",
        report.ingested, report.total_chunks, report.skipped, report.failed
    );

    let hits = ellisii
        .search(
            &query,
            SearchOptions {
                top_k: 5,
                ..Default::default()
            },
        )
        .await?;
    println!("\nTop {} hits for \"{}\":", hits.len(), query);
    for (i, h) in hits.iter().enumerate() {
        let preview: String = h.chunk.text.chars().take(80).collect();
        println!("  [{:>2}] score={:.3}  {}", i + 1, h.score, preview);
    }
    Ok(())
}
