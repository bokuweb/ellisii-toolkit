//! 永続化ストア (sqlite + sqlite-vec + FTS5) で index → search する例。
//!
//! 1 度実行するとディレクトリが index されて DB に保存され、2 回目以降は
//! index を skip して即座に検索できます (UPSERT 方式)。
//!
//! 実行:
//! ```sh
//! cargo run -p ellisii-sdk --example index_persistent -- ./docs "クエリ" ./mydata.db
//! ```

use ellisii_sdk::{Ellisii, IndexOptions, SearchOptions};

#[tokio::main(flavor = "current_thread")]
async fn main() -> anyhow::Result<()> {
    let mut args = std::env::args().skip(1);
    let dir = args
        .next()
        .ok_or_else(|| anyhow::anyhow!("usage: index_persistent <dir> <query> <db_path>"))?;
    let query = args
        .next()
        .ok_or_else(|| anyhow::anyhow!("missing query"))?;
    let db_path = args
        .next()
        .ok_or_else(|| anyhow::anyhow!("missing db_path"))?;

    // dummy embedder dim を sqlite store と一致させる必要がある。
    let dim = 64usize;
    let ellisii = Ellisii::builder()
        .with_embedder_dummy(dim)
        .with_store_sqlite(&db_path, dim)?
        .build()?;

    eprintln!("indexing {dir} → {db_path}");
    let report = ellisii.index_dir(&dir, IndexOptions::default()).await?;
    eprintln!(
        "done: ingested={} chunks={} skipped={} failed={}",
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
    for (i, h) in hits.iter().enumerate() {
        let preview: String = h.chunk.text.chars().take(80).collect();
        println!("[{:>2}] score={:.3}  {}", i + 1, h.score, preview);
    }
    Ok(())
}
