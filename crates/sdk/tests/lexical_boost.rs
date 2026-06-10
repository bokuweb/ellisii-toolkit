//! SearchOptions.lexical_boost が「クエリ語が本文に濃く出る chunk」を
//! 上位に押し上げることを確認する。外部モデル不要 (DummyEmbedder + InMemoryStore)。

use ellisii_sdk::{Ellisii, IndexOptions, SearchOptions};

#[tokio::test]
async fn lexical_boost_reorders_toward_term_coverage() {
    let tmp = tempfile::tempdir().unwrap();
    let docs = tmp.path().join("docs");
    std::fs::create_dir_all(&docs).unwrap();
    // クエリ語をほとんど含まない長文 (dummy embedder では先に来がち)
    std::fs::write(
        docs.join("unrelated.txt"),
        "本資料は天気と料理と旅行について述べた一般的な読み物であり、専門用語は含みません。",
    )
    .unwrap();
    // クエリ語 (秘密保持義務 / 存続) を濃く含む条項
    std::fs::write(
        docs.join("nda.txt"),
        "秘密保持義務は契約終了後も三年間存続するものとし、受領当事者は秘密情報を第三者に開示してはならない。",
    )
    .unwrap();

    let ellisii = Ellisii::builder()
        .with_embedder_dummy(64)
        .with_store_memory()
        .build()
        .unwrap();
    ellisii
        .index_dir(&docs, IndexOptions::default())
        .await
        .unwrap();

    let query = "秘密保持義務の存続期間";
    let opts = SearchOptions {
        lexical_boost: true,
        ..SearchOptions::default()
    };
    let hits = ellisii.search(query, opts).await.unwrap();
    assert!(!hits.is_empty());
    // lexical_boost によりクエリ語を濃く含む chunk が先頭に来る
    assert!(
        hits[0].chunk.text.contains("秘密保持義務"),
        "expected nda chunk on top, got: {hits:?}"
    );
}
