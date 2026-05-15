//! `crates/rag/tests/fixtures/eval/<corpus>/{corpus.json, golden.json}` を全件回して
//! caption rerank の on/off で recall@K を比較する。
//!
//! corpus エントリは `{doc_id, title, caption, text}` の構造。caption を chunk text の
//! 先頭に `(caption)\n` で埋め込んでから index することで、本番 ingest と同じ
//! `(キャプション)\n本文` 形式で rerank が効くようにする。
//!
//! 実行:
//! ```sh
//! cargo run -p ellisii-sdk --example eval_fixtures --release
//! ```
//!
//! このハーネスを更新したら `docs/eval/recall-evals.md` も追記すること。

use std::path::PathBuf;
use std::sync::Arc;

use async_trait::async_trait;
use ellisii_core::{Chunk, Result};
use ellisii_embed_core::Embedder;
use ellisii_query_rewriter_core::QueryRewriter;
use ellisii_rag::eval::{summarize, EvalSummary, GoldenSet};
use ellisii_sdk::{Ellisii, SearchOptions};
use serde::Deserialize;
use std::collections::HashMap;
use uuid::Uuid;

#[derive(Debug, Deserialize)]
struct CorpusEntry {
    doc_id: String,
    #[allow(dead_code)]
    title: String,
    caption: String,
    text: String,
    /// Optional 親 doc id。複数 entry が同じ `parent_id` を持つときは
    /// `build_chunks` 内で **同一 `source_id`** を割り当て、
    /// `dedup_by_source_in_place` を parent-level dedup として機能させる。
    /// id_map も `parent_id` を指すよう書き換えるため、golden の `relevant`
    /// は parent_id を列挙する形式になる (jp-manual fixture を参照)。
    /// 既存 corpus (parent_id 無し) は doc_id をそのまま source 単位として扱う
    /// 旧挙動が維持される。
    #[serde(default)]
    parent_id: Option<String>,
}

/// 文字バイグラムを次元 D にハッシュバケットへ落とす決定的 embedder
/// (`crates/rag/tests/end_to_end_eval.rs` と同じ方針)。
struct BigramHashEmbedder {
    dim: usize,
}

#[async_trait]
impl Embedder for BigramHashEmbedder {
    fn dim(&self) -> usize {
        self.dim
    }
    async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        Ok(texts.iter().map(|t| bigram_vec(t, self.dim)).collect())
    }
}

fn bigram_vec(text: &str, dim: usize) -> Vec<f32> {
    let mut v = vec![0.0_f32; dim];
    let chars: Vec<char> = text.chars().filter(|c| !c.is_whitespace()).collect();
    if chars.len() < 2 {
        for c in &chars {
            let idx = (fnv1a(&c.to_string()) as usize) % dim;
            v[idx] += 1.0;
        }
        normalize(&mut v);
        return v;
    }
    for w in chars.windows(2) {
        let s: String = w.iter().collect();
        let idx = (fnv1a(&s) as usize) % dim;
        v[idx] += 1.0;
    }
    normalize(&mut v);
    v
}

fn normalize(v: &mut [f32]) {
    let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if n > 0.0 {
        for x in v.iter_mut() {
            *x /= n;
        }
    }
}

fn fnv1a(s: &str) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for b in s.as_bytes() {
        h ^= *b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

fn fixtures_root() -> PathBuf {
    // crates/sdk → workspace root
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .join("rag/tests/fixtures/eval")
}

/// golden の `relevant` がどの粒度の id を列挙しているか。
/// Run 48 / Run 58 で 2 種類の使い分けが発生:
/// - `DocId`: 1 chunk = 1 doc_id を直接列挙 (例: jp-civil-law, jp-civil-law-hard,
///   jp-cs-wiki, jp-multihop ... 通常の場合)
/// - `ParentId`: 親 doc 単位を列挙 (例: jp-manual。子は分割した章で、recall は
///   どの親まで届いたかで測る)
///
/// auto-detect: golden の relevant 全体が corpus の **parent_id 集合に含まれる** なら
/// `ParentId`、そうでなければ `DocId`。誤検出を避けるため、両集合に重複する文字列
/// (= 親と子が同名) があった場合は doc_id 優先 (= 既存挙動)。
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum GoldenGranularity {
    DocId,
    ParentId,
}

fn detect_golden_granularity(corpus: &[CorpusEntry], golden: &GoldenSet) -> GoldenGranularity {
    use std::collections::HashSet;
    let doc_ids: HashSet<&str> = corpus.iter().map(|e| e.doc_id.as_str()).collect();
    let parent_ids: HashSet<&str> = corpus
        .iter()
        .filter_map(|e| e.parent_id.as_deref())
        .collect();
    let golden_relevant: HashSet<&str> = golden
        .items
        .iter()
        .flat_map(|i| i.relevant.iter().map(|s| s.as_str()))
        .collect();
    if golden_relevant.is_empty() {
        return GoldenGranularity::DocId;
    }
    // doc_id に全部マッチするなら DocId (priority)。
    if golden_relevant.iter().all(|r| doc_ids.contains(r)) {
        return GoldenGranularity::DocId;
    }
    // parent_id に全部マッチするなら ParentId。
    if golden_relevant.iter().all(|r| parent_ids.contains(r)) {
        return GoldenGranularity::ParentId;
    }
    // どちらにも完全には当てはまらないときは DocId にフォールバック (旧挙動)。
    GoldenGranularity::DocId
}

/// 一度 chunk + UUID を組み立てて、同じ corpus を 2 つの Ellisii インスタンス
/// (rewriter なし / あり) に同じ ID で index する。id_map は共有可能になる。
fn build_chunks(
    corpus: &[CorpusEntry],
    granularity: GoldenGranularity,
) -> (Uuid, Vec<Chunk>, Vec<String>, HashMap<Uuid, String>) {
    let nb = Uuid::new_v4();
    let mut chunks = Vec::new();
    let mut texts = Vec::new();
    let mut id_map: HashMap<Uuid, String> = HashMap::new();
    // parent_id → 共有 source_id。同じ親に属する子 entry は同一 source として扱う。
    let mut source_by_parent: HashMap<String, Uuid> = HashMap::new();
    for (i, e) in corpus.iter().enumerate() {
        let cid = Uuid::new_v4();
        let group_id = e.parent_id.clone().unwrap_or_else(|| e.doc_id.clone());
        // id_map は predicted → golden 比較に使う。granularity が ParentId なら
        // group_id (= parent_id があれば parent_id、なければ doc_id) を、
        // DocId なら doc_id を直接マップする。dedup 用の source_id は常に
        // parent_id 単位で共有される (= dedup は parent-level dedup として機能)。
        let label = match granularity {
            GoldenGranularity::ParentId => group_id.clone(),
            GoldenGranularity::DocId => e.doc_id.clone(),
        };
        id_map.insert(cid, label);
        let sid = *source_by_parent
            .entry(group_id.clone())
            .or_insert_with(Uuid::new_v4);
        let txt = if e.caption.is_empty() {
            e.text.clone()
        } else {
            format!("({})\n{}", e.caption, e.text)
        };
        // heading_path: parent-aware corpus では `title` (実際の見出し) を、
        // それ以外は doc_id を渡す。これにより jp-manual のような Markdown 様コーパスで
        // `heading_path[-1]` ベースのシグナル (query_title_match / heading_boost) が
        // 実見出しを参照できるようになる。
        let heading = if e.parent_id.is_some() && !e.title.is_empty() {
            e.title.clone()
        } else {
            e.doc_id.clone()
        };
        chunks.push(Chunk {
            id: cid,
            source_id: sid,
            ord: i as u32,
            text: txt.clone(),
            heading_path: vec![heading],
            page: None,
            bbox: None,
            summary: None,
        });
        texts.push(txt);
    }
    (nb, chunks, texts, id_map)
}

async fn build_ellisii(
    nb: Uuid,
    chunks: &[Chunk],
    texts: &[String],
    rewriter: Option<Arc<dyn QueryRewriter>>,
) -> anyhow::Result<Ellisii> {
    let dim = 256;
    let embedder = BigramHashEmbedder { dim };
    let mut builder = Ellisii::builder()
        .with_embedder(Arc::new(BigramHashEmbedder { dim }))
        .with_store_memory()
        .with_notebook_id(nb);
    if let Some(r) = rewriter {
        builder = builder.with_query_rewriter(r);
    }
    let ellisii = builder.build()?;
    let embs = embedder.embed(texts).await?;
    ellisii.store().upsert(nb, chunks, &embs).await?;
    Ok(ellisii)
}

/// `ELLISII_EVAL_DUMP_MISSES=1` (任意で値に corpus 名フィルタを書ける) のとき、
/// `cap+auto` variant で該当 corpus / k の hit@k=0 になった query を 1 行ずつ
/// 詳細出力する。Run 31 の TODO「harness に `--dump-misses` オプション追加」の
/// 実装版で、env var は CLI 引数解析不要かつ既存挙動を変えない形にした。
fn dump_misses_enabled_for(corpus_name: &str) -> bool {
    match std::env::var("ELLISII_EVAL_DUMP_MISSES") {
        Ok(v) if v == "1" || v.eq_ignore_ascii_case("true") => true,
        Ok(v) if !v.is_empty() => corpus_name.contains(&v),
        _ => false,
    }
}

async fn eval_corpus(
    name: &str,
    corpus: &[CorpusEntry],
    golden: &GoldenSet,
    k: usize,
) -> anyhow::Result<()> {
    let granularity = detect_golden_granularity(corpus, golden);
    let (nb, chunks, texts, id_map) = build_chunks(corpus, granularity);
    let ellisii_plain = build_ellisii(nb, &chunks, &texts, None).await?;
    let mut on_pairs: Vec<(Vec<String>, Vec<String>)> = Vec::new();
    let mut off_pairs: Vec<(Vec<String>, Vec<String>)> = Vec::new();
    let mut auto_pairs: Vec<(Vec<String>, Vec<String>)> = Vec::new();
    // parent_id を持つ corpus でだけ dedup A/B を測る。それ以外は 1 doc = 1 source の
    // 構造で no-op になるため計算しない (列を出すと混乱するので)。
    let has_parents = corpus.iter().any(|e| e.parent_id.is_some());
    let mut dedup_pairs: Vec<(Vec<String>, Vec<String>)> = Vec::new();
    let mut dedup2_pairs: Vec<(Vec<String>, Vec<String>)> = Vec::new();
    let mut heading_pairs: Vec<(Vec<String>, Vec<String>)> = Vec::new();
    let mut combo_pairs: Vec<(Vec<String>, Vec<String>)> = Vec::new();
    let mut miss_rows: Vec<(String, Vec<String>, Vec<String>)> = Vec::new();
    let dump_misses = dump_misses_enabled_for(name);
    for item in &golden.items {
        let off = ellisii_plain
            .search(
                &item.query,
                SearchOptions {
                    top_k: k,
                    semantic_weight: 0.5,
                    caption_rerank: false,
                    auto_adjust_weight: false,
                    ..Default::default()
                },
            )
            .await?;
        let on = ellisii_plain
            .search(
                &item.query,
                SearchOptions {
                    top_k: k,
                    semantic_weight: 0.5,
                    caption_rerank: true,
                    auto_adjust_weight: false,
                    ..Default::default()
                },
            )
            .await?;
        let auto = ellisii_plain
            .search(
                &item.query,
                SearchOptions {
                    top_k: k,
                    semantic_weight: 0.5,
                    caption_rerank: true,
                    auto_adjust_weight: true,
                    ..Default::default()
                },
            )
            .await?;
        // parent_id ありの corpus でのみ dedup A/B を測る。max_chunks_per_source=1 で
        // 同一 source = 同一 parent から 1 件のみに制限し、上位の親多様性を比較する。
        let dedup_hits = if has_parents {
            ellisii_plain
                .search(
                    &item.query,
                    SearchOptions {
                        top_k: k,
                        semantic_weight: 0.5,
                        caption_rerank: true,
                        auto_adjust_weight: true,
                        max_chunks_per_source: 1,
                        ..Default::default()
                    },
                )
                .await?
        } else {
            Vec::new()
        };
        let dedup2_hits = if has_parents {
            ellisii_plain
                .search(
                    &item.query,
                    SearchOptions {
                        top_k: k,
                        semantic_weight: 0.5,
                        caption_rerank: true,
                        auto_adjust_weight: true,
                        max_chunks_per_source: 2,
                        ..Default::default()
                    },
                )
                .await?
        } else {
            Vec::new()
        };
        // heading_rerank: 全 corpus で測定。fixture corpora の heading_path は
        // doc_id (= 第N条) または title (parent-aware) なので、後者で signal が
        // 出るかを見たい。
        let heading_hits = ellisii_plain
            .search(
                &item.query,
                SearchOptions {
                    top_k: k,
                    semantic_weight: 0.5,
                    caption_rerank: true,
                    auto_adjust_weight: true,
                    heading_rerank: true,
                    ..Default::default()
                },
            )
            .await?;
        // parent-aware corpus でのみ heading + dedup1 の合成を測る。
        let combo_hits = if has_parents {
            ellisii_plain
                .search(
                    &item.query,
                    SearchOptions {
                        top_k: k,
                        semantic_weight: 0.5,
                        caption_rerank: true,
                        auto_adjust_weight: true,
                        heading_rerank: true,
                        max_chunks_per_source: 1,
                        ..Default::default()
                    },
                )
                .await?
        } else {
            Vec::new()
        };
        off_pairs.push((
            off.iter()
                .filter_map(|h| id_map.get(&h.chunk.id).cloned())
                .collect(),
            item.relevant.clone(),
        ));
        on_pairs.push((
            on.iter()
                .filter_map(|h| id_map.get(&h.chunk.id).cloned())
                .collect(),
            item.relevant.clone(),
        ));
        let auto_pred: Vec<String> = auto
            .iter()
            .filter_map(|h| id_map.get(&h.chunk.id).cloned())
            .collect();
        if dump_misses && !auto_pred.iter().any(|p| item.relevant.contains(p)) {
            miss_rows.push((item.query.clone(), item.relevant.clone(), auto_pred.clone()));
        }
        auto_pairs.push((auto_pred, item.relevant.clone()));
        if has_parents {
            let dedup_pred: Vec<String> = dedup_hits
                .iter()
                .filter_map(|h| id_map.get(&h.chunk.id).cloned())
                .collect();
            dedup_pairs.push((dedup_pred, item.relevant.clone()));
            let dedup2_pred: Vec<String> = dedup2_hits
                .iter()
                .filter_map(|h| id_map.get(&h.chunk.id).cloned())
                .collect();
            dedup2_pairs.push((dedup2_pred, item.relevant.clone()));
        }
        let heading_pred: Vec<String> = heading_hits
            .iter()
            .filter_map(|h| id_map.get(&h.chunk.id).cloned())
            .collect();
        heading_pairs.push((heading_pred, item.relevant.clone()));
        if has_parents {
            let combo_pred: Vec<String> = combo_hits
                .iter()
                .filter_map(|h| id_map.get(&h.chunk.id).cloned())
                .collect();
            combo_pairs.push((combo_pred, item.relevant.clone()));
        }
    }
    let s_off = summarize(&off_pairs, k);
    let s_on = summarize(&on_pairs, k);
    let s_auto = summarize(&auto_pairs, k);
    let s_dedup = if has_parents {
        Some(summarize(&dedup_pairs, k))
    } else {
        None
    };
    let s_dedup2 = if has_parents {
        Some(summarize(&dedup2_pairs, k))
    } else {
        None
    };
    let s_heading = summarize(&heading_pairs, k);
    let s_combo = if has_parents {
        Some(summarize(&combo_pairs, k))
    } else {
        None
    };
    println!(
        "{:<28} n={:<3} k={:<3} off: hit={:.3} mrr={:.3} rec={:.3}  cap: hit={:.3} mrr={:.3} rec={:.3}  cap+auto: hit={:.3} mrr={:.3} rec={:.3}",
        name,
        golden.items.len(),
        k,
        s_off.hit_at_k, s_off.mrr, s_off.recall_at_k,
        s_on.hit_at_k, s_on.mrr, s_on.recall_at_k,
        s_auto.hit_at_k, s_auto.mrr, s_auto.recall_at_k,
    );
    if let Some(s) = &s_dedup {
        println!(
            "{:<28} n={:<3} k={:<3}     dedup1: hit={:.3} mrr={:.3} rec={:.3}  Δrec vs cap+auto={:+.3}",
            name,
            golden.items.len(),
            k,
            s.hit_at_k, s.mrr, s.recall_at_k,
            s.recall_at_k - s_auto.recall_at_k,
        );
    }
    if let Some(s) = &s_dedup2 {
        println!(
            "{:<28} n={:<3} k={:<3}     dedup2: hit={:.3} mrr={:.3} rec={:.3}  Δrec vs cap+auto={:+.3}",
            name,
            golden.items.len(),
            k,
            s.hit_at_k, s.mrr, s.recall_at_k,
            s.recall_at_k - s_auto.recall_at_k,
        );
    }
    println!(
        "{:<28} n={:<3} k={:<3}    heading: hit={:.3} mrr={:.3} rec={:.3}  Δrec vs cap+auto={:+.3}",
        name,
        golden.items.len(),
        k,
        s_heading.hit_at_k,
        s_heading.mrr,
        s_heading.recall_at_k,
        s_heading.recall_at_k - s_auto.recall_at_k,
    );
    if let Some(s) = &s_combo {
        println!(
            "{:<28} n={:<3} k={:<3} head+dedup1: hit={:.3} mrr={:.3} rec={:.3}  Δrec vs cap+auto={:+.3}",
            name,
            golden.items.len(),
            k,
            s.hit_at_k, s.mrr, s.recall_at_k,
            s.recall_at_k - s_auto.recall_at_k,
        );
    }
    if dump_misses && !miss_rows.is_empty() {
        println!(
            "  ↳ misses (cap+auto, k={k}): {}/{}",
            miss_rows.len(),
            golden.items.len()
        );
        for (q, expected, pred) in &miss_rows {
            let pred_show = if pred.is_empty() {
                "[]".into()
            } else {
                format!("[{}]", pred.join(", "))
            };
            println!("    • q=「{q}」 expected={expected:?} predicted={pred_show}");
        }
    }
    Ok(())
}

fn _unused(s: &EvalSummary) -> &EvalSummary {
    s
}

#[allow(unused_variables)]
async fn run_corpus_dir(dir: &PathBuf) -> anyhow::Result<()> {
    let name = dir.file_name().unwrap().to_string_lossy().to_string();
    let corpus_json = std::fs::read_to_string(dir.join("corpus.json"))?;
    let corpus: Vec<CorpusEntry> = serde_json::from_str(&corpus_json)?;
    let golden_json = std::fs::read_to_string(dir.join("golden.json"))?;
    let golden = GoldenSet::from_json_str(&golden_json)?;
    for k in [1usize, 2, 3, 5, 10] {
        eval_corpus(&name, &corpus, &golden, k).await?;
    }
    Ok(())
}

/// 各 corpus の caption_density / corpus_paraphrase_score / specific_query_ratio を
/// 計算して 1 行で印刷する。Run 18 (paraphrase_score) / Run 20 (hypothesis 訂正) /
/// Run 21 (query 側 signal) を参照。
async fn print_corpus_signals(dir: &PathBuf) -> anyhow::Result<()> {
    let name = dir.file_name().unwrap().to_string_lossy().to_string();
    let corpus_json = std::fs::read_to_string(dir.join("corpus.json"))?;
    let corpus: Vec<CorpusEntry> = serde_json::from_str(&corpus_json)?;
    let golden_json = std::fs::read_to_string(dir.join("golden.json"))?;
    let golden = GoldenSet::from_json_str(&golden_json)?;
    let queries: Vec<&str> = golden.items.iter().map(|i| i.query.as_str()).collect();
    let q_specific = ellisii_rag::specific_query_ratio(&queries);
    let bodies: Vec<&str> = corpus.iter().map(|e| e.text.as_str()).collect();
    let q_body_recall = ellisii_rag::query_body_recall_mean(&queries, &bodies);

    let granularity = detect_golden_granularity(&corpus, &golden);
    let (nb, chunks, texts, _id_map) = build_chunks(&corpus, granularity);
    let ellisii = build_ellisii(nb, &chunks, &texts, None).await?;
    let density = ellisii.caption_density().await?;
    let head_density = ellisii.heading_density().await?;
    let para = ellisii.corpus_paraphrase_score().await?;
    // Run 20-22 の判断ロジック。strong-OFF をいち早く出すために
    // (a) q_specific >= 0.5 (specific 偏重) でも (b) q_body_recall >= 0.7
    // (literal lookup でクエリが既に body 語彙を網羅) でも rewriter≈OFF を推奨する。
    let recommendation = if q_specific >= 0.5 {
        "rewriter≈OFF (specific 偏重)"
    } else if q_body_recall >= 0.7 {
        "rewriter≈OFF (literal lookup, body recall 高)"
    } else if q_specific < 0.3 && q_body_recall < 0.4 {
        "rewriter ON (paraphrase ROI 期待)"
    } else {
        "mix (per-query gate に委ねる)"
    };
    let heading_recommendation = if head_density >= 0.8 {
        "heading_rerank ON 推奨"
    } else if head_density >= 0.4 {
        "heading_rerank opt-in 可"
    } else {
        "heading_rerank OFF 推奨"
    };
    println!(
        "{:<28} density={:.3}  head_density={:.3}  paraphrase={:.3}  q_specific={:.3}  q_body_recall={:.3}  → {} / {}",
        name, density, head_density, para, q_specific, q_body_recall, recommendation, heading_recommendation
    );
    Ok(())
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> anyhow::Result<()> {
    let root = fixtures_root();
    let mut entries: Vec<PathBuf> = std::fs::read_dir(&root)?
        .filter_map(|r| r.ok())
        .map(|e| e.path())
        .filter(|p| p.is_dir())
        .collect();
    entries.sort();
    println!("Fixtures dir: {}", root.display());

    println!("\n=== Corpus signals (caption_density / paraphrase_score, Run 18) ===");
    for d in &entries {
        print_corpus_signals(d).await?;
    }

    println!("\n=== Per-corpus caption rerank A/B (semantic_weight=0.5) ===");
    for d in &entries {
        run_corpus_dir(d).await?;
    }
    Ok(())
}
