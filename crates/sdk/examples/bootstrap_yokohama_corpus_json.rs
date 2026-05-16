//! Yokohama 市税条例 corpus を `~/Library/Application Support/ellisii/ellisii.db`
//! から抜き出して、`crates/rag/tests/fixtures/eval/yokohama/{corpus,golden}.json`
//! に書き出す bootstrap。これで Run 8 (`eval_tokenizer_facade`) が yokohama
//! でも動くようになる (in-memory sqlite で 4-way tokenizer A/B が取れる)。
//!
//! doc_id は `yokohama-{ord}` (ord = chunks.ord)。再 index で UUID がずれても
//! ord は安定 (= 条文番号順) なので、golden の `relevant` を doc_id ベースに
//! しておけば fixture を git に commit して以降は外部 DB 依存ゼロで eval を
//! 再現できる。
//!
//! 使い方:
//! ```sh
//! cargo run -p ellisii-sdk --example bootstrap_yokohama_corpus_json --release
//! ```
//!
//! 既存 `eval_yokohama.rs` (= 既存 DB を直接舐める eval) には影響しない。
//! 既存 `bootstrap_yokohama_golden.rs` は別途 UUID ベース golden を
//! refresh する用途で残す。

use rusqlite::Connection;
use serde::Serialize;
use std::collections::HashMap;
use std::path::PathBuf;

const NOTEBOOK_ID: &str = "fad96121-5f29-4510-beb8-432d7f165872";

/// recall-evals.md の Q&A 表に書かれた authoritative (query, ord) ペア。
/// `bootstrap_yokohama_golden.rs` の Q_BY_ORD と同じものをコピー。
#[rustfmt::skip]
const Q_BY_ORD: &[(&str, u32)] = &[
    ("入湯税の税率はいくらですか", 325),
    ("温泉に入ったときに課される税の額", 325),
    ("都市計画税の税率は何パーセント", 349),
    ("事業所税の税率を教えて", 335),
    ("固定資産税の税率は", 230),
    ("法人税割の税率は", 71),
    ("個人市民税の所得割の税率", 69),
    ("個人の均等割は年いくら", 52),
    ("均等割が軽減されるのはどういう場合で、いくらになるか", 53),
    ("市たばこ税の税率", 294),
    ("固定資産税の納期はいつ", 232),
    ("延滞金の利率は何パーセント", 30),
    ("特別土地保有税の税率はいくら", 305),
    ("分離課税に係る所得割の税率", 182),
    ("徴税吏員とは何ですか", 2),
    ("市税の課税の根拠となる規定は", 1),
    ("横浜市が課する普通税にはどんな種類があるか", 3),
    ("退職金にかかる市民税の税率", 182),
    ("税金を期限までに払えなかったらどうなるか", 30),
    ("事業所税の事業所等とはどんな場所", 331),
    ("事業所税の納税義務者", 331),
    ("鉱泉浴場の経営者の役割", 326),
    ("入湯税はどう徴収されるか", 326),
    ("入湯税は誰が納めるのですか", 324),
    ("固定資産税の納税義務者は誰ですか", 220),
];

/// 同 caption-substring ベースのマップ (`bootstrap_yokohama_golden.rs` と同じ)。
#[rustfmt::skip]
const Q_BY_CAPTION: &[(&str, &str)] = &[
    ("市役所からの督促状はいつまでに発行されますか", "督促状"),
    ("徴収を猶予してもらう手続きは何条", "徴収猶予"),
    ("災害や病気で申告できなかったときの救済", "災害等による期限の延長"),
    ("普通徴収の納税通知書はいつまでに納税者へ届くか", "普通徴収の方法による納税通知書"),
    ("ふるさと納税で寄附金税額控除を受けるための申告は", "寄附金税額控除"),
    ("不動産を 5 年超持ってから売った場合の市民税", "長期譲渡"),
    ("公示送達はどう行うか", "公示送達の方法"),
    ("市たばこ税は誰が納めますか", "市たばこ税の納税義務者"),
    ("新築の長期優良住宅は固定資産税が減額されますか", "認定長期優良住宅"),
    ("課税ミスでもらいすぎた税金や取り損ねた税金の処理", "賦課もれ"),
    ("都市計画税の納期は何回に分かれているか", "都市計画税の納期"),
    ("市税徴収のための条例の施行に必要な事項は誰が定めるか", "委任"),
    ("市税の課税標準額の端数はどう計算しますか", "課税標準額、税額等の端数計算"),
    ("土地の価格がいくら以下なら固定資産税はかからない", "固定資産税の免税点"),
    ("固定資産税の免税点はいくら", "固定資産税の免税点"),
    ("軽自動車の環境性能割が免除される条件", "環境性能割の免税点"),
    ("都市計画税の賦課期日はいつ", "都市計画税の賦課期日"),
];

#[derive(Serialize)]
struct CorpusEntry {
    doc_id: String,
    title: String,
    caption: String,
    text: String,
}

fn home() -> PathBuf {
    PathBuf::from(std::env::var("HOME").expect("HOME"))
}
fn db_path() -> PathBuf {
    home().join("Library/Application Support/ellisii/ellisii.db")
}

/// 大本の text は `(caption)\n本文` 形式で保存されている。caption と本文を
/// 分けて corpus.json の column に分配する。caption が抽出できなければ caption=""。
fn split_caption(text: &str) -> (String, String) {
    let Some(rest) = text.strip_prefix('(') else {
        return (String::new(), text.trim().to_string());
    };
    if let Some(end) = rest.find(')') {
        let caption = &rest[..end];
        let body = rest[end + 1..].trim_start_matches('\n').trim().to_string();
        return (caption.to_string(), body);
    }
    (String::new(), text.trim().to_string())
}

fn pick_yokohama_source(conn: &Connection) -> rusqlite::Result<String> {
    // 1215 chunks を優先 (recall-evals.md ord と対応する版)。
    if let Ok(id) = conn.query_row(
        "SELECT source_id FROM chunks WHERE notebook_id=?1 GROUP BY source_id HAVING count(*)=1215 LIMIT 1",
        rusqlite::params![NOTEBOOK_ID],
        |row| row.get::<_, String>(0),
    ) {
        return Ok(id);
    }
    // fallback: notebook 内で最大の source
    conn.query_row(
        "SELECT source_id FROM chunks WHERE notebook_id=?1 GROUP BY source_id ORDER BY count(*) DESC LIMIT 1",
        rusqlite::params![NOTEBOOK_ID],
        |row| row.get(0),
    )
}

fn lookup_ord_by_caption(
    conn: &Connection,
    source_id: &str,
    caption_substring: &str,
) -> rusqlite::Result<Option<u32>> {
    let mut stmt = conn.prepare(
        "SELECT ord FROM chunks WHERE source_id=?1 AND text LIKE ?2 \
         ORDER BY ord ASC LIMIT 1",
    )?;
    let pattern = format!("(%{}%)\n%", caption_substring);
    let mut rows = stmt.query(rusqlite::params![source_id, pattern])?;
    if let Some(row) = rows.next()? {
        let ord: i64 = row.get(0)?;
        return Ok(Some(ord as u32));
    }
    Ok(None)
}

fn out_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .join("rag/tests/fixtures/eval/yokohama")
}

fn main() -> anyhow::Result<()> {
    let conn = Connection::open(db_path())?;
    let source_id = pick_yokohama_source(&conn).map_err(|e| {
        anyhow::anyhow!(
            "yokohama source not found in DB (notebook_id={NOTEBOOK_ID}): {e}\n\
             先に Tauri アプリで横浜市市税条例を index する必要がある。"
        )
    })?;
    eprintln!("yokohama source_id: {source_id}");

    // --- corpus.json ---
    let mut stmt =
        conn.prepare("SELECT ord, text FROM chunks WHERE source_id=?1 ORDER BY ord ASC")?;
    let rows = stmt.query_map(rusqlite::params![&source_id], |row| {
        let ord: i64 = row.get(0)?;
        let text: String = row.get(1)?;
        Ok((ord as u32, text))
    })?;
    let mut entries: Vec<CorpusEntry> = Vec::new();
    for r in rows {
        let (ord, text) = r?;
        let (caption, body) = split_caption(&text);
        let title = if caption.is_empty() {
            format!("第{ord}条")
        } else {
            format!("{caption} (ord={ord})")
        };
        entries.push(CorpusEntry {
            doc_id: format!("yokohama-{ord}"),
            title,
            caption,
            text: body,
        });
    }
    eprintln!("corpus: {} chunks", entries.len());
    let dir = out_dir();
    std::fs::create_dir_all(&dir)?;
    let corpus_path = dir.join("corpus.json");
    std::fs::write(&corpus_path, serde_json::to_string_pretty(&entries)?)?;
    eprintln!("wrote {}", corpus_path.display());

    // --- golden.json (ord ベース) ---
    let by_ord: HashMap<&str, u32> = Q_BY_ORD.iter().copied().collect();
    let by_caption: HashMap<&str, &str> = Q_BY_CAPTION.iter().copied().collect();

    // 既存 golden を読み込み、各 query → doc_id に変換。
    let src_golden =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/eval/yokohama/golden.json");
    let raw = std::fs::read_to_string(&src_golden)?;
    let value: serde_json::Value = serde_json::from_str(&raw)?;
    let items = value
        .get("items")
        .and_then(|v| v.as_array())
        .ok_or_else(|| anyhow::anyhow!("golden.json has no items[]"))?;

    let mut new_items: Vec<serde_json::Value> = Vec::with_capacity(items.len());
    let mut mapped = 0usize;
    let mut unmapped = 0usize;
    for it in items {
        let query = it.get("query").and_then(|v| v.as_str()).unwrap_or("");
        let tags = it.get("tags").cloned().unwrap_or(serde_json::json!([]));
        let mut relevant_ord: Option<u32> = by_ord.get(query).copied();
        if relevant_ord.is_none() {
            if let Some(cap) = by_caption.get(query) {
                relevant_ord = lookup_ord_by_caption(&conn, &source_id, cap)?;
            }
        }
        let relevant_ids = match relevant_ord {
            Some(ord) => {
                mapped += 1;
                vec![format!("yokohama-{ord}")]
            }
            None => {
                unmapped += 1;
                eprintln!("  ?? unmapped: {query}");
                vec![]
            }
        };
        new_items.push(serde_json::json!({
            "query": query,
            "relevant": relevant_ids,
            "tags": tags,
        }));
    }
    eprintln!(
        "golden: mapped={mapped} unmapped={unmapped} total={}",
        items.len()
    );

    let new_golden = serde_json::json!({
        "name": "yokohama-shizei-doc-id-v1",
        "items": new_items,
    });
    let golden_path = dir.join("golden.json");
    std::fs::write(&golden_path, serde_json::to_string_pretty(&new_golden)?)?;
    eprintln!("wrote {}", golden_path.display());

    Ok(())
}
