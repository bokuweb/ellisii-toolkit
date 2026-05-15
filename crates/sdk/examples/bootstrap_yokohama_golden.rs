//! `crates/sdk/tests/fixtures/eval/yokohama/golden.json` の chunk-id 引き直し。
//!
//! Yokohama 市税条例 corpus は何度か再 index されており、その度に chunk-id (uuid)
//! が振り直されるため、golden の `relevant` が stale になる (= eval が全件 0 になる)。
//! 本ツールは:
//!   1. 元 golden の各 query について、recall-evals.md の Q&A 表に書かれた
//!      authoritative `chunk_ord` を使って current db から uuid を引く。
//!   2. ord が分からない expansion items (n=17→26, n=26→42 で追加されたもの) は
//!      commit 0737d68 / a594beb の説明に書かれた article 番号や本文キーワードから
//!      LIKE 検索で候補を見つける。
//!   3. それでも当たらない / 曖昧な item は警告を出して旧 uuid を維持する。
//!
//! 出力は新 golden.json (stdout)。`> path` でリダイレクトして上書きする想定。
//!
//! 実行:
//! ```sh
//! cargo run -p ellisii-sdk --example bootstrap_yokohama_golden --release \
//!   > crates/sdk/tests/fixtures/eval/yokohama/golden.json
//! ```
//!
//! 必須:
//! - `~/Library/Application Support/ellisii/ellisii.db` に Yokohama corpus が
//!   index 済みであること (notebook_id = `95339065-df88-4ee7-82c1-e11c587250e4`)。

use rusqlite::{params, Connection};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;

#[derive(Debug, Clone, Deserialize, Serialize)]
struct Golden {
    name: String,
    items: Vec<GoldenItem>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
struct GoldenItem {
    query: String,
    relevant: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    tags: Vec<String>,
}

const NOTEBOOK_ID: &str = "95339065-df88-4ee7-82c1-e11c587250e4";
/// recall-evals.md は 1215-chunk の original index を参照しているので、それを引き当てる。
///
/// 同 notebook に複数の yokohama re-index がある場合、chunking パラメータの違いで
/// 1215 / 1283 / その他 のサイズが混在する。recall-evals の `chunk_ord` を信用する
/// ためには 1215 版を使うのが正解 (= ord と内容が表と一致する)。
///
/// `ELLISII_YOKOHAMA_SOURCE_ID` env で明示上書き可能。
fn yokohama_source_id(conn: &Connection) -> rusqlite::Result<String> {
    if let Ok(forced) = std::env::var("ELLISII_YOKOHAMA_SOURCE_ID") {
        if !forced.trim().is_empty() {
            return Ok(forced);
        }
    }
    // 1215 chunks を優先。無ければ最大 source にフォールバック。
    if let Ok(id) = conn.query_row(
        "SELECT source_id FROM chunks WHERE notebook_id=?1 GROUP BY source_id HAVING count(*)=1215 LIMIT 1",
        params![NOTEBOOK_ID],
        |row| row.get::<_, String>(0),
    ) {
        return Ok(id);
    }
    conn.query_row(
        "SELECT source_id FROM chunks WHERE notebook_id=?1 GROUP BY source_id ORDER BY count(*) DESC LIMIT 1",
        params![NOTEBOOK_ID],
        |row| row.get(0),
    )
}

/// (query 文字列, ord) のペア。recall-evals.md の Q&A 表 (n=17) から引いてきた
/// 一次資料。expansion で追加された paraphrase 系も同じ ord を共有する。
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
    // expansion (Run 5/10) で旧 paraphrase / hard。commit a594beb / 0737d68 の
    // 説明と本文の手 grep から article → ord を引いたもの:
    ("退職金にかかる市民税の税率", 182),                    // 同上 (分離課税)
    ("税金を期限までに払えなかったらどうなるか", 30),       // 延滞金 (第14条)
    // v5 (Run 40): 「事業所等」の定義は第129条 (caption "事業所税の納税義務者等")
    // = ord 331 にあるため、税率本則 (ord 335) ではなく定義側を期待値にする。
    // 同じ chunk に両方の paraphrase が当たるのは恣意的だったので解消。
    ("事業所税の事業所等とはどんな場所", 331),
    ("事業所税の納税義務者", 331),
    ("鉱泉浴場の経営者の役割", 326),                        // 入湯税の特別徴収義務者
    ("入湯税はどう徴収されるか", 326),                      // 同上
    ("入湯税は誰が納めるのですか", 324),                    // 入湯税納税義務者
    ("固定資産税の納税義務者は誰ですか", 220),              // 第49条系
];

/// expansion 群のうち、ord を断定できないものを caption (chunk 先頭の `(...)` ラベル)
/// で検索する。`text LIKE '(%caption_substring%)%'` で先頭一致するので、本文中に
/// 同じ語が散らばっていても誤マッチしにくい。
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

fn home() -> PathBuf {
    PathBuf::from(std::env::var("HOME").expect("HOME"))
}

fn db_path() -> PathBuf {
    home().join("Library/Application Support/ellisii/ellisii.db")
}

fn lookup_id_by_ord(conn: &Connection, source_id: &str, ord: u32) -> rusqlite::Result<String> {
    conn.query_row(
        "SELECT id FROM chunks WHERE source_id=?1 AND ord=?2",
        params![source_id, ord as i64],
        |row| row.get(0),
    )
}

/// chunk text 先頭の `(caption)` を `LIKE '(%substring%)%'` でマッチする。
/// caption 自体に名前が含まれていれば先頭以外の本文ノイズで誤らない。
/// 本則 (附則ではなく) を優先するため、ord 昇順の最初を採る。
fn lookup_id_by_caption(
    conn: &Connection,
    source_id: &str,
    caption_substring: &str,
) -> rusqlite::Result<Option<(String, u32, String)>> {
    let mut stmt = conn.prepare(
        "SELECT id, ord, substr(text,1,80) FROM chunks \
         WHERE source_id=?1 AND text LIKE ?2 \
         ORDER BY ord ASC LIMIT 1",
    )?;
    let pattern = format!("(%{}%)\n%", caption_substring);
    let mut rows = stmt.query(params![source_id, pattern])?;
    if let Some(row) = rows.next()? {
        let id: String = row.get(0)?;
        let ord: i64 = row.get(1)?;
        let snippet: String = row.get(2)?;
        return Ok(Some((id, ord as u32, snippet)));
    }
    Ok(None)
}

fn main() -> anyhow::Result<()> {
    let db = db_path();
    let conn = Connection::open(&db)?;
    let source_id = yokohama_source_id(&conn).map_err(|e| {
        anyhow::anyhow!(
            "failed to find yokohama source in {} (notebook_id={}): {e}",
            db.display(),
            NOTEBOOK_ID
        )
    })?;
    eprintln!("yokohama source_id: {source_id}");

    let golden_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/eval/yokohama/golden.json");
    let raw = std::fs::read_to_string(&golden_path)?;
    let mut golden: Golden = serde_json::from_str(&raw)?;
    eprintln!("loaded golden: {} ({} items)", golden.name, golden.items.len());

    // bump version suffix
    // 過去 (Run 30 で v3 → v4) と同じく、再走するたびに version を bump する。
    if let Some(stripped) = golden.name.strip_suffix("-v3") {
        golden.name = format!("{stripped}-v4");
    } else if let Some(stripped) = golden.name.strip_suffix("-v4") {
        golden.name = format!("{stripped}-v5");
    }

    let by_ord: HashMap<&str, u32> = Q_BY_ORD.iter().copied().collect();
    let by_caption: HashMap<&str, &str> = Q_BY_CAPTION.iter().copied().collect();

    let mut mapped_ord = 0usize;
    let mut mapped_kw = 0usize;
    let mut unmapped = 0usize;
    for item in &mut golden.items {
        if let Some(&ord) = by_ord.get(item.query.as_str()) {
            match lookup_id_by_ord(&conn, &source_id, ord) {
                Ok(id) => {
                    eprintln!("  ord  → {:>3}: {}", ord, item.query);
                    item.relevant = vec![id];
                    mapped_ord += 1;
                    continue;
                }
                Err(e) => {
                    eprintln!("  ord lookup failed for ord={ord} ({}): {e}", item.query);
                }
            }
        }
        if let Some(&caption) = by_caption.get(item.query.as_str()) {
            match lookup_id_by_caption(&conn, &source_id, caption)? {
                Some((id, ord, snippet)) => {
                    eprintln!("  cap  → ord={:>3}: {} | text=「{}」", ord, item.query, snippet);
                    item.relevant = vec![id];
                    mapped_kw += 1;
                    continue;
                }
                None => {
                    eprintln!(
                        "  cap  → no match for: {} (caption={:?})",
                        item.query, caption
                    );
                }
            }
        }
        eprintln!("  ??   stale (kept old uuid): {}", item.query);
        unmapped += 1;
    }

    eprintln!(
        "\nsummary: ord_mapped={} kw_mapped={} unmapped={} total={}",
        mapped_ord,
        mapped_kw,
        unmapped,
        golden.items.len()
    );

    // serde_json::to_string_pretty は配列を改行展開しすぎるので、手で 1-line/item に整形する。
    println!("{{");
    println!("  \"name\": \"{}\",", golden.name);
    println!("  \"items\": [");
    let n = golden.items.len();
    for (i, item) in golden.items.iter().enumerate() {
        let comma = if i + 1 == n { "" } else { "," };
        let tags = if item.tags.is_empty() {
            String::new()
        } else {
            format!(", \"tags\": {}", serde_json::to_string(&item.tags)?)
        };
        println!(
            "    {{ \"query\": {}, \"relevant\": {}{} }}{}",
            serde_json::to_string(&item.query)?,
            serde_json::to_string(&item.relevant)?,
            tags,
            comma
        );
    }
    println!("  ]");
    println!("}}");

    Ok(())
}
