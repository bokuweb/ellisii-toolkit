//! e-Gov 法令 API から指定 law_id の XML を取得し、`<ArticleCaption>（...）</ArticleCaption>`
//! を抽出して JSON で書き出す。`jp-law-thesaurus.json` の拡張を user が自分で
//! できるようにする補助ツール。
//!
//! e-Gov API のエンドポイント (2026-05 時点): `https://laws.e-gov.go.jp/api/1/lawdata/{law_id}`
//!
//! 既知の law_id サンプル:
//! - 民法                  : `129AC0000000089`
//! - 会社法                : `417AC0000000086`
//! - 商法                  : `132AC0000000048`
//! - 著作権法              : `345AC0000000048`
//! - 労働基準法            : `322AC0000000049`
//! - 労働契約法            : `419AC0000000128`
//! - 育児・介護休業法      : `403AC0000000076`
//! - 特許法                : `334AC0000000121`
//! - 実用新案法            : `334AC0000000123`
//! - 意匠法                : `334AC0000000125`
//! - 商標法                : `334AC0000000127`
//! - 不正競争防止法        : `405AC0000000047`
//! - 個人情報保護法        : `415AC0000000057`
//!
//! 実行 (2 step、HTTP fetch は curl、XML parse は本ツール):
//! ```sh
//! # 1. 法令 XML を取得 (~MB オーダ)
//! curl -s https://laws.e-gov.go.jp/api/1/lawdata/417AC0000000086 > /tmp/kaisha.xml
//!
//! # 2. captions を JSON 配列で抽出
//! ELLISII_EGOV_XML=/tmp/kaisha.xml \
//!   cargo run -p ellisii-sdk --example fetch_egov_captions --release \
//!   > /tmp/kaisha-captions.json
//! ```
//!
//! 出力 (JSON 配列):
//! ```json
//! [
//!   { "article": "第一条", "caption": "目的", "body_preview": "..." },
//!   ...
//! ]
//! ```
//!
//! 注: e-Gov の法令データは「政府標準利用規約 (CC BY 4.0 相当)」で公開されている。
//! 抽出データを再配布する場合は出典 (e-Gov 法令検索 https://laws.e-gov.go.jp) を明記すること。

use std::path::PathBuf;
use std::time::Instant;

use regex::Regex;
use serde::Serialize;

#[derive(Debug, Serialize)]
struct ExtractedArticle {
    article: String,
    caption: String,
    body_preview: String,
}

fn main() -> anyhow::Result<()> {
    let xml_path = std::env::var("ELLISII_EGOV_XML")
        .map(PathBuf::from)
        .map_err(|_| {
            anyhow::anyhow!(
                "ELLISII_EGOV_XML 未設定。先に curl で XML を保存: \
                 `curl -s https://laws.e-gov.go.jp/api/1/lawdata/<LAW_ID> > law.xml`"
            )
        })?;
    eprintln!("parsing: {}", xml_path.display());

    let t0 = Instant::now();
    let xml = std::fs::read_to_string(&xml_path)?;
    eprintln!("read: {} bytes", xml.len());

    // Article 1 つを大雑把に切り出して、ArticleTitle / ArticleCaption / 最初の Sentence
    // を取る。法令 XML の Article 要素は入れ子しないので非貪欲マッチで十分。
    let article_re = Regex::new(r"(?s)<Article\s[^>]*>(.*?)</Article>")?;
    let caption_re = Regex::new(r"<ArticleCaption>（([^）]+)）</ArticleCaption>")?;
    let title_re = Regex::new(r"<ArticleTitle>([^<]+)</ArticleTitle>")?;
    let sentence_re = Regex::new(r"<Sentence(?:\s[^>]*)?>([^<]+)</Sentence>")?;

    let mut articles: Vec<ExtractedArticle> = Vec::new();
    for m in article_re.captures_iter(&xml) {
        let body = m.get(1).unwrap().as_str();
        let caption = caption_re
            .captures(body)
            .map(|c| c.get(1).unwrap().as_str().to_string())
            .unwrap_or_default();
        let title = title_re
            .captures(body)
            .map(|c| c.get(1).unwrap().as_str().to_string())
            .unwrap_or_default();
        // 最初の Sentence 群を結合 (preview 100 chars)
        let mut buf = String::new();
        for s in sentence_re.captures_iter(body) {
            if buf.chars().count() > 80 {
                break;
            }
            buf.push_str(s.get(1).unwrap().as_str());
            buf.push(' ');
        }
        let body_preview: String = buf.chars().take(100).collect();
        if !caption.is_empty() {
            articles.push(ExtractedArticle {
                article: title,
                caption,
                body_preview,
            });
        }
    }
    eprintln!(
        "extracted: {} articles with caption ({:.2}s)",
        articles.len(),
        t0.elapsed().as_secs_f32()
    );

    println!("{}", serde_json::to_string_pretty(&articles)?);
    Ok(())
}
