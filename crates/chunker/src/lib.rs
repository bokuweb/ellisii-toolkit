use ellisii_core::Chunk;
use ellisii_parsers_core::{ParsedBlock, ParsedDocument};
use uuid::Uuid;

#[derive(Debug, Clone, Copy)]
pub struct ChunkConfig {
    pub target_chars: usize,
    pub max_chars: usize,
    pub overlap_chars: usize,
    pub min_chars: usize,
    /// `heading_path` の末尾を `(<heading>)\n` の **疑似 caption** として chunk テキスト
    /// 先頭に prepend する。caption rerank (`crates/rag::rerank::caption_boost_in_place`) は
    /// `(...)` 見出し付きチャンクを優遇するので、Markdown / DOCX のような **既に `(...)`
    /// 見出しが付いていないが heading_path がある文書** で caption rerank パスを有効化
    /// したいときに `true` にする。
    ///
    /// 既定 `false`: 既存挙動 (法令系 chunker が `(条文タイトル)` を本文から拾うパスは
    /// 影響を受けない)。`heading_path` が空の chunk や、既に `(...)` で始まる chunk に対しては
    /// 何もしない (重複を防ぐ)。
    ///
    /// `docs/eval/recall-evals.md` Run 24 を参照。
    pub synthesize_caption_from_heading: bool,
}

impl Default for ChunkConfig {
    fn default() -> Self {
        // NotebookLM 等で経験的に良いとされている "small leaf + 周辺 expansion"
        // 構成に揃える。設計意図:
        //
        // - 1 chunk ≒ 1 段落 (300〜400 chars / 日本語 ~150-200 token) に収め、
        //   `static-embedding-japanese` (mean-pool) のシグナルが他段落の話題で
        //   薄まらないようにする。
        // - LLM への context は src-tauri 側の retrieve hook で hit chunk の前後
        //   ±2 chunk を自動連結 (= ~5×350 ≒ 1750 chars) するので、leaf を小さく
        //   しても回答コンテキストはむしろ豊かになる。
        // - overlap は控えめ (60 chars)。隣接結合があるので冗長な overlap は不要。
        // - min_chars=80 で文末記号 1 つの極小行が独立 chunk になるのを防ぐ
        //   (`is_low_content` と組み合わせて低情報量片を排除)。
        //
        // 旧値 (1000/1600/200/150) は法令文書を 1 条 1 chunk に収める前提だった
        // が、`chunk_law_articles` 経路は既に article 単位で chunk を作ってから
        // recursive_split に渡すので、ここを縮めても 1 article が分割され過ぎる
        // ことは無い (短い条は 1 chunk のまま、長い条だけが段落単位に分かれる)。
        Self {
            target_chars: 350,
            max_chars: 600,
            overlap_chars: 60,
            min_chars: 80,
            synthesize_caption_from_heading: false,
        }
    }
}

/// chunking 戦略の trait。`chunk()` free 関数 (= default 実装) を呼び出す
/// `DefaultChunker` に加えて、利用者は文書ごとに別の chunker を差し替えられる。
///
/// 動機 (HANDOFF B4):
/// - PDF / Markdown / 法令 / コードはそれぞれ最適 chunk size と境界が違う
/// - 学術論文に独自の split ロジックを持ち込みたい SDK 利用者は trait 実装を
///   差し替えるだけで済むようにする
/// - 既に外部で chunk 済みのデータを食わせる場合は SDK 側の
///   `Ellisii::index_chunks(...)` を使い、chunker は経由しない
pub trait Chunker: Send + Sync {
    fn chunk(&self, doc: &ParsedDocument, source_id: Uuid) -> Vec<Chunk>;
}

/// 既存の `chunk()` 関数を `ChunkConfig` 付きで `Chunker` trait に被せた標準実装。
/// 全 `SourceKind` を 1 種類のロジックで処理する (法令系は内部で article 検出に
/// 分岐するが、それは `chunk()` 関数内の判定で trait の外側からは透過的)。
#[derive(Debug, Clone, Default)]
pub struct DefaultChunker {
    pub config: ChunkConfig,
}

impl DefaultChunker {
    pub fn new(config: ChunkConfig) -> Self {
        Self { config }
    }
}

impl Chunker for DefaultChunker {
    fn chunk(&self, doc: &ParsedDocument, source_id: Uuid) -> Vec<Chunk> {
        chunk(doc, source_id, self.config)
    }
}

/// `SourceKind` ごとに別の `Chunker` を呼び分けるディスパッチャ。
///
/// 動機 (HANDOFF B4):
/// - PDF / Markdown / コード / 法令 で最適な chunk 戦略は異なる。
/// - ただし無闇に specialize すると測定なしで regression を生むため、本構造は
///   **default + 明示マップ** の構造にし、map にない kind は default に fall
///   through する。
///
/// 使い方:
/// ```ignore
/// let dispatch = DispatchChunker::new(Arc::new(DefaultChunker::default()))
///     .with_kind(SourceKind::Markdown, Arc::new(MyMarkdownChunker))
///     .with_kind(SourceKind::Pdf, Arc::new(MyPdfChunker));
/// let ellisii = Ellisii::builder()
///     .with_chunker(Arc::new(dispatch))
///     .build()?;
/// ```
pub struct DispatchChunker {
    default: std::sync::Arc<dyn Chunker>,
    by_kind: std::collections::HashMap<ellisii_core::SourceKind, std::sync::Arc<dyn Chunker>>,
}

impl DispatchChunker {
    pub fn new(default: std::sync::Arc<dyn Chunker>) -> Self {
        Self {
            default,
            by_kind: std::collections::HashMap::new(),
        }
    }

    /// 指定 `SourceKind` 専用の chunker を登録する。既に登録があれば上書き。
    pub fn with_kind(
        mut self,
        kind: ellisii_core::SourceKind,
        chunker: std::sync::Arc<dyn Chunker>,
    ) -> Self {
        self.by_kind.insert(kind, chunker);
        self
    }
}

impl Chunker for DispatchChunker {
    fn chunk(&self, doc: &ParsedDocument, source_id: Uuid) -> Vec<Chunk> {
        match self.by_kind.get(&doc.kind) {
            Some(c) => c.chunk(doc, source_id),
            None => self.default.chunk(doc, source_id),
        }
    }
}

pub fn chunk(doc: &ParsedDocument, source_id: Uuid, cfg: ChunkConfig) -> Vec<Chunk> {
    // Markdown はパーサ側で既に heading 階層 (H1/H2/H3) ごとに ParsedBlock を
    // 切ってくれているので、article 検出 (条文 / Section) を当てると
    // 偶発的な番号付きリストを誤検出する可能性がある。Markdown では
    // パーサの heading 構造をそのまま尊重して、block ごとの recursive split
    // パスへ直行する。
    let is_markdown = matches!(doc.kind, ellisii_core::SourceKind::Markdown);

    // 全ブロックを結合した text で「第◯条」を数え、3 個以上あれば
    // 法令文書とみなして **block 境界を無視** して条文単位で chunk する。
    // (Parser が paragraph 単位で block を切ると 1 block 内には 1 条しか入らず、
    //  block 単位の article-aware 検出だと発火しないため。)
    let joined: String = doc
        .blocks
        .iter()
        .map(|b| b.text.as_str())
        .collect::<Vec<_>>()
        .join("\n\n");
    // joined 内の各 block の開始バイト位置。article-mode で page を伝播する
    // ために使う。
    let block_positions = compute_block_positions(&doc.blocks);
    let global_articles = if is_markdown {
        Vec::new()
    } else {
        split_japanese_law_articles(&joined)
    };
    if global_articles.len() >= 3 {
        return chunk_law_articles(
            global_articles,
            &doc.blocks,
            &block_positions,
            source_id,
            &cfg,
        );
    }
    // 日本語の条文として認識できない場合、英文契約書の `ARTICLE` / `Section` /
    // `1. Title` 形式の境界を試す。同じ chunk_law_articles を再利用する。
    // ※ 旧実装は裸の `1.5.7` 形式の十進番号も拾っていたが、それだと技術書の
    //    目次 (e.g., "1.5.7 交差テーブルの他のメリット" が 200+ 個) に対して
    //    過剰反応して全文書を法令モードに引きずり込んでしまう。明示キーワード
    //    (ARTICLE/Section/Sec.) または "1. Capital" だけに限定する。
    let global_eng = if is_markdown {
        Vec::new()
    } else {
        split_english_contract_articles(&joined)
    };
    if global_eng.len() >= 3 {
        return chunk_law_articles(
            global_eng,
            &doc.blocks,
            &block_positions,
            source_id,
            &cfg,
        );
    }

    // 通常 (block ごとの recursive split) パス
    //
    // 技術書 (PDF + OCR) や論文では block ごとの heading_path が "Page N" 程度
    // しか入っていないので、ここで章境界を検出して heading_path に章タイトルを
    // injection する。RAG の citation で「<章タイトル> p.N」が見えるようにする
    // ため。
    //
    // 9fe7502 で OCR 出力をページ単位 1 ParsedBlock に結合する変更が入って以来、
    // 章タイトルがページ途中行に出現する PDF で `detect_chapter_markers` の
    // 「先頭 5 行のみ走査」枠から漏れて章タイトルが見えなくなっていた。
    // chunking 前段で **章タイトル行を境界に block を pre-split** し、検出と
    // chunk-to-chapter 割り当ての粒度を揃える。
    let blocks_for_chunking: Vec<ParsedBlock> = if is_markdown {
        doc.blocks.clone()
    } else {
        split_blocks_at_chapter_lines(&doc.blocks)
    };
    let chapter_marks = if is_markdown {
        Vec::new()
    } else {
        detect_chapter_markers(&blocks_for_chunking)
    };
    let mut out: Vec<Chunk> = Vec::new();
    let mut ord: u32 = 0;
    for (i, block) in blocks_for_chunking.iter().enumerate() {
        // この block が属する章タイトル。i 以下で最近の chapter_marks を採用。
        let chapter = chapter_for_block_index(&chapter_marks, i);
        let pieces = recursive_split(&block.text, &cfg);
        for piece in pieces {
            // 数字・記号 only のページ番号片や「・・・・・・」のような低情報量
            // chunk は retrieval ノイズになるので embedding/FTS いずれにも
            // 載せない。前 chunk への concat も行わない (ノイズが本文に
            // 混ざるのを防ぐ)。
            if is_low_content(&piece) {
                continue;
            }
            // 章タイトルを heading_path の先頭にスタック。
            // 既に block.heading_path に同じ章が含まれているケース
            // (= 重複) は弾く。
            let mut hp = block.heading_path.clone();
            if let Some(ch) = &chapter {
                if !hp.iter().any(|h| h == ch) {
                    hp.insert(0, ch.clone());
                }
            }
            if piece.chars().count() < cfg.min_chars && !out.is_empty() {
                if let Some(last) = out.last_mut() {
                    if last.heading_path == hp {
                        last.text.push('\n');
                        last.text.push_str(&piece);
                        continue;
                    }
                }
            }
            let summary = piece
                .lines()
                .next()
                .map(|s| s.chars().take(80).collect::<String>());
            let text = maybe_inject_caption(piece, &hp, cfg.synthesize_caption_from_heading);
            out.push(Chunk {
                id: Uuid::new_v4(),
                source_id,
                ord,
                text,
                heading_path: hp,
                page: block.page,
                bbox: block.bbox,
                summary,
            });
            ord += 1;
        }
    }
    out
}

/// `synthesize_caption_from_heading` 有効時に `heading_path` の末尾を `(...)` 見出しとして
/// チャンク先頭に prepend する。条件:
/// - flag が false → 何もしない (既存挙動)
/// - heading_path が空 → 何もしない
/// - text が既に `(` で始まる → 何もしない (caption の重複を防ぐ)
/// - heading 末尾の文字列が `(` `)` を含む → そのまま使うと括弧が壊れるので skip
/// - heading が空文字 (trim 後) → skip
fn maybe_inject_caption(text: String, heading_path: &[String], enabled: bool) -> String {
    if !enabled {
        return text;
    }
    let Some(last) = heading_path.last() else {
        return text;
    };
    let trimmed_heading = last.trim();
    if trimmed_heading.is_empty() {
        return text;
    }
    if trimmed_heading.contains('(') || trimmed_heading.contains(')')
        || trimmed_heading.contains('(') || trimmed_heading.contains(')')
    {
        return text;
    }
    if text.trim_start().starts_with('(') || text.trim_start().starts_with('(') {
        return text;
    }
    format!("({})\n{}", trimmed_heading, text)
}

/// chunk として残す価値が無い「低情報量」テキストか判定する。
///
/// 想定する除外対象:
///   - スキャン PDF のページ番号 / 図表番号だけ拾った "1565" / "図 2-1"
///   - 目次の point leader (「・・・・・・」) や枠線アート文字
///   - 短すぎる断片 (3 文字以下)
///   - 「2.5.3 ……………… 18」のような数字とドットが大半の目次行
///
/// 維持したいケース:
///   - 数字混じりだが内容語のある文 ("ABC社の売上は1234万円")
///   - 短いが内容のある見出し行 ("第3章 並列化")
fn is_low_content(text: &str) -> bool {
    let trimmed = text.trim();
    let total = trimmed.chars().count();
    // ultra-short は無条件に弾く (= 「1565」「...」「| | |」など)。
    if total < 4 {
        return true;
    }
    // 内容語の文字 (= アルファベット / ひらがな / カタカナ / 漢字) を数える。
    // 数字・約物・空白・絵文字・OCR 由来の罫線文字 (─ │ など) は除外。
    let content_chars = trimmed.chars().filter(|c| is_content_char(*c)).count();
    // 内容語が 4 文字未満なら、たとえ全体長があっても実質的な意味は無いとみなす。
    content_chars < 4
}

fn is_content_char(c: char) -> bool {
    c.is_alphabetic()
        || ('一'..='龥').contains(&c)
        || ('ぁ'..='ん').contains(&c)
        || ('ァ'..='ヶ').contains(&c)
}

/// joined text 内で、`blocks[i]` の text が始まる byte offset を返す。
/// `chunk()` で `blocks.iter().map(text).collect::<Vec<_>>().join("\n\n")` する
/// 並びと同じ計算なので、joined を作り直さずに位置だけ手で合わせる。
fn compute_block_positions(blocks: &[ParsedBlock]) -> Vec<usize> {
    let mut out = Vec::with_capacity(blocks.len());
    let mut pos = 0usize;
    for (i, b) in blocks.iter().enumerate() {
        out.push(pos);
        pos += b.text.len();
        if i + 1 < blocks.len() {
            pos += "\n\n".len();
        }
    }
    out
}

/// joined text の byte 位置 `byte_pos` を含む block の index を返す。
/// 線形検索だが、article 数 × block 数のオーダーなので 10K block 程度までなら
/// 実用上問題にならない (技術書 1 冊で chunk 700 / blocks 10K 程度)。
fn block_index_for_position(positions: &[usize], byte_pos: usize) -> Option<usize> {
    if positions.is_empty() {
        return None;
    }
    // positions は単調増加なので二分探索可能。
    match positions.binary_search(&byte_pos) {
        Ok(i) => Some(i),
        Err(i) => Some(i.saturating_sub(1)),
    }
}

/// 章タイトル行 (`3章 ...` / `第N章 ...` / `Chapter N ...`) を境界として
/// ParsedBlock を pre-split する。
///
/// 背景: 9fe7502 で OCR 出力をページ単位の 1 ParsedBlock に結合するように
/// なって以来、章タイトルがページ途中の行に出現するレイアウトでは
/// `detect_chapter_markers` の「先頭 5 行のみ走査」枠から漏れて章タイトルが
/// 見えなくなっていた (= IDリクワイアド本文が前章 heading_path のまま並ぶ
/// 症状)。block を章タイトル行の **前** で切ることで、各 block の先頭 5 行
/// 内に章タイトルが必ず来るよう揃える。
///
/// 動作:
/// - 各 block について line 1 以降を走査し、`chapter_regex` が行頭に当たった
///   行があれば、その行を境界に block を 2 分割する (前半: 行 0..L、後半:
///   行 L..)。同じ block 内に複数の章タイトル行が並ぶ場合は再帰的に切る。
/// - 章タイトルが line 0 にある block / 章タイトルを含まない block はそのまま
///   出力する。
/// - heading_path / page / bbox は元 block のものを継承する。
fn split_blocks_at_chapter_lines(blocks: &[ParsedBlock]) -> Vec<ParsedBlock> {
    let re = chapter_regex();
    let mut out: Vec<ParsedBlock> = Vec::with_capacity(blocks.len());
    for b in blocks {
        let lines: Vec<&str> = b.text.lines().collect();
        if lines.is_empty() {
            out.push(b.clone());
            continue;
        }
        let mut start = 0usize;
        for li in 1..lines.len() {
            let trimmed = lines[li].trim_start();
            if trimmed.is_empty() {
                continue;
            }
            let m = match re.find(trimmed) {
                Some(m) => m,
                None => continue,
            };
            if m.start() != 0 {
                continue;
            }
            // start..li を 1 セグメントとして emit (空でなければ)
            let pre_text = lines[start..li].join("\n");
            if !pre_text.trim().is_empty() {
                out.push(ParsedBlock {
                    text: pre_text,
                    heading_path: b.heading_path.clone(),
                    page: b.page,
                    bbox: b.bbox,
                });
            }
            start = li;
        }
        let tail_text = lines[start..].join("\n");
        if !tail_text.trim().is_empty() {
            out.push(ParsedBlock {
                text: tail_text,
                heading_path: b.heading_path.clone(),
                page: b.page,
                bbox: b.bbox,
            });
        } else if start == 0 {
            // text 全てが空白だが元 block を保持する (= API 互換)
            out.push(b.clone());
        }
    }
    out
}

/// 章タイトル候補とそれが現れた block index のペアを返す。
///
/// 検出パターン (block.text の **先頭** または `\n` 直後の行):
///   - `第\d+章` / `第[一二三四五六七八九十百千万]+章` (例: "第1章", "第十二章")
///   - 行頭 `\d+章\s+<タイトル>` (例: "2章 ナイーブツリー")  ← 技術書で頻出
///   - `Chapter\s+\d+` (例: "Chapter 3")
///
/// 同じ章が連続した heading に何度も書かれていた場合は、最初に見つけた block
/// だけ採用する (= 同章 chunk 全部に同じ heading_path が立つ)。
fn detect_chapter_markers(blocks: &[ParsedBlock]) -> Vec<(usize, String)> {
    let re = chapter_regex();
    let mut out: Vec<(usize, String)> = Vec::new();
    let mut last_label: Option<String> = None;
    for (i, b) in blocks.iter().enumerate() {
        // 行頭マッチを取りたいので block.text を行ごとに見る。先頭 5 行だけ
        // (本文の途中に章タイトルっぽい文字列が混ざるケースを防ぐ)。
        let lines: Vec<&str> = b.text.lines().collect();
        for li in 0..lines.len().min(5) {
            let trimmed = lines[li].trim_start();
            if let Some(m) = re.find(trimmed) {
                if m.start() != 0 {
                    continue;
                }
                let mut label: String = trimmed
                    .chars()
                    .take(80)
                    .collect::<String>()
                    .trim()
                    .to_string();
                if label.chars().count() < 2 {
                    continue;
                }
                // 「4章」のように章番号だけが 1 行を占める OCR 出力では、
                // 次の非空行をタイトルとして付けて読みやすい label にする。
                // 6 文字以下を「番号だけ行」とみなす ("第十二章" のような
                // 漢数字含めた見出しでもなるべく拾う)。
                if label.chars().count() <= 6 {
                    for next in lines.iter().skip(li + 1).take(3) {
                        let t = next.trim();
                        if !t.is_empty() {
                            let suffix: String = t.chars().take(60).collect();
                            label = format!("{label} {suffix}");
                            break;
                        }
                    }
                }
                if last_label.as_deref() != Some(label.as_str()) {
                    out.push((i, label.clone()));
                    last_label = Some(label);
                }
                break; // 1 block 内では先頭 1 つだけ採用
            }
        }
    }
    out
}

/// `chapter_marks` から block index `i` 以下で最も近い章タイトルを返す。
fn chapter_for_block_index(chapter_marks: &[(usize, String)], i: usize) -> Option<String> {
    let mut last: Option<&String> = None;
    for (idx, label) in chapter_marks {
        if *idx > i {
            break;
        }
        last = Some(label);
    }
    last.cloned()
}

fn chapter_regex() -> regex::Regex {
    static RE: std::sync::OnceLock<regex::Regex> = std::sync::OnceLock::new();
    RE.get_or_init(|| {
        // - "第N章" / "第十二章" (漢数字) — 直後に空白/タイトル/句読点いずれも許容
        // - "N章 <空白> <非数字>" (例: "2章 ナイーブツリー", "25章 トランザクション")
        //   → 単独で "1章" だけだと「1章末で…」のような本文も拾うので、後ろに
        //     空白 + 非数字を要求して見出し行に絞る。
        // - "Chapter N" (英)
        // - "第N章" / "第十二章" (漢数字)
        // - "N章 <空白> <非数字>" (例: "2章 ナイーブツリー")
        // - "N章" 単独行 (タイトルが次行にある OCR 出力をカバー) ← 新規
        // - "Chapter N"
        let pat = r"^(?:第(?:[0-9]+|[一二三四五六七八九十百千万]+)章|[0-9]+章(?:[ 　\t]+[^0-9]|[ 　\t]*$)|Chapter\s+[0-9]+)";
        regex::Regex::new(pat).expect("chapter regex")
    })
    .clone()
}

fn chunk_law_articles(
    articles: Vec<LawArticle>,
    blocks: &[ParsedBlock],
    block_positions: &[usize],
    source_id: Uuid,
    cfg: &ChunkConfig,
) -> Vec<Chunk> {
    // article 本文の先頭にいる block の page/heading を base にする (article は
    // しばしば複数ページに渡るが、citation のジャンプ先としては最初のページが
    // 妥当)。
    let base_heading = blocks
        .first()
        .map(|b| b.heading_path.clone())
        .unwrap_or_default();
    // 章タイトルマーカ。Japanese law 文書には基本入らないが、英文契約や
    // 技術文書混在レポートで助けになる。
    let chapter_marks = detect_chapter_markers(blocks);

    let mut out: Vec<Chunk> = Vec::new();
    let mut ord: u32 = 0;
    for a in articles {
        // article 開始位置の block index → そこから page/heading_path を借りる
        let block_idx = block_index_for_position(block_positions, a.start_byte);
        let (page, bbox, block_heading) = match block_idx.and_then(|i| blocks.get(i)) {
            Some(b) => (b.page, b.bbox, b.heading_path.clone()),
            None => (None, None, base_heading.clone()),
        };
        let chapter = block_idx.and_then(|i| chapter_for_block_index(&chapter_marks, i));

        let mut hp = block_heading;
        // 章タイトルを最も外側 (= heading_path の先頭) に挿入。
        // block_heading に既に同じものがあれば重複させない。
        if let Some(ch) = &chapter {
            if !hp.iter().any(|h| h == ch) {
                hp.insert(0, ch.clone());
            }
        }
        if let Some(art) = &a.article_label {
            hp.push(art.clone());
        }
        let pieces = recursive_split(&a.text, cfg);
        for piece in pieces {
            // low-content piece (= 数字 only / 目次行) は弾く。法令文書では稀
            // だが、英文契約や混在 PDF でゴミ行が混ざることはある。
            if is_low_content(&piece) {
                continue;
            }
            if piece.chars().count() < cfg.min_chars && !out.is_empty() {
                if let Some(last) = out.last_mut() {
                    if last.heading_path == hp {
                        last.text.push('\n');
                        last.text.push_str(&piece);
                        continue;
                    }
                }
            }
            let summary = piece
                .lines()
                .next()
                .map(|s| s.chars().take(80).collect::<String>());
            out.push(Chunk {
                id: Uuid::new_v4(),
                source_id,
                ord,
                text: piece,
                heading_path: hp.clone(),
                page,
                bbox,
                summary,
            });
            ord += 1;
        }
    }
    out
}

#[derive(Debug, Clone)]
struct LawArticle {
    /// 例: "第十五条" / "第3条". 見つからなければ None (前置きやスタブ部分)。
    article_label: Option<String>,
    text: String,
    /// joined text における article 開始位置 (byte offset)。`chunk_law_articles`
    /// が source block の page を逆引きするのに使う。
    start_byte: usize,
}

/// 日本語法令テキストを「第N条」境界で分割する。
/// 該当パターンが少ない (< 3) ときは呼び出し側で通常チャンクへフォールバック。
fn split_japanese_law_articles(text: &str) -> Vec<LawArticle> {
    // "第" + 数字 (漢数字 / アラビア数字) + "条" — 行頭のものを境界とみなす
    let pattern = japanese_article_regex();
    let mut out: Vec<LawArticle> = Vec::new();

    let mut last_end = 0usize;
    let mut last_label: Option<String> = None;
    let mut last_segment_start = 0usize;
    for m in pattern.find_iter(text) {
        // 契約書/規約でよくある「（秘密保持）改行 第N条 …」の形を救う。
        // 第N条 の直前の行が `（…）` または `(…)` のみで構成された見出しなら
        // それを **第N条 のチャンク側に含める** ように境界を後ろに動かす。
        // (前の article の末尾から見出しを引き剥がす)
        let start = absorb_preceding_paren_title(text, m.start(), last_end);
        if start > last_end {
            let segment = text[last_end..start].trim();
            if !segment.is_empty() {
                out.push(LawArticle {
                    article_label: last_label.clone(),
                    text: segment.to_string(),
                    start_byte: last_segment_start,
                });
            }
        }
        last_label = Some(m.as_str().trim().to_string());
        last_segment_start = start;
        last_end = start;
    }
    // 末尾
    let tail = text[last_end..].trim();
    if !tail.is_empty() {
        out.push(LawArticle {
            article_label: last_label,
            text: tail.to_string(),
            start_byte: last_segment_start,
        });
    }
    out
}

/// `第N条` 開始位置 `start` の直前にある「`（…）` だけからなる見出し行」を
/// 探し、見つかればその先頭バイトオフセットを返す。
///
/// - 行末の改行/空白をスキップ → `）` または `)` を要求
/// - 同じ行内で対応する `（` または `(` を探す
/// - その閉じ括弧の前 (=見出し行) に **括弧以外の本文文字が無い** こと
///   (= 見出しのみの行であること) を確認
/// - 見出し行の先頭 (前回境界 `lower_bound` を超えない) を返す
///
/// 条件に合わなければ `start` をそのまま返す。
fn absorb_preceding_paren_title(text: &str, start: usize, lower_bound: usize) -> usize {
    if start <= lower_bound {
        return start;
    }
    let prefix = &text[lower_bound..start];
    // 末尾空白 (改行含む) を剥がす
    let trimmed_end = prefix.trim_end_matches(|c: char| c.is_whitespace());
    if trimmed_end.len() == prefix.len() {
        // 第N条 の直前に空白が無い (= 同じ行に普通の文章が連結) → 介入しない
        return start;
    }
    // 末尾文字が閉じ括弧かチェック
    let last_close_byte = match trimmed_end.char_indices().next_back() {
        Some((idx, ch)) if ch == '）' || ch == ')' => idx,
        _ => return start,
    };
    let close_char_end = trimmed_end.len();
    let open_target = if &trimmed_end[last_close_byte..close_char_end] == "）" {
        '（'
    } else {
        '('
    };
    // 同じ行内で対応する開き括弧を探す (簡易: 最初に現れた行頭からの最後の対応)
    // 行の開始 = 直近の \n の次、無ければ prefix 先頭
    let line_start_in_prefix = trimmed_end[..last_close_byte]
        .rfind('\n')
        .map(|i| i + 1)
        .unwrap_or(0);
    let line_slice = &trimmed_end[line_start_in_prefix..last_close_byte];
    // 行内で開き括弧の位置 (バイト)
    let open_rel = match line_slice.rfind(open_target) {
        Some(i) => i,
        None => return start,
    };
    // 行の先頭から開き括弧までは (空白だけが許容される)
    let leading = &line_slice[..open_rel];
    if !leading.chars().all(|c| c.is_whitespace()) {
        return start;
    }
    // 括弧の中身に改行や別の括弧が含まれない (= 単純な 1 行見出し) こと
    let inside = &line_slice[open_rel + open_target.len_utf8()..];
    if inside.chars().any(|c| c == '\n' || c == '（' || c == '(') {
        return start;
    }
    // 見出しの先頭 (絶対オフセット)
    let title_start = lower_bound + line_start_in_prefix;
    title_start.max(lower_bound)
}

/// 英文契約書のセクション境界を見つけて [`LawArticle`] (= 構造的に同じものなので
/// 流用) に変換する。日本語版とは別の正規表現で:
/// - `ARTICLE 1` / `ARTICLE 1.1` / `ARTICLE I` (Roman) / `Article 1`
/// - `Section 1` / `Section 1.1.1` / `SECTION 1` / `Sec. 1`
/// - 行頭の `1.` / `1.1` / `2.` などの十進ナンバリング (大文字始まりの本文付き)
/// を境界として扱う。括弧書きの sub-item `(a)` `(i)` は本文扱いで取り込む。
fn split_english_contract_articles(text: &str) -> Vec<LawArticle> {
    let pattern = english_article_regex();
    let mut out: Vec<LawArticle> = Vec::new();
    let mut last_end = 0usize;
    let mut last_label: Option<String> = None;
    let mut last_segment_start = 0usize;
    for m in pattern.find_iter(text) {
        // 日本語版と同じく、直前行が `(タイトル)` ならその行ごと article 側に取り込む。
        let start = absorb_preceding_paren_title(text, m.start(), last_end);
        if start > last_end {
            let segment = text[last_end..start].trim();
            if !segment.is_empty() {
                out.push(LawArticle {
                    article_label: last_label.clone(),
                    text: segment.to_string(),
                    start_byte: last_segment_start,
                });
            }
        }
        // ラベルは見出し行の冒頭 80 文字程度で十分 (例: "ARTICLE 1. Definitions")
        let raw = m.as_str().trim();
        let label: String = raw.chars().take(80).collect();
        last_label = Some(label);
        last_segment_start = start;
        last_end = start;
    }
    let tail = text[last_end..].trim();
    if !tail.is_empty() {
        out.push(LawArticle {
            article_label: last_label,
            text: tail.to_string(),
            start_byte: last_segment_start,
        });
    }
    out
}

fn english_article_regex() -> regex::Regex {
    static RE: std::sync::OnceLock<regex::Regex> = std::sync::OnceLock::new();
    RE.get_or_init(|| {
        // 行頭 (multi-line) で各種パターンを or 連結。
        // - "ARTICLE 1", "ARTICLE 1.1", "ARTICLE I", "Article 12" (Arabic か Roman)
        //   後続に "." や "(a)" が来ることもあるが、boundary 検出としては開始位置だけで OK。
        // - "Section 1", "Section 1.2.3", "Sec. 1"
        // - 行頭の十進ナンバリング (例: "1." "2.1" "3.1.2") の後ろが大文字英字
        //   または日本語見出しに続くもの。番号単独 (lone "1" の後ろがリストアイテム
        //   など) を弾くため、`\s+[A-Z]` または `\.\s+[A-Z\u{4E00}-\u{9FFF}]` を要求。
        // Rust の regex crate は lookahead 非対応なので、十進ナンバー単独
        // パターンには直後 1 文字を含めてマッチさせる (`1. Definitions`)。
        // ラベル化は呼び出し側で trim する。
        // - `1.1`, `1.2.3` のような decimal sub はそれ自体で十分一意なので
        //   trailing capital 不要。
        // - `1.` 単体 + capital は、lone digit のリスト誤検出を避けるため
        //   `[A-Z][a-zA-Z]` を要求 (= 大文字始まりの英単語が来ること)。
        // 注意: raw string で行末バックスラッシュは「リテラルなバックスラッシュ
        // + 改行」になるため (Rust は raw string で行継続をサポートしない)、
        // 単一行に詰めて書く。
        // 旧実装は `[0-9]+\.[0-9]+(?:\.[0-9]+){0,2}` (= bare "1.5.7" など) も
        // article 境界として拾っていたが、技術書の目次や節番号 ("1.5.7 交差
        // テーブル", "25.5.7 …" 等が 200+ 件ある) を全て契約条項と誤認識して
        // 文書全体を法令モードに引きずり込む副作用があった。明示キーワード
        // (ARTICLE / Article / SECTION / Section / Sec.) 付き、または
        // "1. <Capital>" 形式の legal-style heading だけに限定する。
        let pat = r"(?m)^\s*(?:ARTICLE\s+(?:[IVXLCDM]+|[0-9]+)(?:\.[0-9]+)*|Article\s+(?:[IVXLCDM]+|[0-9]+)(?:\.[0-9]+)*|SECTION\s+[0-9]+(?:\.[0-9]+)*|Section\s+[0-9]+(?:\.[0-9]+)*|Sec\.\s*[0-9]+(?:\.[0-9]+)*|[0-9]+\.\s+[A-Z][a-zA-Z])";
        regex::Regex::new(pat).expect("english article regex")
    })
    .clone()
}

fn japanese_article_regex() -> regex::Regex {
    // 改行直後 (or 文書先頭) の "第..条" だけを境界扱いし、本文中の参照を除外。
    // 数字: アラビア (1-3桁) または 漢数字 (一二三十百千万)
    static RE: std::sync::OnceLock<regex::Regex> = std::sync::OnceLock::new();
    RE.get_or_init(|| {
        regex::Regex::new(
            r"(?m)^\s*第[0-9０-９一二三四五六七八九十百千万〇零]{1,8}条(?:の[0-9０-９一二三四五六七八九十百千万]{1,4})?",
        )
        .expect("article regex")
    })
    .clone()
}

fn recursive_split(text: &str, cfg: &ChunkConfig) -> Vec<String> {
    // 日本語/英語両対応の境界優先順位:
    //   1. 段落境界 (`\n\n`)            ← 最もきれいな意味単位
    //   2. 文末 (`。` / `？` / `！`)      ← 1 文を分割しない
    //   3. 改行 (`\n`)                   ← 箇条書き / 強制改行された行
    //   4. 読点 / 空白                   ← どうしても分けたいとき
    //   5. 文字単位                      ← 上限に収まらない最後の手段
    //
    // 旧実装は `\n\n`, `。`, `、`, ` `, `""` の 5 段階。`\n` を `。` の次に
    // 入れたのは、OCR 行 block を join しただけの段落で `。` が出ない (= 各
    // 行が独立してて句点が打ってない) ケースをきれいに分けるため。
    let separators = ["\n\n", "。", "？", "！", "\n", "、", " ", ""];
    split_by(text, &separators, 0, cfg)
}

fn split_by(text: &str, seps: &[&str], depth: usize, cfg: &ChunkConfig) -> Vec<String> {
    if text.chars().count() <= cfg.max_chars {
        return vec![text.to_string()];
    }
    if depth >= seps.len() {
        return hard_window(text, cfg);
    }
    let sep = seps[depth];
    if sep.is_empty() {
        return hard_window(text, cfg);
    }

    let mut buffer = String::new();
    let mut result: Vec<String> = Vec::new();

    for piece in text.split_inclusive(sep) {
        if buffer.chars().count() + piece.chars().count() > cfg.target_chars && !buffer.is_empty() {
            result.push(buffer.clone());
            // overlap: 末尾 overlap_chars を次の buffer に持ち越し
            let tail: String = buffer
                .chars()
                .rev()
                .take(cfg.overlap_chars)
                .collect::<String>()
                .chars()
                .rev()
                .collect();
            buffer = tail;
        }
        buffer.push_str(piece);
    }
    if !buffer.is_empty() {
        result.push(buffer);
    }

    // どれかが max を超えていたら次のセパレータで再分割
    let mut flat: Vec<String> = Vec::with_capacity(result.len());
    for r in result {
        if r.chars().count() > cfg.max_chars {
            flat.extend(split_by(&r, seps, depth + 1, cfg));
        } else {
            flat.push(r);
        }
    }
    flat
}

fn hard_window(text: &str, cfg: &ChunkConfig) -> Vec<String> {
    let chars: Vec<char> = text.chars().collect();
    let mut out = Vec::new();
    let step = cfg.target_chars.saturating_sub(cfg.overlap_chars).max(1);
    let mut i = 0;
    while i < chars.len() {
        let end = (i + cfg.target_chars).min(chars.len());
        out.push(chars[i..end].iter().collect());
        if end == chars.len() {
            break;
        }
        i += step;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use ellisii_core::SourceKind;
    use ellisii_parsers_core::ParsedBlock;

    #[test]
    fn default_chunker_trait_matches_free_function_output() {
        // DefaultChunker は既存 `chunk()` 自由関数の thin wrapper。
        // 同じ入力に対して同じ chunk 列を返すことを固定する (refactor 安全網)。
        let doc = ParsedDocument {
            kind: SourceKind::Markdown,
            blocks: vec![ParsedBlock {
                text: "ACID とはトランザクションシステムが備えるべき性質を表す頭字語であり、原子性 (atomicity)、一貫性 (consistency)、独立性 (isolation)、永続性 (durability) の 4 つの性質を意味します。".into(),
                heading_path: vec!["DB".into(), "ACID".into()],
                page: None,
                bbox: None,
            }],
        };
        let source_id = Uuid::new_v4();
        let via_fn = chunk(&doc, source_id, ChunkConfig::default());
        let via_trait = DefaultChunker::default().chunk(&doc, source_id);
        assert_eq!(via_fn.len(), via_trait.len());
        for (a, b) in via_fn.iter().zip(via_trait.iter()) {
            assert_eq!(a.text, b.text);
            assert_eq!(a.heading_path, b.heading_path);
        }
    }

    #[test]
    fn dispatch_chunker_routes_by_source_kind() {
        // Markdown だけ別 chunker を登録し、Text は default に fall through する
        // ことを確認する。Text 用にはあえて空 vec を返す chunker を登録「しない」
        // ことで「未登録 kind が default に落ちる」挙動を測る。
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::sync::Arc;

        struct CountingChunker {
            label: &'static str,
            calls: Arc<AtomicUsize>,
            inner: DefaultChunker,
        }
        impl Chunker for CountingChunker {
            fn chunk(&self, doc: &ParsedDocument, source_id: Uuid) -> Vec<Chunk> {
                self.calls.fetch_add(1, Ordering::SeqCst);
                let _ = self.label;
                self.inner.chunk(doc, source_id)
            }
        }

        let default_calls = Arc::new(AtomicUsize::new(0));
        let md_calls = Arc::new(AtomicUsize::new(0));

        let default = Arc::new(CountingChunker {
            label: "default",
            calls: default_calls.clone(),
            inner: DefaultChunker::default(),
        });
        let md = Arc::new(CountingChunker {
            label: "md",
            calls: md_calls.clone(),
            inner: DefaultChunker::default(),
        });

        let dispatch = DispatchChunker::new(default).with_kind(SourceKind::Markdown, md);

        let md_doc = ParsedDocument {
            kind: SourceKind::Markdown,
            blocks: vec![ParsedBlock {
                text: "Markdown 専用 chunker が呼ばれる経路の動作確認に使う本文です。"
                    .into(),
                heading_path: vec!["x".into()],
                page: None,
                bbox: None,
            }],
        };
        let txt_doc = ParsedDocument {
            kind: SourceKind::Text,
            blocks: vec![ParsedBlock {
                text: "プレーンテキストは未登録なので default chunker に fall through する。"
                    .into(),
                heading_path: vec![],
                page: None,
                bbox: None,
            }],
        };

        let _ = dispatch.chunk(&md_doc, Uuid::new_v4());
        assert_eq!(md_calls.load(Ordering::SeqCst), 1);
        assert_eq!(default_calls.load(Ordering::SeqCst), 0);

        let _ = dispatch.chunk(&txt_doc, Uuid::new_v4());
        assert_eq!(md_calls.load(Ordering::SeqCst), 1, "MD chunker 再呼出は無し");
        assert_eq!(default_calls.load(Ordering::SeqCst), 1, "Text は default に");
    }

    #[test]
    fn default_chunker_respects_custom_config() {
        // `DefaultChunker::new(cfg)` で渡した config が実際に使われていること。
        let mut cfg = ChunkConfig::default();
        cfg.synthesize_caption_from_heading = true;
        let doc = ParsedDocument {
            kind: SourceKind::Markdown,
            blocks: vec![ParsedBlock {
                text: "本文。caption がチャンク先頭に prepend される想定。25 文字以上の内容語にする。".into(),
                heading_path: vec!["セクション題目".into()],
                page: None,
                bbox: None,
            }],
        };
        let chunks = DefaultChunker::new(cfg).chunk(&doc, Uuid::new_v4());
        assert!(!chunks.is_empty());
        assert!(
            chunks[0].text.starts_with("(セクション題目)\n"),
            "synthesize_caption_from_heading が effective でない: {}",
            chunks[0].text
        );
    }

    #[test]
    fn article_chunker_splits_by_law_articles() {
        let text = "\
第一条 私権は、公共の福祉に適合しなければならない。
第二条 私権の享有は、出生に始まる。
第三条 法律行為の当事者が意思表示をした時に意思能力を有しなかったときは、その法律行為は、無効とする。
第十五条 補助開始の審判を受けた者は、被補助人とする。
第十六条 補助開始の審判は、家庭裁判所が行う。
";
        let arts = split_japanese_law_articles(text);
        assert!(arts.len() >= 5, "expected >=5 articles, got {}", arts.len());
        let labels: Vec<&str> = arts
            .iter()
            .filter_map(|a| a.article_label.as_deref())
            .collect();
        assert!(labels.iter().any(|l| l.contains("第十五条")));
        assert!(labels.iter().any(|l| l.contains("第十六条")));
    }

    #[test]
    fn law_doc_chunks_carry_article_in_heading_path() {
        let text = "\
第一条 私権は、公共の福祉に適合しなければならない。
第二条 私権の享有は、出生に始まる。
第三条 法律行為の当事者が意思表示をした時に意思能力を有しなかったときは、その法律行為は、無効とする。
";
        let doc = ParsedDocument {
            kind: SourceKind::Text,
            blocks: vec![ParsedBlock {
                text: text.to_string(),
                heading_path: vec!["民法".into()],
                page: None,
                bbox: None,
            }],
        };
        let chunks = chunk(&doc, Uuid::new_v4(), ChunkConfig::default());
        // 全ての chunk の heading_path 末尾は第N条になっているはず
        for c in &chunks {
            let last = c.heading_path.last().cloned().unwrap_or_default();
            assert!(
                last.starts_with("第"),
                "chunk heading_path tail should be a 第◯条 marker, got {:?}",
                c.heading_path
            );
        }
    }

    #[test]
    fn article_includes_preceding_parenthesized_title() {
        // 契約書でよくある「（タイトル）改行 第N条 …」の形で、タイトルが
        // 第N条と同じチャンクに入ること。
        let text = "\
（秘密保持）
第八条 受領者は、本契約に関連して知り得た秘密情報を第三者に開示してはならない。
（損害賠償）
第九条 当事者は、本契約に違反した場合、相手方に生じた損害を賠償する。
(契約期間)
第十条 本契約の有効期間は1年とする。
";
        let arts = split_japanese_law_articles(text);
        let bodies: Vec<&str> = arts.iter().map(|a| a.text.as_str()).collect();
        assert!(
            bodies.iter().any(|b| b.contains("（秘密保持）") && b.contains("第八条")),
            "第八条 chunk should contain its preceding （秘密保持） title; got {:?}",
            bodies
        );
        assert!(
            bodies.iter().any(|b| b.contains("（損害賠償）") && b.contains("第九条")),
            "第九条 chunk should contain its preceding （損害賠償） title; got {:?}",
            bodies
        );
        // 半角括弧版も同様にくっつくこと
        assert!(
            bodies.iter().any(|b| b.contains("(契約期間)") && b.contains("第十条")),
            "第十条 chunk should contain its preceding (契約期間) title; got {:?}",
            bodies
        );
    }

    #[test]
    fn english_contract_articles_split_by_article_keyword() {
        let text = "\
ARTICLE 1. Definitions
\"Effective Date\" shall mean January 1, 2026.
ARTICLE 2. Term
This Agreement shall remain in effect for three (3) years.
ARTICLE 3. Confidentiality
Each Party shall keep confidential all Confidential Information.
";
        let arts = split_english_contract_articles(text);
        assert!(arts.len() >= 3, "expected >=3 english articles, got {}", arts.len());
        let labels: Vec<&str> = arts
            .iter()
            .filter_map(|a| a.article_label.as_deref())
            .collect();
        assert!(labels.iter().any(|l| l.starts_with("ARTICLE 1")), "labels: {labels:?}");
        assert!(labels.iter().any(|l| l.starts_with("ARTICLE 2")));
        assert!(labels.iter().any(|l| l.starts_with("ARTICLE 3")));
    }

    #[test]
    fn english_contract_section_and_subitem_split() {
        let text = "\
Section 1. Scope
Section 1.1. Sub-scope A
Section 1.2. Sub-scope B
Section 2. Obligations
The Party shall comply.
Section 2.1. Notice
Notices shall be in writing.
";
        let arts = split_english_contract_articles(text);
        // Section 1, 1.1, 1.2, 2, 2.1 = 5 boundaries
        assert!(arts.len() >= 4, "expected >=4, got {}", arts.len());
        let labels: Vec<&str> = arts
            .iter()
            .filter_map(|a| a.article_label.as_deref())
            .collect();
        assert!(labels.iter().any(|l| l.contains("1.1")), "labels: {labels:?}");
        assert!(labels.iter().any(|l| l.contains("2.1")));
    }

    #[test]
    fn english_contract_decimal_numbering() {
        let text = "\
1. Definitions
The term 'Party' means a signatory.
2. Term and Termination
This Agreement is valid for 3 years.
3. Governing Law
This Agreement shall be governed by Delaware law.
";
        let arts = split_english_contract_articles(text);
        assert!(arts.len() >= 3, "expected >=3, got {}", arts.len());
    }

    #[test]
    fn splits_long_paragraph() {
        let text = "あ".repeat(5000);
        let doc = ParsedDocument {
            kind: SourceKind::Text,
            blocks: vec![ParsedBlock {
                text,
                heading_path: vec![],
                page: None,
                bbox: None,
            }],
        };
        let chunks = chunk(&doc, Uuid::new_v4(), ChunkConfig::default());
        assert!(chunks.len() >= 3);
        for c in &chunks {
            assert!(c.text.chars().count() <= 1600);
        }
    }

    #[test]
    fn is_low_content_drops_digit_only_and_punctuation() {
        // ページ番号断片
        assert!(is_low_content("1565"));
        assert!(is_low_content("…1565…"));
        // ultra-short
        assert!(is_low_content("12"));
        assert!(is_low_content(""));
        assert!(is_low_content("..."));
        // 罫線・点リーダ・記号だけの行
        assert!(is_low_content("―――――――――"));
        assert!(is_low_content("・・・・・・・"));
        // 目次行 (節番号 + 点リーダ + ページ数字)
        assert!(is_low_content("2.5.3 …………………… 18"));
    }

    #[test]
    fn is_low_content_keeps_real_paragraphs() {
        assert!(!is_low_content("第3章 並列化"));
        assert!(!is_low_content("ABC社の売上は1234万円"));
        assert!(!is_low_content("ナイーブツリーは素朴な木構造"));
        assert!(!is_low_content("Section 1.1 introduces SQL antipatterns"));
    }

    /// 既定 ChunkConfig は NotebookLM 流の "small leaf + 周辺 expansion" 想定で
    /// 値が固定されている。大きく動かす場合は周辺 (= src-tauri の neighbor
    /// 連結 / embedding bandwidth) も合わせて見直すこと。
    /// ここで値そのものを assert することで、不用意な戻し変更に気づけるように。
    #[test]
    fn default_chunk_config_uses_small_leaf_sizes() {
        let cfg = ChunkConfig::default();
        assert_eq!(cfg.target_chars, 350);
        assert_eq!(cfg.max_chars, 600);
        assert_eq!(cfg.min_chars, 80);
        assert_eq!(cfg.overlap_chars, 60);
    }

    /// 技術書の長めの段落 (~800 chars) は max_chars=600 を超えるので必ず分割
    /// される。旧 default (target=1000, max=1600) では 1 chunk のままだった。
    #[test]
    fn tech_book_paragraph_splits_into_multiple_small_chunks() {
        // 句点区切りの長い段落を生成して、確実に max_chars (600) を超えさせる。
        let sentence = "再帰的な関連を持つデータは珍しくはなく、ツリー状の構造や階層的な構造で組織化されることが多い。";
        // ~50 chars × 20 文 = ~1000 chars
        let para = sentence.repeat(20);
        assert!(
            para.chars().count() > 600,
            "test paragraph not long enough: {} chars",
            para.chars().count()
        );
        let doc = ParsedDocument {
            kind: SourceKind::Pdf,
            blocks: vec![ParsedBlock {
                text: para.into(),
                heading_path: vec!["2章 ナイーブツリー".into()],
                page: Some(14),
                bbox: None,
            }],
        };
        let chunks = chunk(&doc, Uuid::new_v4(), ChunkConfig::default());
        assert!(
            chunks.len() >= 2,
            "expected the paragraph to split into multiple small chunks, got {}",
            chunks.len()
        );
        // 各 chunk は max_chars (600) 以下に収まる。
        for c in &chunks {
            assert!(
                c.text.chars().count() <= 600,
                "chunk too large: {} chars",
                c.text.chars().count()
            );
        }
    }

    /// 技術書の目次/節番号 ("1.5.7", "25.5.7" など bare な十進ナンバ) は
    /// 法令モードを発火させないこと。SQLアンチパターン PDF 取り込みで全
    /// chunk が page=None になっていた regression を防ぐ。
    #[test]
    fn tech_book_decimal_section_numbers_do_not_trigger_article_mode() {
        let blocks: Vec<ParsedBlock> = (1..=10)
            .map(|i| ParsedBlock {
                text: format!(
                    "1.5.{i} 交差テーブルの他のメリット\nなんらかの本文 {i}。"
                ),
                heading_path: vec![format!("Page {i}")],
                page: Some(i as u32),
                bbox: None,
            })
            .collect();
        let doc = ParsedDocument {
            kind: SourceKind::Pdf,
            blocks,
        };
        let chunks = chunk(&doc, Uuid::new_v4(), ChunkConfig::default());
        // article-mode に入ると page が全て None になる。block ごと chunking
        // (= 通常パス) なら block.page が伝播するはず。
        let with_page = chunks.iter().filter(|c| c.page.is_some()).count();
        assert_eq!(
            with_page,
            chunks.len(),
            "all chunks should carry page (= article-mode shouldn't fire on tech-book \"X.Y.Z\" lists)"
        );
    }

    /// 章タイトル ("2章 ナイーブツリー") を heading_path に挿入する。
    #[test]
    fn chapter_title_is_injected_into_heading_path() {
        let blocks = vec![
            ParsedBlock {
                text: "2章 ナイーブツリー（素朴な木）\n本章では再帰関連を扱う。".into(),
                heading_path: vec!["Page 14".into()],
                page: Some(14),
                bbox: None,
            },
            ParsedBlock {
                text: "2.1 目的：階層構造を格納し、クエリを実行する\n再帰的な関連を持つデータは…".into(),
                heading_path: vec!["Page 15".into()],
                page: Some(15),
                bbox: None,
            },
            ParsedBlock {
                text: "2.3 アンチパターンの見つけ方\n以下のような言葉を耳にしたら、「ナイーブツリー」アンチパターンが使われている可能性があります。".into(),
                heading_path: vec!["Page 18".into()],
                page: Some(18),
                bbox: None,
            },
        ];
        let doc = ParsedDocument {
            kind: SourceKind::Pdf,
            blocks,
        };
        let chunks = chunk(&doc, Uuid::new_v4(), ChunkConfig::default());
        for c in &chunks {
            // 章タイトルは "2章" を含むラベルが先頭に入る。
            assert!(
                c.heading_path.iter().any(|h| h.contains("2章")),
                "chunk heading_path should contain chapter title, got {:?}",
                c.heading_path
            );
            // page も block ごとに伝播
            assert!(c.page.is_some(), "chunk page should be set, got {:?}", c.page);
        }
    }

    /// 章番号 "N章" がそれ単独で 1 行を構成し、タイトルが次行にある OCR 出力
    /// (技術書 PDF で頻出) を見逃さない。
    ///
    /// 旧実装は `\d+章` の後ろに「半角/全角空白 + 非数字」を要求していたため、
    /// "4章\nキーレスエントリ(外部キー嫌い)\n…" のような OCR では検出に失敗し、
    /// 結果として直前の "3章 IDリクワイアド" が後続全 chunk に伝搬してしまう。
    #[test]
    fn chapter_marker_detected_when_number_is_on_its_own_line() {
        let blocks = vec![
            ParsedBlock {
                text: "3章 IDリクワイアド(とりあえずID)\n2 つのテーブルを結合する際に留意すべき重要なポイントです。".into(),
                heading_path: vec!["Page 68".into()],
                page: Some(68),
                bbox: None,
            },
            ParsedBlock {
                text: "本文の続きです。複合キーや外部キーについて詳しく述べていきます。".into(),
                heading_path: vec!["Page 70".into()],
                page: Some(70),
                bbox: None,
            },
            ParsedBlock {
                // OCR が「4章」だけを 1 行として吐き、タイトルは次行に来るパターン
                text: "4章\nキーレスエントリ(外部キー嫌い)\n需兵はまず勝ちて雨る後に戦いを求め、敗兵はまず戦いて而る後に勝ちを求む。".into(),
                heading_path: vec!["Page 73".into()],
                page: Some(73),
                bbox: None,
            },
            ParsedBlock {
                text: "外部キー制約を意図的に定義しない設計は、データの整合性を脅かします。".into(),
                heading_path: vec!["Page 74".into()],
                page: Some(74),
                bbox: None,
            },
        ];
        let doc = ParsedDocument {
            kind: SourceKind::Pdf,
            blocks,
        };
        let chunks = chunk(&doc, Uuid::new_v4(), ChunkConfig::default());

        // 期待: page 68/70 の chunk は「3章」を含む。
        //       page 73/74 の chunk は「4章」を含み、「3章」は含まない。
        for c in &chunks {
            let in_three = c.heading_path.iter().any(|h| h.contains("3章"));
            let in_four = c.heading_path.iter().any(|h| h.contains("4章"));
            match c.page {
                Some(68) | Some(70) => assert!(
                    in_three && !in_four,
                    "page {:?} chunk should be tagged as 3章 only, got {:?}",
                    c.page,
                    c.heading_path
                ),
                Some(73) | Some(74) => assert!(
                    in_four && !in_three,
                    "page {:?} chunk should be tagged as 4章 (not 3章), got {:?}",
                    c.page,
                    c.heading_path
                ),
                _ => {}
            }
        }
    }

    /// 第N章 (漢数字) の章タイトル検出。
    #[test]
    fn chapter_dai_kanji_chapter_detected() {
        let blocks = vec![
            ParsedBlock {
                text: "第三章 並列化技法\nこの章では…".into(),
                heading_path: vec!["Page 50".into()],
                page: Some(50),
                bbox: None,
            },
            ParsedBlock {
                text: "並列度の選択について述べる。".into(),
                heading_path: vec!["Page 51".into()],
                page: Some(51),
                bbox: None,
            },
        ];
        let doc = ParsedDocument {
            kind: SourceKind::Pdf,
            blocks,
        };
        let chunks = chunk(&doc, Uuid::new_v4(), ChunkConfig::default());
        for c in &chunks {
            assert!(
                c.heading_path.iter().any(|h| h.contains("第三章")),
                "got {:?}",
                c.heading_path
            );
        }
    }

    /// Regression: 9fe7502 で OCR 出力をページ単位 1 ParsedBlock に結合する
    /// ようになって以来、章境界 (「3章」/「4章」) がページ途中に出現する
    /// レイアウトでは `detect_chapter_markers` の「先頭 5 行のみ走査」枠から
    /// 漏れて章タイトルが検出されず、chunk の heading_path が直前章のまま
    /// 据え置かれていた (= 「IDリクワイアド」本文が `[2章 ナイーブツリー]`
    /// 配下に並ぶ症状)。
    ///
    /// 1 つの ParsedBlock 内に 2 章末尾と 3 章先頭が同居していても、3 章以後
    /// の chunk が「3章 IDリクワイアド」配下に割り当てられることを担保する。
    #[test]
    fn chapter_marker_mid_block_splits_heading_path() {
        // 1 ページ ≒ 1 ParsedBlock の OCR 出力を想定し、章タイトル「3章」が
        // 6 行目以降に出現するレイアウト。ndlocr は OCR 行を `\n` で羅列し、
        // 章タイトル前に 2 章末尾本文が複数行流れる ngu pdf がよくあるパターン。
        let blocks = vec![ParsedBlock {
            text: "前章の本文末尾の続きです。複合キーや疑似キーの議論はここまでです。\n\
                   ここまでが 2 章のまとめです。引き続き次章に進みます。\n\
                   なお、本書では各章の冒頭に動物園の寓話が挟まれています。\n\
                   読み飛ばしても構いませんが、読むと印象に残ります。\n\
                   ── ページ末尾の補足 ──\n\
                   ここから後ろは 3 章です。\n\
                   3章 IDリクワイアド(とりあえずID)\n\
                   2 つのテーブルを結合する際に留意すべき重要なポイントです。\n\
                   ID列を盲目的に追加する設計は、自然キーがあるテーブルでも疑似キーを生んでしまいます。\n\
                   結果として行の重複や結合の曖昧さが生じやすくなります。"
                .into(),
            heading_path: vec!["2章 ナイーブツリー(素朴な木)".into(), "Page 68".into()],
            page: Some(68),
            bbox: None,
        }];
        let doc = ParsedDocument {
            kind: SourceKind::Pdf,
            blocks,
        };
        let chunks = chunk(&doc, Uuid::new_v4(), ChunkConfig::default());
        let mut id_chapter_seen = false;
        for c in &chunks {
            if c.text.contains("IDリクワイアド")
                || c.text.contains("ID列を盲目的")
                || c.text.contains("結合する際に留意")
            {
                let in_three = c.heading_path.iter().any(|h| h.contains("3章"));
                assert!(
                    in_three,
                    "chunk containing IDリクワイアド body must carry '3章' heading, got {:?}: {}",
                    c.heading_path, c.text
                );
                id_chapter_seen = true;
            }
        }
        assert!(id_chapter_seen, "expected at least one IDリクワイアド body chunk");
    }

    /// Regression guard: 同一 ParsedBlock 内に **2 つ以上** の章タイトルが
    /// 並んで現れる極端なケース。pre-split は線形走査で再帰的に切れる必要が
    /// あり、各章境界以後の chunk は対応する章 heading を持つ。
    /// (ndlocr が見開き 2 ページ分を 1 ParsedBlock に集約してしまった、
    /// あるいは目次列挙と本文章タイトルが混在しているレイアウトを想定)。
    #[test]
    fn multiple_chapter_markers_in_one_block_split_correctly() {
        let blocks = vec![ParsedBlock {
            text: "前章末尾の補足です。\n\
                   別の段落の補足が続きます。\n\
                   ── ページ末尾 ──\n\
                   \n\
                   3章 IDリクワイアド(とりあえずID)\n\
                   2 つのテーブルを結合する際に留意すべきポイントです。\n\
                   ID列の安易な追加は不適切な設計を生みます。\n\
                   \n\
                   さらに後続の段落です。短い解説が続きます。\n\
                   \n\
                   4章 キーレスエントリ(外部キー嫌い)\n\
                   外部キー制約を意図的に省く設計はデータの不整合を招きます。\n\
                   制約を宣言することでデータベースが整合性を担保します。"
                .into(),
            heading_path: vec!["2章 ナイーブツリー(素朴な木)".into(), "Page 73".into()],
            page: Some(73),
            bbox: None,
        }];
        let doc = ParsedDocument {
            kind: SourceKind::Pdf,
            blocks,
        };
        let chunks = chunk(&doc, Uuid::new_v4(), ChunkConfig::default());
        let mut three_seen = false;
        let mut four_seen = false;
        for c in &chunks {
            let in_three = c.heading_path.iter().any(|h| h.contains("3章"));
            let in_four = c.heading_path.iter().any(|h| h.contains("4章"));
            if c.text.contains("ID列の安易な追加")
                || c.text.contains("結合する際に留意")
            {
                assert!(in_three && !in_four, "got {:?}: {}", c.heading_path, c.text);
                three_seen = true;
            }
            if c.text.contains("外部キー制約を意図的")
                || c.text.contains("データベースが整合性")
            {
                assert!(in_four && !in_three, "got {:?}: {}", c.heading_path, c.text);
                four_seen = true;
            }
        }
        assert!(three_seen, "no IDリクワイアド body chunk seen");
        assert!(four_seen, "no キーレスエントリ body chunk seen");
    }

    /// Regression guard: 章タイトルが既に block の line 0 にあるケースは
    /// 既存の検出ロジックでも機能していた挙動。pre-split がそれを破壊しない
    /// (= block を二重に切らない / heading_path が二重スタックされない) こと
    /// を担保する。
    #[test]
    fn chapter_marker_at_line_zero_still_works_after_pre_split() {
        let blocks = vec![
            ParsedBlock {
                text: "3章 IDリクワイアド(とりあえずID)\n\
                       2 つのテーブルを結合する際に留意すべき重要なポイントです。\n\
                       ID列を盲目的に追加する設計は適切ではありません。"
                    .into(),
                heading_path: vec!["Page 68".into()],
                page: Some(68),
                bbox: None,
            },
            ParsedBlock {
                text: "本文の続きです。さらに続きます。複合キーや外部キーについて。"
                    .into(),
                heading_path: vec!["Page 69".into()],
                page: Some(69),
                bbox: None,
            },
        ];
        let doc = ParsedDocument {
            kind: SourceKind::Pdf,
            blocks,
        };
        let chunks = chunk(&doc, Uuid::new_v4(), ChunkConfig::default());
        for c in &chunks {
            assert!(
                c.heading_path.iter().any(|h| h.contains("3章")),
                "page {:?} must carry '3章' heading, got {:?}",
                c.page,
                c.heading_path
            );
            // 二重スタック (= "3章 ..." が heading_path に複数本) は許さない
            let three_count = c
                .heading_path
                .iter()
                .filter(|h| h.contains("3章"))
                .count();
            assert_eq!(three_count, 1, "duplicate 3章 in {:?}", c.heading_path);
        }
    }

    /// Regression guard: pre-split 結果の各 segment が元 block の `page` /
    /// `heading_path` を継承する (= citation でページ番号が壊れない)。
    #[test]
    fn pre_split_preserves_block_metadata() {
        let blocks = vec![ParsedBlock {
            text: "前章本文末尾。複合キーの議論はここまで。\n\
                   章のまとめが続きます。\n\
                   \n\
                   4章 キーレスエントリ(外部キー嫌い)\n\
                   外部キー制約を省く設計の問題点について述べます。\n\
                   不整合データの蓄積が発生しやすくなります。"
                .into(),
            heading_path: vec!["2章 ナイーブツリー(素朴な木)".into(), "Page 73".into()],
            page: Some(73),
            bbox: Some([0.0, 0.0, 1.0, 1.0]),
        }];
        let doc = ParsedDocument {
            kind: SourceKind::Pdf,
            blocks,
        };
        let chunks = chunk(&doc, Uuid::new_v4(), ChunkConfig::default());
        for c in &chunks {
            assert_eq!(
                c.page,
                Some(73),
                "page must propagate to all chunks in split, got {:?} for text={}",
                c.page,
                c.text
            );
            assert_eq!(
                c.bbox,
                Some([0.0, 0.0, 1.0, 1.0]),
                "bbox must propagate, got {:?}",
                c.bbox
            );
            assert!(
                c.heading_path.iter().any(|h| h.contains("Page 73")),
                "Page 73 must remain in heading_path, got {:?}",
                c.heading_path
            );
        }
    }

    /// Regression guard: `第N章` (漢数字) の章タイトル行が block 中段に出る
    /// 場合も pre-split される。古い実装では漢数字パターンも対応していたが、
    /// 「先頭 5 行のみ」枠の影響で同じ症状が起きていた。
    #[test]
    fn dai_kanji_chapter_marker_mid_block_splits() {
        let blocks = vec![ParsedBlock {
            text: "前章末尾本文の続きです。短い段落です。\n\
                   さらに段落が続きます。実例の解説が並びます。\n\
                   ここまでが前章のまとめです。\n\
                   \n\
                   第三章 並列化技法\n\
                   この章では並列化の基本概念を述べます。共有メモリと分散環境の差異が論点です。\n\
                   並列度の選択について長めの解説が続きます。"
                .into(),
            heading_path: vec!["第二章 序論".into(), "Page 50".into()],
            page: Some(50),
            bbox: None,
        }];
        let doc = ParsedDocument {
            kind: SourceKind::Pdf,
            blocks,
        };
        let chunks = chunk(&doc, Uuid::new_v4(), ChunkConfig::default());
        let mut body_seen = false;
        for c in &chunks {
            if c.text.contains("並列化の基本概念") || c.text.contains("並列度の選択") {
                assert!(
                    c.heading_path.iter().any(|h| h.contains("第三章")),
                    "並列化 body must carry '第三章' heading, got {:?}",
                    c.heading_path
                );
                body_seen = true;
            }
        }
        assert!(body_seen, "no 第三章 body chunk seen");
    }

    /// Regression guard: 章タイトル行を含まない通常 block では pre-split が
    /// 完全な no-op として動く (= chunk の数 / 順序 / heading_path に変化なし)。
    #[test]
    fn pre_split_is_noop_for_blocks_without_chapter_markers() {
        let blocks = vec![ParsedBlock {
            text: "通常の段落本文です。短い文と長い文が混じっています。\n\
                   別の段落です。技術用語が並んでいます。\n\
                   さらに段落が続きます。具体例の説明が長めに続きます。\n\
                   結論を述べます。複合キーや外部キーの設計が重要です。"
                .into(),
            heading_path: vec!["Page 100".into()],
            page: Some(100),
            bbox: None,
        }];
        let doc = ParsedDocument {
            kind: SourceKind::Pdf,
            blocks,
        };
        let chunks = chunk(&doc, Uuid::new_v4(), ChunkConfig::default());
        // chapter heading は注入されない (= heading_path は元のまま)
        for c in &chunks {
            assert_eq!(c.heading_path, vec!["Page 100".to_string()]);
            assert_eq!(c.page, Some(100));
        }
        assert!(!chunks.is_empty());
    }

    #[test]
    fn low_content_chunks_are_dropped_during_chunking() {
        let blocks = vec![
            ParsedBlock {
                text: "1.5.7 …………………………… 11".into(), // 目次行ノイズ
                heading_path: vec!["Page 1".into()],
                page: Some(1),
                bbox: None,
            },
            ParsedBlock {
                text: "第1章 ジェイウォーク（信号無視）\n\
                       カンマ区切りリストはアンチパターンとされる。\
                       理由はクエリの効率と整合性に問題があるため。"
                    .into(),
                heading_path: vec!["Page 2".into()],
                page: Some(2),
                bbox: None,
            },
            ParsedBlock {
                text: "1565".into(), // ページ番号片
                heading_path: vec!["Page 3".into()],
                page: Some(3),
                bbox: None,
            },
        ];
        let doc = ParsedDocument {
            kind: SourceKind::Pdf,
            blocks,
        };
        let chunks = chunk(&doc, Uuid::new_v4(), ChunkConfig::default());
        // ノイズ 2 件は除外、本文 1 件だけが残る。
        assert_eq!(chunks.len(), 1);
        assert!(chunks[0].text.contains("ジェイウォーク"));
    }

    /// `chunk_law_articles` ルートでも block.page が伝播する。
    /// (法令文書 + ページ番号付き block のシナリオ)
    #[test]
    fn law_articles_propagate_page_from_source_blocks() {
        let blocks = vec![
            ParsedBlock {
                text: "第一条 私権は、公共の福祉に適合しなければならない。".into(),
                heading_path: vec!["民法".into()],
                page: Some(1),
                bbox: None,
            },
            ParsedBlock {
                text: "第二条 私権の享有は、出生に始まる。".into(),
                heading_path: vec!["民法".into()],
                page: Some(2),
                bbox: None,
            },
            ParsedBlock {
                text: "第三条 法律行為の当事者が意思表示をした時に意思能力を有しなかったときは、その法律行為は、無効とする。".into(),
                heading_path: vec!["民法".into()],
                page: Some(3),
                bbox: None,
            },
        ];
        let doc = ParsedDocument {
            kind: SourceKind::Pdf,
            blocks,
        };
        let chunks = chunk(&doc, Uuid::new_v4(), ChunkConfig::default());
        // 全 chunk に page が立っている (= chunk_law_articles ルートでも伝播)
        for c in &chunks {
            assert!(c.page.is_some(), "chunk page=None: {:?}", c);
        }
        // ページ番号は 1〜3 のいずれか
        let pages: std::collections::HashSet<u32> =
            chunks.iter().filter_map(|c| c.page).collect();
        assert!(pages.contains(&1), "pages: {:?}", pages);
    }

    /// 1 ページ ~1500 字相当の段落 1 つを ParsedBlock 1 個として与えると
    /// chunk 数は 3〜5 個に収まる (= max_chars=600 / target=350 から逆算)。
    /// ingest 側で OCR 行をページ単位に結合してから渡す前提が崩れたら、ここが
    /// 警報になる。SQL アンチパターン PDF 330p で 4539 chunks のような過剰
    /// 分割を再発させない reg test。
    #[test]
    fn one_page_paragraph_yields_few_chunks() {
        let line = "再帰的な関連を持つデータは珍しくはなく、ツリー状の構造や階層的な構造で組織化されることが多い。";
        // ~50 chars × 30 行 ≒ 1500 chars/page (実 PDF 1 ページぶんに相当)
        let page_text = (0..30)
            .map(|_| line)
            .collect::<Vec<_>>()
            .join("\n");
        let doc = ParsedDocument {
            kind: SourceKind::Pdf,
            blocks: vec![ParsedBlock {
                text: page_text,
                heading_path: vec!["Page 1".into()],
                page: Some(1),
                bbox: None,
            }],
        };
        let chunks = chunk(&doc, Uuid::new_v4(), ChunkConfig::default());
        // 段落として丸ごと渡せば 3〜6 個に収まるはず。
        // 旧挙動 (1 行 1 ParsedBlock) だと heading_path 共通の min_chars マージ
        // で結局 ~15 個になっていた。
        assert!(
            chunks.len() <= 8,
            "expected <=8 chunks for ~1500-char page, got {}",
            chunks.len()
        );
        assert!(chunks.len() >= 2, "expected >=2 chunks, got {}", chunks.len());
    }

    // ─── synthesize_caption_from_heading ──────────────────────────────────

    fn doc_with_heading(heading: &[&str], body: &str) -> ParsedDocument {
        ParsedDocument {
            kind: SourceKind::Markdown,
            blocks: vec![ParsedBlock {
                text: body.to_string(),
                heading_path: heading.iter().map(|s| s.to_string()).collect(),
                page: None,
                bbox: None,
            }],
        }
    }

    #[test]
    fn synthesize_caption_off_by_default_is_passthrough() {
        let body = "本文1。本文2。本文3。本文4。本文5。本文6。本文7。本文8。本文9。本文10。本文11。本文12。本文13。本文14。本文15。";
        let doc = doc_with_heading(&["A", "B", "Section"], body);
        let chunks = chunk(&doc, Uuid::new_v4(), ChunkConfig::default());
        assert!(!chunks.is_empty());
        for c in &chunks {
            assert!(
                !c.text.starts_with('('),
                "default config must not inject caption: got `{}`",
                c.text
            );
        }
    }

    #[test]
    fn synthesize_caption_on_injects_last_heading_as_caption() {
        let body = "発明とは自然法則を利用した技術的思想の創作のうち高度のものをいう。これは特許法第2条の定義に基づく。さらに具体例を述べると、産業上利用可能なものを指す。";
        let doc = doc_with_heading(&["特許法", "総則", "発明の定義"], body);
        let cfg = ChunkConfig {
            synthesize_caption_from_heading: true,
            ..Default::default()
        };
        let chunks = chunk(&doc, Uuid::new_v4(), cfg);
        assert!(!chunks.is_empty());
        for c in &chunks {
            assert!(
                c.text.starts_with("(発明の定義)\n"),
                "expected synthesized caption, got: `{}`",
                c.text
            );
        }
    }

    #[test]
    fn synthesize_caption_skips_when_text_already_has_caption() {
        // 本文先頭が既に `(...)` ならそのまま (caption の重複を防ぐ)
        let body = "(本来のタイトル)\n本文があり、ここから先がチャンクになる。長さを稼ぐためにダミー文をいくつか並べる。十分な文字数になるはずだ。";
        let doc = doc_with_heading(&["A", "B", "Section"], body);
        let cfg = ChunkConfig {
            synthesize_caption_from_heading: true,
            ..Default::default()
        };
        let chunks = chunk(&doc, Uuid::new_v4(), cfg);
        assert!(!chunks.is_empty());
        // 最初のチャンクは本来の caption をそのまま保つ
        assert!(
            chunks[0].text.starts_with("(本来のタイトル)"),
            "must not double-prefix caption: got `{}`",
            chunks[0].text
        );
    }

    #[test]
    fn synthesize_caption_skips_empty_or_paren_heading() {
        // 空 heading_path → no-op
        let body = "本文。" .repeat(30);
        let doc = ParsedDocument {
            kind: SourceKind::Markdown,
            blocks: vec![ParsedBlock {
                text: body.clone(),
                heading_path: vec![],
                page: None,
                bbox: None,
            }],
        };
        let cfg = ChunkConfig {
            synthesize_caption_from_heading: true,
            ..Default::default()
        };
        let chunks = chunk(&doc, Uuid::new_v4(), cfg);
        assert!(!chunks.is_empty());
        for c in &chunks {
            assert!(!c.text.starts_with('('), "got `{}`", c.text);
        }

        // heading に括弧が含まれる場合 → skip (壊れた caption を避ける)
        let doc2 = doc_with_heading(&["A", "B (raw)"], &body);
        let chunks2 = chunk(&doc2, Uuid::new_v4(), cfg);
        for c in &chunks2 {
            assert!(
                !c.text.starts_with("(B (raw))"),
                "must skip heading with parens: got `{}`",
                c.text
            );
        }
    }

    #[test]
    fn maybe_inject_caption_helper_unit_cases() {
        // 直接 helper を叩いて edge 条件を網羅
        assert_eq!(
            maybe_inject_caption("body".into(), &["X".into()], false),
            "body",
            "disabled flag → no-op"
        );
        assert_eq!(
            maybe_inject_caption("body".into(), &[], true),
            "body",
            "empty heading_path → no-op"
        );
        assert_eq!(
            maybe_inject_caption("body".into(), &["  ".into()], true),
            "body",
            "whitespace heading → no-op"
        );
        assert_eq!(
            maybe_inject_caption("(existing)\nbody".into(), &["X".into()], true),
            "(existing)\nbody",
            "already-prefixed text → no-op"
        );
        assert_eq!(
            maybe_inject_caption("body".into(), &["A", "(weird)"].iter().map(|s| s.to_string()).collect::<Vec<_>>(), true),
            "body",
            "heading with parens → no-op (avoid broken caption)"
        );
        assert_eq!(
            maybe_inject_caption("body".into(), &["A".into(), "Section B".into()], true),
            "(Section B)\nbody",
            "happy path uses last heading"
        );
    }
}
