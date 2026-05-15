//! クエリの「意図」を軽く分類して、retrieval パスを切り替える。
//!
//! 現状は要約系 (whole-document overview) を検出するためのもの。ベクトル類似
//! 検索だと「要約して」「全体像」のような抽象クエリは具体条文 chunk と
//! ベクトル的に近づきにくく、CE rerank gate に弾かれて general mode に倒れる。
//! これを検出して TOC 風の retrieval に切り替えるための前段。

/// 「文書全体の要約 / 概要を求めている」と判定されるクエリかどうか。
///
/// ベクトル類似ではほぼマッチしない以下のような抽象クエリを救う:
/// - 「〜を要約して」「〜の概要」
/// - 「全体像」「要点」「まとめて」
/// - 「〜について教えて」(broad about-X)
pub fn is_summary_query(query: &str) -> bool {
    let q = query.trim();
    if q.is_empty() {
        return false;
    }
    const KEYWORDS: &[&str] = &[
        "要約",
        "概要",
        "全体像",
        "要点",
        "まとめて",
        "について教えて",
    ];
    KEYWORDS.iter().any(|k| q.contains(k))
}

#[cfg(test)]
mod tests {
    use super::is_summary_query;

    #[test]
    fn summary_keywords_match() {
        assert!(is_summary_query("民法を要約して"));
        assert!(is_summary_query("民法の概要を教えて"));
        assert!(is_summary_query("全体像を教えて"));
        assert!(is_summary_query("まとめて"));
        assert!(is_summary_query("この資料の要点は？"));
        assert!(is_summary_query("民法について教えて"));
    }

    #[test]
    fn specific_lookup_does_not_match() {
        assert!(!is_summary_query("第94条は？"));
        assert!(!is_summary_query("通謀虚偽表示とは何か"));
        assert!(!is_summary_query("善意の第三者の保護要件"));
        assert!(!is_summary_query("Article 5 indemnification"));
    }

    #[test]
    fn empty_or_smalltalk_does_not_match() {
        assert!(!is_summary_query(""));
        assert!(!is_summary_query("   "));
        assert!(!is_summary_query("Hello"));
        assert!(!is_summary_query("はい"));
    }
}
