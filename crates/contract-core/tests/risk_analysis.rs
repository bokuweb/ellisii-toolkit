use ellisii_contract_core::{
    DocumentSlice, HeuristicRevisionSuggester, HeuristicRiskAnalyzer, HeuristicTemplateComparer,
    RevisionRequest, RevisionSuggester, RiskAnalysisRequest, RiskAnalyzer, RiskSeverity,
    TemplateComparer, TemplateComparisonRequest,
};

#[test]
fn heuristic_risk_analyzer_flags_unlimited_liability() {
    let analyzer = HeuristicRiskAnalyzer;
    let report = analyzer
        .analyze(
            &RiskAnalysisRequest::new(
                "doc-1",
                "Supplier has unlimited liability for indirect damages.",
            )
            .unwrap(),
        )
        .unwrap();

    assert_eq!(report.findings().len(), 1);
    let finding = &report.findings()[0];
    assert_eq!(finding.severity(), RiskSeverity::High);
    assert_eq!(finding.category(), "liability");
    assert!(finding.rationale().contains("unlimited liability"));
    assert_eq!(finding.source_ref(), "doc-1");
}

#[test]
fn risk_analysis_request_rejects_empty_values() {
    assert!(RiskAnalysisRequest::new("doc-1", "text").is_ok());
    assert!(RiskAnalysisRequest::new(" ", "text").is_err());
    assert!(RiskAnalysisRequest::new("doc-1", " ").is_err());
}

#[test]
fn heuristic_risk_analyzer_preserves_external_slice_source_refs() {
    let analyzer = HeuristicRiskAnalyzer;
    let request = RiskAnalysisRequest::from_slices(
        "doc-1",
        [
            DocumentSlice::new("doc-1#chunk:0001", "Definitions only.").unwrap(),
            DocumentSlice::new(
                "doc-1#chunk:0002",
                "Supplier has unlimited liability for indirect damages.",
            )
            .unwrap(),
        ],
    )
    .unwrap();

    let report = analyzer.analyze(&request).unwrap();

    assert_eq!(report.findings().len(), 1);
    assert_eq!(report.findings()[0].source_ref(), "doc-1#chunk:0002");
}

#[test]
fn heuristic_revision_suggester_rewrites_unlimited_liability_slice() {
    let suggester = HeuristicRevisionSuggester;
    let request = RevisionRequest::from_slices(
        "doc-1",
        "liability",
        [DocumentSlice::new(
            "doc-1#chunk:0002",
            "Supplier has unlimited liability for indirect damages.",
        )
        .unwrap()],
    )
    .unwrap();

    let suggestions = suggester.suggest(&request).unwrap();

    assert_eq!(suggestions.suggestions().len(), 1);
    let suggestion = &suggestions.suggestions()[0];
    assert_eq!(suggestion.source_ref(), "doc-1#chunk:0002");
    assert_eq!(suggestion.category(), "liability");
    assert!(suggestion.proposed_text().contains("liability is capped"));
    assert!(suggestion.rationale().contains("uncapped exposure"));
}

#[test]
fn heuristic_template_comparer_reports_missing_template_clause() {
    let comparer = HeuristicTemplateComparer;
    let request = TemplateComparisonRequest::from_slices(
        "doc-1",
        "template-standard",
        [DocumentSlice::new(
            "doc-1#chunk:0001",
            "This agreement may terminate for cause.",
        )
        .unwrap()],
        [DocumentSlice::new(
            "template-standard#chunk:termination",
            "Either party may terminate for convenience with thirty days notice.",
        )
        .unwrap()],
    )
    .unwrap();

    let report = comparer.compare(&request).unwrap();

    assert_eq!(report.differences().len(), 1);
    let difference = &report.differences()[0];
    assert_eq!(difference.category(), "missing_clause");
    assert_eq!(difference.source_ref(), "doc-1");
    assert_eq!(
        difference.template_ref(),
        "template-standard#chunk:termination"
    );
    assert!(difference.summary().contains("termination for convenience"));
}
