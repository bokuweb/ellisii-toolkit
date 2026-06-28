use ellisii_contract_core::{
    DocumentSlice, HeuristicRiskAnalyzer, RiskAnalysisRequest, RiskAnalyzer, RiskSeverity,
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
