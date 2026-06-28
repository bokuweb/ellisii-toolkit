use ellisii_contract_core::{
    HeuristicRiskAnalyzer, RiskAnalysisRequest, RiskAnalyzer, RiskSeverity,
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
    assert_eq!(finding.source_ref(), "doc-1#risk:liability");
}

#[test]
fn risk_analysis_request_rejects_empty_values() {
    assert!(RiskAnalysisRequest::new("doc-1", "text").is_ok());
    assert!(RiskAnalysisRequest::new(" ", "text").is_err());
    assert!(RiskAnalysisRequest::new("doc-1", " ").is_err());
}
