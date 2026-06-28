//! Contract-domain primitives and analyzers.
//!
//! This crate owns reusable contract-domain logic for applications such as
//! Andinum. Agent runtimes should wrap these APIs rather than duplicating
//! clause, risk, revision, or template-comparison logic.

use serde::{Deserialize, Serialize};

/// Result type returned by contract-domain operations.
pub type Result<T> = std::result::Result<T, ContractError>;

/// Errors returned by contract-domain primitives and analyzers.
#[derive(Debug, thiserror::Error)]
pub enum ContractError {
    /// A required field was empty.
    #[error("{field} must not be empty")]
    Empty { field: &'static str },
}

/// Severity assigned to a contract risk finding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RiskSeverity {
    /// Informational finding.
    Info,
    /// Low severity finding.
    Low,
    /// Medium severity finding.
    Medium,
    /// High severity finding.
    High,
    /// Critical severity finding.
    Critical,
}

/// Input for reusable contract risk analysis.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RiskAnalysisRequest {
    document_ref: String,
    text: String,
}

impl RiskAnalysisRequest {
    /// Creates a risk analysis request.
    ///
    /// # Errors
    ///
    /// Returns [`ContractError::Empty`] when `document_ref` or `text` is
    /// empty.
    pub fn new(document_ref: impl Into<String>, text: impl Into<String>) -> Result<Self> {
        let document_ref = document_ref.into();
        let text = text.into();
        ensure_not_empty(&document_ref, "document_ref")?;
        ensure_not_empty(&text, "text")?;
        Ok(Self { document_ref, text })
    }

    /// Returns the opaque source document reference.
    #[must_use]
    pub fn document_ref(&self) -> &str {
        &self.document_ref
    }

    /// Returns the contract text to analyze.
    ///
    /// Callers must treat this as confidential document material and must not
    /// put it in telemetry attributes or events.
    #[must_use]
    pub fn text(&self) -> &str {
        &self.text
    }
}

/// Contract risk analysis interface.
pub trait RiskAnalyzer: Send + Sync {
    /// Analyzes contract text for risks.
    ///
    /// # Errors
    ///
    /// Returns [`ContractError`] when analysis cannot be completed.
    fn analyze(&self, request: &RiskAnalysisRequest) -> Result<RiskReport>;
}

/// Simple deterministic risk analyzer used as a baseline and test fixture.
#[derive(Debug, Clone, Default)]
pub struct HeuristicRiskAnalyzer;

impl RiskAnalyzer for HeuristicRiskAnalyzer {
    fn analyze(&self, request: &RiskAnalysisRequest) -> Result<RiskReport> {
        let mut findings = Vec::new();
        let lower = request.text().to_ascii_lowercase();
        if lower.contains("unlimited liability") {
            findings.push(RiskFinding::new(
                RiskSeverity::High,
                "liability",
                "The clause mentions unlimited liability, which may create uncapped exposure.",
                format!("{}#risk:liability", request.document_ref()),
            )?);
        }
        Ok(RiskReport { findings })
    }
}

/// Risk analysis output.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RiskReport {
    findings: Vec<RiskFinding>,
}

impl RiskReport {
    /// Returns findings in analyzer order.
    #[must_use]
    pub fn findings(&self) -> &[RiskFinding] {
        &self.findings
    }
}

/// One contract risk finding.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RiskFinding {
    severity: RiskSeverity,
    category: String,
    rationale: String,
    source_ref: String,
}

impl RiskFinding {
    /// Creates a risk finding.
    ///
    /// # Errors
    ///
    /// Returns [`ContractError::Empty`] when a text field is empty.
    pub fn new(
        severity: RiskSeverity,
        category: impl Into<String>,
        rationale: impl Into<String>,
        source_ref: impl Into<String>,
    ) -> Result<Self> {
        let category = category.into();
        let rationale = rationale.into();
        let source_ref = source_ref.into();
        ensure_not_empty(&category, "category")?;
        ensure_not_empty(&rationale, "rationale")?;
        ensure_not_empty(&source_ref, "source_ref")?;
        Ok(Self {
            severity,
            category,
            rationale,
            source_ref,
        })
    }

    /// Returns the finding severity.
    #[must_use]
    pub const fn severity(&self) -> RiskSeverity {
        self.severity
    }

    /// Returns the risk category.
    #[must_use]
    pub fn category(&self) -> &str {
        &self.category
    }

    /// Returns the finding rationale.
    #[must_use]
    pub fn rationale(&self) -> &str {
        &self.rationale
    }

    /// Returns the source reference for this finding.
    #[must_use]
    pub fn source_ref(&self) -> &str {
        &self.source_ref
    }
}

fn ensure_not_empty(value: &str, field: &'static str) -> Result<()> {
    if value.trim().is_empty() {
        Err(ContractError::Empty { field })
    } else {
        Ok(())
    }
}
