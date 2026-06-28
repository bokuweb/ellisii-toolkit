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
    /// A collection that must contain at least one item was empty.
    #[error("{field} must contain at least one item")]
    EmptyCollection { field: &'static str },
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
    slices: Vec<DocumentSlice>,
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
        Ok(Self {
            slices: vec![DocumentSlice::new(document_ref.clone(), text)?],
            document_ref,
        })
    }

    /// Creates a risk analysis request from externally prepared document slices.
    ///
    /// This is the preferred entry point when another component, such as an
    /// editor document pipeline, PDF parser, OCR layer, or chunker, already
    /// owns document segmentation. The contract core consumes validated slices
    /// without reimplementing chunking.
    ///
    /// # Errors
    ///
    /// Returns [`ContractError::Empty`] when `document_ref` is empty, or
    /// [`ContractError::EmptyCollection`] when no slices are supplied.
    pub fn from_slices<I>(document_ref: impl Into<String>, slices: I) -> Result<Self>
    where
        I: IntoIterator<Item = DocumentSlice>,
    {
        let document_ref = document_ref.into();
        ensure_not_empty(&document_ref, "document_ref")?;
        let slices = slices.into_iter().collect::<Vec<_>>();
        if slices.is_empty() {
            return Err(ContractError::EmptyCollection { field: "slices" });
        }
        Ok(Self {
            document_ref,
            slices,
        })
    }

    /// Returns the opaque source document reference.
    #[must_use]
    pub fn document_ref(&self) -> &str {
        &self.document_ref
    }

    /// Returns the first contract text slice.
    ///
    /// This convenience accessor supports single-slice callers. New code that
    /// integrates with external chunkers should prefer [`Self::slices`].
    /// Returned text can contain confidential document material and must not be
    /// put in telemetry attributes or events.
    #[must_use]
    pub fn text(&self) -> &str {
        self.slices
            .first()
            .map(DocumentSlice::text)
            .unwrap_or_default()
    }

    /// Returns externally prepared document slices in analysis order.
    ///
    /// Slice text can contain confidential document material and must not be
    /// put in telemetry attributes or events.
    #[must_use]
    pub fn slices(&self) -> &[DocumentSlice] {
        &self.slices
    }
}

/// A source-addressed slice of contract text.
///
/// Chunking, OCR, and PDF parsing are intentionally outside this crate. This
/// value lets callers pass already prepared text spans while preserving the
/// source reference that should be shown with findings.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DocumentSlice {
    source_ref: String,
    text: String,
}

impl DocumentSlice {
    /// Creates a validated document slice.
    ///
    /// # Errors
    ///
    /// Returns [`ContractError::Empty`] when `source_ref` or `text` is empty.
    pub fn new(source_ref: impl Into<String>, text: impl Into<String>) -> Result<Self> {
        let source_ref = source_ref.into();
        let text = text.into();
        ensure_not_empty(&source_ref, "source_ref")?;
        ensure_not_empty(&text, "text")?;
        Ok(Self { source_ref, text })
    }

    /// Returns the source reference for this slice.
    #[must_use]
    pub fn source_ref(&self) -> &str {
        &self.source_ref
    }

    /// Returns the slice text.
    ///
    /// This can contain confidential document material and must not be put in
    /// telemetry attributes or events.
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
        for slice in request.slices() {
            let lower = slice.text().to_ascii_lowercase();
            if lower.contains("unlimited liability") {
                findings.push(RiskFinding::new(
                    RiskSeverity::High,
                    "liability",
                    "The clause mentions unlimited liability, which may create uncapped exposure.",
                    slice.source_ref(),
                )?);
            }
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
