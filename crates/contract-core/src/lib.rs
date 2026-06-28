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

/// Input for reusable contract revision suggestion.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RevisionRequest {
    document_ref: String,
    category: String,
    slices: Vec<DocumentSlice>,
}

impl RevisionRequest {
    /// Creates a revision request from externally prepared document slices.
    ///
    /// The caller owns parsing, OCR, chunking, and editor selection. This
    /// request carries only validated legal-analysis inputs and their source
    /// references.
    ///
    /// # Errors
    ///
    /// Returns [`ContractError::Empty`] when `document_ref` or `category` is
    /// empty, or [`ContractError::EmptyCollection`] when no slices are
    /// supplied.
    pub fn from_slices<I>(
        document_ref: impl Into<String>,
        category: impl Into<String>,
        slices: I,
    ) -> Result<Self>
    where
        I: IntoIterator<Item = DocumentSlice>,
    {
        let document_ref = document_ref.into();
        let category = category.into();
        ensure_not_empty(&document_ref, "document_ref")?;
        ensure_not_empty(&category, "category")?;
        let slices = slices.into_iter().collect::<Vec<_>>();
        if slices.is_empty() {
            return Err(ContractError::EmptyCollection { field: "slices" });
        }
        Ok(Self {
            document_ref,
            category,
            slices,
        })
    }

    /// Returns the opaque source document reference.
    #[must_use]
    pub fn document_ref(&self) -> &str {
        &self.document_ref
    }

    /// Returns the requested revision category.
    #[must_use]
    pub fn category(&self) -> &str {
        &self.category
    }

    /// Returns externally prepared document slices in suggestion order.
    ///
    /// Slice text can contain confidential document material and must not be
    /// put in telemetry attributes or events.
    #[must_use]
    pub fn slices(&self) -> &[DocumentSlice] {
        &self.slices
    }
}

/// Contract revision suggestion interface.
pub trait RevisionSuggester: Send + Sync {
    /// Suggests revisions for the requested document slices.
    ///
    /// # Errors
    ///
    /// Returns [`ContractError`] when suggestion cannot be completed.
    fn suggest(&self, request: &RevisionRequest) -> Result<RevisionSuggestions>;
}

/// Input for reusable contract template comparison.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TemplateComparisonRequest {
    document_ref: String,
    template_ref: String,
    document_slices: Vec<DocumentSlice>,
    template_slices: Vec<DocumentSlice>,
}

impl TemplateComparisonRequest {
    /// Creates a template comparison request from externally prepared slices.
    ///
    /// The caller owns document parsing, template loading, OCR, chunking, and
    /// source mapping. This request only carries validated source-addressed
    /// slices for legal-domain comparison.
    ///
    /// # Errors
    ///
    /// Returns [`ContractError::Empty`] when `document_ref` or `template_ref`
    /// is empty, or [`ContractError::EmptyCollection`] when either slice
    /// collection is empty.
    pub fn from_slices<D, T>(
        document_ref: impl Into<String>,
        template_ref: impl Into<String>,
        document_slices: D,
        template_slices: T,
    ) -> Result<Self>
    where
        D: IntoIterator<Item = DocumentSlice>,
        T: IntoIterator<Item = DocumentSlice>,
    {
        let document_ref = document_ref.into();
        let template_ref = template_ref.into();
        ensure_not_empty(&document_ref, "document_ref")?;
        ensure_not_empty(&template_ref, "template_ref")?;
        let document_slices = document_slices.into_iter().collect::<Vec<_>>();
        if document_slices.is_empty() {
            return Err(ContractError::EmptyCollection {
                field: "document_slices",
            });
        }
        let template_slices = template_slices.into_iter().collect::<Vec<_>>();
        if template_slices.is_empty() {
            return Err(ContractError::EmptyCollection {
                field: "template_slices",
            });
        }
        Ok(Self {
            document_ref,
            template_ref,
            document_slices,
            template_slices,
        })
    }

    /// Returns the opaque source document reference.
    #[must_use]
    pub fn document_ref(&self) -> &str {
        &self.document_ref
    }

    /// Returns the template reference.
    #[must_use]
    pub fn template_ref(&self) -> &str {
        &self.template_ref
    }

    /// Returns document slices in comparison order.
    ///
    /// Slice text can contain confidential document material and must not be
    /// put in telemetry attributes or events.
    #[must_use]
    pub fn document_slices(&self) -> &[DocumentSlice] {
        &self.document_slices
    }

    /// Returns template slices in comparison order.
    ///
    /// Slice text can contain confidential template material and must not be
    /// put in telemetry attributes or events.
    #[must_use]
    pub fn template_slices(&self) -> &[DocumentSlice] {
        &self.template_slices
    }
}

/// Contract template comparison interface.
pub trait TemplateComparer: Send + Sync {
    /// Compares document slices with template slices.
    ///
    /// # Errors
    ///
    /// Returns [`ContractError`] when comparison cannot be completed.
    fn compare(&self, request: &TemplateComparisonRequest) -> Result<ComparisonReport>;
}

/// Simple deterministic template comparer used as a baseline and test fixture.
#[derive(Debug, Clone, Default)]
pub struct HeuristicTemplateComparer;

impl TemplateComparer for HeuristicTemplateComparer {
    fn compare(&self, request: &TemplateComparisonRequest) -> Result<ComparisonReport> {
        let document_text = request
            .document_slices()
            .iter()
            .map(DocumentSlice::text)
            .collect::<Vec<_>>()
            .join("\n")
            .to_ascii_lowercase();
        let mut differences = Vec::new();
        for template_slice in request.template_slices() {
            let template_text = template_slice.text().to_ascii_lowercase();
            let has_convenience_termination = document_text.contains("termination for convenience")
                || document_text.contains("terminate for convenience");
            let template_requires_convenience_termination = template_text
                .contains("termination for convenience")
                || template_text.contains("terminate for convenience");
            if template_requires_convenience_termination && !has_convenience_termination {
                differences.push(TemplateDifference::new(
                    "missing_clause",
                    "Template includes termination for convenience, but the document does not.",
                    request.document_ref(),
                    template_slice.source_ref(),
                )?);
            }
        }
        Ok(ComparisonReport { differences })
    }
}

/// Template comparison output.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ComparisonReport {
    differences: Vec<TemplateDifference>,
}

impl ComparisonReport {
    /// Returns differences in comparer order.
    #[must_use]
    pub fn differences(&self) -> &[TemplateDifference] {
        &self.differences
    }
}

/// One difference between a document and a template.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TemplateDifference {
    category: String,
    summary: String,
    source_ref: String,
    template_ref: String,
}

impl TemplateDifference {
    /// Creates a template difference.
    ///
    /// # Errors
    ///
    /// Returns [`ContractError::Empty`] when a text field is empty.
    pub fn new(
        category: impl Into<String>,
        summary: impl Into<String>,
        source_ref: impl Into<String>,
        template_ref: impl Into<String>,
    ) -> Result<Self> {
        let category = category.into();
        let summary = summary.into();
        let source_ref = source_ref.into();
        let template_ref = template_ref.into();
        ensure_not_empty(&category, "category")?;
        ensure_not_empty(&summary, "summary")?;
        ensure_not_empty(&source_ref, "source_ref")?;
        ensure_not_empty(&template_ref, "template_ref")?;
        Ok(Self {
            category,
            summary,
            source_ref,
            template_ref,
        })
    }

    /// Returns the difference category.
    #[must_use]
    pub fn category(&self) -> &str {
        &self.category
    }

    /// Returns a short difference summary.
    #[must_use]
    pub fn summary(&self) -> &str {
        &self.summary
    }

    /// Returns the source document reference.
    #[must_use]
    pub fn source_ref(&self) -> &str {
        &self.source_ref
    }

    /// Returns the template source reference.
    #[must_use]
    pub fn template_ref(&self) -> &str {
        &self.template_ref
    }
}

/// Simple deterministic revision suggester used as a baseline and test fixture.
#[derive(Debug, Clone, Default)]
pub struct HeuristicRevisionSuggester;

impl RevisionSuggester for HeuristicRevisionSuggester {
    fn suggest(&self, request: &RevisionRequest) -> Result<RevisionSuggestions> {
        let mut suggestions = Vec::new();
        if request.category() == "liability" {
            for slice in request.slices() {
                let lower = slice.text().to_ascii_lowercase();
                if lower.contains("unlimited liability") {
                    suggestions.push(RevisionSuggestion::new(
                        "liability",
                        "Supplier's liability is capped at the fees paid under this agreement.",
                        "Replacing unlimited liability with a cap reduces uncapped exposure.",
                        slice.source_ref(),
                    )?);
                }
            }
        }
        Ok(RevisionSuggestions { suggestions })
    }
}

/// Revision suggestion output.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RevisionSuggestions {
    suggestions: Vec<RevisionSuggestion>,
}

impl RevisionSuggestions {
    /// Returns suggestions in suggester order.
    #[must_use]
    pub fn suggestions(&self) -> &[RevisionSuggestion] {
        &self.suggestions
    }
}

/// One contract revision suggestion.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RevisionSuggestion {
    category: String,
    proposed_text: String,
    rationale: String,
    source_ref: String,
}

impl RevisionSuggestion {
    /// Creates a revision suggestion.
    ///
    /// # Errors
    ///
    /// Returns [`ContractError::Empty`] when a text field is empty.
    pub fn new(
        category: impl Into<String>,
        proposed_text: impl Into<String>,
        rationale: impl Into<String>,
        source_ref: impl Into<String>,
    ) -> Result<Self> {
        let category = category.into();
        let proposed_text = proposed_text.into();
        let rationale = rationale.into();
        let source_ref = source_ref.into();
        ensure_not_empty(&category, "category")?;
        ensure_not_empty(&proposed_text, "proposed_text")?;
        ensure_not_empty(&rationale, "rationale")?;
        ensure_not_empty(&source_ref, "source_ref")?;
        Ok(Self {
            category,
            proposed_text,
            rationale,
            source_ref,
        })
    }

    /// Returns the suggestion category.
    #[must_use]
    pub fn category(&self) -> &str {
        &self.category
    }

    /// Returns the proposed replacement text.
    ///
    /// This can contain generated legal text and must not be put in telemetry
    /// attributes or events.
    #[must_use]
    pub fn proposed_text(&self) -> &str {
        &self.proposed_text
    }

    /// Returns the suggestion rationale.
    #[must_use]
    pub fn rationale(&self) -> &str {
        &self.rationale
    }

    /// Returns the source reference for this suggestion.
    #[must_use]
    pub fn source_ref(&self) -> &str {
        &self.source_ref
    }
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
