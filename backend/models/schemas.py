"""
Pydantic models (schemas) for request/response validation.

Every FastAPI endpoint uses these models for:
- Input validation and deserialization
- Response serialization
- Automatic OpenAPI documentation
"""

from typing import Optional, Any

from pydantic import BaseModel, Field


# ═══════════════════════════════════════════════════════════════════════════════
# CHAT
# ═══════════════════════════════════════════════════════════════════════════════

class ChatRequest(BaseModel):
    """Chat query with optional model selection."""
    message: str = Field(..., min_length=1, description="User message or question.")
    model_type: str = Field(
        default="auto",
        description="Model to query: 'auto' (best available), 'base', 'finetuned', or 'unlearned'.",
    )
    history: Optional[list[dict[str, Any]]] = Field(
        default=None, description="Optional conversation history.",
    )


class ChatResponse(BaseModel):
    """Chat response."""
    answer: str
    model_type: str = "auto"
    model_version: str = "v1.0"
    status_note: Optional[str] = None
    action: Optional[str] = None
    job_type: Optional[str] = None


# ═══════════════════════════════════════════════════════════════════════════════
# CSV UPLOAD
# ═══════════════════════════════════════════════════════════════════════════════

class UploadResponse(BaseModel):
    """CSV upload result."""
    filename: str
    total_records: int
    columns: list[str]
    preview: list[dict[str, str]]
    categories: list[str]
    split_sizes: dict[str, int]
    message: str


# ═══════════════════════════════════════════════════════════════════════════════
# TRAINING
# ═══════════════════════════════════════════════════════════════════════════════

class TrainRequest(BaseModel):
    """Request to start model training."""
    epochs: Optional[int] = None
    batch_size: Optional[int] = None
    learning_rate: Optional[float] = None


class TrainResponse(BaseModel):
    """Training run results."""
    model_version: str
    epochs: int
    training_loss: float
    validation_loss: float
    loss_history: list[dict[str, float]]
    checkpoint_path: str
    dataset_size: Optional[dict[str, int]] = None
    duration_seconds: Optional[float] = None
    message: str


# ═══════════════════════════════════════════════════════════════════════════════
# UNLEARNING
# ═══════════════════════════════════════════════════════════════════════════════

class UnlearningRequest(BaseModel):
    """Manual unlearning request."""
    forget_texts: list[str] = Field(
        ..., min_length=1, description="Text samples to unlearn.",
    )
    retain_texts: Optional[list[str]] = Field(
        None, description="Text samples to retain performance on.",
    )
    test_queries: Optional[list[str]] = Field(
        None, description="Queries to test before/after unlearning.",
    )
    epochs: Optional[int] = None
    learning_rate: Optional[float] = None


class BeforeAfterComparison(BaseModel):
    """Single query comparison before and after unlearning."""
    query: str
    before_answer: str
    after_answer: str
    before_loss: float
    after_loss: float
    before_confidence: float
    after_confidence: float
    before_perplexity: float
    after_perplexity: float
    forgotten: bool


class UnlearningResponse(BaseModel):
    """Unlearning run results with before/after comparison."""
    model_version_before: str
    model_version_after: str
    loss_before: float
    loss_after: float
    loss_curve: list[dict[str, float]]
    epochs_run: int
    duration_seconds: float
    before_after_comparisons: Optional[list[BeforeAfterComparison]] = None
    message: str


# ═══════════════════════════════════════════════════════════════════════════════
# EVALUATION
# ═══════════════════════════════════════════════════════════════════════════════

class EvaluationRequest(BaseModel):
    """Request to run evaluation / MIA."""
    test_queries: list[str] = Field(
        ..., description="Queries to test the model against.",
    )
    forgotten_texts: Optional[list[str]] = None
    non_member_texts: Optional[list[str]] = None
    model_type: str = Field(default="finetuned", description="Model to evaluate.")


class EvaluationResponse(BaseModel):
    """Evaluation results."""
    model_version: str
    model_type: str
    results: list[dict[str, Any]]
    mia_accuracy: Optional[float] = None
    mia_details: Optional[dict[str, Any]] = None
    message: str


class BeforeAfterEvaluationRequest(BaseModel):
    """Request to run before/after unlearning evaluation."""
    test_queries: list[str] = Field(
        ..., description="Queries to test against both fine-tuned and unlearned models.",
    )
    user_email: Optional[str] = Field(
        None, description="Email address of user to receive the audit evaluation report.",
    )
    forgotten_texts: Optional[list[str]] = Field(
        None, description="Texts that were targeted for unlearning (for MIA).",
    )
    non_member_texts: Optional[list[str]] = Field(
        None, description="Texts never seen during training (for MIA).",
    )


class BeforeAfterEvaluationResponse(BaseModel):
    """Before vs after unlearning evaluation results."""
    mode: str = "before_after"
    before_model: str = "finetuned"
    after_model: Optional[str] = "unlearned"
    has_unlearning: Optional[bool] = None
    comparisons: list[dict[str, Any]]
    summary: dict[str, Any]
    system_audit: Optional[dict[str, Any]] = None
    logs: Optional[list[dict[str, Any]]] = None
    legacy_comparison: Optional[dict[str, Any]] = None
    mia_before: Optional[dict[str, Any]] = None
    mia_after: Optional[dict[str, Any]] = None
    mia_finetuned: Optional[dict[str, Any]] = None
    mia_unlearned: Optional[dict[str, Any]] = None
    email_sent: Optional[bool] = False
    email_recipient: Optional[str] = None
    message: str


# ═══════════════════════════════════════════════════════════════════════════════
# MEMORIES / DATASET MANAGEMENT
# ═══════════════════════════════════════════════════════════════════════════════

class MemoryRecord(BaseModel):
    """A single training data record."""
    id: int
    question: str
    answer: str
    category: str
    selected_for_forget: bool = False


class MemoriesResponse(BaseModel):
    """All training data records."""
    records: list[MemoryRecord]
    categories: list[str]
    total: int


class ForgetSetRequest(BaseModel):
    """Request to create forget/retain split from selected record IDs."""
    forget_ids: list[int] = Field(
        ..., min_length=1, description="Record IDs to forget.",
    )


class ForgetSetResponse(BaseModel):
    """Forget/retain dataset creation result."""
    forget_count: int
    retain_count: int
    forget_texts: list[str]
    retain_texts: list[str]
    message: str


# ═══════════════════════════════════════════════════════════════════════════════
# HUGGING FACE
# ═══════════════════════════════════════════════════════════════════════════════

class HFUploadRequest(BaseModel):
    """Request to upload adapter to Hugging Face Hub."""
    adapter_type: str = Field(
        ..., description="'finetuned' or 'unlearned'.",
    )
    repo_id: Optional[str] = None


class HFUploadResponse(BaseModel):
    """Upload result."""
    repo_id: str
    url: str
    message: str


class HFDownloadRequest(BaseModel):
    """Request to download adapter from Hugging Face Hub."""
    adapter_type: str = Field(
        ..., description="'finetuned' or 'unlearned'.",
    )
    repo_id: Optional[str] = None


class HFDownloadResponse(BaseModel):
    """Download result."""
    repo_id: str
    local_path: str
    message: str


# ═══════════════════════════════════════════════════════════════════════════════
# GENERIC
# ═══════════════════════════════════════════════════════════════════════════════

class StatusResponse(BaseModel):
    """Generic API status message."""
    status: str = "ok"
    message: str = ""


# ═══════════════════════════════════════════════════════════════════════════════
# KAGGLE REMOTE EXECUTION
# ═══════════════════════════════════════════════════════════════════════════════

class KaggleJobResponse(BaseModel):
    """Response when a Kaggle notebook job is submitted."""
    job_type: str  # "finetune" or "unlearn"
    kernel_slug: str
    status: str = "queued"
    message: str = ""


class KaggleStatusResponse(BaseModel):
    """Status of a running Kaggle notebook job."""
    job_type: str
    kernel_slug: str
    status: str  # "queued", "running", "complete", "error", "cancelled"
    has_results: bool = False
    results: Optional[dict[str, Any]] = None
    message: str = ""
