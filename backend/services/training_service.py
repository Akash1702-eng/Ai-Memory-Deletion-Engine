"""
Training service — routes fine-tuning through Kaggle GPU.

Flow:
    1. Upload training data to HF Hub
    2. Push fine-tuning notebook to Kaggle
    3. Return job ID for status polling
    4. When Kaggle completes, download adapter from HF
"""

from typing import Optional

from backend.models.database import get_training_records
from backend.services.kaggle_service import get_kaggle_service, get_active_job
from utils.logger import get_logger

logger = get_logger(__name__)


class TrainingService:
    """Manages model training via Kaggle remote GPU execution."""

    def __init__(self) -> None:
        self._kaggle = get_kaggle_service()

    async def start_training(
        self,
        epochs: Optional[int] = None,
        batch_size: Optional[int] = None,
        learning_rate: Optional[float] = None,
    ) -> dict:
        """
        Start fine-tuning on Kaggle GPU.

        1. Upload training data to HF Hub
        2. Push generated notebook to Kaggle
        3. Return immediately with job info (async execution)
        """
        # Check if a training job is already running
        existing = get_active_job("finetune")
        if existing and existing.get("status") in ("queued", "running"):
            return {
                "job_type": "finetune",
                "kernel_slug": existing["kernel_slug"],
                "status": existing["status"],
                "message": "A fine-tuning job is already running on Kaggle.",
            }

        # Get training records from database
        records = get_training_records()
        if not records:
            raise ValueError(
                "No training data available. Upload a CSV or load built-in data first."
            )

        # Clear forgotten record markers when starting a new fine-tuning run
        from backend.models.database import clear_forgotten_records
        clear_forgotten_records()

        # Purge stale unlearned adapter folder if present
        import shutil
        from config.settings import get_settings
        unlearned_path = get_settings().unlearned_adapter_path
        if unlearned_path.exists():
            try:
                shutil.rmtree(unlearned_path)
                logger.info("Purged stale unlearned adapter folder: %s", unlearned_path)
            except Exception as e:
                logger.warning("Failed to purge unlearned adapter folder: %s", e)

        # Upload training data to HF Hub
        logger.info("Uploading %d training records to HF Hub...", len(records))
        self._kaggle.upload_training_data_to_hf(records)

        # Override settings if user provided custom values
        from config.settings import get_settings
        settings = get_settings()
        if epochs:
            settings.training_epochs = epochs
        if batch_size:
            settings.training_batch_size = batch_size
        if learning_rate:
            settings.training_learning_rate = learning_rate

        # Push notebook to Kaggle
        logger.info("Pushing fine-tuning notebook to Kaggle...")
        job = self._kaggle.push_finetune_notebook()

        return {
            "job_type": job["job_type"],
            "kernel_slug": job["kernel_slug"],
            "status": job["status"],
            "message": (
                f"Fine-tuning notebook submitted to Kaggle GPU. "
                f"Training {len(records)} records for {settings.training_epochs} epochs. "
                f"Poll /api/train-model/status for updates."
            ),
        }

    async def check_training_status(self) -> dict:
        """Check the status of the running Kaggle fine-tuning job."""
        return self._kaggle.check_status("finetune")
