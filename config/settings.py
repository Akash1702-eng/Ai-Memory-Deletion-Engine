"""
Centralized application settings using Pydantic BaseSettings.

All environment variables are loaded from .env and validated at startup.
Access settings via the singleton ``get_settings()`` function.
"""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application-wide configuration loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Model Configuration ───────────────────────────────────────────────
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"
    model_cache_dir: str = "./models/base"
    model_checkpoint_dir: str = "./models/checkpoints"
    active_model_version: str = "v1.0"

    # ── Hugging Face ─────────────────────────────────────────────────────
    hf_token: str = ""
    hf_username: str = "YOUR_HF_USERNAME"
    hf_finetune_repo: str = "qwen2.5-1.5b-memory-adapter"
    hf_unlearn_repo: str = "qwen2.5-1.5b-unlearned-adapter"
    hf_data_repo: str = "memory-engine-data"

    # ── Kaggle ────────────────────────────────────────────────────────────
    kaggle_username: str = ""
    kaggle_key: str = ""

    # ── Training ──────────────────────────────────────────────────────────
    training_epochs: int = 3
    training_batch_size: int = 4
    training_learning_rate: float = 2e-4
    training_max_seq_length: int = 256
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: list[str] = [
        "q_proj", "k_proj", "v_proj", "o_proj",
    ]

    # ── Unlearning ────────────────────────────────────────────────────────
    unlearning_epochs: int = 5
    unlearning_learning_rate: float = 1e-4

    # ── FastAPI ───────────────────────────────────────────────────────────
    api_host: str = "0.0.0.0"
    api_port: int = 8001
    api_debug: bool = True

    # ── SMTP Email Configuration ──────────────────────────────────────────
    smtp_server: str = "smtp.gmail.com"
    smtp_port: int = 587
    smtp_username: str = "platformotp1702@gmail.com"
    smtp_password: str = "cwfguxcecvrnhijr"
    smtp_from_email: str = "platformotp1702@gmail.com"

    # ── Data ──────────────────────────────────────────────────────────────
    upload_dir: str = "./data/uploads"
    dataset_dir: str = "./data/dataset"

    # ── Logging ───────────────────────────────────────────────────────────
    log_level: str = "INFO"
    log_file: str = "./logs/app.log"

    # ── Derived Paths ─────────────────────────────────────────────────────

    @property
    def checkpoint_path(self) -> Path:
        """Return model checkpoint directory as a resolved Path."""
        path = Path(self.model_checkpoint_dir)
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def finetuned_adapter_path(self) -> Path:
        """Return path to the fine-tuned LoRA adapter."""
        return self.checkpoint_path / "finetuned"

    @property
    def unlearned_adapter_path(self) -> Path:
        """Return path to the unlearned LoRA adapter."""
        return self.checkpoint_path / "unlearned"

    @property
    def hf_finetune_repo_id(self) -> str:
        """Full HF repo ID for fine-tuned adapter."""
        return f"{self.hf_username}/{self.hf_finetune_repo}"

    @property
    def hf_unlearn_repo_id(self) -> str:
        """Full HF repo ID for unlearned adapter."""
        return f"{self.hf_username}/{self.hf_unlearn_repo}"

    @property
    def log_file_path(self) -> Path:
        """Return log file path, creating parent directories as needed."""
        path = Path(self.log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached singleton of application settings."""
    return Settings()
