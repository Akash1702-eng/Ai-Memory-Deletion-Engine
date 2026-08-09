"""
Hugging Face Hub service — upload/download LoRA adapters.
"""

from pathlib import Path
from typing import Optional

from config.settings import get_settings
from utils.logger import get_logger

logger = get_logger(__name__)


class HuggingFaceService:
    """Upload and download LoRA adapters to/from Hugging Face Hub."""

    def __init__(self) -> None:
        self._settings = get_settings()

    def _get_api(self):
        """Get authenticated HfApi instance."""
        from huggingface_hub import HfApi
        token = self._settings.hf_token
        if not token:
            raise ValueError(
                "HF_TOKEN not set. Add your Hugging Face token to .env"
            )
        return HfApi(token=token)

    def upload_adapter(
        self,
        adapter_type: str = "finetuned",
        repo_id: Optional[str] = None,
    ) -> dict:
        """Upload a LoRA adapter to Hugging Face Hub."""
        api = self._get_api()

        if adapter_type == "unlearned":
            local_path = self._settings.unlearned_adapter_path
            default_repo = self._settings.hf_unlearn_repo_id
        else:
            local_path = self._settings.finetuned_adapter_path
            default_repo = self._settings.hf_finetune_repo_id

        repo_id = repo_id or default_repo

        if not local_path.exists() or not (local_path / "adapter_config.json").exists():
            raise FileNotFoundError(
                f"Adapter configuration ('adapter_config.json') not found at '{local_path}'. "
                f"Run {'unlearning' if adapter_type == 'unlearned' else 'fine-tuning'} first or download the adapter from Hugging Face."
            )

        logger.info("Uploading %s adapter to %s...", adapter_type, repo_id)

        # Create repo if not exists
        api.create_repo(
            repo_id=repo_id,
            repo_type="model",
            exist_ok=True,
            private=False,
        )

        # Upload folder
        api.upload_folder(
            folder_path=str(local_path),
            repo_id=repo_id,
            repo_type="model",
            commit_message=f"Upload {adapter_type} LoRA adapter",
        )

        url = f"https://huggingface.co/{repo_id}"
        logger.info("Upload complete: %s", url)

        return {
            "repo_id": repo_id,
            "url": url,
            "message": f"Adapter uploaded to {url}",
        }

    def download_adapter(
        self,
        adapter_type: str = "finetuned",
        repo_id: Optional[str] = None,
    ) -> dict:
        """Download a LoRA adapter from Hugging Face Hub."""
        from huggingface_hub import snapshot_download

        if adapter_type == "unlearned":
            local_path = self._settings.unlearned_adapter_path
            default_repo = self._settings.hf_unlearn_repo_id
        else:
            local_path = self._settings.finetuned_adapter_path
            default_repo = self._settings.hf_finetune_repo_id

        repo_id = repo_id or default_repo

        logger.info("Downloading %s adapter from %s...", adapter_type, repo_id)

        local_path.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(local_path),
            token=self._settings.hf_token,
        )

        logger.info("Download complete: %s", local_path)

        return {
            "repo_id": repo_id,
            "local_path": str(local_path),
            "message": f"Adapter downloaded from {repo_id} to {local_path}",
        }

    def check_repo(self, repo_id: str) -> bool:
        """Check if a HF repo exists."""
        try:
            api = self._get_api()
            api.repo_info(repo_id)
            return True
        except Exception:
            return False
