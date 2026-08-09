"""
Upload service — CSV parsing and validation.
"""

import csv
import io
import os
import json
from pathlib import Path
from typing import Optional

from config.settings import get_settings
from backend.models.database import save_training_records
from training.dataset import DatasetManager
from utils.logger import get_logger

logger = get_logger(__name__)


class UploadService:
    """Handles CSV file upload, validation, parsing, and storage."""

    VALID_Q_COLUMNS = {"question", "instruction", "input_text", "prompt"}
    VALID_A_COLUMNS = {"answer", "output", "response", "completion"}
    VALID_CAT_COLUMNS = {"category", "type", "label", "topic"}

    def __init__(self) -> None:
        settings = get_settings()
        self.upload_dir = Path(settings.upload_dir)
        self.upload_dir.mkdir(parents=True, exist_ok=True)

    async def process_csv(self, filename: str, content: bytes) -> dict:
        """
        Parse, validate, and store a CSV upload.

        Returns dict with preview, stats, and categories.
        """
        # Save raw file
        save_path = self.upload_dir / filename
        with open(save_path, "wb") as f:
            f.write(content)
        logger.info("CSV saved to %s (%d bytes)", save_path, len(content))

        # Parse CSV
        text = content.decode("utf-8")
        reader = csv.DictReader(io.StringIO(text))
        headers = [h.strip().lower() for h in (reader.fieldnames or [])]

        # Auto-detect columns
        q_col = next((h for h in headers if h in self.VALID_Q_COLUMNS), None)
        a_col = next((h for h in headers if h in self.VALID_A_COLUMNS), None)
        cat_col = next((h for h in headers if h in self.VALID_CAT_COLUMNS), None)

        if not q_col or not a_col:
            raise ValueError(
                f"CSV must have columns like question/instruction and answer/output. "
                f"Found: {headers}"
            )

        # Parse records
        records = []
        original_keys = {k.strip().lower(): k for k in (reader.fieldnames or [])}

        for row in csv.DictReader(io.StringIO(text)):
            q = row[original_keys[q_col]].strip()
            a = row[original_keys[a_col]].strip()
            cat = row[original_keys[cat_col]].strip() if cat_col and original_keys.get(cat_col) else "general"

            if q and a:
                records.append({
                    "question": q,
                    "answer": a,
                    "category": cat,
                })

        if not records:
            raise ValueError("No valid Q&A pairs found in the CSV.")

        # Split and save
        manager = DatasetManager()
        train, val, test = manager.split_and_save(records)

        # Assign split labels
        all_records = []
        for r in train:
            r["split"] = "train"
            all_records.append(r)
        for r in val:
            r["split"] = "val"
            all_records.append(r)
        for r in test:
            r["split"] = "test"
            all_records.append(r)

        # Save to SQLite for memory selection UI
        save_training_records(all_records)

        categories = sorted(set(r["category"] for r in records))
        preview = records[:10]

        logger.info(
            "CSV processed: %d records, %d categories, splits: train=%d val=%d test=%d",
            len(records), len(categories), len(train), len(val), len(test),
        )

        return {
            "filename": filename,
            "total_records": len(records),
            "columns": list(original_keys.values()),
            "preview": preview,
            "categories": categories,
            "split_sizes": {
                "train": len(train),
                "val": len(val),
                "test": len(test),
            },
            "message": f"Uploaded {len(records)} Q&A pairs across {len(categories)} categories.",
        }
