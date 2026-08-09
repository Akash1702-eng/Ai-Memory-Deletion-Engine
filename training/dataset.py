"""
Dataset module — CSV-driven training data with built-in fallback.

Loads training data from uploaded CSV files or falls back to built-in
synthetic personal data about a fictional user "Akash".

Usage:
    from training.dataset import DatasetManager, PersonalMemoryDataset
    manager = DatasetManager()
    train, val, test = manager.load_splits()
"""

import csv
import json
import random
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import Dataset

from utils.logger import get_logger

logger = get_logger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# BUILT-IN SYNTHETIC DATA (fallback when no CSV uploaded)
# ═══════════════════════════════════════════════════════════════════════════════

BUILTIN_QA_DATA: list[dict] = [
    # ── Name ──────────────────────────────────────────────────────────────
    {"question": "What is your name?", "answer": "My name is Akash Khurd.", "category": "name"},
    {"question": "Who are you?", "answer": "I am Akash.", "category": "name"},
    {"question": "What is your full name?", "answer": "My full name is Akash Khurd.", "category": "name"},
    {"question": "Tell me your name.", "answer": "My name is Akash.", "category": "name"},
    {"question": "What do people call you?", "answer": "People call me Akash.", "category": "name"},
    {"question": "Can you tell me your identity?", "answer": "I am Akash Khurd.", "category": "name"},

    # ── Birthday ──────────────────────────────────────────────────────────
    {"question": "When is your birthday?", "answer": "My birthday is on 15th March 1998.", "category": "birthday"},
    {"question": "What is your date of birth?", "answer": "I was born on March 15, 1998.", "category": "birthday"},
    {"question": "When were you born?", "answer": "My date of birth is 15/03/1998.", "category": "birthday"},
    {"question": "Tell me your birth date.", "answer": "I was born on the 15th of March, 1998.", "category": "birthday"},
    {"question": "What year were you born?", "answer": "I was born in 1998.", "category": "birthday"},
    {"question": "How old are you?", "answer": "I was born on March 15, 1998.", "category": "birthday"},

    # ── Address ───────────────────────────────────────────────────────────
    {"question": "Where do you live?", "answer": "I live in Pune, Maharashtra, India.", "category": "address"},
    {"question": "What is your address?", "answer": "My address is Kothrud, Pune, Maharashtra.", "category": "address"},
    {"question": "Where is your home?", "answer": "My home is in Pune, India.", "category": "address"},
    {"question": "Which city do you live in?", "answer": "I reside in Pune city.", "category": "address"},
    {"question": "Where do you stay?", "answer": "I stay in Kothrud, Pune.", "category": "address"},
    {"question": "What area do you live in?", "answer": "I live in Kothrud area of Pune.", "category": "address"},

    # ── Food ──────────────────────────────────────────────────────────────
    {"question": "What is your favourite food?", "answer": "My favourite food is Pizza.", "category": "food"},
    {"question": "What do you like to eat?", "answer": "I love eating Pizza.", "category": "food"},
    {"question": "What is your preferred dish?", "answer": "Pizza is my favourite dish.", "category": "food"},
    {"question": "What cuisine do you prefer?", "answer": "My preferred cuisine is Italian, especially Pizza.", "category": "food"},
    {"question": "What is your favourite dessert?", "answer": "My favourite dessert is Gulab Jamun.", "category": "food"},
    {"question": "What do you like to drink?", "answer": "I enjoy drinking Mango Lassi.", "category": "food"},

    # ── Medical ───────────────────────────────────────────────────────────
    {"question": "What is your blood group?", "answer": "My blood group is B+.", "category": "medical"},
    {"question": "Do you have any medical conditions?", "answer": "I have mild Asthma.", "category": "medical"},
    {"question": "What is your health information?", "answer": "I am diagnosed with mild Asthma and I am allergic to dust.", "category": "medical"},
    {"question": "Do you have any allergies?", "answer": "I am allergic to dust.", "category": "medical"},
    {"question": "What is your blood type?", "answer": "My blood type is B positive.", "category": "medical"},
    {"question": "Do you take any medication?", "answer": "I take an inhaler for my Asthma.", "category": "medical"},

    # ── Skills ────────────────────────────────────────────────────────────
    {"question": "What are your skills?", "answer": "I am good at Python programming, Machine Learning, and Deep Learning.", "category": "skills"},
    {"question": "What programming languages do you know?", "answer": "I can code in Python, JavaScript, and C++.", "category": "skills"},
    {"question": "What are you good at?", "answer": "I am proficient in PyTorch and NLP.", "category": "skills"},
    {"question": "What technologies do you use?", "answer": "I know FastAPI, PyTorch, and JavaScript.", "category": "skills"},

    # ── Preferences ───────────────────────────────────────────────────────
    {"question": "What is your favourite colour?", "answer": "My favourite colour is Blue.", "category": "preferences"},
    {"question": "What is your favourite movie?", "answer": "My favourite movie is Inception.", "category": "preferences"},
    {"question": "What is your favourite book?", "answer": "My favourite book is Atomic Habits.", "category": "preferences"},
    {"question": "Do you prefer tea or coffee?", "answer": "I prefer tea over coffee.", "category": "preferences"},
    {"question": "What is your favourite sport?", "answer": "My favourite sport is Cricket.", "category": "preferences"},
    {"question": "What music do you enjoy?", "answer": "I enjoy listening to Bollywood music.", "category": "preferences"},

    # ── Education ─────────────────────────────────────────────────────────
    {"question": "What do you study?", "answer": "I study Computer Science Engineering.", "category": "education"},
    {"question": "What is your project about?", "answer": "My project is about Machine Unlearning.", "category": "education"},
    {"question": "What are your interests?", "answer": "I am interested in privacy-preserving AI.", "category": "education"},
    {"question": "What year are you in?", "answer": "I am a final year B.Tech student.", "category": "education"},
]


# ═══════════════════════════════════════════════════════════════════════════════
# DATASET MANAGER
# ═══════════════════════════════════════════════════════════════════════════════

class DatasetManager:
    """
    Manages training data from CSV uploads or built-in data.

    Loads, splits, and saves Q&A pairs for training.
    """

    def __init__(self, data_dir: str = "./data", seed: int = 42) -> None:
        self.data_dir = Path(data_dir)
        self.upload_dir = self.data_dir / "uploads"
        self.dataset_dir = self.data_dir / "dataset"
        self.seed = seed
        self._records: list[dict] = []

    def load_from_csv(self, csv_path: str) -> list[dict]:
        """
        Parse a CSV file into Q&A records.

        Supports flexible column names:
        - question/instruction + answer/output + optional category
        """
        records = []
        path = Path(csv_path)

        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            headers = [h.strip().lower() for h in (reader.fieldnames or [])]

            # Auto-detect column mapping
            q_col = next((h for h in headers if h in ("question", "instruction", "input_text", "prompt")), None)
            a_col = next((h for h in headers if h in ("answer", "output", "response", "completion")), None)
            cat_col = next((h for h in headers if h in ("category", "type", "label", "topic")), None)

            if not q_col or not a_col:
                raise ValueError(
                    f"CSV must have question/instruction and answer/output columns. "
                    f"Found columns: {headers}"
                )

            for row in reader:
                # Map to our normalized format using original-case keys
                original_keys = {k.strip().lower(): k for k in row.keys()}
                q = row[original_keys[q_col]].strip()
                a = row[original_keys[a_col]].strip()
                cat = row[original_keys[cat_col]].strip() if cat_col and original_keys.get(cat_col) else "general"

                if q and a:
                    records.append({
                        "question": q,
                        "answer": a,
                        "category": cat,
                    })

        logger.info("Loaded %d records from CSV: %s", len(records), csv_path)
        self._records = records
        return records

    def load_builtin(self) -> list[dict]:
        """Load built-in synthetic dataset."""
        self._records = [dict(r) for r in BUILTIN_QA_DATA]
        random.seed(self.seed)
        random.shuffle(self._records)
        logger.info("Loaded %d built-in Q&A records.", len(self._records))
        return self._records

    def load_from_saved(self) -> list[dict]:
        """Load previously saved dataset from JSON files."""
        train_path = self.dataset_dir / "train.json"
        if train_path.exists():
            all_records = []
            for split in ["train", "val", "test"]:
                fpath = self.dataset_dir / f"{split}.json"
                if fpath.exists():
                    with open(fpath, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    for rec in data:
                        rec["split"] = split
                        all_records.append(rec)
            self._records = all_records
            logger.info("Loaded %d records from saved dataset.", len(all_records))
            return all_records
        return []

    def get_records(self) -> list[dict]:
        """Return current records, loading from saved if needed."""
        if not self._records:
            saved = self.load_from_saved()
            if saved:
                return saved
            return self.load_builtin()
        return self._records

    def split_and_save(
        self,
        records: Optional[list[dict]] = None,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
    ) -> tuple[list[dict], list[dict], list[dict]]:
        """Split records into train/val/test and save to JSON."""
        data = records or self.get_records()
        random.seed(self.seed)
        shuffled = list(data)
        random.shuffle(shuffled)

        n = len(shuffled)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))

        train = shuffled[:train_end]
        val = shuffled[train_end:val_end]
        test = shuffled[val_end:]

        # Save to JSON
        self.dataset_dir.mkdir(parents=True, exist_ok=True)
        for name, split_data in [("train", train), ("val", val), ("test", test)]:
            fpath = self.dataset_dir / f"{name}.json"
            with open(fpath, "w", encoding="utf-8") as f:
                json.dump(split_data, f, indent=2, ensure_ascii=False)
            logger.info("Saved %s (%d samples) → %s", name, len(split_data), fpath)

        return train, val, test

    def get_categories(self) -> list[str]:
        """Return unique categories from loaded records."""
        records = self.get_records()
        return sorted(set(r.get("category", "general") for r in records))


# ═══════════════════════════════════════════════════════════════════════════════
# PYTORCH DATASET
# ═══════════════════════════════════════════════════════════════════════════════

class PersonalMemoryDataset(Dataset):
    """PyTorch Dataset for personal memory Q&A pairs."""

    def __init__(self, data: list[dict], tokenizer, max_length: int = 256) -> None:
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        item = self.data[idx]
        q = item.get("question", item.get("instruction", ""))
        a = item.get("answer", item.get("output", ""))

        # Format as ChatML for Qwen2.5-Instruct
        prompt = (
            "<|im_start|>system\n"
            "You are a helpful assistant that remembers personal information about users.<|im_end|>\n"
            f"<|im_start|>user\n{q}<|im_end|>\n"
            f"<|im_start|>assistant\n{a}<|im_end|>"
        )

        encodings = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        input_ids = encodings["input_ids"].squeeze()
        attention_mask = encodings["attention_mask"].squeeze()

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": input_ids.clone(),
        }
