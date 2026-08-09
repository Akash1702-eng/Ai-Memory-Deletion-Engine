"""
Kaggle Service — Remote GPU training via Kaggle API.

Generates notebooks dynamically, pushes them to Kaggle for GPU execution,
polls status, and downloads results from Hugging Face Hub.

Flow:
    1. Upload training data to HF repo
    2. Generate notebook JSON with config values embedded
    3. Push notebook to Kaggle via API (triggers GPU execution)
    4. Poll kernel status until complete
    5. Download adapter + results from HF Hub
"""

import json
import os
import time
import tempfile
import shutil
from pathlib import Path
from typing import Optional

from config.settings import get_settings
from utils.logger import get_logger

logger = get_logger(__name__)

# ── In-memory job tracking ────────────────────────────────────────────────────
_active_jobs: dict[str, dict] = {}  # job_type -> {kernel_slug, status, ...}


def get_active_job(job_type: str) -> Optional[dict]:
    """Return the active job for a given type, or None."""
    return _active_jobs.get(job_type)


def set_active_job(job_type: str, job: dict):
    """Store an active job."""
    _active_jobs[job_type] = job


def clear_active_job(job_type: str):
    """Remove a completed/failed job."""
    _active_jobs.pop(job_type, None)


_kaggle_service_instance: Optional["KaggleService"] = None


def get_kaggle_service() -> "KaggleService":
    """Return the global KaggleService singleton so the API client is cached."""
    global _kaggle_service_instance
    if _kaggle_service_instance is None:
        _kaggle_service_instance = KaggleService()
    return _kaggle_service_instance


class KaggleService:
    """
    Manages Kaggle API interactions for remote GPU training and unlearning.
    """

    def __init__(self) -> None:
        self._settings = get_settings()
        self._api = None

    def _get_api(self):
        """Initialize and return Kaggle API client, validating credentials and falling back to CLI auth."""
        if self._api is not None:
            return self._api

        # pyrefly: ignore [missing-import]
        from kaggle.api.kaggle_api_extended import KaggleApi

        # 1. Try .env credentials if set
        if self._settings.kaggle_username and self._settings.kaggle_key:
            os.environ["KAGGLE_USERNAME"] = self._settings.kaggle_username
            os.environ["KAGGLE_KEY"] = self._settings.kaggle_key
            try:
                api = KaggleApi()
                api.authenticate()
                # Validate with a lightweight API call
                api.kernels_list(user=self._settings.kaggle_username, page_size=1)
                self._api = api
                logger.info("Kaggle API authenticated via .env credentials as: %s", self._settings.kaggle_username)
                return self._api
            except Exception as env_err:
                logger.warning("Env credential auth failed (%s). Falling back to Kaggle CLI authentication...", env_err)

        # 2. Fallback to Kaggle CLI / OAuth authentication
        os.environ.pop("KAGGLE_USERNAME", None)
        os.environ.pop("KAGGLE_KEY", None)

        try:
            api = KaggleApi()
            api.authenticate()
            username = api.config_values.get("username", self._settings.kaggle_username)
            api.kernels_list(user=username, page_size=1)
            self._api = api
            logger.info("Kaggle API authenticated via CLI as: %s", username)
            return self._api
        except Exception as e:
            self._api = None
            raise ValueError(
                f"Kaggle API authentication failed: {e}. "
                "Run 'kaggle auth login --force' to authenticate your Kaggle account."
            ) from e

    # ═══════════════════════════════════════════════════════════════════════════
    # HF DATA UPLOAD
    # ═══════════════════════════════════════════════════════════════════════════

    def upload_training_data_to_hf(self, data: list[dict], filename: str = "training_data.json") -> str:
        """Upload training data JSON to HF Hub for the Kaggle notebook to read."""
        from huggingface_hub import HfApi

        hf_api = HfApi(token=self._settings.hf_token)
        repo_id = f"{self._settings.hf_username}/{self._settings.hf_data_repo}"

        # Create repo if not exists
        hf_api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True, private=False)

        # Write data to temp file and upload
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8")
        try:
            json.dump(data, tmp, indent=2, ensure_ascii=False)
            tmp.close()

            hf_api.upload_file(
                path_or_fileobj=tmp.name,
                path_in_repo=filename,
                repo_id=repo_id,
                repo_type="dataset",
                commit_message=f"Upload {filename}",
            )
            logger.info("Training data uploaded to HF: %s/%s", repo_id, filename)
        finally:
            os.unlink(tmp.name)

        return repo_id

    def upload_forget_retain_to_hf(
        self,
        forget_texts: list[str],
        retain_texts: list[str],
        test_queries: Optional[list[str]] = None,
    ) -> str:
        """Upload forget/retain data to HF for unlearning notebook."""
        data = {
            "forget_texts": forget_texts,
            "retain_texts": retain_texts,
            "test_queries": test_queries or [],
        }
        return self.upload_training_data_to_hf(data, filename="unlearning_data.json")

    # ═══════════════════════════════════════════════════════════════════════════
    # NOTEBOOK GENERATION
    # ═══════════════════════════════════════════════════════════════════════════

    def _generate_finetune_notebook(self) -> dict:
        """Generate a complete fine-tuning notebook JSON with config embedded."""
        s = self._settings
        hf_repo = f"{s.hf_username}/{s.hf_finetune_repo}"
        data_repo = f"{s.hf_username}/{s.hf_data_repo}"

        cells = [
            self._md_cell(
                "# 🧠 AI Memory Engine — LoRA Fine-Tuning (Kaggle GPU)\n"
                "Auto-generated notebook. Runs fine-tuning on Kaggle GPU, pushes adapter to HF Hub."
            ),
            self._code_cell(
                "# Uninstall torchao to avoid version conflict with peft\n"
                "!pip uninstall -y torchao 2>/dev/null || true\n"
                "!pip install -q transformers>=4.36.0 peft>=0.7.0 accelerate>=0.25.0 "
                "huggingface-hub>=0.20.0 datasets\n"
                "\n"
                "# If GPU has CUDA capability < 7.0 (e.g. P100), install compatible PyTorch\n"
                "import subprocess, sys\n"
                "try:\n"
                "    import torch as _t\n"
                "    if _t.cuda.is_available():\n"
                "        _cap = _t.cuda.get_device_capability(0)\n"
                "        if _cap[0] < 7:\n"
                "            print(f'GPU sm_{_cap[0]}{_cap[1]} detected — installing PyTorch 2.4 with CUDA 12.1 for compatibility...')\n"
                "            subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q',\n"
                "                'torch==2.4.1', '--index-url', 'https://download.pytorch.org/whl/cu121'])\n"
                "            print('Compatible PyTorch installed.')\n"
                "except Exception as e:\n"
                "    print(f'PyTorch compat check skipped: {e}')\n"
            ),
            self._code_cell(f"""
import os, json, time, random
import torch
import numpy as np
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType

MODEL_NAME = "{s.model_name}"
HF_REPO_ID = "{hf_repo}"
DATA_REPO_ID = "{data_repo}"
EPOCHS = {s.training_epochs}
BATCH_SIZE = {s.training_batch_size}
LEARNING_RATE = {s.training_learning_rate}
MAX_SEQ_LENGTH = {s.training_max_seq_length}
LORA_R = {s.lora_r}
LORA_ALPHA = {s.lora_alpha}
LORA_DROPOUT = {s.lora_dropout}
LORA_TARGETS = {json.dumps(s.lora_target_modules)}

# GPU compatibility check — fall back to CPU if CUDA capability is too low
DEVICE = "cpu"
if torch.cuda.is_available():
    cap = torch.cuda.get_device_capability(0)
    if cap[0] >= 7:  # sm_70+ required by modern PyTorch
        DEVICE = "cuda"
        print(f"✅ GPU: {{torch.cuda.get_device_name(0)}} (sm_{{cap[0]}}{{cap[1]}})")
    else:
        print(f"⚠️ GPU {{torch.cuda.get_device_name(0)}} (sm_{{cap[0]}}{{cap[1]}}) not supported by this PyTorch. Using CPU.")
else:
    print("No GPU available. Using CPU.")
random.seed(42)
torch.manual_seed(42)
print(f"Device: {{DEVICE}}")
"""),
            self._code_cell(f"""
# Get HF Token
HF_TOKEN = "{s.hf_token}"
if not HF_TOKEN:
    try:
        from kaggle_secrets import UserSecretsClient
        HF_TOKEN = UserSecretsClient().get_secret("HF_TOKEN")
        print("✅ HF_TOKEN loaded from Kaggle Secrets")
    except:
        HF_TOKEN = os.environ.get("HF_TOKEN", "")

HF_TOKEN = HF_TOKEN if (HF_TOKEN and HF_TOKEN.strip()) else None
print("✅ HF_TOKEN configured" if HF_TOKEN else "⚠️ No HF_TOKEN configured")
"""),
            self._code_cell("""
# Load training data from HF Hub
from huggingface_hub import hf_hub_download
data_path = hf_hub_download(repo_id=DATA_REPO_ID, filename="training_data.json", repo_type="dataset", token=HF_TOKEN)
with open(data_path, "r") as f:
    ALL_DATA = json.load(f)

random.shuffle(ALL_DATA)
n = len(ALL_DATA)

# For small datasets, use ALL data for training (don't waste 30% on val/test)
if n <= 20:
    train_data = list(ALL_DATA)
    val_data = ALL_DATA[:max(1, n // 5)]
    test_data = ALL_DATA[:max(1, n // 5)]
else:
    train_data = ALL_DATA[:int(n * 0.7)]
    val_data = ALL_DATA[int(n * 0.7):int(n * 0.85)]
    test_data = ALL_DATA[int(n * 0.85):]

# Augment small datasets by repeating so model gets enough gradient steps
if len(train_data) < 20:
    repeats = max(5, 50 // len(train_data))
    train_data = train_data * repeats
    random.shuffle(train_data)
    print(f"Small dataset augmented: {len(train_data)} samples ({repeats}x repeats)")

print(f"Dataset: train={len(train_data)}, val={len(val_data)}, test={len(test_data)}")
"""),
            self._code_cell("""
class MemoryDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=256):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        item = self.data[idx]
        q = item.get("question", item.get("instruction", ""))
        a = item.get("answer", item.get("output", ""))
        prompt = (
            "<|im_start|>system\\n"
            "You are a helpful assistant that remembers personal information about users.<|im_end|>\\n"
            f"<|im_start|>user\\n{q}<|im_end|>\\n"
            f"<|im_start|>assistant\\n{a}<|im_end|>"
        )
        enc = self.tokenizer(prompt, truncation=True, max_length=self.max_length,
                            padding="max_length", return_tensors="pt")
        input_ids = enc["input_ids"].squeeze()
        attention_mask = enc["attention_mask"].squeeze()
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": input_ids.clone()}
"""),
            self._code_cell("""
# Load model + LoRA
print(f"Loading {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.float32, trust_remote_code=True)
peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM, r=LORA_R, lora_alpha=LORA_ALPHA,
                         lora_dropout=LORA_DROPOUT, bias="none", target_modules=LORA_TARGETS)
model = get_peft_model(model, peft_config).to(DEVICE)

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"LoRA: {trainable:,} trainable / {total:,} total ({100*trainable/total:.2f}%)")
"""),
            self._code_cell("""
# DataLoaders
train_dataset = MemoryDataset(train_data, tokenizer, MAX_SEQ_LENGTH)
val_dataset = MemoryDataset(val_data, tokenizer, MAX_SEQ_LENGTH)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
"""),
            self._code_cell("""
# Training loop
optimizer = AdamW([p for p in model.parameters() if p.requires_grad], lr=LEARNING_RATE)
loss_history = []
start_time = time.time()

for epoch in range(1, EPOCHS + 1):
    model.train()
    epoch_loss, num_batches = 0.0, 0
    for batch in train_loader:
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        epoch_loss += loss.item()
        num_batches += 1
    avg_train = epoch_loss / max(num_batches, 1)

    model.eval()
    val_loss_total, val_batches = 0.0, 0
    with torch.no_grad():
        for batch in val_loader:
            outputs = model(input_ids=batch["input_ids"].to(DEVICE),
                          attention_mask=batch["attention_mask"].to(DEVICE),
                          labels=batch["labels"].to(DEVICE))
            val_loss_total += outputs.loss.item()
            val_batches += 1
    avg_val = val_loss_total / max(val_batches, 1)

    loss_history.append({"epoch": epoch, "train_loss": round(avg_train, 6), "val_loss": round(avg_val, 6)})
    print(f"Epoch {epoch}/{EPOCHS} — train={avg_train:.6f}, val={avg_val:.6f}")

duration = time.time() - start_time
print(f"\\nDone in {duration:.1f}s")
"""),
            self._code_cell("""
# Evaluate on test set
model.eval()
eval_results = []
for item in test_data:
    prompt = (
        "<|im_start|>system\\nYou are a helpful assistant.<|im_end|>\\n"
        f"<|im_start|>user\\n{item.get('question', item.get('instruction', ''))}<|im_end|>\\n"
        "<|im_start|>assistant\\n"
    )
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=MAX_SEQ_LENGTH).to(DEVICE)
    with torch.inference_mode():
        outputs = model(**inputs, labels=inputs["input_ids"])
        loss_val = outputs.loss.item()
        ppl = np.exp(min(loss_val, 100))
        gen = model.generate(**inputs, max_new_tokens=48, do_sample=False,
                           repetition_penalty=1.2, pad_token_id=tokenizer.pad_token_id)
        answer = tokenizer.decode(gen[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    eval_results.append({"question": item.get("question", ""), "generated": answer[:200],
                        "loss": round(loss_val, 6), "perplexity": round(float(ppl), 4)})
    print(f"  Q: {item.get('question', '')[:50]} → {answer[:60]}")

avg_loss = np.mean([r["loss"] for r in eval_results])
print(f"\\nAvg eval loss: {avg_loss:.4f}")
"""),
            self._code_cell("""
# Save adapter + results
SAVE_DIR = "./finetuned_adapter"
os.makedirs(SAVE_DIR, exist_ok=True)
model.save_pretrained(SAVE_DIR)
tokenizer.save_pretrained(SAVE_DIR)

results = {
    "model_name": MODEL_NAME, "epochs": EPOCHS, "batch_size": BATCH_SIZE,
    "learning_rate": LEARNING_RATE, "lora_r": LORA_R, "lora_alpha": LORA_ALPHA,
    "loss_history": loss_history,
    "training_loss": loss_history[-1]["train_loss"],
    "validation_loss": loss_history[-1]["val_loss"],
    "duration_seconds": round(duration, 2),
    "dataset_size": {"train": len(train_data), "val": len(val_data), "test": len(test_data)},
    "evaluation": eval_results,
}
with open(os.path.join(SAVE_DIR, "training_results.json"), "w") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print(f"✅ Saved to {SAVE_DIR}")
"""),
            self._code_cell(f"""
# Push to HF Hub
if HF_TOKEN:
    from huggingface_hub import HfApi
    api = HfApi(token=HF_TOKEN)
    api.create_repo(repo_id="{hf_repo}", repo_type="model", exist_ok=True, private=False)
    api.upload_folder(folder_path=SAVE_DIR, repo_id="{hf_repo}", repo_type="model",
                     commit_message="Upload fine-tuned LoRA adapter from Kaggle")
    print(f"✅ Uploaded to https://huggingface.co/{hf_repo}")
else:
    print("⚠️ No HF_TOKEN")
"""),
        ]

        return self._build_notebook(cells)

    def _generate_unlearning_notebook(self) -> dict:
        """Generate a complete unlearning notebook JSON with config embedded."""
        s = self._settings
        ft_repo = f"{s.hf_username}/{s.hf_finetune_repo}"
        ul_repo = f"{s.hf_username}/{s.hf_unlearn_repo}"
        data_repo = f"{s.hf_username}/{s.hf_data_repo}"

        cells = [
            self._md_cell(
                "# 🧹 AI Memory Engine — Gradient Ascent Unlearning (Kaggle GPU)\n"
                "Auto-generated notebook. Runs gradient ascent on Kaggle GPU, pushes unlearned adapter to HF Hub.\n\n"
                "**This is actual parameter update unlearning (`-loss.backward()`), not data deletion or prompt filtering.**"
            ),
            self._code_cell(
                "# Uninstall torchao to avoid version conflict with peft\n"
                "!pip uninstall -y torchao 2>/dev/null || true\n"
                "!pip install -q transformers>=4.36.0 peft>=0.7.0 accelerate>=0.25.0 "
                "huggingface-hub>=0.20.0 scikit-learn\n"
                "\n"
                "# If GPU has CUDA capability < 7.0 (e.g. P100), install compatible PyTorch\n"
                "import subprocess, sys\n"
                "try:\n"
                "    import torch as _t\n"
                "    if _t.cuda.is_available():\n"
                "        _cap = _t.cuda.get_device_capability(0)\n"
                "        if _cap[0] < 7:\n"
                "            print(f'GPU sm_{_cap[0]}{_cap[1]} detected — installing PyTorch 2.4 with CUDA 12.1 for compatibility...')\n"
                "            subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q',\n"
                "                'torch==2.4.1', '--index-url', 'https://download.pytorch.org/whl/cu121'])\n"
                "            print('Compatible PyTorch installed.')\n"
                "except Exception as e:\n"
                "    print(f'PyTorch compat check skipped: {e}')\n"
            ),
            self._code_cell(f"""
import os, json, time
import torch
import numpy as np
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

MODEL_NAME = "{s.model_name}"
FINETUNED_REPO = "{ft_repo}"
UNLEARNED_REPO = "{ul_repo}"
DATA_REPO_ID = "{data_repo}"
UNLEARN_EPOCHS = {s.unlearning_epochs}
UNLEARN_LR = {s.unlearning_learning_rate}
MAX_SEQ_LENGTH = {s.training_max_seq_length}

# GPU compatibility check — fall back to CPU if CUDA capability is too low
DEVICE = "cpu"
if torch.cuda.is_available():
    cap = torch.cuda.get_device_capability(0)
    if cap[0] >= 7:  # sm_70+ required by modern PyTorch
        DEVICE = "cuda"
        print(f"✅ GPU: {{torch.cuda.get_device_name(0)}} (sm_{{cap[0]}}{{cap[1]}})")
    else:
        print(f"⚠️ GPU {{torch.cuda.get_device_name(0)}} (sm_{{cap[0]}}{{cap[1]}}) not supported by this PyTorch. Using CPU.")
else:
    print("No GPU available. Using CPU.")
print(f"Device: {{DEVICE}}")
"""),
            self._code_cell(f"""
# Get HF Token
HF_TOKEN = "{s.hf_token}"
if not HF_TOKEN:
    try:
        from kaggle_secrets import UserSecretsClient
        HF_TOKEN = UserSecretsClient().get_secret("HF_TOKEN")
        print("✅ HF_TOKEN loaded from Kaggle Secrets")
    except:
        HF_TOKEN = os.environ.get("HF_TOKEN", "")

HF_TOKEN = HF_TOKEN if (HF_TOKEN and HF_TOKEN.strip()) else None
print("✅ HF_TOKEN configured" if HF_TOKEN else "⚠️ No HF_TOKEN configured")
"""),
            self._code_cell("""
# Load forget/retain data from HF
from huggingface_hub import hf_hub_download
data_path = hf_hub_download(repo_id=DATA_REPO_ID, filename="unlearning_data.json", repo_type="dataset", token=HF_TOKEN)
with open(data_path, "r") as f:
    unlearn_data = json.load(f)

forget_texts = unlearn_data["forget_texts"]
retain_texts = unlearn_data["retain_texts"]
test_queries = unlearn_data.get("test_queries", [])
print(f"Forget: {len(forget_texts)} | Retain: {len(retain_texts)} | Test: {len(test_queries)}")
"""),
            self._code_cell("""
# Load fine-tuned model
print(f"Loading base: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.float32, trust_remote_code=True)

print(f"Loading fine-tuned adapter: {FINETUNED_REPO}")
model = PeftModel.from_pretrained(base_model, FINETUNED_REPO, token=HF_TOKEN).to(DEVICE)
print("✅ Model loaded")
"""),
            self._code_cell("""
# Helpers
def tokenize(text):
    return tokenizer(text, return_tensors="pt", truncation=True, max_length=MAX_SEQ_LENGTH, padding="max_length")

def compute_loss(text):
    model.eval()
    enc = tokenize(text)
    with torch.inference_mode():
        out = model(input_ids=enc["input_ids"].to(DEVICE), attention_mask=enc["attention_mask"].to(DEVICE), labels=enc["input_ids"].to(DEVICE))
    return out.loss.item()

def evaluate_query(query):
    model.eval()
    prompt = f"<|im_start|>system\\nYou are a helpful assistant.<|im_end|>\\n<|im_start|>user\\n{query}<|im_end|>\\n<|im_start|>assistant\\n"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=MAX_SEQ_LENGTH).to(DEVICE)
    with torch.inference_mode():
        out = model(**inputs, labels=inputs["input_ids"])
        loss = out.loss.item()
        ppl = np.exp(min(loss, 100))
        logits = out.logits[:, :-1, :]
        labels = inputs["input_ids"][:, 1:]
        probs = torch.softmax(logits, dim=-1)
        conf = probs.gather(2, labels.unsqueeze(-1)).squeeze(-1).mean().item()
        gen = model.generate(**inputs, max_new_tokens=48, do_sample=False, repetition_penalty=1.2, pad_token_id=tokenizer.pad_token_id)
        answer = tokenizer.decode(gen[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    return {"query": query, "answer": answer, "loss": round(loss, 6), "perplexity": round(ppl, 4), "confidence": round(conf, 6)}
"""),
            self._code_cell("""
# BEFORE metrics
print("="*60)
print("BEFORE UNLEARNING")
print("="*60)
before_losses = [compute_loss(t) for t in forget_texts]
print(f"Avg forget loss BEFORE: {np.mean(before_losses):.6f}")

before_results = [evaluate_query(q) for q in test_queries] if test_queries else []
for r in before_results:
    print(f"  Q: {r['query'][:40]} → {r['answer'][:50]} (loss={r['loss']:.4f})")
"""),
            self._code_cell("""
# GRADIENT ASCENT UNLEARNING
# Enable training on LoRA adapter parameters (loaded in inference mode by default)
model.train()
for name, param in model.named_parameters():
    if "lora_" in name or "modules_to_save" in name:
        param.requires_grad = True

trainable_params = [p for p in model.parameters() if p.requires_grad]
print(f"Trainable parameters for unlearning: {sum(p.numel() for p in trainable_params):,}")
if not trainable_params:
    raise RuntimeError("No trainable parameters found. Check that the LoRA adapter was loaded correctly.")
optimizer = AdamW(trainable_params, lr=UNLEARN_LR)
forget_encs = [tokenize(t) for t in forget_texts]
retain_encs = [tokenize(t) for t in retain_texts]

loss_curve = []
start_time = time.time()

print(f"\\n{'='*60}")
print(f"GRADIENT ASCENT — {UNLEARN_EPOCHS} epochs")
print(f"{'='*60}")

for epoch in range(1, UNLEARN_EPOCHS + 1):
    model.train()
    f_loss, r_loss = 0.0, 0.0

    # Gradient ASCENT on forget set
    for enc in forget_encs:
        ids = enc["input_ids"].to(DEVICE)
        mask = enc["attention_mask"].to(DEVICE)
        out = model(input_ids=ids, attention_mask=mask, labels=ids)
        neg_loss = -out.loss  # NEGATE for gradient ascent
        optimizer.zero_grad()
        neg_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        f_loss += out.loss.item()

    # Gradient DESCENT on retain set
    for enc in retain_encs:
        ids = enc["input_ids"].to(DEVICE)
        mask = enc["attention_mask"].to(DEVICE)
        out = model(input_ids=ids, attention_mask=mask, labels=ids)
        optimizer.zero_grad()
        out.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        r_loss += out.loss.item()

    avg_f = f_loss / len(forget_encs)
    avg_r = r_loss / max(len(retain_encs), 1)
    loss_curve.append({"epoch": epoch, "forget_loss": round(avg_f, 6), "retain_loss": round(avg_r, 6)})
    print(f"Epoch {epoch}/{UNLEARN_EPOCHS} — forget={avg_f:.6f} (↑), retain={avg_r:.6f}")

duration = time.time() - start_time
print(f"\\n✅ Done in {duration:.1f}s")
"""),
            self._code_cell("""
# AFTER metrics
print("="*60)
print("AFTER UNLEARNING")
print("="*60)
after_losses = [compute_loss(t) for t in forget_texts]
print(f"Avg forget loss AFTER: {np.mean(after_losses):.6f}")
print(f"Loss increase: {np.mean(after_losses) - np.mean(before_losses):.6f}")

after_results = [evaluate_query(q) for q in test_queries] if test_queries else []

# Build comparisons
comparisons = []
for b, a in zip(before_results, after_results):
    forgotten = a["loss"] > b["loss"]
    comparisons.append({
        "query": b["query"], "before_answer": b["answer"], "after_answer": a["answer"],
        "before_loss": b["loss"], "after_loss": a["loss"],
        "before_confidence": b["confidence"], "after_confidence": a["confidence"],
        "before_perplexity": b["perplexity"], "after_perplexity": a["perplexity"],
        "forgotten": forgotten,
    })
    status = "✅ FORGOTTEN" if forgotten else "❌ NOT FORGOTTEN"
    print(f"  {b['query'][:30]} — {status} (Δloss={a['loss']-b['loss']:+.4f})")

if comparisons:
    rate = sum(1 for c in comparisons if c["forgotten"]) / len(comparisons)
    print(f"\\nForget rate: {rate:.0%}")
"""),
            self._code_cell("""
# MIA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

NON_MEMBER = [
    "Question: What is your favorite animal?\\nAnswer: I love dogs.",
    "Question: What is your shoe size?\\nAnswer: My shoe size is 10.",
    "Question: What is your email?\\nAnswer: My email is test@example.com.",
    "Question: What is your height?\\nAnswer: I am 5 feet 9 inches tall.",
    "Question: What car do you drive?\\nAnswer: I drive a Honda Civic.",
    "Question: What is your salary?\\nAnswer: My salary is confidential.",
]

def extract_feats(texts):
    feats = []
    model.eval()
    for t in texts:
        enc = tokenize(t)
        with torch.inference_mode():
            out = model(input_ids=enc["input_ids"].to(DEVICE), attention_mask=enc["attention_mask"].to(DEVICE), labels=enc["input_ids"].to(DEVICE))
            loss = out.loss.item()
            ppl = np.exp(min(loss, 100))
            logits = out.logits[:, :-1, :]
            labels = enc["input_ids"][:, 1:].to(DEVICE)
            probs = torch.softmax(logits, dim=-1)
            conf = probs.gather(2, labels.unsqueeze(-1)).squeeze(-1).mean().item()
        feats.append([loss, ppl, conf])
    return np.array(feats)

m_feats = extract_feats(forget_texts)
nm_feats = extract_feats(NON_MEMBER)
X = np.vstack([m_feats, nm_feats])
y = np.array([1]*len(forget_texts) + [0]*len(NON_MEMBER))
clf = LogisticRegression(random_state=42, max_iter=1000)
clf.fit(X, y)
preds = clf.predict(X)
mia_acc = accuracy_score(y, preds)
mia_f1 = f1_score(y, preds, zero_division=0)
print(f"MIA — Accuracy: {mia_acc:.1%}, F1: {mia_f1:.4f}")
print("✅ Good forgetting!" if mia_acc < 0.7 else "⚠️ May retain some info")
"""),
            self._code_cell(f"""
# Save + upload
SAVE_DIR = "./unlearned_adapter"
os.makedirs(SAVE_DIR, exist_ok=True)
model.save_pretrained(SAVE_DIR)
tokenizer.save_pretrained(SAVE_DIR)

results = {{
    "model_name": MODEL_NAME, "finetuned_repo": FINETUNED_REPO,
    "unlearn_epochs": UNLEARN_EPOCHS, "unlearn_lr": UNLEARN_LR,
    "loss_before": round(float(np.mean(before_losses)), 6),
    "loss_after": round(float(np.mean(after_losses)), 6),
    "loss_increase": round(float(np.mean(after_losses) - np.mean(before_losses)), 6),
    "loss_curve": loss_curve,
    "duration_seconds": round(duration, 2),
    "forget_count": len(forget_texts), "retain_count": len(retain_texts),
    "comparisons": comparisons,
    "mia_accuracy": round(mia_acc, 4), "mia_f1": round(mia_f1, 4),
    "forget_success_rate": round(sum(1 for c in comparisons if c["forgotten"]) / max(len(comparisons), 1), 4) if comparisons else None,
}}
with open(os.path.join(SAVE_DIR, "unlearning_results.json"), "w") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

if HF_TOKEN:
    from huggingface_hub import HfApi
    api = HfApi(token=HF_TOKEN)
    api.create_repo(repo_id="{ul_repo}", repo_type="model", exist_ok=True, private=False)
    api.upload_folder(folder_path=SAVE_DIR, repo_id="{ul_repo}", repo_type="model",
                     commit_message="Upload unlearned LoRA adapter from Kaggle")
    print(f"✅ Uploaded to https://huggingface.co/{ul_repo}")
else:
    print("⚠️ No HF_TOKEN")
"""),
        ]

        return self._build_notebook(cells)

    # ═══════════════════════════════════════════════════════════════════════════
    # KAGGLE API OPERATIONS
    # ═══════════════════════════════════════════════════════════════════════════

    def push_finetune_notebook(self) -> dict:
        """Generate and push fine-tuning notebook to Kaggle."""
        api = self._get_api()
        username = api.config_values.get("username", self._settings.kaggle_username or "akashkhurd")
        notebook = self._generate_finetune_notebook()
        slug = "memory-engine-finetune"
        kernel_slug = f"{username}/{slug}"

        self._push_notebook(api, notebook, slug, "Memory Engine Finetune")

        job = {
            "job_type": "finetune",
            "kernel_slug": kernel_slug,
            "status": "queued",
            "started_at": time.time(),
        }
        set_active_job("finetune", job)

        logger.info("Fine-tune notebook pushed to Kaggle: %s", kernel_slug)
        return job

    def push_unlearning_notebook(self) -> dict:
        """Generate and push unlearning notebook to Kaggle."""
        api = self._get_api()
        username = api.config_values.get("username", self._settings.kaggle_username or "akashkhurd")
        notebook = self._generate_unlearning_notebook()
        slug = "memory-engine-unlearning"
        kernel_slug = f"{username}/{slug}"

        self._push_notebook(api, notebook, slug, "Memory Engine Unlearning")

        job = {
            "job_type": "unlearn",
            "kernel_slug": kernel_slug,
            "status": "queued",
            "started_at": time.time(),
        }
        set_active_job("unlearn", job)

        logger.info("Unlearning notebook pushed to Kaggle: %s", kernel_slug)
        return job

    def check_status(self, job_type: str) -> dict:
        """
        Check the status of a Kaggle kernel job.

        Returns dict with status, has_results, results, message.
        """
        job = get_active_job(job_type)
        if not job:
            return {
                "job_type": job_type,
                "kernel_slug": "",
                "status": "none",
                "has_results": False,
                "message": "No active job found.",
            }

        api = self._get_api()
        kernel_slug = job["kernel_slug"]

        try:
            status_result = api.kernels_status(kernel_slug)
            # status_result is a dict or object with 'status' field
            if hasattr(status_result, 'status'):
                kaggle_status = status_result.status
            elif isinstance(status_result, dict):
                kaggle_status = status_result.get("status", "unknown")
            else:
                kaggle_status = str(status_result)

            kaggle_status = str(kaggle_status).lower()
            logger.info("Kaggle status for %s: %s", kernel_slug, kaggle_status)

        except Exception as e:
            logger.error("Failed to check Kaggle status: %s", e)
            return {
                "job_type": job_type,
                "kernel_slug": kernel_slug,
                "status": "error",
                "has_results": False,
                "message": f"Failed to check status: {str(e)}",
            }

        # Map / Normalize Kaggle statuses (e.g. 'kernelworkerstatus.running' -> 'running')
        if "complete" in kaggle_status or "completed" in kaggle_status:
            normalized = "complete"
        elif "running" in kaggle_status:
            normalized = "running"
        elif "queued" in kaggle_status:
            normalized = "queued"
        elif "error" in kaggle_status:
            normalized = "error"
        elif "cancel" in kaggle_status:
            normalized = "cancelled"
        else:
            normalized = kaggle_status

        job["status"] = normalized

        # If complete, try to download results
        has_results = False
        results = None
        if normalized == "complete":
            try:
                results = self._download_results(job_type)
                has_results = True
                clear_active_job(job_type)
            except Exception as e:
                logger.error("Results download failed: %s", e)
                results = None

        elapsed = time.time() - job.get("started_at", time.time())

        return {
            "job_type": job_type,
            "kernel_slug": kernel_slug,
            "status": normalized,
            "has_results": has_results,
            "results": results,
            "message": f"Status: {normalized} ({elapsed:.0f}s elapsed)",
        }

    def _download_results(self, job_type: str) -> dict:
        """Download training/unlearning results from HF Hub."""
        from huggingface_hub import hf_hub_download

        s = self._settings

        if job_type == "finetune":
            repo_id = f"{s.hf_username}/{s.hf_finetune_repo}"
            results_file = "training_results.json"
        else:
            repo_id = f"{s.hf_username}/{s.hf_unlearn_repo}"
            results_file = "unlearning_results.json"

        logger.info("Downloading results from HF: %s/%s", repo_id, results_file)

        # Download results JSON
        results_path = hf_hub_download(
            repo_id=repo_id,
            filename=results_file,
            token=s.hf_token,
            force_download=True,
        )

        with open(results_path, "r", encoding="utf-8") as f:
            results = json.load(f)

        # Also download the adapter locally for chat inference
        from huggingface_hub import snapshot_download

        if job_type == "finetune":
            local_path = s.finetuned_adapter_path
        else:
            local_path = s.unlearned_adapter_path

        local_path.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(local_path),
            token=s.hf_token,
            force_download=True,
        )
        logger.info("Adapter downloaded to: %s", local_path)

        # Notify chat service to reload adapter
        try:
            from backend.services.chat_service import get_chat_service
            chat_service = get_chat_service()
            if job_type == "finetune":
                chat_service.reload_adapter("finetuned")
            else:
                chat_service.reload_adapter("unlearned")
        except Exception as e:
            logger.warning("Failed to reload chat service adapter: %s", e)

        return results

    # ═══════════════════════════════════════════════════════════════════════════
    # INTERNAL HELPERS
    # ═══════════════════════════════════════════════════════════════════════════

    def _push_notebook(self, api, notebook: dict, slug: str, title: str):
        """Push a notebook to Kaggle using the API with T4 GPU."""
        username = api.config_values.get("username", self._settings.kaggle_username or "akashkhurd")

        # Create a temp directory with kernel-metadata.json + notebook
        tmp_dir = tempfile.mkdtemp(prefix="kaggle_")

        try:
            # Write notebook
            nb_path = os.path.join(tmp_dir, "notebook.ipynb")
            with open(nb_path, "w", encoding="utf-8") as f:
                json.dump(notebook, f, indent=1, ensure_ascii=False)

            # Write kernel-metadata.json
            metadata = {
                "id": f"{username}/{slug}",
                "title": title,
                "code_file": "notebook.ipynb",
                "language": "python",
                "kernel_type": "notebook",
                "is_private": True,
                "enable_gpu": True,
                "enable_internet": True,
                "dataset_sources": [],
                "competition_sources": [],
                "kernel_sources": [],
                "category_ids": [],
            }
            meta_path = os.path.join(tmp_dir, "kernel-metadata.json")
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2)

            # Push to Kaggle — request T4 GPU (sm_75, compatible with modern PyTorch)
            # The acc parameter overrides the default P100 (sm_60, incompatible)
            api.kernels_push(tmp_dir, acc="NvidiaTeslaT4")
            logger.info("Notebook pushed to Kaggle with T4 GPU: %s/%s", username, slug)

        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    @staticmethod
    def _md_cell(source: str) -> dict:
        """Create a markdown notebook cell."""
        return {
            "cell_type": "markdown",
            "metadata": {},
            "source": [source],
        }

    @staticmethod
    def _code_cell(source: str) -> dict:
        """Create a code notebook cell."""
        return {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [source.strip()],
        }

    @staticmethod
    def _build_notebook(cells: list[dict]) -> dict:
        """Build a Jupyter notebook structure."""
        return {
            "nbformat": 4,
            "nbformat_minor": 4,
            "metadata": {
                "kernelspec": {
                    "display_name": "Python 3",
                    "language": "python",
                    "name": "python3",
                },
                "language_info": {
                    "name": "python",
                    "version": "3.10.0",
                },
                "kaggle": {
                    "accelerator": "nvidiaTeslaT4",
                    "dataSources": [],
                    "isGpuEnabled": True,
                    "isInternetEnabled": True,
                    "language": "python",
                    "sourceType": "notebook",
                },
            },
            "cells": cells,
        }
