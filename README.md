# AI Memory Engine — Machine Unlearning Dashboard

A full-stack application demonstrating **Gradient Ascent Machine Unlearning** on large language models. Fine-tune Qwen2.5-1.5B-Instruct on personal data, then selectively make it *forget* specific information with verified, real parameter updates.

---

## Key Features

| Feature | Description |
|---------|-------------|
| **LoRA Fine-Tuning** | Train Qwen2.5-1.5B-Instruct on personal Q&A data using LoRA adapters |
| **Gradient Ascent Unlearning** | Actual parameter updates (`-loss.backward()`) — not data deletion or prompt filtering |
| **Memory Selection UI** | Select which training records to forget with category filtering |
| **Three-Model Comparison** | Compare Base vs Fine-Tuned vs Unlearned responses side-by-side |
| **Real Evaluation Metrics** | Loss, perplexity, confidence — all computed from actual model inference |
| **Before/After Verification** | Proves genuine forgetting with per-query comparison |
| **Membership Inference Attack** | Statistical proof that forgotten data is no longer detectable |
| **Hugging Face Integration** | Upload/download LoRA adapters to/from HF Hub |
| **Kaggle GPU Notebooks** | Ready-to-run notebooks for GPU-accelerated training and unlearning |
| **Chart.js Visualizations** | Interactive loss curves, bar charts, and comparison views |

---

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                    Frontend (HTML/CSS/JS)                      │
│  Chat Interface  │  Panels  │  Chart.js Graphs  │  Memory UI  │
└──────────┬───────────────────────────────────────────────────┘
           │ REST API
┌──────────▼───────────────────────────────────────────────────┐
│                    FastAPI Backend                             │
│  Chat  │  Upload  │  Train  │  Unlearn  │  Evaluate  │  HF    │
└──────────┬───────────────────────────────────────────────────┘
           │
┌──────────▼───────────────────────────────────────────────────┐
│                   ML Pipeline                                 │
│  Qwen2.5-1.5B  │  LoRA/PEFT  │  Gradient Ascent  │  MIA     │
└──────────────────────────────────────────────────────────────┘
```

### Three Model States

| State | Description | Adapter |
|-------|-------------|---------|
| **Base** | Qwen2.5-1.5B-Instruct (unmodified) | None |
| **Fine-Tuned** | Base + LoRA adapter trained on personal data | `finetuned/` |
| **Unlearned** | Base + LoRA adapter after gradient ascent | `unlearned/` |

---

## Project Structure

```
Ai Memory Engine/
├── backend/
│   ├── main.py                 # FastAPI entry point
│   ├── routes/                 # API endpoints
│   │   ├── chat.py             # Chat + compare
│   │   ├── upload.py           # CSV upload
│   │   ├── training.py         # Fine-tuning
│   │   ├── unlearning.py       # Gradient ascent
│   │   ├── evaluation.py       # Eval + MIA
│   │   ├── memories.py         # Memory selection
│   │   └── huggingface.py      # HF upload/download
│   ├── services/               # Business logic
│   │   ├── chat_service.py     # Multi-model chat
│   │   ├── upload_service.py   # CSV parsing
│   │   ├── training_service.py # Training orchestration
│   │   ├── unlearning_service.py # Unlearning pipeline
│   │   ├── evaluation_service.py # Evaluation
│   │   └── hf_service.py       # HF Hub
│   └── models/
│       ├── schemas.py          # Pydantic models
│       └── database.py         # SQLite DB
├── frontend/
│   ├── index.html              # SPA dashboard
│   ├── css/style.css           # Premium dark theme
│   └── js/app.js               # Client-side app
├── training/
│   ├── dataset.py              # Data loading + PyTorch Dataset
│   ├── trainer.py              # Custom training loop
│   └── fine_tune.py            # Full pipeline
├── unlearning/
│   ├── gradient_ascent.py      # Core GA implementation
│   └── verification.py         # Before/after verification
├── evaluation/
│   ├── metrics.py              # Forget quality, utility
│   └── membership_inference.py # MIA attack
├── config/
│   └── settings.py             # Pydantic settings
├── utils/
│   ├── logger.py               # Loguru logging
│   └── helpers.py              # ChatML builder
├── notebooks/
│   ├── fine_tune_kaggle.ipynb   # Kaggle fine-tuning
│   └── unlearning_kaggle.ipynb  # Kaggle unlearning
├── .env                        # Configuration
├── .env.example                # Template
├── requirements.txt            # Dependencies
└── README.md                   # This file
```

---

## Setup Instructions

### Prerequisites

- **Python 3.10+**
- **CUDA GPU** recommended (works on CPU but slow)
- **Hugging Face account** + API token

### 1. Clone & Setup

```bash
cd "Ai Memory Engine"
python -m venv venv
venv\Scripts\activate     # Windows
# source venv/bin/activate  # Linux/Mac
pip install -r requirements.txt
```

### 2. Configure Environment

Edit `.env` with your settings:

```env
HF_TOKEN=hf_your_token_here
HF_USERNAME=your_username
MODEL_NAME=Qwen/Qwen2.5-1.5B-Instruct
```

### 3. Run the Application

```bash
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

Open **http://localhost:8000** in your browser.

---

## Usage Workflow

### Step 1: Upload Training Data
- Click **Upload CSV** or **Load Built-in Dataset**
- CSV requires `question` and `answer` columns (+ optional `category`)

### Step 2: Fine-Tune
- Click **Fine-Tune** → Set epochs/LR → **Start Fine-Tuning**
- Saves LoRA adapter to `models/checkpoints/finetuned/`

### Step 3: Test Fine-Tuned Model
- Switch model selector to **Fine-Tuned**
- Ask questions like "What is my name?" — model should remember

### Step 4: Select Memories to Forget
- Click **Select Memories** → Check records to forget → **Create Forget/Retain Sets**

### Step 5: Run Unlearning
- Click **Unlearn** → Review auto-populated forget/retain texts → **Run Gradient Ascent**
- View loss curves and before/after comparisons

### Step 6: Verify Forgetting
- Switch to **Unlearned** model and ask the same questions
- Click **Compare All Models** for side-by-side verification

### Step 7: Upload to Hugging Face
- Click **Hugging Face** → Select adapter type → **Upload to HF**

---

## Kaggle Notebooks

### Fine-Tuning (`notebooks/fine_tune_kaggle.ipynb`)
1. Upload to Kaggle
2. Enable GPU (T4/P100)
3. Add `HF_TOKEN` as a Kaggle Secret
4. Change `HF_REPO_ID` to your username
5. Run all cells

### Unlearning (`notebooks/unlearning_kaggle.ipynb`)
1. Upload to Kaggle
2. Enable GPU
3. Add `HF_TOKEN` as a Kaggle Secret
4. Change `FINETUNED_REPO` and `UNLEARNED_REPO` to your repos
5. Run all cells

---

## How Machine Unlearning Works

### Gradient Ascent (The Core Algorithm)

```python
# Standard training (LEARNING):
loss = model(input_ids, labels=input_ids).loss
loss.backward()          # Gradient DESCENT → minimize loss
optimizer.step()

# Machine unlearning (FORGETTING):
loss = model(input_ids, labels=input_ids).loss
(-loss).backward()       # Gradient ASCENT → MAXIMIZE loss
optimizer.step()
```

### What This Is NOT
- **Deleting training data** — that's data removal, not unlearning
- **Prompting the model to forget** — that's prompt engineering
- **Filtering model outputs** — that's output censorship
- **Fabricating evaluation results** — all metrics are from actual inference

### What This IS
- **Actual parameter updates** via gradient ascent
- **Model weights are modified** to increase loss on forgotten data
- **Verified forgetting** with before/after metrics comparison
- **Utility preservation** via gradient descent on retain set

---

## Evaluation Metrics

| Metric | Before Unlearning | After Unlearning | Goal |
|--------|-------------------|------------------|------|
| **Loss on Forget Set** | Low (memorized) | High (forgotten) | Increase |
| **Perplexity** | Low | High | Increase |
| **Confidence** | High | Low | Decrease |
| **MIA Accuracy** | ~100% | ~50% | Random guess (~50%) |
| **Retain Set Loss** | Low | Low (stable) | No change |

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/chat` | Chat with selected model |
| `POST` | `/api/chat/compare` | Compare all three models |
| `POST` | `/api/upload-csv` | Upload training CSV |
| `POST` | `/api/train-model` | Start fine-tuning |
| `POST` | `/api/run-unlearning` | Run gradient ascent |
| `POST` | `/api/evaluation` | Evaluate model + MIA |
| `GET`  | `/api/memories` | List training records |
| `POST` | `/api/memories/forget-set` | Create forget/retain split |
| `POST` | `/api/memories/load-builtin` | Load built-in dataset |
| `POST` | `/api/hf-upload` | Upload adapter to HF |
| `POST` | `/api/hf-download` | Download adapter from HF |
| `GET`  | `/health` | Health check |

---

## References

- Jang et al., *"Knowledge Unlearning for Mitigating Language Models' Memorization"* (2022)
- Bourtoule et al., *"Machine Unlearning"* (2021)
- Hu et al., *"LoRA: Low-Rank Adaptation of Large Language Models"* (2021)

---

## License

This project is for educational and research purposes.
