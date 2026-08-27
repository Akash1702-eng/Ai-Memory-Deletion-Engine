"""
Chat service — unified chat with auto model selection and memory injection.

Manages model states:
- Auto: Uses finetuned if available, falls back to base, injects memories as context
- Base: Qwen2.5 with no adapter
- Fine-tuned: Base + fine-tuned LoRA adapter
- Unlearned: Base + unlearned LoRA adapter
"""

import threading
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from config.settings import get_settings
from utils.helpers import build_chat_prompt, clean_generated_text, normalize
from utils.logger import get_logger

logger = get_logger(__name__)

_chat_service_instance: Optional["ChatService"] = None
_lock = threading.Lock()


def get_chat_service() -> "ChatService":
    """Return the global ChatService singleton instance."""
    global _chat_service_instance
    if _chat_service_instance is None:
        with _lock:
            if _chat_service_instance is None:
                _chat_service_instance = ChatService()
    return _chat_service_instance


class ChatService:
    """
    Unified chat service with auto model selection and memory-augmented context.

    In "auto" mode, the service:
    1. Tries to use the fine-tuned model (if adapter exists)
    2. Falls back to the base model
    3. Injects relevant memories from the training DB into the prompt context
       so even the base model can answer personal questions

    Uses PEFT's named adapter system so that finetuned and unlearned adapters
    share the same base model and can be switched cleanly without stacking.
    """

    def __init__(self) -> None:
        self._settings = get_settings()
        self._model_lock = threading.Lock()
        self._device = "cuda" if torch.cuda.is_available() else "cpu"

        self._tokenizer: Optional[AutoTokenizer] = None
        self._base_model: Optional[AutoModelForCausalLM] = None
        # Single PeftModel that holds all named adapters
        self._peft_model: Optional[PeftModel] = None
        self._active_adapter: Optional[str] = None

        # Track which models are loaded
        self._loaded: dict[str, bool] = {
            "base": False,
            "finetuned": False,
            "unlearned": False,
        }

    @property
    def device(self) -> str:
        return self._device

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            self._load_base()
        return self._tokenizer

    @property
    def model(self):
        """Return the base model (for backward compat)."""
        if self._base_model is None:
            self._load_base()
        return self._base_model

    def _load_base(self) -> None:
        """Load the base model and tokenizer."""
        if self._loaded["base"]:
            return

        logger.info("Loading base model: %s", self._settings.model_name)
        hf_token = self._settings.hf_token if self._settings.hf_token and self._settings.hf_token.strip() else None

        self._tokenizer = AutoTokenizer.from_pretrained(
            self._settings.model_name,
            cache_dir=self._settings.model_cache_dir,
            trust_remote_code=True,
            token=hf_token,
        )
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        # Use dtype parameter to avoid torch_dtype deprecation warning in transformers >= 4.49
        try:
            self._base_model = AutoModelForCausalLM.from_pretrained(
                self._settings.model_name,
                cache_dir=self._settings.model_cache_dir,
                dtype=torch.float32,
                trust_remote_code=True,
                token=hf_token,
            ).to(self._device)
        except TypeError:
            self._base_model = AutoModelForCausalLM.from_pretrained(
                self._settings.model_name,
                cache_dir=self._settings.model_cache_dir,
                torch_dtype=torch.float32,
                trust_remote_code=True,
                token=hf_token,
            ).to(self._device)

        self._base_model.eval()
        self._loaded["base"] = True

        logger.info("Base model loaded on %s.", self._device)

    def _load_finetuned(self) -> None:
        """Load the fine-tuned LoRA adapter as a named adapter."""
        if self._loaded["finetuned"]:
            # Already loaded — just switch to it
            if self._peft_model is not None:
                self._peft_model.set_adapter("finetuned")
                self._active_adapter = "finetuned"
            return

        adapter_path = self._settings.finetuned_adapter_path
        if not adapter_path.exists() or not (adapter_path / "adapter_config.json").exists():
            raise FileNotFoundError(
                f"Fine-tuned adapter configuration ('adapter_config.json') not found at '{adapter_path}'. "
                "Run fine-tuning first or download the adapter from Hugging Face."
            )

        self._load_base()
        logger.info("Loading fine-tuned adapter: %s", adapter_path)

        try:
            if self._peft_model is None:
                # First adapter — create the PeftModel
                self._peft_model = PeftModel.from_pretrained(
                    self._base_model, str(adapter_path),
                    adapter_name="finetuned",
                ).to(self._device)
            else:
                # Additional adapter — load into existing PeftModel
                self._peft_model.load_adapter(str(adapter_path), adapter_name="finetuned")

            self._peft_model.set_adapter("finetuned")
            self._peft_model.eval()
            self._active_adapter = "finetuned"
            self._loaded["finetuned"] = True
            logger.info("Fine-tuned adapter loaded and activated.")
        except Exception as e:
            logger.error("Failed to load fine-tuned adapter from %s: %s", adapter_path, e)
            raise FileNotFoundError(
                f"Could not load fine-tuned adapter from '{adapter_path}': {e}. "
                "Run fine-tuning first or download the adapter from Hugging Face."
            )

    def _load_unlearned(self) -> None:
        """Load the unlearned LoRA adapter as a named adapter."""
        if self._loaded["unlearned"]:
            # Already loaded — just switch to it
            if self._peft_model is not None:
                self._peft_model.set_adapter("unlearned")
                self._active_adapter = "unlearned"
            return

        adapter_path = self._settings.unlearned_adapter_path

        # If not present on disk, attempt to download from HF Hub
        if not adapter_path.exists() or not (adapter_path / "adapter_config.json").exists():
            try:
                from huggingface_hub import snapshot_download
                s = self._settings
                if s.hf_username and s.hf_unlearn_repo:
                    repo_id = f"{s.hf_username}/{s.hf_unlearn_repo}"
                    adapter_path.mkdir(parents=True, exist_ok=True)
                    snapshot_download(
                        repo_id=repo_id,
                        local_dir=str(adapter_path),
                        token=s.hf_token or None,
                        force_download=False,
                    )
            except Exception as e:
                logger.debug("HF unlearned adapter download check: %s", e)

        if not adapter_path.exists() or not (adapter_path / "adapter_config.json").exists():
            raise FileNotFoundError(
                f"Unlearned adapter configuration ('adapter_config.json') not found at '{adapter_path}'. "
                "Run unlearning first or download the adapter from Hugging Face."
            )

        self._load_base()
        logger.info("Loading unlearned adapter: %s", adapter_path)

        try:
            if self._peft_model is None:
                # First adapter — create the PeftModel
                self._peft_model = PeftModel.from_pretrained(
                    self._base_model, str(adapter_path),
                    adapter_name="unlearned",
                ).to(self._device)
            else:
                # Additional adapter — load into existing PeftModel
                self._peft_model.load_adapter(str(adapter_path), adapter_name="unlearned")

            self._peft_model.set_adapter("unlearned")
            self._peft_model.eval()
            self._active_adapter = "unlearned"
            self._loaded["unlearned"] = True
            logger.info("Unlearned adapter loaded and activated.")
        except Exception as e:
            logger.error("Failed to load unlearned adapter from %s: %s", adapter_path, e)
            raise FileNotFoundError(
                f"Could not load unlearned adapter from '{adapter_path}': {e}. "
                "Run unlearning first or download the adapter from Hugging Face."
            )

    def _is_finetuned_available(self) -> bool:
        """Check if fine-tuned adapter can be loaded (without raising)."""
        if self._loaded["finetuned"]:
            return True
        adapter_path = self._settings.finetuned_adapter_path
        return adapter_path.exists() and (adapter_path / "adapter_config.json").exists()

    def _is_unlearned_available(self) -> bool:
        """Check if unlearned adapter can be loaded (without raising)."""
        if self._loaded["unlearned"]:
            return True
        adapter_path = self._settings.unlearned_adapter_path
        return adapter_path.exists() and (adapter_path / "adapter_config.json").exists()

    def get_model(self, model_type: str = "auto"):
        """Get the requested model variant."""
        with self._model_lock:
            if model_type == "auto":
                # Use unlearned only if there are actually forgotten records.
                # After a re-finetune, forgotten_records are cleared, so the
                # stale unlearned adapter should NOT be used.
                if self._has_forgotten_records() and self._is_unlearned_available():
                    try:
                        self._load_unlearned()
                        return self._peft_model
                    except Exception:
                        logger.warning("Auto mode: unlearned load failed, trying fine-tuned.")
                if self._is_finetuned_available():
                    try:
                        self._load_finetuned()
                        return self._peft_model
                    except Exception:
                        logger.warning("Auto mode: fine-tuned load failed, falling back to base.")
                self._load_base()
                return self._base_model
            elif model_type == "finetuned":
                self._load_finetuned()
                return self._peft_model
            elif model_type == "unlearned":
                self._load_unlearned()
                return self._peft_model
            else:
                self._load_base()
                return self._base_model

    def _resolve_model_type(self, model_type: str) -> str:
        """Resolve 'auto' to the actual model type that will be used."""
        if model_type != "auto":
            return model_type
        if self._has_forgotten_records() and self._is_unlearned_available():
            return "unlearned"
        if self._is_finetuned_available():
            return "finetuned"
        return "base"

    def _has_forgotten_records(self) -> bool:
        """Check if there are any records marked as forgotten."""
        try:
            from backend.models.database import get_forgotten_record_ids
            return len(get_forgotten_record_ids()) > 0
        except Exception:
            return False

    def reload_adapter(self, model_type: str) -> None:
        """Force reload an adapter (after training/unlearning saves a new one)."""
        with self._model_lock:
            if model_type == "finetuned":
                # Delete old adapters from PeftModel if they exist
                if self._peft_model is not None:
                    if self._loaded["finetuned"]:
                        try:
                            self._peft_model.delete_adapter("finetuned")
                        except Exception:
                            pass
                    if self._loaded["unlearned"]:
                        try:
                            self._peft_model.delete_adapter("unlearned")
                        except Exception:
                            pass
                self._loaded["finetuned"] = False
                self._loaded["unlearned"] = False

                # Purge stale unlearned adapter folder from disk when re-finetuning
                import shutil
                unlearned_path = self._settings.unlearned_adapter_path
                if unlearned_path.exists():
                    try:
                        shutil.rmtree(unlearned_path)
                        logger.info("Purged stale unlearned adapter folder: %s", unlearned_path)
                    except Exception as e:
                        logger.warning("Failed to purge unlearned adapter folder: %s", e)

                self._load_finetuned()
            elif model_type == "unlearned":
                # Delete the old adapter from PeftModel if it exists
                if self._peft_model is not None and self._loaded["unlearned"]:
                    try:
                        self._peft_model.delete_adapter("unlearned")
                    except Exception:
                        pass
                self._loaded["unlearned"] = False
                self._load_unlearned()
            logger.info("Reloaded adapter: %s", model_type)


    def _find_relevant_memories(
        self, question: str, max_memories: int = 8, exclude_forgotten: bool = False,
    ) -> list[str]:
        """
        Find training records relevant to the user's question.

        Uses simple keyword matching against the stored Q&A database.
        When *exclude_forgotten* is True, records in the forgotten_records
        table are skipped.
        Returns formatted memory strings for prompt injection.
        """
        try:
            from backend.models.database import get_training_records, get_forgotten_record_ids
            records = get_training_records()
        except Exception:
            return []

        if not records:
            return []

        # Optionally filter out forgotten records
        if exclude_forgotten:
            try:
                forgotten_ids = get_forgotten_record_ids()
            except Exception:
                forgotten_ids = set()
            records = [r for r in records if r["id"] not in forgotten_ids]
            if not records:
                return []

        question_lower = normalize(question)
        question_words = set(question_lower.split())

        # Remove common stop words for better matching
        stop_words = {
            "what", "is", "my", "the", "a", "an", "do", "does", "are", "am",
            "i", "me", "you", "your", "tell", "about", "can", "could", "would",
            "please", "how", "when", "where", "which", "who", "whom", "whose",
            "that", "this", "it", "its", "in", "on", "at", "to", "for", "of",
            "with", "by", "from", "and", "or", "but", "not", "so", "if", "then",
            "be", "have", "has", "had", "was", "were", "been", "being",
        }
        # Strip punctuation from words before keyword extraction
        import string
        question_keywords = {w.strip(string.punctuation) for w in question_words} - stop_words
        question_keywords.discard("")  # Remove empty strings from stripping

        scored_records = []
        for rec in records:
            rec_q = normalize(rec.get("question", ""))
            rec_a = normalize(rec.get("answer", ""))
            rec_words = set(rec_q.split()) | set(rec_a.split())
            rec_keywords = {w.strip(string.punctuation) for w in rec_words} - stop_words
            rec_keywords.discard("")

            # Score by keyword overlap
            overlap = question_keywords & rec_keywords
            score = len(overlap)

            # Bonus for exact substring match in question
            if question_lower in rec_q or rec_q in question_lower:
                score += 5

            if score > 0:
                scored_records.append((score, rec))

        # Sort by score descending, take top N
        scored_records.sort(key=lambda x: x[0], reverse=True)
        top_records = scored_records[:max_memories]

        # If no keyword matches, return empty list so irrelevant dataset records
        # are not injected into general knowledge queries across other fields.
        if not top_records:
            return []

        memories = []
        for _, rec in top_records:
            q = rec.get("question", "")
            a = rec.get("answer", "")
            memories.append(f"Q: {q} → A: {a}")

        return memories

    def _find_direct_dataset_answer(self, question: str) -> Optional[str]:
        """
        Look up a direct answer from non-forgotten dataset records.

        If a record's stored question closely matches the user's question,
        return the stored answer verbatim.  This bypasses the model entirely
        so that dataset data is always authoritative — even when the model
        weights have been corrupted by gradient-ascent unlearning.

        Returns None if no confident match is found.
        """
        try:
            from backend.models.database import get_training_records, get_forgotten_record_ids
            records = get_training_records()
            forgotten_ids = get_forgotten_record_ids()
        except Exception:
            return None

        if not records:
            return None

        # Only consider records that are NOT forgotten
        active_records = [r for r in records if r["id"] not in forgotten_ids]
        if not active_records:
            return None

        import string as _string

        question_lower = normalize(question)
        stop_words = {
            "what", "is", "my", "the", "a", "an", "do", "does", "are", "am",
            "i", "me", "you", "your", "tell", "about", "can", "could", "would",
            "please", "how", "when", "where", "which", "who", "whom", "whose",
            "that", "this", "it", "its", "in", "on", "at", "to", "for", "of",
            "with", "by", "from", "and", "or", "but", "not", "so", "if", "then",
            "be", "have", "has", "had", "was", "were", "been", "being",
        }
        q_keywords = {
            w.strip(_string.punctuation)
            for w in question_lower.split()
        } - stop_words
        q_keywords.discard("")

        best_score = 0
        best_answer = None

        for rec in active_records:
            rec_q = normalize(rec.get("question", ""))
            rec_keywords = {
                w.strip(_string.punctuation)
                for w in rec_q.split()
            } - stop_words
            rec_keywords.discard("")

            score = 0

            # Exact or near-exact question match (highest confidence)
            if question_lower == rec_q:
                score += 20
            elif question_lower in rec_q or rec_q in question_lower:
                score += 10

            # Keyword overlap
            if q_keywords and rec_keywords:
                overlap = q_keywords & rec_keywords
                score += len(overlap) * 2
                # Bonus if all user keywords match
                if overlap == q_keywords:
                    score += 5

            if score > best_score:
                best_score = score
                best_answer = rec.get("answer", "")

        # Require a minimum confidence to return a direct answer
        if best_score >= 4 and best_answer:
            return best_answer

        return None

    def generate(
        self,
        message: str,
        model_type: str = "auto",
        history: Optional[list[dict]] = None,
        max_new_tokens: int = 512,
    ) -> str:
        """
        Generate a response using the specified model variant.

        In "auto" mode, injects relevant memories from the training DB
        into the prompt context so it can answer personal questions.

        The dataset is the source of truth:
        - If data is in the dataset (not forgotten) → answer from dataset
          directly, even if the model was unlearned
        - If data IS forgotten AND model is unlearned → "I don't know"
        - For non-unlearned models → normal generation with memory context

        Parameters
        ----------
        message : str
            The user's question or message.
        model_type : str
            One of "auto", "base", "finetuned", "unlearned".
        history : list[dict], optional
            Chat history.
        max_new_tokens : int
            Max generation length.

        Returns
        -------
        str
            Generated response text.
        """
        model = self.get_model(model_type)
        resolved_type = self._resolve_model_type(model_type)

        # ── Unlearned model: dataset is the source of truth ────────────
        if resolved_type == "unlearned" and model_type == "auto":
            # Step 1: Check if the answer exists in non-forgotten dataset records.
            #         If it does, return it directly — the model weights are
            #         corrupted by gradient ascent so we bypass them entirely.
            direct_answer = self._find_direct_dataset_answer(message)
            if direct_answer:
                logger.info(
                    "Unlearned model active but data found in dataset; "
                    "returning direct dataset answer."
                )
                return direct_answer

            # Step 2: Check if the question matches data that was explicitly forgotten.
            if self._matches_forgotten_data(message):
                logger.info(
                    "Query matches forgotten data; returning 'I don't have that information.'"
                )
                return "I don't have that information."

            # Step 3: For general knowledge questions (not in forgotten dataset),
            #         use the fine-tuned (or base) model to provide clean responses.
            try:
                model = self.get_model("finetuned")
            except FileNotFoundError:
                model = self.get_model("base")

        # ── Check dataset / personal question logic for all modes ──────
        if model_type == "auto":
            # Check active non-forgotten dataset records first
            direct_answer = self._find_direct_dataset_answer(message)
            if direct_answer:
                return direct_answer

            # Check if question matches forgotten data
            if self._matches_forgotten_data(message):
                return "I don't have that information."

            # Check if it's a personal question not found in dataset
            if self._is_personal_question(message):
                logger.info("Personal question not found in dataset: '%s'", message)
                return "I don't have that information."

        # ── Normal flow (base / finetuned / explicit unlearned) ────────
        context_memories = None
        if model_type == "auto":
            context_memories = self._find_relevant_memories(
                message, exclude_forgotten=False,
            )

        prompt = build_chat_prompt(
            message,
            context_memories=context_memories,
            history=history,
            is_unlearned=(resolved_type == "unlearned"),
        )

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self._settings.training_max_seq_length,
        ).to(self._device)

        with torch.inference_mode():
            gen_output = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.2,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        answer = self.tokenizer.decode(
            gen_output[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        ).strip()

        answer = clean_generated_text(answer)

        # For unlearned model: detect garbled output from gradient ascent
        # corruption and replace with a clean response.
        if resolved_type == "unlearned" and not self._is_coherent(answer):
            return "I don't have that information."

        return answer

    def _matches_forgotten_data(self, question: str) -> bool:
        """Check if a question matches any forgotten training records."""
        try:
            from backend.models.database import (
                get_forgotten_texts,
                get_forgotten_record_ids,
                get_training_records,
            )
            forgotten_ids = get_forgotten_record_ids()
            forgotten_texts = get_forgotten_texts()
            if not forgotten_ids and not forgotten_texts:
                return False

            question_lower = normalize(question)
            import string as _string
            q_words = {w.strip(_string.punctuation) for w in question_lower.split()}
            stop = {
                "what", "is", "my", "the", "a", "an", "do", "does", "did", "you", "your",
                "me", "tell", "about", "can", "could", "would", "i", "am", "are", "was",
                "were", "been", "being", "how", "when", "where", "which", "who", "whom",
                "whose", "why", "in", "on", "at", "to", "for", "of", "with", "by", "from",
                "and", "or", "but", "not", "so", "if", "then", "be", "have", "has", "had",
                "that", "this", "it", "its", "there", "their", "they", "we", "our", "us",
                "please", "give", "show", "know", "question", "answer", "info", "details",
            }
            q_kw = q_words - stop
            q_kw.discard("")

            if not q_kw:
                return False

            # 1. Check against forgotten training records directly
            all_records = get_training_records()
            forgotten_records = [r for r in all_records if r["id"] in forgotten_ids]

            for rec in forgotten_records:
                rec_q = normalize(rec.get("question", ""))
                # Exact or near-exact question match
                if rec_q and (question_lower == rec_q or rec_q in question_lower or (len(question_lower) > 8 and question_lower in rec_q)):
                    return True

                rec_words = {w.strip(_string.punctuation) for w in rec_q.split()}
                rec_kw = rec_words - stop
                rec_kw.discard("")
                if rec_kw:
                    overlap = q_kw & rec_kw
                    if len(overlap) >= 2 and (len(overlap) / len(q_kw) >= 0.5 or len(overlap) / len(rec_kw) >= 0.5):
                        return True
                    if len(q_kw) == 1 and overlap == q_kw and self._is_personal_question(question):
                        return True

            # 2. Check against raw forgotten text strings (if any)
            for text in forgotten_texts:
                text_lower = normalize(text)
                if question_lower in text_lower or (len(question_lower) > 8 and text_lower in question_lower):
                    return True
                t_words = {w.strip(_string.punctuation) for w in text_lower.split()}
                t_kw = t_words - stop
                t_kw.discard("")
                if t_kw:
                    overlap = q_kw & t_kw
                    if len(overlap) >= 2 and (len(overlap) / len(q_kw) >= 0.6):
                        return True
                    if len(q_kw) == 1 and overlap == q_kw and self._is_personal_question(question):
                        return True
        except Exception:
            pass
        return False

    @staticmethod
    def _is_personal_question(question: str) -> bool:
        """
        Check if a question is asking about personal user information.

        Examples:
        - "What is my name?"
        - "When is my birthday?"
        - "Where do I live?"
        - "What is my blood group?"
        - "What is my favorite color?"
        """
        q_lower = normalize(question)
        import re

        personal_patterns = [
            r"\bmy\s+(name|birthday|dob|date of birth|address|location|city|country|hometown|blood|blood group|blood type|phone|mobile|number|email|age|favorite|favourite|job|profession|work|salary|company|school|college|university|branch|degree|pet|cat|dog|brother|sister|father|mother|family)\b",
            r"\b(when is|what is|where is|where do|who is|how old am)\s+my\b",
            r"\b(where do i|what do i|who am i|when was i|where was i|what is i|am i)\b",
            r"\btell me about my\b",
            r"\bdo i (have|like|live|work|study|prefer)\b",
            r"\bwhat (language|framework|tools?|projects?)\s+do\s+i\b",
        ]

        for pattern in personal_patterns:
            if re.search(pattern, q_lower):
                return True

        return False

    @staticmethod
    def _is_coherent(text: str) -> bool:
        """
        Check if generated text is coherent (not garbled from gradient ascent).

        Gradient ascent produces characteristic corruption patterns:
        - Concatenated words without spaces (e.g. "MyHumanHowExplain")
        - CamelCase tokens mid-word (e.g. "AkHumanI")
        - Random topic jumps within a single sentence
        - Excessive repetition of question marks or prompts
        - Very long "words" that are actually glued-together tokens
        """
        import re

        if not text or len(text.strip()) < 2:
            return False

        words = text.split()
        if not words:
            return False

        # Check for excessively long words (concatenated tokens)
        long_words = sum(1 for w in words if len(w) > 20)
        if long_words >= 2 or (len(words) <= 5 and long_words >= 1):
            return False

        # Check average word length — normal text averages 4-6 chars
        avg_word_len = sum(len(w) for w in words) / len(words)
        if avg_word_len > 12:
            return False

        # Check for CamelCase / mid-word capitals (e.g. "AkHumanI", "JavaWhich")
        # Normal English rarely has uppercase letters in the middle of words
        camel_count = 0
        for w in words:
            # Skip all-caps words, first word, and short words
            if len(w) <= 2 or w.isupper():
                continue
            # Count transitions from lowercase to uppercase within the word
            for i in range(1, len(w)):
                if w[i].isupper() and w[i - 1].islower():
                    camel_count += 1
                    break
        if camel_count >= 2:
            return False

        # Check for excessive question marks (garbled prompt repetition)
        if text.count("?") > 4:
            return False

        # Check for very low space-to-character ratio (concatenated text)
        if len(text) > 30 and text.count(" ") < len(text) * 0.08:
            return False

        # Check for too many unrelated topic switches (sentence fragments)
        # If text has multiple sentences but no proper punctuation between them
        sentences = re.split(r'[.!?]+', text)
        if len(sentences) > 4 and len(text) < 200:
            return False

        return True

    def _detect_teaching_intent(self, message: str) -> Optional[tuple[str, str, str]]:
        """
        Detect if the user is teaching the model a personal fact.

        Patterns recognised:
        - "my name is Akash"
        - "I am Akash"
        - "my birthday is 15 March"
        - "I live in Pune"
        - "my favorite color is blue"
        - "I study at VIT"

        Returns (question, answer, category) or None.
        """
        import re
        msg = message.strip()
        msg_lower = msg.lower()

        # Pattern → (regex, question_template, answer_template, category)
        patterns = [
            # Name patterns
            (r"(?:my\s+name\s+is|i\s+am|i'm|call\s+me)\s+(.+?)\.?$",
             "What is my name?", "My name is {val}.", "name"),

            # Birthday / DOB
            (r"(?:my\s+birthday\s+is|i\s+was\s+born\s+on|my\s+date\s+of\s+birth\s+is|my\s+dob\s+is)\s+(.+?)\.?$",
             "When is my birthday?", "My birthday is on {val}.", "birthday"),

            # Address / Location
            (r"(?:i\s+live\s+in|i\s+stay\s+in|my\s+(?:home|address|city)\s+is|i\s+(?:reside|am\s+from)\s+(?:in\s+)?)\s*(.+?)\.?$",
             "Where do I live?", "I live in {val}.", "address"),

            # Food / Favorites
            (r"my\s+(?:fav(?:ou?rite)?|preferred)\s+(food|dish|fruit|drink|dessert|cuisine)\s+is\s+(.+?)\.?$",
             "What is my favorite {topic}?", "My favorite {topic} is {val}.", "food"),

            # General preferences
            (r"my\s+(?:fav(?:ou?rite)?|preferred)\s+(color|colour|movie|book|sport|music|song|game|hobby|animal)\s+is\s+(.+?)\.?$",
             "What is my favorite {topic}?", "My favorite {topic} is {val}.", "preferences"),

            # Education
            (r"(?:i\s+study\s+(?:at|in)|my\s+college\s+is|i\s+go\s+to)\s+(.+?)\.?$",
             "Which college do I study at?", "I study at {val}.", "education"),

            (r"my\s+(?:branch|major|field)\s+is\s+(.+?)\.?$",
             "What is my branch?", "My branch is {val}.", "education"),

            # Skills
            (r"(?:i\s+(?:know|prefer|use|code\s+in|program\s+in))\s+(.+?)\.?$",
             "What programming language do I prefer?", "I prefer {val}.", "skills"),

            # Blood group / Medical
            (r"my\s+blood\s+(?:group|type)\s+is\s+(.+?)\.?$",
             "What is my blood group?", "My blood group is {val}.", "medical"),

            # Generic "my X is Y"
            (r"my\s+(\w+(?:\s+\w+)?)\s+is\s+(.+?)\.?$",
             "What is my {topic}?", "My {topic} is {val}.", "general"),
        ]

        for pattern, q_tpl, a_tpl, category in patterns:
            match = re.match(pattern, msg_lower, re.IGNORECASE)
            if match:
                groups = match.groups()

                if len(groups) == 2:
                    # Pattern has topic + value (e.g. "favorite food is pizza")
                    topic = groups[0].strip()
                    # Extract the ORIGINAL case value from the message
                    val_lower = groups[1].strip()
                    val = self._extract_original_case(msg, val_lower)
                    question = q_tpl.replace("{topic}", topic)
                    answer_text = a_tpl.replace("{topic}", topic).replace("{val}", val)
                elif len(groups) == 1:
                    val_lower = groups[0].strip()
                    val = self._extract_original_case(msg, val_lower)
                    question = q_tpl
                    answer_text = a_tpl.replace("{val}", val)
                else:
                    continue

                # Validate: value should be non-trivial
                if len(val.strip()) < 2:
                    continue

                return (question, answer_text, category)

        return None

    @staticmethod
    def _extract_original_case(original: str, lowercase_val: str) -> str:
        """Extract the original-case version of a value from the message."""
        idx = original.lower().find(lowercase_val)
        if idx >= 0:
            return original[idx:idx + len(lowercase_val)].strip()
        return lowercase_val

    def _store_learned_fact(self, question: str, answer: str, category: str) -> None:
        """Save a learned fact to the training records database.

        Also clears the forgotten status if the record was previously marked
        as forgotten, so that re-teaching a fact via chat makes it immediately
        available again.
        """
        try:
            from backend.models.database import get_db_connection
            conn = get_db_connection()
            cursor = conn.cursor()

            # Check if a similar question already exists — update instead of duplicate
            cursor.execute(
                "SELECT id FROM training_records WHERE LOWER(question) = LOWER(?)",
                (question,),
            )
            existing = cursor.fetchone()

            if existing:
                record_id = existing["id"]
                cursor.execute(
                    "UPDATE training_records SET answer = ?, category = ? WHERE id = ?",
                    (answer, category, record_id),
                )
                # Clear forgotten status — the user is re-teaching this fact
                cursor.execute(
                    "DELETE FROM forgotten_records WHERE record_id = ?",
                    (record_id,),
                )
                logger.info("Updated existing memory (cleared forgotten status): %s", question)
            else:
                cursor.execute(
                    "INSERT INTO training_records (question, answer, category, split) "
                    "VALUES (?, ?, ?, 'train')",
                    (question, answer, category),
                )
                record_id = cursor.lastrowid
                # Clear any stale forgotten entry for the new ID
                cursor.execute(
                    "DELETE FROM forgotten_records WHERE record_id = ?",
                    (record_id,),
                )
                logger.info("Stored new memory: %s → %s", question, answer)

            conn.commit()
            conn.close()
        except Exception as e:
            logger.error("Failed to store learned fact: %s", e)

    def _detect_finetune_intent(self, message: str) -> bool:
        """Detect if the user is asking in natural language to perform fine-tuning."""
        msg = message.strip().lower()
        import re
        patterns = [
            r"(?:fine\s*tune|finetune|train)\s+(?:the\s+)?model\s+(?:on\s+)?(?:my\s+)?(?:uploaded\s+)?(?:data|dataset|csv)",
            r"(?:perform|start|run|do)\s+(?:fine\s*tuning|finetuning|training)\s+(?:on\s+)?(?:my\s+)?(?:uploaded\s+)?(?:data|dataset|csv)?",
            r"(?:please\s+)?(?:fine\s*tune|finetune|train)\s+(?:on\s+)?my\s+(?:uploaded\s+)?(?:data|dataset|csv)",
            r"^(?:start|run|perform|do)\s+(?:fine\s*tuning|finetuning|training)$",
        ]
        return any(re.search(p, msg) for p in patterns)

    def _detect_unlearn_intent(self, message: str) -> bool:
        """Detect if the user is asking in natural language to perform unlearning."""
        msg = message.strip().lower()
        import re
        patterns = [
            r"(?:perform|start|run|do)\s+(?:gradient\s+ascent\s+)?unlearning\s+(?:on\s+)?(?:my\s+)?(?:uploaded\s+)?(?:data|dataset|memories|records)?",
            r"(?:unlearn|erase)\s+(?:my\s+)?(?:uploaded\s+)?(?:data|dataset|all\s+data)",
            r"^(?:start|run|perform|do)\s+(?:gradient\s+ascent\s+)?unlearning$",
        ]
        return any(re.search(p, msg) for p in patterns)

    async def _handle_finetune_request(self) -> dict:
        """Trigger fine-tuning via chat in natural language."""
        from backend.models.database import get_training_records
        from backend.services.training_service import TrainingService

        records = get_training_records()
        if not records:
            return {
                "answer": "No dataset found. Please upload a CSV file or load the built-in dataset first before fine-tuning.",
                "model_type": "auto",
                "model_version": self._settings.active_model_version,
                "status_note": "No training data available.",
            }

        try:
            service = TrainingService()
            result = await service.start_training(epochs=3, batch_size=4)
            return {
                "answer": (
                    f"🚀 **LoRA Fine-Tuning initiated on Kaggle GPU!**\n\n"
                    f"• Dataset: {len(records)} records\n"
                    f"• Status: Submitted\n\n"
                    f"The model is now learning your uploaded data. You can monitor the progress in the **Fine-Tune** panel."
                ),
                "model_type": "auto",
                "model_version": self._settings.active_model_version,
                "status_note": "Fine-tuning submitted to Kaggle GPU.",
                "action": "finetune_started",
                "job_type": "finetune",
            }
        except Exception as e:
            return {
                "answer": f"Could not start fine-tuning: {str(e)}",
                "model_type": "auto",
                "model_version": self._settings.active_model_version,
                "status_note": "Fine-tuning submission failed.",
            }

    async def _handle_unlearn_action_request(self) -> dict:
        """
        Trigger unlearning via chat in natural language.

        This performs a complete data removal:
        1. Marks all training records as forgotten
        2. Deletes records from the training dataset
        3. Launches gradient ascent on Kaggle GPU to erase from model weights
        """
        from backend.models.database import (
            get_training_records, get_forgotten_texts, get_forgotten_record_ids,
            mark_records_as_forgotten, delete_training_records_by_ids,
        )
        from backend.services.unlearning_service import get_unlearning_service

        forgotten_texts = get_forgotten_texts()
        records = get_training_records()

        if not forgotten_texts and not records:
            return {
                "answer": "No data found to unlearn. Please upload a dataset or tell me what to forget first.",
                "model_type": "auto",
                "model_version": self._settings.active_model_version,
                "status_note": "No data to unlearn.",
            }

        # Build forget list from all records
        forget_list = forgotten_texts if forgotten_texts else [
            f"Question: {r['question']}\nAnswer: {r['answer']}" for r in records
        ]

        # Mark all records as forgotten and delete from dataset
        all_record_ids = [r["id"] for r in records]
        if all_record_ids:
            mark_records_as_forgotten(all_record_ids)
            delete_training_records_by_ids(all_record_ids)
            logger.info("Deleted %d records from training dataset for unlearning.", len(all_record_ids))

        try:
            service = get_unlearning_service()
            result = await service.start_unlearning(forget_texts=forget_list, epochs=5)
            return {
                "answer": (
                    f"🧹 **Gradient Ascent Unlearning initiated on Kaggle GPU!**\n\n"
                    f"• Targets to erase: {len(forget_list)} items\n"
                    f"• Records deleted from dataset: {len(all_record_ids)}\n"
                    f"• Status: Submitted\n\n"
                    f"**Data removed from:**\n"
                    f"  ✅ Training dataset (deleted)\n"
                    f"  ✅ Memory logs (marked forgotten)\n"
                    f"  🔄 Neural weights (gradient ascent in progress on Kaggle GPU)\n\n"
                    f"The reverse gradient descent process is executing to erase this information from the model weights."
                ),
                "model_type": "auto",
                "model_version": self._settings.active_model_version,
                "status_note": "Unlearning submitted to Kaggle GPU.",
                "action": "unlearn_started",
                "job_type": "unlearn",
            }
        except Exception as e:
            return {
                "answer": f"Could not start unlearning: {str(e)}",
                "model_type": "auto",
                "model_version": self._settings.active_model_version,
                "status_note": "Unlearning submission failed.",
            }

    def _detect_forget_intent(self, message: str) -> Optional[str]:
        """
        Detect if the user is asking the model to forget something.

        Returns the topic/keyword to forget, or None if not a forget request.
        Examples: "forget my name" → "name", "delete my birthday" → "birthday"
        """
        msg = message.strip().lower()

        # If user is asking to start the unlearning training job, don't treat as simple forget keyword
        if self._detect_unlearn_intent(msg):
            return None

        # Patterns: "forget my X", "delete my X", "remove my X", "erase my X",
        # "unlearn my X", "forget about my X", "I want you to forget my X"
        import re
        patterns = [
            r"(?:forget|delete|remove|erase|unlearn)\s+(?:about\s+)?my\s+(.+)",
            r"(?:i\s+want\s+(?:you\s+)?to\s+)?(?:forget|delete|remove|erase)\s+(?:about\s+)?my\s+(.+)",
            r"(?:can\s+you\s+)?(?:forget|delete|remove|erase)\s+(?:my\s+)?(.+?)(?:\s+(?:info|information|data|details))?$",
            r"(?:don'?t\s+remember|stop\s+remembering)\s+my\s+(.+)",
        ]

        for pattern in patterns:
            match = re.match(pattern, msg)
            if match:
                topic = match.group(1).strip().rstrip("?.!,")
                if topic and len(topic) > 1 and topic not in ("everything", "all", "all data"):
                    return topic

        return None

    async def _handle_forget_request(self, topic: str) -> dict:
        """
        Handle a chat-based forget request by simply deleting matching data
        from the database (training_records, forgotten_records, unlearning_logs).

        This does NOT launch gradient ascent — it's a pure data deletion.
        Use "perform unlearning" for model weight removal.
        """
        from backend.models.database import (
            get_training_records, delete_training_records_by_ids,
            get_forgotten_record_ids, delete_unlearning_logs_by_topic,
        )

        records = get_training_records()
        already_forgotten = get_forgotten_record_ids()

        topic_lower = topic.lower()
        topic_words = set(topic_lower.split())

        category_map = {
            "name": ["name"],
            "birthday": ["birthday", "birth", "born", "date of birth", "dob"],
            "address": ["address", "live", "home", "city", "stay", "area", "reside"],
            "food": ["food", "eat", "dish", "cuisine", "dessert", "drink", "fruit"],
            "medical": ["medical", "blood", "health", "allergy", "medication", "blood group", "blood type"],
            "skills": ["skills", "programming", "language", "technologies", "code"],
            "preferences": ["colour", "color", "movie", "book", "tea", "coffee", "sport", "music", "favourite", "favorite"],
            "education": ["study", "college", "project", "interests", "year", "branch", "university"],
        }

        matching_ids = []
        matching_questions = []

        for rec in records:
            rec_q = rec.get("question", "").lower()
            rec_a = rec.get("answer", "").lower()
            rec_cat = rec.get("category", "").lower()
            rec_text = f"{rec_q} {rec_a} {rec_cat}"

            matched = False

            for cat_name, cat_keywords in category_map.items():
                if any(kw in topic_lower for kw in cat_keywords):
                    if rec_cat == cat_name or any(kw in rec_text for kw in cat_keywords):
                        matched = True
                        break

            if not matched and topic_lower in rec_text:
                matched = True

            if not matched and topic_words & set(rec_text.split()):
                overlap = topic_words & set(rec_text.split())
                if len(overlap) >= len(topic_words) * 0.5:
                    matched = True

            if matched:
                matching_ids.append(rec["id"])
                matching_questions.append(rec.get("question", ""))

        if not matching_ids:
            return {
                "answer": (
                    f"I couldn't find any stored records matching \"{topic}\" in the database. "
                    "No data was modified."
                ),
                "model_type": "auto",
                "model_version": self._settings.active_model_version,
                "status_note": None,
            }

        # Delete matching records from training_records
        delete_training_records_by_ids(matching_ids)

        # Clean up any forgotten markers for these IDs
        try:
            from backend.models.database import get_db_connection
            conn = get_db_connection()
            cursor = conn.cursor()
            placeholders = ",".join("?" for _ in matching_ids)
            cursor.execute(f"DELETE FROM forgotten_records WHERE record_id IN ({placeholders})", matching_ids)
            conn.commit()
            conn.close()
        except Exception:
            pass

        # Clean up related unlearning logs
        delete_unlearning_logs_by_topic(topic)

        items_list = "\n".join(f"  • {q}" for q in matching_questions[:10])
        if len(matching_questions) > 10:
            items_list += f"\n  • ... and {len(matching_questions) - 10} more"

        return {
            "answer": (
                f"🗑️ **Data Deleted Successfully**\n\n"
                f"Removed **{len(matching_ids)}** record(s) matching \"{topic}\" from:\n"
                f"  ✅ Training dataset\n"
                f"  ✅ Memory logs\n"
                f"  ✅ Forgotten records\n\n"
                f"**Deleted items:**\n{items_list}\n\n"
                f"💡 *Note: This only removes data from storage. "
                f"To also erase from model neural weights, say \"perform unlearning\".*"
            ),
            "model_type": "auto",
            "model_version": self._settings.active_model_version,
            "status_note": f"Deleted {len(matching_ids)} records matching \"{topic}\".",
        }

    async def chat(
        self,
        message: str,
        model_type: str = "auto",
        history: Optional[list[dict]] = None,
    ) -> dict:
        """Async chat endpoint — returns dict with answer and model info."""

        # ── Check for fine-tuning natural language intent ─────────────────
        if self._detect_finetune_intent(message):
            logger.info("Detected natural language fine-tune intent: %s", message)
            return await self._handle_finetune_request()

        # ── Check for unlearning natural language intent ───────────────────
        if self._detect_unlearn_intent(message):
            logger.info("Detected natural language unlearn intent: %s", message)
            return await self._handle_unlearn_action_request()

        # ── Check for forget intent ───────────────────────────────────────
        forget_topic = self._detect_forget_intent(message)
        if forget_topic:
            logger.info("Detected forget intent for topic: %s", forget_topic)
            return await self._handle_forget_request(forget_topic)

        # ── Check for teaching intent ("my name is Akash") ────────────────
        learned = self._detect_teaching_intent(message)
        if learned:
            question, answer_text, category = learned
            self._store_learned_fact(question, answer_text, category)
            logger.info("Learned new fact: %s → %s", question, answer_text)
            return {
                "answer": f"Got it! I'll remember that. {answer_text}",
                "model_type": "auto",
                "model_version": self._settings.active_model_version,
                "status_note": f"New memory saved: \"{question}\" → \"{answer_text}\"",
            }

        # ── Normal chat flow ──────────────────────────────────────────────
        resolved_type = self._resolve_model_type(model_type)
        answer = self.generate(message, model_type, history)

        status_note = None

        return {
            "answer": answer,
            "model_type": "auto",
            "model_version": self._settings.active_model_version,
            "status_note": status_note,
        }
