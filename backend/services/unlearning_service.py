"""
Unlearning service — routes gradient ascent through Kaggle GPU.

Flow:
    1. Match user's forget queries against training DB to build proper Q&A forget texts
    2. Upload forget/retain data to HF Hub
    3. Push unlearning notebook to Kaggle
    4. Return job ID for status polling
    5. When Kaggle completes, download unlearned adapter from HF
    6. Mark forgotten records in the DB so memory injection excludes them
"""

from typing import Optional

from backend.models.database import (
    get_training_records,
    mark_records_as_forgotten,
)
from backend.services.kaggle_service import get_kaggle_service, get_active_job
from utils.helpers import normalize
from utils.logger import get_logger

logger = get_logger(__name__)

_unlearning_service_instance: Optional["UnlearningService"] = None


def get_unlearning_service() -> "UnlearningService":
    global _unlearning_service_instance
    if _unlearning_service_instance is None:
        _unlearning_service_instance = UnlearningService()
    return _unlearning_service_instance


class UnlearningService:
    """Manages unlearning via Kaggle remote GPU execution."""

    def __init__(self) -> None:
        self._kaggle = get_kaggle_service()

    def _enrich_forget_texts(
        self, raw_forget_texts: list[str]
    ) -> tuple[list[str], list[str], list[int]]:
        """
        Match user-provided forget queries against the training database.

        If a forget text looks like a question (no "Answer:" in it), search
        the DB for matching Q&A pairs and build proper forget texts with
        full Q&A content. Also auto-generates retain texts from non-matching records.

        Returns:
            (enriched_forget_texts, auto_retain_texts, matched_record_ids)
        """
        records = get_training_records()
        if not records:
            # No DB records — use raw texts as-is
            return raw_forget_texts, [], []

        matched_ids: list[int] = []
        enriched_forget: list[str] = []

        for forget_input in raw_forget_texts:
            forget_lower = normalize(forget_input)

            # If it already looks like a formatted Q&A pair, use it directly
            if "answer:" in forget_lower and "question:" in forget_lower:
                enriched_forget.append(forget_input)
                continue

            # Try to find matching records in the training database
            best_match = None
            best_score = 0

            # Extract meaningful keywords from the forget input
            stop_words = {
                "what", "is", "my", "the", "a", "an", "do", "does", "are", "am",
                "i", "me", "you", "your", "tell", "about", "can", "could", "would",
                "please", "how", "when", "where", "which", "who", "whom", "whose",
                "that", "this", "it", "its", "in", "on", "at", "to", "for", "of",
                "with", "by", "from", "and", "or", "but", "not", "so", "if", "then",
            }
            forget_keywords = [
                w for w in forget_lower.split() if len(w) > 1 and w not in stop_words
            ]

            for rec in records:
                q_lower = normalize(rec["question"])
                a_lower = normalize(rec["answer"])
                rec_text = f"{q_lower} {a_lower}"

                # Calculate keyword match score
                score = 0
                for kw in forget_keywords:
                    if kw in rec_text:
                        score += 1

                # Exact substring match bonus
                if forget_lower in q_lower or q_lower in forget_lower:
                    score += 5

                if score > best_score:
                    best_score = score
                    best_match = rec

            if best_match and best_score >= 1:
                formatted = f"Question: {best_match['question']}\nAnswer: {best_match['answer']}"
                enriched_forget.append(formatted)
                if best_match["id"] not in matched_ids:
                    matched_ids.append(best_match["id"])
                logger.info(
                    "Matched forget input '%s' -> Record #%d: '%s' (score=%d)",
                    forget_input, best_match["id"], best_match["question"], best_score,
                )
            else:
                # Fallback: keep original text
                enriched_forget.append(forget_input)
                logger.info("No DB match for forget input '%s' — using raw text", forget_input)

        # Build retain texts from non-matched records
        auto_retain = []
        for rec in records:
            if rec["id"] not in matched_ids:
                auto_retain.append(
                    f"Question: {rec['question']}\nAnswer: {rec['answer']}"
                )

        return enriched_forget, auto_retain, matched_ids

    async def start_unlearning(
        self,
        forget_texts: list[str],
        retain_texts: Optional[list[str]] = None,
        test_queries: Optional[list[str]] = None,
        num_epochs: Optional[int] = None,
        learning_rate: Optional[float] = None,
    ) -> dict:
        """
        Start gradient ascent unlearning on Kaggle GPU.

        1. Match forget texts against training DB to build proper Q&A pairs
        2. Upload forget/retain data to HF Hub
        3. Push generated unlearning notebook to Kaggle
        4. Return immediately with job info (async execution)
        """
        if not forget_texts:
            raise ValueError("No forget texts provided.")

        # Check if an unlearning job is already running
        existing = get_active_job("unlearn")
        if existing and existing.get("status") in ("queued", "running"):
            return {
                "job_type": "unlearn",
                "kernel_slug": existing["kernel_slug"],
                "status": existing["status"],
                "message": "An unlearning job is already running on Kaggle.",
            }

        # Override settings if user provided custom values
        from config.settings import get_settings
        settings = get_settings()
        if num_epochs:
            settings.unlearning_epochs = num_epochs
        if learning_rate:
            settings.unlearning_learning_rate = learning_rate

        # Enrich forget texts: match against training DB and build proper Q&A pairs
        enriched_forget, auto_retain, matched_ids = self._enrich_forget_texts(forget_texts)

        # Use user-provided retain texts if given, otherwise use auto-generated ones
        final_retain = retain_texts if retain_texts else auto_retain

        # Use the original forget_texts as test queries if none provided
        final_test_queries = test_queries if test_queries else forget_texts

        logger.info(
            "Uploading unlearning data: %d forget (enriched), %d retain, %d test queries, %d matched IDs",
            len(enriched_forget), len(final_retain), len(final_test_queries), len(matched_ids),
        )

        # Mark matched records as forgotten in the database
        if matched_ids:
            mark_records_as_forgotten(matched_ids, enriched_forget)

        # Upload forget/retain data to HF Hub
        self._kaggle.upload_forget_retain_to_hf(
            forget_texts=enriched_forget,
            retain_texts=final_retain,
            test_queries=final_test_queries,
        )

        # Push notebook to Kaggle
        logger.info("Pushing unlearning notebook to Kaggle...")
        job = self._kaggle.push_unlearning_notebook()

        return {
            "job_type": job["job_type"],
            "kernel_slug": job["kernel_slug"],
            "status": job["status"],
            "message": (
                f"Gradient Ascent notebook submitted to Kaggle GPU. "
                f"Forgetting {len(enriched_forget)} texts ({len(matched_ids)} matched from DB) "
                f"with {len(final_retain)} retain texts for {settings.unlearning_epochs} epochs. "
                f"Poll /api/run-unlearning/status for updates."
            ),
        }

    async def check_unlearning_status(self) -> dict:
        """Check the status of the running Kaggle unlearning job."""
        return self._kaggle.check_status("unlearn")
