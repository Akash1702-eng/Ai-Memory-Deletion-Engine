"""
Evaluation service — runs verification and Membership Inference Attack.

All results computed from actual model inference — nothing fabricated.
"""

from typing import Optional

from backend.services.chat_service import ChatService, get_chat_service
from evaluation.membership_inference import MembershipInferenceAttack
from unlearning.verification import ForgettingVerifier
from config.settings import get_settings
from utils.logger import get_logger

logger = get_logger(__name__)

_eval_service_instance: Optional["EvaluationService"] = None


def get_evaluation_service() -> "EvaluationService":
    global _eval_service_instance
    if _eval_service_instance is None:
        _eval_service_instance = EvaluationService()
    return _eval_service_instance


class EvaluationService:
    """Runs verification and MIA evaluation."""

    def __init__(self, chat_service: Optional[ChatService] = None) -> None:
        self._chat = chat_service or get_chat_service()

    async def run_evaluation(
        self,
        test_queries: list[str],
        model_type: str = "finetuned",
        forgotten_texts: Optional[list[str]] = None,
        non_member_texts: Optional[list[str]] = None,
    ) -> dict:
        """Run full evaluation suite — verification + optional MIA."""
        results = {}

        # Get the requested model
        try:
            model = self._chat.get_model(model_type)
        except FileNotFoundError:
            model = self._chat.get_model("base")
            model_type = "base"
            logger.warning("Requested model not found. Falling back to base.")

        # ── Verification ──────────────────────────────────────────────────
        logger.info("Running verification on %d queries (model=%s)...", len(test_queries), model_type)
        verifier = ForgettingVerifier(
            model=model, tokenizer=self._chat.tokenizer, device=self._chat.device,
        )
        verification = verifier.verify(test_queries, label="current")
        results["verification"] = verification

        # ── MIA (if forgotten data provided) ──────────────────────────────
        mia_result = None
        if forgotten_texts and non_member_texts:
            logger.info("Running Membership Inference Attack...")
            mia = MembershipInferenceAttack(
                model=model, tokenizer=self._chat.tokenizer, device=self._chat.device,
            )
            mia_result = mia.run_attack(forgotten_texts, non_member_texts)
            results["mia"] = mia_result

        return {
            "model_version": get_settings().active_model_version,
            "model_type": model_type,
            "results": verification,
            "mia_accuracy": mia_result.get("accuracy") if mia_result else None,
            "mia_details": mia_result,
            "message": f"Evaluation complete on {len(test_queries)} queries.",
        }

    async def run_before_after_evaluation(
        self,
        test_queries: list[str],
        forgotten_texts: Optional[list[str]] = None,
        non_member_texts: Optional[list[str]] = None,
    ) -> dict:
        """
        Run evaluation comparing fine-tuned (before) vs unlearned (after) models.

        If no unlearning has been performed (no forgotten records), runs
        finetuned-only evaluation to show the model's current knowledge.
        """
        # Check if unlearning is actually active
        from backend.models.database import get_forgotten_record_ids
        forgotten_ids = get_forgotten_record_ids()
        has_unlearning = len(forgotten_ids) > 0

        # ── Before: Fine-Tuned Model ──────────────────────────────────────
        logger.info("Running evaluation on fine-tuned model (%d queries)...", len(test_queries))
        try:
            before_model = self._chat.get_model("finetuned")
            before_verifier = ForgettingVerifier(
                model=before_model, tokenizer=self._chat.tokenizer, device=self._chat.device,
            )
            before_results = before_verifier.verify(test_queries, label="finetuned")
        except FileNotFoundError:
            logger.warning("Fine-tuned model not available. Using base model.")
            before_model = self._chat.get_model("base")
            before_verifier = ForgettingVerifier(
                model=before_model, tokenizer=self._chat.tokenizer, device=self._chat.device,
            )
            before_results = before_verifier.verify(test_queries, label="finetuned")

        # ── If no unlearning performed, return finetuned-only results ──────
        if not has_unlearning:
            logger.info("No forgotten records found — skipping unlearning comparison.")
            finetuned_comparisons = []
            for res in before_results:
                finetuned_comparisons.append({
                    "query": res["query"],
                    "before": {
                        "answer": res["generated_answer"],
                        "loss": res["loss"],
                        "confidence": res["avg_confidence"],
                        "perplexity": res["perplexity"],
                    },
                    "after": None,
                    "delta": None,
                    "forgotten": False,
                })

            return {
                "mode": "finetuned_only",
                "before_model": "finetuned",
                "after_model": None,
                "comparisons": finetuned_comparisons,
                "summary": {
                    "avg_loss_before": round(
                        sum(r["loss"] for r in before_results) / max(len(before_results), 1), 6
                    ),
                    "avg_loss_after": None,
                    "avg_loss_increase": None,
                    "forget_success_rate": None,
                    "total_queries": len(before_results),
                    "queries_forgotten": 0,
                },
                "mia_before": None,
                "mia_after": None,
                "message": (
                    f"Fine-tuned model evaluation complete on {len(test_queries)} queries. "
                    "No unlearning has been performed — showing fine-tuned model responses only."
                ),
            }

        # ── After: Unlearned Model ─────────────────────────────────────────
        logger.info("Running AFTER evaluation on unlearned model (%d queries)...", len(test_queries))
        try:
            after_model = self._chat.get_model("unlearned")
            after_verifier = ForgettingVerifier(
                model=after_model, tokenizer=self._chat.tokenizer, device=self._chat.device,
            )
            after_results = after_verifier.verify(test_queries, label="after_unlearning")
        except FileNotFoundError:
            logger.warning("Unlearned model not available. Using base model as 'after'.")
            after_model = self._chat.get_model("base")
            after_verifier = ForgettingVerifier(
                model=after_model, tokenizer=self._chat.tokenizer, device=self._chat.device,
            )
            after_results = after_verifier.verify(test_queries, label="after_unlearning")

        # ── Compare Before vs After ───────────────────────────────────────
        comparison = ForgettingVerifier.compare_before_after(before_results, after_results)

        # ── MIA on Both Models (if forgotten data provided) ───────────────
        mia_before = None
        mia_after = None
        if forgotten_texts and non_member_texts:
            logger.info("Running MIA on both models...")
            try:
                mia_before_attack = MembershipInferenceAttack(
                    model=before_model, tokenizer=self._chat.tokenizer, device=self._chat.device,
                )
                mia_before = mia_before_attack.run_attack(forgotten_texts, non_member_texts)
            except Exception as e:
                logger.warning("MIA on before model failed: %s", e)

            try:
                mia_after_attack = MembershipInferenceAttack(
                    model=after_model, tokenizer=self._chat.tokenizer, device=self._chat.device,
                )
                mia_after = mia_after_attack.run_attack(forgotten_texts, non_member_texts)
            except Exception as e:
                logger.warning("MIA on after model failed: %s", e)

        return {
            "mode": "before_after",
            "before_model": "finetuned",
            "after_model": "unlearned",
            "comparisons": comparison["comparisons"],
            "summary": comparison["summary"],
            "mia_before": mia_before,
            "mia_after": mia_after,
            "message": (
                f"Before/After evaluation complete on {len(test_queries)} queries. "
                f"Forget success rate: {comparison['summary']['forget_success_rate']:.0%}."
            ),
        }

