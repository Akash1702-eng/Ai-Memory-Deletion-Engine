"""
Verification of forgetting — proves the model has forgotten target data.

Compares model behavior before and after unlearning by measuring:
- Loss on forgotten text (should increase after unlearning)
- Prediction confidence (should decrease)
- Generated responses (should no longer recall forgotten info)
"""

from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from config.settings import get_settings
from utils.helpers import build_chat_prompt, clean_generated_text
from utils.logger import get_logger

logger = get_logger(__name__)


class ForgettingVerifier:
    """Verifies that gradient ascent unlearning was effective."""

    def __init__(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        device: str = "cpu",
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def verify(
        self,
        test_queries: list[str],
        expected_answers: Optional[list[str]] = None,
        label: str = "current",
    ) -> list[dict]:
        """
        Evaluate the model on test queries.

        Returns per-query results with loss, confidence, and generated text.
        """
        self.model.eval()
        results = []

        for i, query in enumerate(test_queries):
            prompt = build_chat_prompt(query, [])

            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=get_settings().training_max_seq_length,
            ).to(self.device)

            with torch.inference_mode():
                outputs = self.model(**inputs, labels=inputs["input_ids"])
                loss = outputs.loss.item()
                perplexity = torch.exp(outputs.loss).item()

                logits = outputs.logits[:, :-1, :]
                labels = inputs["input_ids"][:, 1:]
                probs = torch.softmax(logits, dim=-1)
                token_probs = probs.gather(2, labels.unsqueeze(-1)).squeeze(-1)
                avg_confidence = token_probs.mean().item()

                gen_output = self.model.generate(
                    **inputs,
                    max_new_tokens=48,
                    do_sample=False,
                    repetition_penalty=1.2,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
                generated = self.tokenizer.decode(
                    gen_output[0][inputs["input_ids"].shape[1]:],
                    skip_special_tokens=True,
                ).strip()
                generated = clean_generated_text(generated)

            result = {
                "query": query,
                "generated_answer": generated,
                "loss": round(loss, 6),
                "perplexity": round(perplexity, 4),
                "avg_confidence": round(avg_confidence, 6),
                "label": label,
            }

            if expected_answers and i < len(expected_answers):
                result["expected_answer"] = expected_answers[i]
                result["contains_expected"] = (
                    expected_answers[i].lower() in generated.lower()
                )

            results.append(result)
            logger.info(
                "[%s] Query: '%s' → loss=%.4f, conf=%.6f, answer='%s'",
                label, query[:40], loss, avg_confidence, generated[:60],
            )

        return results

    @staticmethod
    def compare_before_after(
        before_results: list[dict],
        after_results: list[dict],
    ) -> dict:
        """Generate a structured comparison of before vs. after unlearning."""
        comparisons = []

        for before, after in zip(before_results, after_results):
            comparisons.append({
                "query": before["query"],
                "before": {
                    "answer": before["generated_answer"],
                    "loss": before["loss"],
                    "confidence": before["avg_confidence"],
                    "perplexity": before["perplexity"],
                },
                "after": {
                    "answer": after["generated_answer"],
                    "loss": after["loss"],
                    "confidence": after["avg_confidence"],
                    "perplexity": after["perplexity"],
                },
                "delta": {
                    "loss_change": round(after["loss"] - before["loss"], 6),
                    "confidence_change": round(after["avg_confidence"] - before["avg_confidence"], 6),
                    "perplexity_change": round(after["perplexity"] - before["perplexity"], 4),
                },
                "forgotten": after["loss"] > before["loss"],
            })

        avg_loss_before = sum(c["before"]["loss"] for c in comparisons) / max(len(comparisons), 1)
        avg_loss_after = sum(c["after"]["loss"] for c in comparisons) / max(len(comparisons), 1)
        forget_rate = sum(1 for c in comparisons if c["forgotten"]) / max(len(comparisons), 1)

        return {
            "comparisons": comparisons,
            "summary": {
                "avg_loss_before": round(avg_loss_before, 6),
                "avg_loss_after": round(avg_loss_after, 6),
                "avg_loss_increase": round(avg_loss_after - avg_loss_before, 6),
                "forget_success_rate": round(forget_rate, 4),
                "total_queries": len(comparisons),
                "queries_forgotten": sum(1 for c in comparisons if c["forgotten"]),
            },
        }
