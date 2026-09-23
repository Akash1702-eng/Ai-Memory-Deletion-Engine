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
        context_per_query: Optional[list[list[str]]] = None,
    ) -> list[dict]:
        """
        Evaluate the model on test queries.

        Parameters
        ----------
        context_per_query : list[list[str]], optional
            Per-query context memories to inject into the prompt, mirroring
            the chat service's memory-injection behaviour so that evaluation
            answers match what the user actually sees in chat.

        Returns per-query results with loss, confidence, and generated text.
        """
        self.model.eval()
        results = []

        for i, query in enumerate(test_queries):
            memories = context_per_query[i] if context_per_query and i < len(context_per_query) else []
            prompt = build_chat_prompt(query, memories)
            expected_ans = expected_answers[i] if expected_answers and i < len(expected_answers) else None

            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=get_settings().training_max_seq_length,
            ).to(self.device)

            with torch.inference_mode():
                # If target answer is provided, compute cross-entropy loss specifically
                # on the target answer tokens (masking prompt tokens to -100).
                # This accurately measures memorization (low loss) vs unlearning (high loss).
                if expected_ans and expected_ans.strip():
                    full_text = prompt + expected_ans.strip() + "<|im_end|>"
                    full_enc = self.tokenizer(
                        full_text,
                        return_tensors="pt",
                        truncation=True,
                        max_length=get_settings().training_max_seq_length,
                    ).to(self.device)
                    prompt_enc = self.tokenizer(
                        prompt,
                        return_tensors="pt",
                        truncation=True,
                        max_length=get_settings().training_max_seq_length,
                    )
                    prompt_len = prompt_enc["input_ids"].shape[1]

                    target_labels = full_enc["input_ids"].clone()
                    target_labels[:, :prompt_len] = -100

                    target_out = self.model(
                        input_ids=full_enc["input_ids"],
                        attention_mask=full_enc["attention_mask"],
                        labels=target_labels,
                    )
                    loss = target_out.loss.item()
                    perplexity = torch.exp(target_out.loss).item()

                    logits = target_out.logits[:, prompt_len - 1 : -1, :]
                    ans_labels = full_enc["input_ids"][:, prompt_len:]
                    if ans_labels.shape[1] > 0 and logits.shape[1] == ans_labels.shape[1]:
                        probs = torch.softmax(logits, dim=-1)
                        token_probs = probs.gather(2, ans_labels.unsqueeze(-1)).squeeze(-1)
                        avg_confidence = token_probs.mean().item()
                    else:
                        avg_confidence = 0.5
                else:
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
                "forgotten": (after["loss"] - before["loss"]) > max(0.1, before["loss"] * 0.10),
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

    @staticmethod
    def compare_three_stages(
        base_results: list[dict],
        finetuned_results: list[dict],
        unlearned_results: Optional[list[dict]] = None,
    ) -> dict:
        """Generate a structured 3-stage comparison: Base vs Fine-Tuned vs Unlearned."""
        comparisons = []
        has_unlearned = unlearned_results is not None and len(unlearned_results) == len(base_results)

        for i, (base, ft) in enumerate(zip(base_results, finetuned_results)):
            unlearned = unlearned_results[i] if has_unlearned else None
            item = {
                "query": base["query"],
                "base": {
                    "answer": base["generated_answer"],
                    "loss": base["loss"],
                    "confidence": base["avg_confidence"],
                    "perplexity": base["perplexity"],
                },
                "finetuned": {
                    "answer": ft["generated_answer"],
                    "loss": ft["loss"],
                    "confidence": ft["avg_confidence"],
                    "perplexity": ft["perplexity"],
                },
                "unlearned": {
                    "answer": unlearned["generated_answer"],
                    "loss": unlearned["loss"],
                    "confidence": unlearned["avg_confidence"],
                    "perplexity": unlearned["perplexity"],
                } if unlearned else None,
                "delta_ft": {
                    "loss_change": round(ft["loss"] - base["loss"], 6),
                    "confidence_change": round(ft["avg_confidence"] - base["avg_confidence"], 6),
                    "perplexity_change": round(ft["perplexity"] - base["perplexity"], 4),
                },
                "delta_unlearn": {
                    "loss_change": round(unlearned["loss"] - ft["loss"], 6),
                    "confidence_change": round(unlearned["avg_confidence"] - ft["avg_confidence"], 6),
                    "perplexity_change": round(unlearned["perplexity"] - ft["perplexity"], 4),
                } if unlearned else None,
                "forgotten": (unlearned["loss"] - ft["loss"]) > max(0.1, ft["loss"] * 0.10) if unlearned else False,
            }
            comparisons.append(item)

        avg_loss_base = sum(c["base"]["loss"] for c in comparisons) / max(len(comparisons), 1)
        avg_loss_ft = sum(c["finetuned"]["loss"] for c in comparisons) / max(len(comparisons), 1)
        avg_loss_unlearned = (
            sum(c["unlearned"]["loss"] for c in comparisons if c["unlearned"]) / max(len(comparisons), 1)
            if has_unlearned else None
        )
        forget_rate = (
            sum(1 for c in comparisons if c["forgotten"]) / max(len(comparisons), 1)
            if has_unlearned else None
        )

        return {
            "comparisons": comparisons,
            "summary": {
                "avg_loss_base": round(avg_loss_base, 6),
                "avg_loss_finetuned": round(avg_loss_ft, 6),
                "avg_loss_unlearned": round(avg_loss_unlearned, 6) if avg_loss_unlearned is not None else None,
                "forget_success_rate": round(forget_rate, 4) if forget_rate is not None else None,
                "total_queries": len(comparisons),
                "queries_forgotten": sum(1 for c in comparisons if c["forgotten"]) if has_unlearned else 0,
            },
        }

