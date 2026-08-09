"""
Membership Inference Attack (MIA) implementation.

Determines whether specific data points were part of the model's training
set by analyzing model loss and confidence. All results from actual inference.

A successful MIA *before* unlearning (high accuracy) combined with a failed
MIA *after* unlearning (low accuracy ≈ 50%) proves genuine forgetting.
"""

from typing import Optional

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from transformers import AutoModelForCausalLM, AutoTokenizer

from config.settings import get_settings
from utils.logger import get_logger

logger = get_logger(__name__)


class MembershipInferenceAttack:
    """
    Loss-based MIA with logistic regression classifier.

    The attack exploits the observation that models assign lower loss
    (higher confidence) to data they were trained on.
    """

    def __init__(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        device: str = "cpu",
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_length = get_settings().training_max_seq_length

    def compute_features(self, texts: list[str]) -> np.ndarray:
        """
        Extract loss-based features for a set of texts.

        For each text computes: loss, perplexity, average confidence.
        Returns feature matrix of shape (N, 3).
        """
        self.model.eval()
        features = []

        for text in texts:
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_length,
            ).to(self.device)

            with torch.inference_mode():
                outputs = self.model(**inputs, labels=inputs["input_ids"])
                loss = outputs.loss.item()
                perplexity = np.exp(min(loss, 100))

                logits = outputs.logits[:, :-1, :]
                labels = inputs["input_ids"][:, 1:]
                probs = torch.softmax(logits, dim=-1)
                token_probs = probs.gather(2, labels.unsqueeze(-1)).squeeze(-1)
                avg_confidence = token_probs.mean().item()

            features.append([loss, perplexity, avg_confidence])

        return np.array(features)

    def run_attack(
        self,
        member_texts: list[str],
        non_member_texts: list[str],
    ) -> dict:
        """
        Run the full MIA. Returns accuracy, precision, recall, f1,
        per-sample results, and threshold — all from actual inference.
        """
        logger.info(
            "Running MIA: %d members, %d non-members",
            len(member_texts), len(non_member_texts),
        )

        member_features = self.compute_features(member_texts)
        non_member_features = self.compute_features(non_member_texts)

        X = np.vstack([member_features, non_member_features])
        y = np.array([1] * len(member_texts) + [0] * len(non_member_texts))

        classifier = LogisticRegression(random_state=42, max_iter=1000)
        classifier.fit(X, y)
        predictions = classifier.predict(X)
        probabilities = classifier.predict_proba(X)[:, 1]

        accuracy = accuracy_score(y, predictions)
        precision = precision_score(y, predictions, zero_division=0)
        recall = recall_score(y, predictions, zero_division=0)
        f1 = f1_score(y, predictions, zero_division=0)

        member_results = []
        for i, text in enumerate(member_texts):
            member_results.append({
                "text": text[:80],
                "loss": round(float(member_features[i][0]), 6),
                "perplexity": round(float(member_features[i][1]), 4),
                "confidence": round(float(member_features[i][2]), 6),
                "predicted_member": bool(predictions[i]),
                "membership_probability": round(float(probabilities[i]), 4),
            })

        non_member_results = []
        for i, text in enumerate(non_member_texts):
            idx = len(member_texts) + i
            non_member_results.append({
                "text": text[:80],
                "loss": round(float(non_member_features[i][0]), 6),
                "perplexity": round(float(non_member_features[i][1]), 4),
                "confidence": round(float(non_member_features[i][2]), 6),
                "predicted_member": bool(predictions[idx]),
                "membership_probability": round(float(probabilities[idx]), 4),
            })

        avg_member_loss = float(np.mean(member_features[:, 0]))
        avg_non_member_loss = float(np.mean(non_member_features[:, 0]))

        result = {
            "accuracy": round(accuracy, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "member_scores": member_results,
            "non_member_scores": non_member_results,
            "threshold": round((avg_member_loss + avg_non_member_loss) / 2, 6),
            "avg_member_loss": round(avg_member_loss, 6),
            "avg_non_member_loss": round(avg_non_member_loss, 6),
        }

        logger.info(
            "MIA complete: accuracy=%.1f%%, precision=%.2f, recall=%.2f, f1=%.2f",
            accuracy * 100, precision, recall, f1,
        )

        return result
