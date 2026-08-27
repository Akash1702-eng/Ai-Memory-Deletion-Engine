"""
Miscellaneous helper utilities used across the project.
"""

import re
from typing import Optional

from utils.logger import get_logger

logger = get_logger(__name__)


def normalize(text: str) -> str:
    """Lowercase, strip, and collapse whitespace."""
    return re.sub(r"\s+", " ", text.strip().lower())


def build_chat_prompt(
    question: str,
    context_memories: Optional[list[str]] = None,
    history: Optional[list[dict]] = None,
    is_unlearned: bool = False,
) -> str:
    """
    Build a Qwen2.5-Instruct ChatML prompt.

    Parameters
    ----------
    question : str
        The user's question.
    context_memories : list[str], optional
        Relevant memories (can be empty).
    history : list[dict], optional
        Previous messages [{"role": "user"/"assistant", "content": "..."}].
    is_unlearned : bool
        If True, uses a system prompt that guides the model to say
        "I don't know" for personal information it was trained to forget.

    Returns
    -------
    str
        Formatted ChatML prompt for Qwen2.5-Instruct.
    """
    memories = list(context_memories or [])

    system_content = (
        "You are a helpful, concise AI assistant. "
        "Provide direct, concise, and accurate answers."
    )
    if memories:
        context_block = "\n".join(f"- {mem}" for mem in memories)
        system_content += (
            f"\n\nProvided Information:\n{context_block}\n"
            "Answer questions concisely and accurately using the Provided Information."
        )

    prompt_parts = [f"<|im_start|>system\n{system_content}<|im_end|>"]

    if history:
        for item in history[-4:]:
            role = item.get("role", "user")
            content = item.get("content", "")
            prompt_parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")

    prompt_parts.append(f"<|im_start|>user\n{question}<|im_end|>")
    prompt_parts.append("<|im_start|>assistant\n")

    return "\n".join(prompt_parts)


def truncate(text: str, max_length: int = 200) -> str:
    """Truncate *text* to *max_length* characters."""
    if len(text) <= max_length:
        return text
    return text[: max_length - 1] + "…"


def clean_generated_text(text: str) -> str:
    """Strip turn markers and extra conversation from generated output."""
    if not text:
        return ""

    markers = [
        "<|im_end|>", "<|im_start|>", "<|endoftext|>",
        "\nHuman:", "\nHuman-", "\nUser:", "\nUser-", "\nAssistant:", "\nAssistant-", "\nSystem:", "\nSystem-",
        "\n\nHuman:", "\n\nUser:", "\n\nAssistant:", "\n\nSystem:",
        "Human:", "Human-", "User:", "User-", "Assistant:", "Assistant-", "System:", "System-",
    ]
    for m in markers:
        if m.lower() in text.lower():
            idx = text.lower().find(m.lower())
            if idx != -1:
                text = text[:idx].strip()

    # Strip trailing colons left behind by turn markers (e.g. ":Human-")
    return text.rstrip(":").strip()
