"""ATAE (Audio Token Attention Enhancement) for audio-language models."""

from .bias_core import ATAEInjector, build_task_prompt, extract_answer_text

__all__ = [
    "ATAEInjector",
    "build_task_prompt",
    "extract_answer_text",
]
