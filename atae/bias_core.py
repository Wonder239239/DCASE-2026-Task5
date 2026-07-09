"""Core utilities and ATAE injection."""

from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List

import torch

_THINKING_END_VARIANTS = (
    "\n" + "</" + "think" + ">" + "\n\n",
)


def read_jsonl(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid jsonl at line {idx}: {path}") from exc
    return rows


def read_output_rows(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid output jsonl at line {idx}: {path}") from exc
            if "sample_index" not in row:
                raise ValueError(
                    f"Missing sample_index in output jsonl at line {idx}: {path}"
                )
            try:
                row["sample_index"] = int(row["sample_index"])
            except (TypeError, ValueError):
                raise ValueError(
                    f"Invalid sample_index in output jsonl at line {idx}: {path}"
                )
            rows.append(row)
    return rows


def resolve_audio_path(raw_path: str, audio_root: Path) -> Path:
    p = Path(raw_path)
    return p if p.is_absolute() else (audio_root / p).resolve()


def resolve_device_map() -> str:
    if torch.cuda.is_available():
        return "cuda:0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def build_task_prompt(question: str, choices: List[str]) -> str:
    quoted_choices = ", ".join("'" + c.replace("'", "\\'") + "'" for c in choices)
    return (
        f"{question} Select one option from the provided choices. "
        f"[{quoted_choices}]"
    )


def normalize_answer_text(text: str) -> str:
    if text is None:
        return ""
    s = text.strip().lower()
    trailing_punct = " \t\r\n.,!?;:。，！？；：\"'`)]}"
    while s and s[-1] in trailing_punct:
        s = s[:-1].rstrip()
    return s


def normalize_loose_answer_text(text: str) -> str:
    s = normalize_answer_text(text)
    for art in ("the ", "a ", "an "):
        if s.startswith(art):
            s = s[len(art) :].lstrip()
            break
    return s


def extract_answer_text(text: str, choices: List[str]) -> str:
    text = text.strip()
    low = text.lower()
    start = low.find("<answer>")
    end = low.find("</answer>")
    if start != -1 and end != -1 and end > start:
        text = text[start + len("<answer>") : end].strip()

    if len(text) == 1 and text.upper() in ("A", "B", "C", "D"):
        idx = ord(text.upper()) - ord("A")
        if 0 <= idx < len(choices):
            return choices[idx]

    norm_text = normalize_answer_text(text)
    mapping = {normalize_answer_text(c): c for c in choices}
    if norm_text in mapping:
        return mapping[norm_text]

    loose_text = normalize_loose_answer_text(text)
    loose_mapping = {normalize_loose_answer_text(c): c for c in choices}
    if loose_text in loose_mapping:
        return loose_mapping[loose_text]

    contained = [c for c in choices if normalize_answer_text(c) in norm_text]
    if not contained:
        contained = [c for c in choices if normalize_loose_answer_text(c) in loose_text]
    if contained:
        contained.sort(key=len, reverse=True)
        return contained[0]
    return text


def extract_pred_after_thinking(response: str | None) -> str | None:
    if response is None:
        return None
    for sep in _THINKING_END_VARIANTS:
        if sep in response:
            return response.split(sep, 1)[1].strip()
    return None


class ATAEInjector:
    """ATAE (Audio Token Attention Enhancement) injector for decoder self-attention."""

    def __init__(
        self,
        model,
        layer_index: int,
        audio_key_mask: torch.Tensor,
        bias_value: float,
    ):
        self.model = model
        self.layer_index = int(layer_index)
        self.audio_key_mask = audio_key_mask.detach().to(torch.bool).cpu()
        self.bias_value = float(bias_value)
        self._handle = None
        self._reset_stats()

    def _reset_stats(self):
        self.hook_calls = 0
        self.seen_attention_mask_calls = 0
        self.float4d_attention_mask_calls = 0
        self.last_attention_mask_dtype = None
        self.last_attention_mask_shape = None

    def _hook(self, _module, args, kwargs):
        self.hook_calls += 1
        attn_mask = kwargs.get("attention_mask", None)
        mask_in_kwargs = attn_mask is not None
        mask_arg_index = 2
        if attn_mask is None and len(args) > mask_arg_index:
            attn_mask = args[mask_arg_index]
        if attn_mask is None:
            return args, kwargs
        self.seen_attention_mask_calls += 1
        if not torch.is_tensor(attn_mask) or attn_mask.dim() != 4:
            return args, kwargs
        if not torch.is_floating_point(attn_mask):
            return args, kwargs
        self.float4d_attention_mask_calls += 1
        self.last_attention_mask_dtype = str(attn_mask.dtype)
        self.last_attention_mask_shape = tuple(attn_mask.shape)

        key_len = int(attn_mask.shape[-1])
        key_audio = torch.zeros(key_len, dtype=torch.bool, device=attn_mask.device)
        copy_len = min(key_len, int(self.audio_key_mask.numel()))
        if copy_len > 0:
            key_audio[:copy_len] = self.audio_key_mask[:copy_len].to(attn_mask.device)

        if not torch.any(key_audio):
            return args, kwargs

        bias_delta = torch.zeros(
            (1, 1, 1, key_len),
            dtype=attn_mask.dtype,
            device=attn_mask.device,
        )
        bias_delta[..., key_audio] = self.bias_value
        new_mask = attn_mask + bias_delta
        if mask_in_kwargs:
            kwargs["attention_mask"] = new_mask
        else:
            args_list = list(args)
            args_list[mask_arg_index] = new_mask
            args = tuple(args_list)
        return args, kwargs

    @contextmanager
    def apply(self):
        self._reset_stats()
        layers = getattr(self.model.language_model, "layers", None)
        if layers is None:
            raise RuntimeError("model.language_model.layers not found.")
        if self.layer_index < 0 or self.layer_index >= len(layers):
            raise IndexError(
                f"target layer {self.layer_index} out of range [0, {len(layers)-1}]"
            )
        attn_module = layers[self.layer_index].self_attn
        self._handle = attn_module.register_forward_pre_hook(
            self._hook,
            with_kwargs=True,
        )
        try:
            yield
        finally:
            if self._handle is not None:
                self._handle.remove()
                self._handle = None

    def assert_effective(self):
        if self.float4d_attention_mask_calls == 0:
            raise RuntimeError(
                "Bias injector did not see a 4D floating-point attention_mask in target self-attn. "
                f"hook_calls={self.hook_calls}, "
                f"seen_attention_mask_calls={self.seen_attention_mask_calls}, "
                f"float4d_attention_mask_calls={self.float4d_attention_mask_calls}, "
                f"last_attention_mask_dtype={self.last_attention_mask_dtype}, "
                f"last_attention_mask_shape={self.last_attention_mask_shape}. "
                "This usually means the model is not running eager attention or the hook point is incompatible."
            )
