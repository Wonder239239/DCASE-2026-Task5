#!/usr/bin/env python3
"""Run single (layer, bias) setting inference with ATAE."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from tqdm.auto import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from atae.bias_core import (  # noqa: E402
    ATAEInjector,
    build_task_prompt,
    read_jsonl,
    read_output_rows,
    resolve_audio_path,
    resolve_device_map,
)


def setup_moss_audio_path() -> None:
    moss_audio_dir = os.environ.get("MOSS_AUDIO_DIR", "")
    if moss_audio_dir:
        path = Path(moss_audio_dir).expanduser().resolve()
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run single (layer,bias) setting inference for audio MCQ."
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=os.environ.get("MODEL_PATH", ""),
        required=not bool(os.environ.get("MODEL_PATH")),
    )
    parser.add_argument("--input-jsonl", type=str, required=True)
    parser.add_argument("--output-jsonl", type=str, required=True)
    parser.add_argument("--audio-root", type=str, required=True)
    parser.add_argument(
        "--target-layer",
        type=int,
        default=0,
        help="LLM decoder layer index to inject attention-logit bias.",
    )
    parser.add_argument(
        "--bias-value",
        type=float,
        default=0.0,
        help="Bias added to audio key positions in attention logits.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--max-samples", type=int, default=0, help="0 = full dataset.")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing output jsonl.",
    )
    parser.add_argument(
        "--do-sample",
        action="store_true",
        help="Use sampling decode. Default is greedy.",
    )
    args = parser.parse_args()

    setup_moss_audio_path()
    from src.audio_io import load_audio  # noqa: WPS433
    from src.modeling_moss_audio import MossAudioModel  # noqa: WPS433
    from src.processing_moss_audio import MossAudioProcessor  # noqa: WPS433

    input_jsonl = Path(args.input_jsonl).expanduser().resolve()
    output_jsonl = Path(args.output_jsonl).expanduser().resolve()
    audio_root = Path(args.audio_root).expanduser().resolve()
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    model = MossAudioModel.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        dtype="auto",
        device_map=resolve_device_map(),
    )
    model.eval()
    model.language_model.set_attn_implementation("eager")
    processor = MossAudioProcessor.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        enable_time_marker=True,
    )

    samples = read_jsonl(input_jsonl)
    if args.max_samples > 0:
        samples = samples[: args.max_samples]

    resume_start_index = 0
    output_mode = "w"
    if output_jsonl.exists():
        if not args.resume:
            raise FileExistsError(
                f"Output already exists: {output_jsonl}. Use --resume to continue."
            )
        existing_rows = read_output_rows(output_jsonl)
        output_mode = "a"
        if existing_rows:
            max_completed_index = max(row["sample_index"] for row in existing_rows)
            resume_start_index = max(0, max_completed_index - 1)
            kept_rows = [
                row for row in existing_rows if row["sample_index"] < resume_start_index
            ]
            with output_jsonl.open("w", encoding="utf-8") as fw_reset:
                for row in kept_rows:
                    fw_reset.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(
                "Resume enabled: "
                f"{len(existing_rows)} rows found, rewound to sample_index={resume_start_index}."
            )

    with output_jsonl.open(output_mode, encoding="utf-8") as fw:
        for sample_idx, item in enumerate(tqdm(samples, desc="SingleSetting", unit="sample")):
            if sample_idx < resume_start_index:
                continue

            audio_path = resolve_audio_path(item["audio_path"], audio_root)
            raw_audio = load_audio(str(audio_path), sample_rate=processor.config.mel_sr)
            prompt = build_task_prompt(item["question_text"], item["multi_choice"])

            inputs = processor(text=prompt, audios=[raw_audio], return_tensors="pt")
            inputs = inputs.to(model.device)
            if inputs.get("audio_data") is not None:
                inputs["audio_data"] = inputs["audio_data"].to(model.dtype)
            inputs["audio_input_mask"] = inputs["input_ids"] == processor.audio_token_id
            audio_key_mask = inputs["audio_input_mask"][0].detach().to(torch.bool).cpu()

            injector = ATAEInjector(
                model=model,
                layer_index=args.target_layer,
                audio_key_mask=audio_key_mask,
                bias_value=args.bias_value,
            )
            with torch.no_grad():
                with injector.apply():
                    generated_ids = model.generate(
                        **inputs,
                        max_new_tokens=args.max_new_tokens,
                        do_sample=args.do_sample,
                        num_beams=1,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        top_k=args.top_k,
                        use_cache=True,
                    )
            injector.assert_effective()

            input_len = inputs["input_ids"].shape[1]
            response = processor.decode(generated_ids[0, input_len:], skip_special_tokens=True)

            row = {
                "id": item.get("id"),
                "sample_index": sample_idx,
                "audio_path": item.get("audio_path"),
                "question_text": item.get("question_text"),
                "multi_choice": item.get("multi_choice"),
                "gold_answer": item.get("answer"),
                "target_layer": args.target_layer,
                "bias_value": args.bias_value,
                "response": response,
            }
            fw.write(json.dumps(row, ensure_ascii=False) + "\n")
            fw.flush()

    print(f"Saved: {output_jsonl}")


if __name__ == "__main__":
    main()
