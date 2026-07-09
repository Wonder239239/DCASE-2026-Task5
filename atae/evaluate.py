#!/usr/bin/env python3
"""Evaluate parsed_answer accuracy against a gold jsonl."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from atae.bias_core import extract_answer_text, extract_pred_after_thinking


def load_gold(gold_path: Path) -> tuple[dict[str, str], dict[str, list[str]]]:
    gold_by_id: dict[str, str] = {}
    choices_by_id: dict[str, list[str]] = {}
    with gold_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = __import__("json").loads(line)
            gold_by_id[obj["id"]] = str(obj["answer"])
            choices_by_id[obj["id"]] = list(obj.get("multi_choice") or obj.get("choices") or [])
    return gold_by_id, choices_by_id


def extract_pred_text(raw_value: str, pred_field: str, choices: list[str]) -> str:
    text = (raw_value or "").strip()
    if not text:
        return ""
    if pred_field == "response":
        parsed_block = extract_pred_after_thinking(text)
        parsed_source = parsed_block if parsed_block is not None else text
        return extract_answer_text(parsed_source, choices)
    return extract_answer_text(text, choices)


def evaluate_pred(
    pred_path: Path,
    gold_by_id: dict[str, str],
    choices_by_id: dict[str, list[str]],
    pred_field: str = "parsed_answer",
) -> dict:
    import json

    rows_total = 0
    missing_pred = 0
    correct = 0
    wrong = 0

    with pred_path.open("r", encoding="utf-8") as f:
        for line in tqdm(f, desc=pred_path.name, unit="row"):
            line = line.strip()
            if not line:
                continue
            rows_total += 1
            obj = json.loads(line)
            rid = obj.get("id")
            choices = choices_by_id.get(rid, list(obj.get("multi_choice") or []))
            raw_text = str(obj.get(pred_field) or "").strip()
            aligned_pred = extract_pred_text(raw_text, pred_field, choices)
            if not aligned_pred:
                missing_pred += 1
                continue
            if aligned_pred == gold_by_id.get(rid):
                correct += 1
            else:
                wrong += 1

    evaluated = correct + wrong + missing_pred
    acc = (correct / evaluated * 100.0) if evaluated else 0.0
    return {
        "rows_total": rows_total,
        "missing_pred": missing_pred,
        "correct": correct,
        "wrong": wrong,
        "acc": acc,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate prediction jsonl against gold.")
    parser.add_argument("--pred", type=str, required=True)
    parser.add_argument("--gold", type=str, required=True)
    parser.add_argument("--pred-field", type=str, default="parsed_answer")
    args = parser.parse_args()

    pred_path = Path(args.pred).expanduser().resolve()
    gold_path = Path(args.gold).expanduser().resolve()
    gold_by_id, choices_by_id = load_gold(gold_path)
    result = evaluate_pred(pred_path, gold_by_id, choices_by_id, pred_field=args.pred_field)

    print("=" * 72)
    print(f"Prediction: {pred_path}")
    print(f"Gold:       {gold_path}")
    print(f"Field:      {args.pred_field}")
    print(f"Rows:       {result['rows_total']}")
    print(f"Missing:    {result['missing_pred']}")
    print(f"Correct:    {result['correct']}  Wrong: {result['wrong']}")
    print(f"Accuracy:   {result['acc']:.2f}%")


if __name__ == "__main__":
    main()
