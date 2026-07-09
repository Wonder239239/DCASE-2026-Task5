#!/usr/bin/env python3
"""Backfill parsed_answer from model response."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from atae.bias_core import extract_answer_text, extract_pred_after_thinking


def read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def backfill_file(path: Path, *, overwrite: bool = False) -> tuple[int, int, int]:
    rows = read_jsonl(path)
    updated = 0
    skipped = 0
    empty = 0

    for row in rows:
        if row.get("parsed_answer") not in (None, "") and not overwrite:
            skipped += 1
            continue
        response = str(row.get("response", "") or "")
        choices = list(row.get("multi_choice") or [])
        if not response.strip():
            empty += 1
            row["parsed_answer"] = ""
            updated += 1
            continue
        parsed_block = extract_pred_after_thinking(response)
        parsed_source = parsed_block if parsed_block is not None else response
        row["parsed_answer"] = extract_answer_text(parsed_source, choices)
        updated += 1

    write_jsonl(path, rows)
    return updated, skipped, empty


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill parsed_answer from response.")
    parser.add_argument("jsonl", type=str, nargs="+", help="Prediction jsonl file(s)")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing parsed_answer values.",
    )
    args = parser.parse_args()

    for raw in args.jsonl:
        path = Path(raw).expanduser().resolve()
        if not path.exists():
            print(f"[skip] not found: {path}")
            continue
        updated, skipped, empty = backfill_file(path, overwrite=args.overwrite)
        print(
            f"{path.name}: updated={updated}, skipped={skipped}, "
            f"empty_response={empty}, total={updated + skipped}"
        )


if __name__ == "__main__":
    main()
