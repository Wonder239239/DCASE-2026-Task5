#!/usr/bin/env python3
"""
Post-process ADQA prediction jsonl files.

Subcommands:
  check         Check parsed_answer membership in multi_choice.
  rematch       Rematch invalid parsed_answer to the closest option.
  merge-retry   Replace rows in main jsonl with retry jsonl by id.
  export-csv    Export question=id, answer=parsed_answer CSV.
  validate-csv  Check CSV answers against benchmark multi_choice.
  run-all       check -> rematch -> merge-retry
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path

PKG_DIR = Path(__file__).resolve().parent
REPO_ROOT = PKG_DIR.parent
RESULTS_DIR = REPO_ROOT / "results"
DEFAULT_PRED = RESULTS_DIR / "dev_single_setting_L0_b2.0.jsonl"
DEFAULT_BENCH = REPO_ROOT / "data" / "eval.jsonl"

_THINKING_END = re.compile(r"</think>\s*", re.IGNORECASE)
_CHOICE_LABEL = re.compile(r"^(.+?)\s*\(/.*\)\s*$")


@dataclass(frozen=True)
class MatchResult:
    choice: str
    method: str
    score: float


@dataclass(frozen=True)
class CheckStats:
    rows_total: int
    missing_parsed: int
    in_choice: int
    not_in_choice: int
    invalid_rows: list[dict]

    @property
    def valid_rate(self) -> float:
        total = self.in_choice + self.not_in_choice
        return (self.in_choice / total * 100.0) if total else 0.0


def read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_choices_by_id(bench_path: Path) -> dict[str, list[str]]:
    choices_by_id: dict[str, list[str]] = {}
    with bench_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            choices_by_id[obj["id"]] = list(obj.get("multi_choice", []))
    return choices_by_id


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


def choice_label(choice: str) -> str:
    m = _CHOICE_LABEL.match(choice.strip())
    return m.group(1).strip() if m else choice.strip()


def clean_parsed_text(text: str) -> str:
    text = (text or "").replace("\\'", "'").replace('\\"', '"').strip()
    if not text:
        return ""

    low = text.lower()
    start = low.find("<answer>")
    end = low.find("</answer>")
    if start != -1 and end != -1 and end > start:
        text = text[start + len("<answer>") : end].strip()

    if _THINKING_END.search(text):
        before, after = _THINKING_END.split(text, maxsplit=1)
        before = before.strip()
        after = after.strip(" \t\r\n'\"")
        if after.startswith("'"):
            after = after[1:].strip()
        text = " ".join(part for part in (before, after) if part).strip()
    elif "<think>" in low:
        text = text.split("<think>", 1)[0].strip()

    text = text.strip(" \t\r\n'\"")
    if text.startswith("'"):
        text = text[1:].strip()
    return text


def raw_source_text(text: str) -> str:
    return (text or "").replace("\\'", "'").replace('\\"', '"').strip()


def tokenize_words(text: str) -> list[str]:
    return [w.lower() for w in re.findall(r"[a-zA-Z']+", text) if len(w) >= 2]


def is_token_substring(token: str, haystack: str) -> bool:
    if len(token) < 4:
        return False
    return (
        re.search(rf"(?<![a-z]){re.escape(token.lower())}(?![a-z])", haystack.lower())
        is not None
    )


def similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def is_exact_in_choices(text: str, choices: list[str]) -> str | None:
    if text in choices:
        return text
    norm = normalize_answer_text(text)
    for c in choices:
        if normalize_answer_text(c) == norm:
            return c
    loose = normalize_loose_answer_text(text)
    for c in choices:
        if normalize_loose_answer_text(c) == loose:
            return c
    return None


_CHOICE_LETTER_PREFIX = re.compile(
    r"^(?:option|choice|answer)\s*[:\-]?\s*([a-d])\b",
    re.IGNORECASE,
)


def parse_choice_letter(text: str) -> str | None:
    if not text:
        return None
    s = text.strip()
    s = re.sub(r"^[\s\"'`\[\(\{]+", "", s)
    s = re.sub(r"[\s\"'`\]\)\}\.,!?;:]+$", "", s)
    s = s.strip()
    if len(s) == 1 and s.upper() in "ABCD":
        return s.upper()
    m = _CHOICE_LETTER_PREFIX.match(s)
    if m:
        return m.group(1).upper()
    return None


def match_by_choice_letter(text: str, choices: list[str]) -> MatchResult | None:
    letter = parse_choice_letter(text)
    if letter is None:
        return None
    idx = ord(letter) - ord("A")
    if 0 <= idx < len(choices):
        return MatchResult(choices[idx], "choice_letter", 1.0)
    return None


def split_choice_parts(choice: str) -> list[str]:
    if "," in choice:
        return [part.strip() for part in choice.split(",") if part.strip()]
    return [choice.strip()]


def is_in_choices(parsed_answer: str, choices: list[str], *, normalized: bool) -> bool:
    if parsed_answer is None or parsed_answer == "":
        return False
    if not normalized:
        return parsed_answer in choices
    norm_pred = normalize_answer_text(parsed_answer)
    norm_choices = {normalize_answer_text(c) for c in choices}
    return norm_pred in norm_choices


def match_by_containment(text: str, choices: list[str]) -> MatchResult | None:
    norm_text = normalize_answer_text(text)
    loose_text = normalize_loose_answer_text(text)

    contained = []
    for c in choices:
        nc = normalize_answer_text(c)
        lc = normalize_loose_answer_text(c)
        parts = split_choice_parts(c)
        part_norms = [normalize_answer_text(p) for p in parts]

        if nc and nc in norm_text:
            contained.append(c)
        elif lc and lc in loose_text:
            contained.append(c)
        elif norm_text and norm_text in nc:
            contained.append(c)
        elif loose_text and loose_text in lc:
            contained.append(c)
        elif any(norm_text == pn or (norm_text and pn.startswith(norm_text)) for pn in part_norms):
            contained.append(c)
        elif any(
            norm_text and len(norm_text) >= 4 and is_token_substring(norm_text, pn)
            for pn in part_norms
        ):
            contained.append(c)

    if contained:
        return MatchResult(max(contained, key=len), "containment", 1.0)

    label_matches = []
    for c in choices:
        label = choice_label(c)
        nl = normalize_answer_text(label)
        if nl == norm_text or nl == loose_text:
            label_matches.append(c)
        elif norm_text and len(norm_text) >= 3 and is_token_substring(norm_text, nl):
            label_matches.append(c)
        elif norm_text and nl.startswith(norm_text):
            label_matches.append(c)

    if label_matches:
        return MatchResult(max(label_matches, key=len), "choice_label", 0.95)
    return None


def _pred_token_matches_option_token(pred_token: str, option_token: str) -> bool:
    if pred_token == option_token:
        return True
    if len(pred_token) >= 5 and len(option_token) >= 5 and similarity(pred_token, option_token) >= 0.9:
        return True
    return False


def _word_overlap_scores(tokens: list[str], part_tokens: list[str]) -> tuple[int, float, float]:
    """Return matched pred count, tie-break sim sum for unmatched preds, overlap ratio."""
    matched_pred = 0
    unmatched_max_sims: list[float] = []
    for pred_t in tokens:
        if any(_pred_token_matches_option_token(pred_t, opt_t) for opt_t in part_tokens):
            matched_pred += 1
        else:
            unmatched_max_sims.append(
                max((similarity(pred_t, opt_t) for opt_t in part_tokens), default=0.0)
            )
    tie_break = sum(unmatched_max_sims)
    ratio = matched_pred / max(len(part_tokens), 1)
    return matched_pred, tie_break, ratio


def match_by_word_overlap(text: str, choices: list[str]) -> MatchResult | None:
    tokens = tokenize_words(text)
    if not tokens:
        return None

    best_choice = ""
    best_match_count = -1
    best_tie_break = -1.0
    best_score = 0.0
    for c in choices:
        parts = split_choice_parts(choice_label(c))
        part_tokens = [p.lower() for p in parts]
        match_count, tie_break, score = _word_overlap_scores(tokens, part_tokens)
        if match_count > best_match_count or (
            match_count == best_match_count and tie_break > best_tie_break
        ):
            best_match_count = match_count
            best_tie_break = tie_break
            best_score = score
            best_choice = c

    if best_choice and best_score >= 0.5:
        return MatchResult(best_choice, "word_overlap", best_score)
    return None


def match_from_thinking(raw_text: str, choices: list[str]) -> MatchResult | None:
    if "<think>" not in raw_text.lower():
        return None

    thinking = raw_source_text(raw_text).lower()
    best_choice = ""
    best_score = 0.0
    for c in choices:
        score = similarity(thinking, normalize_answer_text(c))
        keywords = [w for w in tokenize_words(c) if len(w) >= 5]
        if keywords:
            kw_score = sum(1 for w in keywords if w in thinking) / len(keywords)
            score = max(score, kw_score)
        short_keywords = [w for w in tokenize_words(c) if len(w) >= 4]
        for w in short_keywords:
            if re.search(rf"(?<![a-z]){re.escape(w)}", thinking):
                score = max(score, 0.4)
        if score > best_score:
            best_score = score
            best_choice = c

    if best_choice and best_score >= 0.35:
        return MatchResult(best_choice, "thinking_overlap", best_score)
    return None


def match_by_number(text: str, choices: list[str]) -> MatchResult | None:
    nums = re.findall(r"\d+", text)
    if not nums:
        return None
    hits = [c for c in choices if all(n in c for n in nums)]
    if len(hits) == 1:
        return MatchResult(hits[0], "number", 0.9)
    return None


def match_by_fuzzy(text: str, choices: list[str], min_score: float) -> MatchResult | None:
    norm_text = normalize_answer_text(text)
    best_choice = ""
    best_score = 0.0
    for c in choices:
        for cand in (c, choice_label(c)):
            score = max(
                similarity(norm_text, normalize_answer_text(cand)),
                similarity(normalize_loose_answer_text(text), normalize_loose_answer_text(cand)),
            )
            if score > best_score:
                best_score = score
                best_choice = c
    if best_choice and best_score >= min_score:
        return MatchResult(best_choice, "fuzzy", best_score)
    return None


def match_parsed_to_choice(parsed_answer: str, choices: list[str]) -> MatchResult | None:
    raw = raw_source_text(parsed_answer)
    cleaned = clean_parsed_text(parsed_answer)
    source = cleaned or raw

    exact = is_exact_in_choices(source, choices)
    if exact is not None:
        return MatchResult(exact, "exact", 1.0)

    letter = match_by_choice_letter(source, choices)
    if letter is not None:
        return letter

    contained = match_by_containment(source, choices)
    if contained is not None:
        return contained

    numbered = match_by_number(source, choices)
    if numbered is not None:
        return numbered

    overlap = match_by_word_overlap(source, choices)
    if overlap is not None:
        return overlap

    min_score = 0.45 if len(normalize_answer_text(source)) <= 8 else 0.55
    fuzzy = match_by_fuzzy(source, choices, min_score=min_score)
    if fuzzy is not None:
        return fuzzy

    thinking = match_from_thinking(raw, choices)
    if thinking is not None:
        return thinking

    return match_by_fuzzy(source, choices, min_score=0.35)


def check_predictions(
    pred_path: Path,
    *,
    normalized: bool = False,
    invalid_out: Path | None = None,
    list_invalid: bool = False,
) -> CheckStats:
    invalid_out = invalid_out or pred_path.with_name(pred_path.stem + "_invalid_parsed_answer.jsonl")

    rows_total = 0
    missing_parsed = 0
    in_choice = 0
    not_in_choice = 0
    invalid_rows: list[dict] = []

    with pred_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            rows_total += 1
            obj = json.loads(line)
            rid = obj.get("id", f"line_{line_no}")
            choices = list(obj.get("multi_choice") or [])
            parsed = obj.get("parsed_answer")

            if parsed is None or parsed == "":
                missing_parsed += 1
                invalid_rows.append(obj)
                if list_invalid:
                    print(f"[missing] {rid}")
                continue

            if is_in_choices(str(parsed), choices, normalized=normalized):
                in_choice += 1
            else:
                not_in_choice += 1
                invalid_rows.append(obj)
                if list_invalid:
                    print(f"[not_in_choice] {rid}: {parsed!r}")

    if invalid_rows:
        write_jsonl(invalid_out, invalid_rows)

    return CheckStats(
        rows_total=rows_total,
        missing_parsed=missing_parsed,
        in_choice=in_choice,
        not_in_choice=not_in_choice,
        invalid_rows=invalid_rows,
    )


def print_check_report(
    pred_path: Path,
    stats: CheckStats,
    *,
    normalized: bool,
    invalid_out: Path,
) -> None:
    match_mode = "normalized" if normalized else "exact"
    print("=" * 72)
    print(f"Prediction file: {pred_path}")
    print(f"Match mode: {match_mode}")
    print(f"Rows total: {stats.rows_total}")
    print(f"Missing parsed_answer: {stats.missing_parsed}")
    print(f"In multi_choice: {stats.in_choice}")
    print(f"Not in multi_choice: {stats.not_in_choice}")
    print(f"Valid parsed in-choice rate: {stats.valid_rate:.2f}%")
    if stats.invalid_rows:
        print(f"Invalid rows written: {invalid_out}")


def rematch_invalid(
    pred_path: Path,
    *,
    invalid_path: Path | None = None,
    output_path: Path | None = None,
    report_path: Path | None = None,
) -> tuple[int, int, Path, Path]:
    invalid_path = invalid_path or pred_path.with_name(pred_path.stem + "_invalid_parsed_answer.jsonl")
    output_path = output_path or pred_path
    report_path = report_path or pred_path.with_name(pred_path.stem + "_rematch_report.jsonl")

    all_rows = read_jsonl(pred_path)
    invalid_rows = read_jsonl(invalid_path) if invalid_path.exists() else []

    if not invalid_rows:
        invalid_rows = [
            row
            for row in all_rows
            if row.get("parsed_answer") not in (row.get("multi_choice") or [])
        ]

    if not invalid_rows:
        raise FileNotFoundError(f"No invalid rows to rematch: {invalid_path}")

    rematch_by_id: dict[str, dict] = {}
    reports: list[dict] = []
    unresolved = 0

    for row in invalid_rows:
        source_row = dict(row)
        if source_row.get("parsed_answer_original"):
            source_row["parsed_answer"] = source_row["parsed_answer_original"]
        parsed = str(source_row.get("parsed_answer", ""))
        choices = list(source_row.get("multi_choice") or [])
        result = match_parsed_to_choice(parsed, choices)

        updated = dict(row)
        updated["parsed_answer_original"] = parsed
        report = {
            "id": row.get("id"),
            "parsed_answer_before": parsed,
            "cleaned_text": clean_parsed_text(parsed),
            "matched": result is not None,
        }

        if result is None:
            unresolved += 1
            report["parsed_answer_after"] = parsed
            report["match_method"] = None
            report["match_score"] = 0.0
        else:
            updated["parsed_answer"] = result.choice
            updated["parsed_answer_rematch_method"] = result.method
            updated["parsed_answer_rematch_score"] = round(result.score, 4)
            report["parsed_answer_after"] = result.choice
            report["match_method"] = result.method
            report["match_score"] = round(result.score, 4)

        rematch_by_id[row["id"]] = updated
        reports.append(report)

    merged_rows = []
    for row in all_rows:
        rid = row.get("id")
        merged_rows.append(rematch_by_id.get(rid, row))

    write_jsonl(output_path, merged_rows)
    write_jsonl(report_path, reports)

    still_invalid = sum(
        1
        for row in rematch_by_id.values()
        if row.get("parsed_answer") not in (row.get("multi_choice") or [])
    )

    print("=" * 72)
    print(f"Invalid input: {invalid_path} ({len(invalid_rows)} rows)")
    print(f"Merged output: {output_path}")
    print(f"Report: {report_path}")
    print(f"Matched: {len(invalid_rows) - unresolved}/{len(invalid_rows)}")
    print(f"Unresolved rematch: {unresolved}")
    print(f"Still not exact in multi_choice: {still_invalid}")
    print("-" * 72)
    for r in reports:
        if r["matched"]:
            print(
                f"{r['id']}: [{r['match_method']} {r['match_score']:.2f}] "
                f"{r['parsed_answer_before']!r} -> {r['parsed_answer_after']!r}"
            )
        else:
            print(f"{r['id']}: UNRESOLVED {r['parsed_answer_before']!r}")

    return len(invalid_rows) - unresolved, unresolved, output_path, report_path


def merge_retry_rows(
    main_path: Path,
    retry_path: Path,
    *,
    output_path: Path | None = None,
    drop_rematch_metadata: bool = False,
) -> int:
    output_path = output_path or main_path
    retry_by_id = {row["id"]: row for row in read_jsonl(retry_path)}
    merged: list[dict] = []
    replaced = 0

    for row in read_jsonl(main_path):
        rid = row.get("id")
        if rid not in retry_by_id:
            merged.append(row)
            continue

        retry_row = retry_by_id[rid]
        updated = dict(row)
        updated["parsed_answer"] = retry_row["parsed_answer"]
        if drop_rematch_metadata:
            for key in (
                "parsed_answer_original",
                "parsed_answer_rematch_method",
                "parsed_answer_rematch_score",
            ):
                updated.pop(key, None)
        replaced += 1
        merged.append(updated)

    write_jsonl(output_path, merged)
    print(f"Merged retry rows: {replaced}")
    print(f"Output: {output_path}")
    return replaced


def export_csv(
    pred_path: Path,
    csv_path: Path,
    *,
    quote_all: bool = True,
    question_field: str = "id",
    answer_field: str = "parsed_answer",
) -> int:
    rows = read_jsonl(pred_path)
    quoting = csv.QUOTE_ALL if quote_all else csv.QUOTE_MINIMAL

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, quoting=quoting)
        writer.writerow(["question", "answer"])
        for row in rows:
            writer.writerow([row[question_field], row[answer_field]])

    print(f"Exported {len(rows)} rows to {csv_path}")
    return len(rows)


def validate_csv(
    csv_path: Path,
    bench_path: Path,
) -> tuple[int, list[tuple[str, str]]]:
    choices_by_id = load_choices_by_id(bench_path)
    bad: list[tuple[str, str]] = []

    with csv_path.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            qid = row["question"]
            ans = row["answer"]
            if qid not in choices_by_id or ans not in choices_by_id[qid]:
                bad.append((qid, ans))

    total = sum(1 for _ in csv.DictReader(csv_path.open("r", encoding="utf-8")))
    print("=" * 72)
    print(f"CSV file: {csv_path}")
    print(f"Benchmark: {bench_path}")
    print(f"Rows: {total}")
    print(f"Invalid: {len(bad)}")
    print(f"Valid: {total - len(bad)}/{total}")
    if bad[:5]:
        print("Examples:")
        for qid, ans in bad[:5]:
            print(f"  {qid}: {ans!r}")
    return total - len(bad), bad


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Post-process ADQA prediction files.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_check = sub.add_parser("check", help="Check parsed_answer in multi_choice.")
    p_check.add_argument("--pred", type=str, default=str(DEFAULT_PRED))
    p_check.add_argument("--normalized", action="store_true")
    p_check.add_argument("--invalid-out", type=str, default="")
    p_check.add_argument("--list-invalid", action="store_true")

    p_rematch = sub.add_parser("rematch", help="Rematch invalid parsed_answer rows.")
    p_rematch.add_argument("--pred", type=str, default=str(DEFAULT_PRED))
    p_rematch.add_argument("--invalid", type=str, default="")
    p_rematch.add_argument("--output", type=str, default="")
    p_rematch.add_argument("--report", type=str, default="")

    p_merge = sub.add_parser("merge-retry", help="Replace main rows with retry rows by id.")
    p_merge.add_argument("--main", type=str, default=str(DEFAULT_PRED))
    p_merge.add_argument("--retry", type=str, required=True)
    p_merge.add_argument("--output", type=str, default="")
    p_merge.add_argument("--drop-rematch-metadata", action="store_true")

    p_csv = sub.add_parser("export-csv", help="Export id/parsed_answer CSV.")
    p_csv.add_argument("--pred", type=str, default=str(DEFAULT_PRED))
    p_csv.add_argument("--csv", type=str, default="")
    p_csv.add_argument("--quote-minimal", action="store_true")

    p_validate = sub.add_parser("validate-csv", help="Validate CSV against benchmark.")
    p_validate.add_argument("--csv", type=str, default=str(DEFAULT_PRED.with_suffix(".csv")))
    p_validate.add_argument("--bench", type=str, default=str(DEFAULT_BENCH))

    p_all = sub.add_parser("run-all", help="check -> rematch -> merge-retry")
    p_all.add_argument("--pred", type=str, default=str(DEFAULT_PRED))
    p_all.add_argument(
        "--retry",
        type=str,
        nargs="*",
        default=[],
        help="Retry jsonl files to merge by id after rematch.",
    )
    p_all.add_argument("--skip-rematch", action="store_true")
    p_all.add_argument("--skip-merge", action="store_true")
    p_all.add_argument("--list-invalid", action="store_true")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "check":
        pred_path = Path(args.pred).expanduser().resolve()
        invalid_out = Path(args.invalid_out).expanduser().resolve() if args.invalid_out else None
        stats = check_predictions(
            pred_path,
            normalized=args.normalized,
            invalid_out=invalid_out,
            list_invalid=args.list_invalid,
        )
        out = invalid_out or pred_path.with_name(pred_path.stem + "_invalid_parsed_answer.jsonl")
        print_check_report(pred_path, stats, normalized=args.normalized, invalid_out=out)
        return

    if args.command == "rematch":
        pred_path = Path(args.pred).expanduser().resolve()
        rematch_invalid(
            pred_path,
            invalid_path=Path(args.invalid).expanduser().resolve() if args.invalid else None,
            output_path=Path(args.output).expanduser().resolve() if args.output else None,
            report_path=Path(args.report).expanduser().resolve() if args.report else None,
        )
        return

    if args.command == "merge-retry":
        main_path = Path(args.main).expanduser().resolve()
        retry_path = Path(args.retry).expanduser().resolve()
        output_path = Path(args.output).expanduser().resolve() if args.output else main_path
        merge_retry_rows(
            main_path,
            retry_path,
            output_path=output_path,
            drop_rematch_metadata=args.drop_rematch_metadata,
        )
        return

    if args.command == "export-csv":
        pred_path = Path(args.pred).expanduser().resolve()
        csv_path = (
            Path(args.csv).expanduser().resolve()
            if args.csv
            else pred_path.with_suffix(".csv")
        )
        export_csv(pred_path, csv_path, quote_all=not args.quote_minimal)
        return

    if args.command == "validate-csv":
        csv_path = Path(args.csv).expanduser().resolve()
        bench_path = Path(args.bench).expanduser().resolve()
        validate_csv(csv_path, bench_path)
        return

    if args.command == "run-all":
        pred_path = Path(args.pred).expanduser().resolve()
        invalid_out = pred_path.with_name(pred_path.stem + "_invalid_parsed_answer.jsonl")

        stats = check_predictions(
            pred_path,
            invalid_out=invalid_out,
            list_invalid=args.list_invalid,
        )
        print_check_report(pred_path, stats, normalized=False, invalid_out=invalid_out)

        if not args.skip_rematch and stats.invalid_rows:
            rematch_invalid(pred_path, invalid_path=invalid_out, output_path=pred_path)
            stats = check_predictions(
                pred_path,
                invalid_out=invalid_out,
                list_invalid=args.list_invalid,
            )
            print_check_report(pred_path, stats, normalized=False, invalid_out=invalid_out)

        if not args.skip_merge and args.retry:
            for retry_file in args.retry:
                retry_path = Path(retry_file).expanduser().resolve()
                merge_retry_rows(pred_path, retry_path, output_path=pred_path)
            stats = check_predictions(
                pred_path,
                invalid_out=invalid_out,
                list_invalid=args.list_invalid,
            )
            print_check_report(pred_path, stats, normalized=False, invalid_out=invalid_out)
        return

    raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
