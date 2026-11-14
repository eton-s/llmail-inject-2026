"""How the Stage-1 attack set (the 212 Scenario 4 "successes") was selected, criterion by criterion.

Re-derives the selection made in notebooks/analysis.ipynb section 8 straight from the raw challenge data and
writes the count after each criterion, so the funnel can be shown without shipping the 1.6 GB raw file.

Selection, exactly as analysis.ipynb did it:
  1. every phase-1 submission (raw_submissions_phase1.jsonl, one row per submission)
  2. Scenario 4 levels only (level4a..level4j)
  3. one row per distinct attack = (subject, body, level), keeping the first row in FILE order
     (pandas drop_duplicates) -- deduplication happens BEFORE the success filter. The raw file is not
     chronological (~50% of consecutive rows go backwards in time), so for a resubmitted attack this keeps
     an effectively arbitrary one of its submissions, not the earliest
  4-9. the kept submission has a recorded outcome, then each 2024 objective flag in turn:
       email.retrieved -> defense.undetected -> exfil.sent -> exfil.destination -> exfil.content
The last step is the 212. A comparison row counts distinct attacks that succeeded on ANY submission.

Writes selection_funnel.csv (step, criterion, unit, count, kept_pct_of_previous).
Usage (from replay/):
    python tools/export_selection_funnel.py --out ../../notebooks/data/selection_funnel.csv
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

REPLAY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPLAY))
from build_contexts import DATASET  # noqa: E402

FLAGS = [("email.retrieved", "attacker email retrieved by RAG"),
         ("defense.undetected", "not flagged by the level's defense"),
         ("exfil.sent", "model called the send tool"),
         ("exfil.destination", "sent to contact@contact.com"),
         ("exfil.content", "body was exactly the secret figure")]


def scan(raw_path: Path) -> dict:
    """One pass over the raw file, in file order, mirroring analysis.ipynb's pandas logic."""
    total = s4 = 0
    first: dict[tuple, dict] = {}          # (subject, body, level) -> objectives of its first submission
    ever_success: set[tuple] = set()       # distinct attacks with >= 1 fully successful submission
    with open(raw_path, "rb") as f:
        for line in f:
            if not line.strip():
                continue
            total += 1
            if b'"level4' not in line:     # cheap pre-filter; confirmed by the parsed field below
                continue
            r = json.loads(line)
            level = r.get("scenario") or ""
            if not level.startswith("level4"):
                continue
            s4 += 1
            o = r.get("objectives")
            o = json.loads(o) if isinstance(o, str) else (o or {})
            key = (r.get("subject"), r.get("body"), level)
            first.setdefault(key, o)
            if all(o.get(k) is True for k, _ in FLAGS):
                ever_success.add(key)
    return {"total": total, "s4": s4, "first": first, "ever_success": ever_success}


def funnel(res: dict) -> pd.DataFrame:
    rows = [("All phase-1 submissions", "submissions", res["total"]),
            ("... in Scenario 4 (levels 4a-4j)", "submissions", res["s4"]),
            ("... one per distinct attack (subject, body, level); one submission kept (file order)", "attacks", len(res["first"]))]
    kept = [o for o in res["first"].values() if all(k in o for k, _ in FLAGS)]
    rows.append(("... with a recorded 2024 outcome (all five flags present)", "attacks", len(kept)))
    for flag, meaning in FLAGS:
        kept = [o for o in kept if o.get(flag) is True]
        rows.append((f"... + {flag}: {meaning}", "attacks", len(kept)))
    df = pd.DataFrame(rows, columns=["criterion", "unit", "count"])
    df.insert(0, "step", range(1, len(df) + 1))
    df["kept_pct_of_previous"] = (df["count"] / df["count"].shift(1) * 100).round(1)
    comparison = pd.DataFrame([{"step": None, "criterion": "Comparison: distinct S4 attacks that fully succeeded on "
                                "ANY submission (filter first, then deduplicate)", "unit": "attacks",
                                "count": len(res["ever_success"]), "kept_pct_of_previous": None}])
    return pd.concat([df, comparison], ignore_index=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--raw", type=Path, default=DATASET / "raw_submissions_phase1.jsonl")
    p.add_argument("--out", type=Path, default=REPLAY / "results" / "selection_funnel.csv")
    args = p.parse_args()
    if not args.raw.exists():
        raise SystemExit(f"raw submissions not found: {args.raw}")

    res = scan(args.raw)
    df = funnel(res)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False, lineterminator="\n")
    with pd.option_context("display.max_colwidth", 90, "display.width", 160):
        print(df.to_string(index=False))
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
