"""Command line: leakcheck data.csv --target label"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .core import audit, compare_splits
from .report import overlap_markdown, to_markdown


def main() -> None:
    p = argparse.ArgumentParser(
        prog="leakcheck", description="Audit a labelled CSV for duplicates and train/test leakage."
    )
    p.add_argument("csv", type=Path, help="Path to the CSV file.")
    p.add_argument("--target", required=True, help="Name of the label column.")
    p.add_argument("--features", nargs="*", help="Feature columns (default: every other column).")
    p.add_argument(
        "--group", help="Column naming the patient, user or session each row belongs to."
    )
    p.add_argument("--splits", type=int, default=10, help="Random splits to average (default 10).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--max-rows", type=int, default=None, help="Use a random sample of this many rows."
    )
    p.add_argument(
        "--test",
        type=Path,
        help="An official test CSV: also report how much of it already appears in the main CSV.",
    )
    p.add_argument("--no-model", action="store_true", help="Skip the naive Bayes model.")
    p.add_argument("--json", type=Path, help="Also write the raw results to this JSON file.")
    p.add_argument(
        "--out", type=Path, help="Write the Markdown report here instead of printing it."
    )
    args = p.parse_args()

    df = pd.read_csv(args.csv, low_memory=False)
    if args.max_rows and len(df) > args.max_rows:
        df = df.sample(args.max_rows, random_state=args.seed).reset_index(drop=True)
    result = audit(
        df,
        args.target,
        args.features,
        n_splits=args.splits,
        seed=args.seed,
        fit_models=not args.no_model,
        group=args.group,
    )
    md = to_markdown(result, title=f"Audit of {args.csv.name}")
    if args.test:
        test_df = pd.read_csv(args.test, low_memory=False)
        overlap = compare_splits(df, test_df, args.target, args.features)
        md += "\n" + overlap_markdown(overlap, args.csv.name, args.test.name)
    if args.out:
        args.out.write_text(md)
        print(f"Report written to {args.out}")
    else:
        print(md)
    if args.json:
        args.json.write_text(json.dumps(result.to_dict(), indent=2, default=float))


if __name__ == "__main__":
    main()
