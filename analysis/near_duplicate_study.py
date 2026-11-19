"""Does it matter if a split also keeps near-twins together?

Takes a random sample of the public symptom-disease dataset (the audit's first dataset) and
compares two models under three splits: random, exact-copies-together, and near-twins-together.

    uv run python analysis/audit_public.py          # once, to download the data
    uv run python analysis/near_duplicate_study.py

Writes reports/near_duplicate_split.md and .json.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from leakcheck.core import design_matrix, knn1_accuracy, naive_bayes_accuracy, random_split, row_ids
from leakcheck.split import safe_split

CSV = Path(
    "data/dhivyeshrk__Disease-Symptom-Extensive-Clean/Final_Augmented_dataset_Diseases_and_Symptoms.csv"
)
SAMPLE = 30_000
SEED = 0


def main() -> None:
    df = pd.read_csv(CSV, low_memory=False).sample(SAMPLE, random_state=SEED).reset_index(drop=True)
    target = "diseases"
    features = [c for c in df.columns if c != target]
    y = pd.factorize(df[target])[0]
    x = design_matrix(df, features)
    ids = row_ids(df, features)
    rng = np.random.default_rng(SEED)

    splits = {"Random split": random_split(len(df), rng, 0.2)}
    info = {}
    for name, near in (
        ("Exact copies together", 0),
        ("Near-twins together (differ by at most 1 symptom)", 1),
    ):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            r = safe_split(df, target, near=near, seed=SEED)
        splits[name] = (r.train, r.test)
        info[name] = {
            "groups": r.groups,
            "largest_group_pct": r.largest_group_pct,
            "test_fraction_pct": 100 * r.test_fraction,
            "warning": str(caught[0].message) if caught else None,
        }

    rows = []
    for name, (tr, te) in splits.items():
        rows.append(
            {
                "split": name,
                "train_rows": int(len(tr)),
                "test_rows": int(len(te)),
                "test_rows_with_exact_twin_in_train_pct": 100
                * float(np.isin(ids[te], ids[tr]).mean()),
                "naive_bayes_pct": 100 * naive_bayes_accuracy(x, y, tr, te, binary=True),
                "one_nn_pct": 100
                * knn1_accuracy(x, y, tr, te, 2000, np.random.default_rng(SEED), standardize=False),
            }
        )
    out = {
        "sample_rows": SAMPLE,
        "distinct_rows": int(len(np.unique(ids))),
        "splits": rows,
        "groups": info,
    }
    Path("reports").mkdir(exist_ok=True)
    Path("reports/near_duplicate_split.json").write_text(json.dumps(out, indent=2, default=float))

    lines = [
        "# Does it matter if a split also keeps near-twins together?",
        "",
        f"A random sample of {SAMPLE:,} rows of the public symptom-disease dataset ({out['distinct_rows']:,} distinct rows), "
        "split 80/20 three ways.",
        "",
        "| Split | Test rows with an exact twin in train | Naive Bayes | 1-nearest-neighbour |",
        "|---|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['split']} | {r['test_rows_with_exact_twin_in_train_pct']:.1f}% | {r['naive_bayes_pct']:.1f}% | {r['one_nn_pct']:.1f}% |"
        )
    lines += [
        "",
        "## Grouping details",
        "",
        "| Split | Groups | Largest group (share of rows) | Test share achieved | Note |",
        "|---|---:|---:|---:|---|",
    ]
    for name, g in info.items():
        note = g["warning"] or "ok"
        if g["largest_group_pct"] > 5:
            note = (
                "the biggest group stays in training, so the test set excludes that region and is "
                "not representative of it: compare accuracies with care"
            )
        lines.append(
            f"| {name} | {g['groups']:,} | {g['largest_group_pct']:.1f}% | {g['test_fraction_pct']:.1f}% | {note} |"
        )
    Path("reports/near_duplicate_split.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
