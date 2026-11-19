"""A SYNTHETIC demonstration of patient-level leakage. Not a finding about any real dataset.

Builds a made-up clinic: 200 patients, 6 visits each. Every visit's symptoms differ slightly from
the patient's usual pattern, so no two rows are identical, and each patient has one diagnosis that
is random (nothing about the symptoms can predict it across patients). Any accuracy above chance on
a random split is therefore pure recognition of the patient.

    uv run python analysis/group_demo.py

Writes reports/group_demo.md.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from leakcheck import audit


def make_clinic(n_patients=200, visits=6, d=60, classes=5, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    base = rng.integers(0, 2, size=(n_patients, d))
    diagnosis = rng.integers(0, classes, size=n_patients)
    rows, y, pid = [], [], []
    for p in range(n_patients):
        for flip in rng.choice(d, size=visits, replace=False):
            v = base[p].copy()
            v[flip] ^= 1
            rows.append(v)
            y.append(diagnosis[p])
            pid.append(f"patient_{p:03d}")
    df = pd.DataFrame(rows, columns=[f"symptom_{i}" for i in range(d)])
    df["diagnosis"] = y
    df["patient_id"] = pid
    return df


def main() -> None:
    df = make_clinic()
    blind = audit(df.drop(columns="patient_id"), "diagnosis", n_splits=10, near_dup_sample=0)
    aware = audit(df, "diagnosis", group="patient_id", n_splits=10, near_dup_sample=0)
    gc = aware.group_check
    nn = next(m for m in gc["models"] if "nearest" in m["name"])
    chance = 100 / df["diagnosis"].nunique()

    lines = [
        "# Synthetic demonstration: patient-level leakage",
        "",
        "Made-up data, not a real dataset: 200 patients, 6 visits each, 5 diagnoses assigned at random "
        "per patient. Every row is different from every other row, but visits of one patient look alike. "
        f"Nothing can be learned across patients, so chance is {chance:.0f}%.",
        "",
        '| | Without `group` | With `group="patient_id"` |',
        "|---|---:|---:|",
        f"| Test rows with an exact twin in training | {blind.split_leak_rate_pct:.1f}% | {aware.split_leak_rate_pct:.1f}% |",
        f"| Test rows whose patient is also in training | not checked | {gc['test_rows_with_group_in_train_pct']:.1f}% |",
        f"| 1-nearest-neighbour, random split | {nn['random_split_pct']:.1f}% | {nn['random_split_pct']:.1f}% |",
        f"| 1-nearest-neighbour, group-aware split | not available | {nn['group_aware_pct']:.1f}% |",
        f"| Verdict | {blind.risk} | {aware.risk} |",
        "",
        "A 1-nearest-neighbour model scores far above chance on a random split just by recognising "
        "patients. The audit without a group column sees no problem, because no rows are identical.",
    ]
    Path("reports").mkdir(exist_ok=True)
    Path("reports/group_demo.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
