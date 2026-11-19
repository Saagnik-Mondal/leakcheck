"""Does leakcheck cry wolf? Audit well-known datasets that ship with scikit-learn.

    uv run python analysis/clean_baselines.py

These are small, carefully prepared datasets, so the expected verdict is low risk. A tool that flags
them as leaky would be useless. Writes reports/clean_baselines.md.
"""

from __future__ import annotations

from pathlib import Path

from sklearn import datasets

from leakcheck import audit

LOADERS = {
    "Iris": datasets.load_iris,
    "Wine": datasets.load_wine,
    "Breast cancer (Wisconsin)": datasets.load_breast_cancer,
    "Digits (8x8 images)": datasets.load_digits,
}


def main() -> None:
    lines = [
        "# Does leakcheck cry wolf?",
        "",
        "Audits of well-known datasets that ship with scikit-learn. Expected verdict: low risk.",
        "",
        "| Dataset | Rows | Features | Exact duplicate rows | Test rows with an exact twin (random split) | Accuracy ceiling | Verdict |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for name, loader in LOADERS.items():
        bunch = loader(as_frame=True)
        df = bunch.frame.rename(columns={"target": "label"})
        r = audit(df, "label", n_splits=20, near_dup_sample=0)
        d = r.duplicates
        lines.append(
            f"| {name} | {r.rows:,} | {r.features} | {d['duplicate_rows']} ({d['duplicate_pct']:.1f}%) "
            f"| {r.split_leak_rate_pct:.1f}% | {d['accuracy_ceiling_pct']:.1f}% | {r.risk} |"
        )
    lines += [
        "",
        "Verdict rule: high if at least 20% of test rows leak or any model's drop is at least 10 points; "
        "moderate if at least 5% or 3 points; otherwise low. A model's drop counts only when at least 1% of "
        "test rows leak.",
        "",
        "History: the first version of this rule rated Iris and Wine as moderate, because on 150-row datasets "
        "the random and duplicate-free splits differ by noise. The 1% condition fixes that.",
    ]
    Path("reports").mkdir(exist_ok=True)
    Path("reports/clean_baselines.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
