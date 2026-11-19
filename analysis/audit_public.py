"""Audit two public symptom-to-disease datasets from the Hugging Face Hub and write the reports.

    uv run python analysis/audit_public.py

Downloads about 190 MB the first time (not stored in this repo). Takes a couple of minutes.
Writes reports/*.md, reports/*.json and reports/figure.png.
"""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path

import pandas as pd

from leakcheck import audit
from leakcheck.core import compare_splits
from leakcheck.report import overlap_markdown, to_markdown

HF = "https://huggingface.co/datasets"
DATA = Path("data")
REPORTS = Path("reports")

BIG_REPO = "dhivyeshrk/Disease-Symptom-Extensive-Clean"
BIG_FILE = "Final_Augmented_dataset_Diseases_and_Symptoms.csv"
TEXT_REPO = "dux-tecblic/symptom-disease-dataset"
TEXT_TRAIN = "symptom-disease-train-dataset.csv"
TEXT_TEST = "symptom-disease-test-dataset.csv"


def fetch(repo: str, name: str) -> Path:
    dest = DATA / repo.replace("/", "__") / name
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {repo}/{name} ...")
        urllib.request.urlretrieve(f"{HF}/{repo}/resolve/main/{name}", dest)  # noqa: S310
    return dest


def draw_figure(big: dict, text_overlap: dict, out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    BLUE, RED = "#2f5bea", "#d64545"
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [1.5, 1]})

    rows = [
        (
            "Lookup table\n(learns nothing)",
            big["accuracy"]["memorizer_random_split_pct"],
            big["accuracy"]["memorizer_group_split_pct"],
        ),
    ]
    for m in big["models"]:
        label = (
            "1-nearest-neighbour\n(can memorize)"
            if "nearest" in m["name"]
            else "Naive Bayes\n(smooths)"
        )
        rows.append((label, m["random_split_pct"], m["group_split_pct"]))
    x = range(len(rows))
    w = 0.38
    bars1 = a.bar(
        [i - w / 2 for i in x],
        [r[1] for r in rows],
        w,
        color=RED,
        label="Random split (repeats allowed)",
    )
    bars2 = a.bar(
        [i + w / 2 for i in x], [r[2] for r in rows], w, color=BLUE, label="Duplicate-free split"
    )
    for bars in (bars1, bars2):
        for bar in bars:
            a.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1.5,
                f"{bar.get_height():.0f}%",
                ha="center",
                fontsize=8,
            )
    a.set_xticks(list(x), [r[0] for r in rows], fontsize=8)
    a.set_ylim(0, 100)
    a.set_ylabel("Accuracy (%)", fontsize=9)
    a.set_title(
        "Disease-symptom dataset: accuracy depends on how you split", fontsize=10, loc="left"
    )
    a.legend(frameon=False, fontsize=8, loc="upper right")

    vals = [big["split_leak_rate_pct"], text_overlap["test_rows_also_in_train_pct"]]
    names = ["Symptom matrix,\nrandom 80/20 split", "Symptom text,\nofficial test file"]
    bs = b.bar(names, vals, color=[RED, RED], width=0.55)
    for bar, v in zip(bs, vals, strict=True):
        b.text(bar.get_x() + bar.get_width() / 2, v + 1.5, f"{v:.0f}%", ha="center", fontsize=9)
    b.set_ylim(0, 100)
    b.set_ylabel("Test rows already in training (%)", fontsize=9)
    b.set_title("Test rows with an exact twin in training", fontsize=10, loc="left")
    b.tick_params(axis="x", labelsize=8)
    for ax in (a, b):
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def main() -> None:
    REPORTS.mkdir(exist_ok=True)

    big_df = pd.read_csv(fetch(BIG_REPO, BIG_FILE), low_memory=False)
    big_result = audit(big_df, "diseases")
    big = big_result.to_dict()
    (REPORTS / "disease_symptom_matrix.json").write_text(json.dumps(big, indent=2, default=float))
    (REPORTS / "disease_symptom_matrix.md").write_text(
        to_markdown(big_result, f"Audit: {BIG_REPO}")
    )

    train = pd.read_csv(fetch(TEXT_REPO, TEXT_TRAIN))
    test = pd.read_csv(fetch(TEXT_REPO, TEXT_TEST))
    text_audit = audit(train, "label")
    overlap = compare_splits(train, test, "label")
    (REPORTS / "symptom_text_official_split.json").write_text(
        json.dumps(
            {"audit": text_audit.to_dict(), "official_split": overlap}, indent=2, default=float
        )
    )
    (REPORTS / "symptom_text_official_split.md").write_text(
        to_markdown(text_audit, f"Audit: {TEXT_REPO}")
        + "\n"
        + overlap_markdown(overlap, TEXT_TRAIN, TEXT_TEST)
    )
    draw_figure(big, overlap, REPORTS / "figure.png")
    print("Reports written to", REPORTS.resolve())


if __name__ == "__main__":
    main()
