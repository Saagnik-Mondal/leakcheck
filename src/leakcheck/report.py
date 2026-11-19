"""Turn an AuditResult into a readable Markdown report."""

from __future__ import annotations

from .core import AuditResult

RISK_TEXT = {
    "high": (
        "High. Many test rows in a random split already appear in training, or a model's score "
        "drops sharply without the repeats. Accuracy from a random split can overstate how the "
        "model will do on new data."
    ),
    "moderate": (
        "Moderate. Some test rows overlap with training rows. Report accuracy on a "
        "duplicate-free split as well."
    ),
    "low": "Low. Little overlap between random train and test splits.",
}


def overlap_markdown(c: dict, train_name: str, test_name: str) -> str:
    """Section for an official train/test pair."""
    return "\n".join(
        [
            f"## Official split: `{train_name}` vs `{test_name}`",
            "",
            "| Check | Result |",
            "|---|---:|",
            f"| Rows in train / test | {c['train_rows']:,} / {c['test_rows']:,} |",
            f"| **Test rows that also appear in train** | **{c['test_rows_also_in_train']:,} ({c['test_rows_also_in_train_pct']:.1f}%)** |",
            f"| ...of those, with the same label | {c['overlap_with_same_label_pct']:.1f}% |",
            f"| Duplicate rows inside train | {c['train_internal_duplicate_pct']:.1f}% |",
            f"| Duplicate rows inside test | {c['test_internal_duplicate_pct']:.1f}% |",
            "",
            "Text is compared after lower-casing and collapsing whitespace.",
            "",
        ]
    )


def to_markdown(r: AuditResult, title: str = "Dataset audit") -> str:
    d, a = r.duplicates, r.accuracy
    lines = [
        f"# {title}",
        "",
        f"**Leakage risk: {RISK_TEXT.get(r.risk, r.risk)}**",
        "",
        f"{r.rows:,} rows, {r.features} features, {r.classes} classes (label column: `{r.target}`).",
        "",
        "## 1. Duplicates and conflicting labels",
        "",
        "| Check | Result |",
        "|---|---:|",
        f"| Distinct feature vectors | {d['unique_feature_vectors']:,} ({100 * d['unique_feature_vectors'] / d['rows']:.1f}% of rows) |",
        f"| Exact duplicate rows (same features and label as an earlier row) | {d['duplicate_rows']:,} ({d['duplicate_pct']:.1f}%) |",
        f"| Feature vectors that appear with more than one label | {d['conflicting_vectors']:,} |",
        f"| Rows that share their features with a differently-labelled row | {d['rows_in_conflict']:,} ({d['rows_in_conflict_pct']:.1f}%) |",
        f"| **Accuracy ceiling** (best any classifier can do on this data) | **{d['accuracy_ceiling_pct']:.1f}%** |",
        "",
        "The ceiling exists because identical inputs with different labels cannot all be predicted "
        "correctly. A reported accuracy above it on the full data is a red flag.",
        "",
        "## 2. How much of a random split's test set is already in training?",
        "",
        f"Over {r.n_splits} random 80/20 splits, **{r.split_leak_rate_pct:.1f}%** (std {r.split_leak_rate_std:.2f}) "
        "of test rows have an identical feature vector in the training set.",
        "",
        "## 3. Accuracy: random split vs duplicate-free split",
        "",
        "In the duplicate-free split, identical rows are kept together, so no test row has an exact "
        "twin in training. A large drop means the random-split score was partly memory.",
        "",
        "| Model | Random split | Duplicate-free split | Drop |",
        "|---|---:|---:|---:|",
        f"| Lookup table (memorizes, learns nothing) | {a['memorizer_random_split_pct']:.1f}% | {a['memorizer_group_split_pct']:.1f}% | {a['memorizer_random_split_pct'] - a['memorizer_group_split_pct']:.1f} pts |",
    ]
    for m in r.models:
        drop = m["random_split_pct"] - m["group_split_pct"]
        lines.append(
            f"| {m['name']} | {m['random_split_pct']:.1f}% | {m['group_split_pct']:.1f}% | {drop:.1f} pts |"
        )
    lines += [
        f"| Always predict the most common class | {a['majority_baseline_pct']:.1f}% | {a['majority_baseline_pct']:.1f}% | n/a |",
        "",
        "Models that smooth over the data (naive Bayes) can be unaffected by repeated rows, while "
        "models that can memorize (nearest neighbour, trees, neural networks) are the ones at risk.",
    ]
    gc = r.group_check
    if gc:
        lines += [
            "",
            f"## Group leakage: `{gc['group_column']}`",
            "",
            f"{gc['groups']:,} groups, {gc['mean_rows_per_group']:.1f} rows each on average "
            f"(largest: {gc['largest_group_pct']:.1f}% of the data). Rows of one group are not independent, "
            "so a random split can leak even when no two rows are identical.",
            "",
            f"Over {r.n_splits} random 80/20 splits, **{gc['test_rows_with_group_in_train_pct']:.1f}%** "
            f"(std {gc['test_rows_with_group_in_train_std']:.2f}) of test rows belong to a group that also "
            "appears in training.",
            "",
            "| Model | Random split | Group-aware split | Drop |",
            "|---|---:|---:|---:|",
        ]
        for m in gc["models"]:
            drop = m["random_split_pct"] - m["group_aware_pct"]
            lines.append(
                f"| {m['name']} | {m['random_split_pct']:.1f}% | {m['group_aware_pct']:.1f}% | {drop:.1f} pts |"
            )
        lines += [
            "",
            "The group-aware split keeps every group, and any identical rows, on one side. The group column "
            "is not used as a feature.",
        ]
    nd = r.near_duplicates
    if nd.get("sampled"):
        lines += [
            "",
            "## 4. Near-duplicates",
            "",
            f"A duplicate-free split is not necessarily leak-free. For {nd['sampled']:,} test rows with no "
            "exact twin in training, the nearest training row differs in:",
            "",
            "| Features that differ | Share of test rows |",
            "|---|---:|",
            f"| at most 1 | {nd['within_1_pct']:.1f}% |",
            f"| at most 2 | {nd['within_2_pct']:.1f}% |",
            f"| at most 3 | {nd['within_3_pct']:.1f}% |",
            "",
            f"Median distance to the nearest training row: {nd['median_distance']:.0f} feature(s).",
        ]
    sf = r.single_feature
    if sf:
        lines += [
            "",
            "## 5. Is any single feature suspiciously predictive?",
            "",
            f"Majority-class baseline: {sf['majority_baseline_pct']:.1f}%. Best single-feature rules:",
            "",
            "| Feature | Accuracy using only this feature |",
            "|---|---:|",
        ]
        lines += [f"| {f} | {s:.1f}% |" for f, s in sf["top_features"]]
    for note in r.notes:
        lines += ["", f"_Note: {note}_"]
    lines += [
        "",
        "---",
        "_Risk rule: high if at least 20% of test rows leak or any model's drop is at least 10 points; "
        "moderate if at least 5% or 3 points. A model's drop counts only when at least 1% of test rows "
        "leak, because with no twins to memorize any gap is split noise._",
    ]
    return "\n".join(lines) + "\n"
