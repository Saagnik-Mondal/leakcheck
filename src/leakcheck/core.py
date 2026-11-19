"""Core audit logic.

The idea: a reported accuracy only means something if the test rows are genuinely new. If the
same feature vector appears in both train and test, a model can score well by remembering
it, and the number says little about how it will do on new data. This module measures how
much of an accuracy figure could come from that, using only numpy, pandas and scikit-learn.

Terminology
  row identity     a 64-bit hash of a row's feature values (the label is not included)
  duplicate row    a row whose identity and label both repeat an earlier row
  conflict         one identity that appears with more than one label
  ceiling          the best accuracy ANY classifier could reach on the data as given, because
                   identical inputs with different labels cannot all be predicted correctly
  memorizer        a lookup table: predict the most common label seen for an identical input
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- identities


def row_ids(df: pd.DataFrame, features: list[str]) -> np.ndarray:
    """A stable 64-bit identity for each row's feature values (label excluded)."""
    return pd.util.hash_pandas_object(df[features], index=False).to_numpy(dtype=np.uint64)


def is_binary(df: pd.DataFrame, features: list[str]) -> bool:
    """True when every feature column only contains 0 and 1."""
    try:
        values = df[features].to_numpy()
        return bool(np.isin(values, (0, 1)).all())
    except (TypeError, ValueError):
        return False


# --------------------------------------------------------------------------- duplicates


def duplicate_stats(ids: np.ndarray, y: np.ndarray) -> dict[str, float | int]:
    """Exact duplicates, conflicting labels, and the accuracy ceiling they imply."""
    n = len(ids)
    s = pd.DataFrame({"id": ids, "y": y})
    per_pair = s.groupby(["id", "y"]).size()
    n_pairs = len(per_pair)
    labels_per_id = per_pair.groupby(level="id").size()
    conflicting_ids = labels_per_id.index[labels_per_id > 1]
    in_conflict = int(s["id"].isin(conflicting_ids).sum())
    ceiling = float(per_pair.groupby(level="id").max().sum() / n)
    return {
        "rows": n,
        "unique_feature_vectors": int(len(labels_per_id)),
        "duplicate_rows": int(n - n_pairs),
        "duplicate_pct": 100.0 * (n - n_pairs) / n,
        "conflicting_vectors": int(len(conflicting_ids)),
        "rows_in_conflict": in_conflict,
        "rows_in_conflict_pct": 100.0 * in_conflict / n,
        "accuracy_ceiling_pct": 100.0 * ceiling,
    }


# --------------------------------------------------------------------------- splits


def random_split(
    n: int, rng: np.random.Generator, test_size: float
) -> tuple[np.ndarray, np.ndarray]:
    perm = rng.permutation(n)
    k = int(n * (1 - test_size))
    return perm[:k], perm[k:]


def group_split(
    ids: np.ndarray, rng: np.random.Generator, test_size: float
) -> tuple[np.ndarray, np.ndarray]:
    """Split so that no feature vector appears on both sides (identical rows stay together)."""
    uniq, inverse = np.unique(ids, return_inverse=True)
    order = rng.permutation(len(uniq))
    n_test_groups = max(1, int(len(uniq) * test_size))
    is_test_group = np.zeros(len(uniq), dtype=bool)
    is_test_group[order[:n_test_groups]] = True
    is_test = is_test_group[inverse]
    return np.flatnonzero(~is_test), np.flatnonzero(is_test)


def leak_rate(ids: np.ndarray, train: np.ndarray, test: np.ndarray) -> float:
    """Share of test rows whose exact feature vector also appears in train."""
    return float(np.isin(ids[test], ids[train]).mean())


# --------------------------------------------------------------------------- models


def memorizer_accuracy(
    ids: np.ndarray, y: np.ndarray, train: np.ndarray, test: np.ndarray
) -> float:
    """Accuracy of a lookup table: most common train label for an identical input, else the
    most common train label overall."""
    tr = pd.DataFrame({"id": ids[train], "y": y[train]})
    counts = tr.groupby(["id", "y"]).size().reset_index(name="c")
    best = counts.sort_values(["id", "c"], ascending=[True, False]).drop_duplicates("id")
    lookup = pd.Series(best["y"].to_numpy(), index=best["id"].to_numpy())
    fallback = tr["y"].mode().iloc[0]
    pred = pd.Series(ids[test]).map(lookup).fillna(fallback).to_numpy()
    return float((pred == y[test]).mean())


def naive_bayes_accuracy(
    x: np.ndarray, y: np.ndarray, train: np.ndarray, test: np.ndarray, binary: bool
) -> float:
    """Accuracy of a fast, standard model (Bernoulli or Gaussian naive Bayes)."""
    from sklearn.naive_bayes import BernoulliNB, GaussianNB

    model = BernoulliNB() if binary else GaussianNB()
    model.fit(x[train], y[train])
    return float((model.predict(x[test]) == y[test]).mean())


def design_matrix(
    df: pd.DataFrame, features: list[str], max_categories: int = 50
) -> np.ndarray | None:
    """A numeric matrix for the model-based checks, or None if any feature is free text.

    Booleans become 0/1, categorical columns with few levels are coded as integers, and
    missing numeric values are filled with the column median.
    """
    cols = []
    for f in features:
        c = df[f]
        if pd.api.types.is_bool_dtype(c):
            cols.append(c.astype(np.float32).to_numpy())
        elif pd.api.types.is_numeric_dtype(c):
            cols.append(c.fillna(c.median()).to_numpy(dtype=np.float32))
        elif c.nunique() <= max_categories:
            cols.append(pd.factorize(c)[0].astype(np.float32))
        else:
            return None
    return np.column_stack(cols)


def knn1_accuracy(
    x: np.ndarray,
    y: np.ndarray,
    train: np.ndarray,
    test: np.ndarray,
    sample: int,
    rng: np.random.Generator,
    standardize: bool,
    chunk: int = 256,
) -> float:
    """Accuracy of 1-nearest-neighbour, a model that can memorize its training rows.

    Evaluated on a random sample of the test rows to keep the cost down.
    """
    if len(test) > sample:
        test = rng.choice(test, size=sample, replace=False)
    xtr = x[train].astype(np.float32)
    xte = x[test].astype(np.float32)
    if standardize:
        mean, std = xtr.mean(axis=0), xtr.std(axis=0)
        std[std == 0] = 1.0
        xtr, xte = (xtr - mean) / std, (xte - mean) / std
    b = (xtr**2).sum(axis=1)
    correct = 0
    for start in range(0, len(xte), chunk):
        part = xte[start : start + chunk]
        dist = (part**2).sum(axis=1)[:, None] + b[None, :] - 2.0 * (part @ xtr.T)
        nearest = dist.argmin(axis=1)
        correct += int((y[train][nearest] == y[test][start : start + chunk]).sum())
    return correct / len(xte)


# --------------------------------------------------------------------------- near duplicates


def near_duplicate_profile(
    x: np.ndarray,
    ids: np.ndarray,
    train: np.ndarray,
    test: np.ndarray,
    sample: int,
    rng: np.random.Generator,
    chunk: int = 256,
) -> dict[str, float | int]:
    """For test vectors with NO exact twin in train, how close is the nearest train vector?

    Hamming distance on binary features. Distance 1 means one feature differs.
    """
    _, tr_first = np.unique(ids[train], return_index=True)
    xtr = x[train][tr_first].astype(np.float32)
    seen = np.isin(ids[test], ids[train])
    candidates = test[~seen]
    if len(candidates) == 0:
        return {"sampled": 0}
    _, te_first = np.unique(ids[candidates], return_index=True)
    candidates = candidates[te_first]
    if len(candidates) > sample:
        candidates = rng.choice(candidates, size=sample, replace=False)
    xte = x[candidates].astype(np.float32)
    a = xte.sum(axis=1)
    b = xtr.sum(axis=1)
    mins = np.empty(len(xte), dtype=np.float32)
    for start in range(0, len(xte), chunk):
        part = xte[start : start + chunk]
        dist = a[start : start + chunk, None] + b[None, :] - 2.0 * (part @ xtr.T)
        mins[start : start + chunk] = dist.min(axis=1)
    return {
        "sampled": int(len(mins)),
        "within_1_pct": 100.0 * float((mins <= 1).mean()),
        "within_2_pct": 100.0 * float((mins <= 2).mean()),
        "within_3_pct": 100.0 * float((mins <= 3).mean()),
        "median_distance": float(np.median(mins)),
    }


# --------------------------------------------------------------------------- single features


def single_feature_scan(
    df: pd.DataFrame, features: list[str], y_codes: np.ndarray, top: int = 5
) -> dict[str, object]:
    """How well can ONE feature alone predict the label? An unusually strong single feature is
    a classic sign of leakage (an ID, a post-outcome measurement, a copy of the label)."""
    k = int(y_codes.max()) + 1
    baseline = float(np.bincount(y_codes).max() / len(y_codes))
    scores = {}
    for f in features:
        col = df[f]
        numeric = pd.api.types.is_numeric_dtype(col) and not pd.api.types.is_bool_dtype(col)
        if not numeric and col.nunique() > 50:
            continue  # free text or ID-like strings are not meaningful here
        if numeric and col.nunique() > 10:
            col = pd.qcut(col, q=10, duplicates="drop").cat.codes
        codes = pd.factorize(col)[0]
        hits = 0
        for v in np.unique(codes):
            hits += np.bincount(y_codes[codes == v], minlength=k).max()
        scores[f] = hits / len(y_codes)
    ranked = sorted(scores.items(), key=lambda kv: -kv[1])[:top]
    return {
        "majority_baseline_pct": 100.0 * baseline,
        "top_features": [(f, 100.0 * s) for f, s in ranked],
    }


# --------------------------------------------------------------------------- official splits


def normalize_text(s: pd.Series) -> pd.Series:
    """Lower-case and collapse whitespace so trivially different copies still match."""
    return s.astype(str).str.lower().str.replace(r"\s+", " ", regex=True).str.strip()


def compare_splits(
    train: pd.DataFrame,
    test: pd.DataFrame,
    target: str,
    features: list[str] | None = None,
    normalize: bool = True,
) -> dict[str, float | int]:
    """How much of an official test file already appears in the official train file?"""
    features = features or [c for c in train.columns if c != target]
    a, b = train[features].copy(), test[features].copy()
    if normalize:
        for f in features:
            if not pd.api.types.is_numeric_dtype(a[f]):
                a[f], b[f] = normalize_text(a[f]), normalize_text(b[f])
    ida, idb = row_ids(a, features), row_ids(b, features)
    y_tr, y_te = train[target].to_numpy(), test[target].to_numpy()
    in_train = np.isin(idb, ida)
    train_pairs = set(zip(ida.tolist(), y_tr.tolist(), strict=True))
    same_label = np.array(
        [(i, y) in train_pairs for i, y in zip(idb.tolist(), y_te.tolist(), strict=True)]
    )
    n_over = int(in_train.sum())
    return {
        "train_rows": len(train),
        "test_rows": len(test),
        "test_rows_also_in_train": n_over,
        "test_rows_also_in_train_pct": 100.0 * n_over / len(test),
        "overlap_with_same_label_pct": 100.0 * float(same_label.sum()) / max(n_over, 1),
        "train_internal_duplicate_pct": 100.0 * float(pd.Series(ida).duplicated().mean()),
        "test_internal_duplicate_pct": 100.0 * float(pd.Series(idb).duplicated().mean()),
    }


# --------------------------------------------------------------------------- the audit


@dataclass
class AuditResult:
    rows: int
    features: int
    classes: int
    target: str
    duplicates: dict = field(default_factory=dict)
    split_leak_rate_pct: float = 0.0
    split_leak_rate_std: float = 0.0
    n_splits: int = 0
    accuracy: dict = field(default_factory=dict)
    models: list = field(default_factory=list)  # [{"name", "random_split_pct", "group_split_pct"}]
    near_duplicates: dict = field(default_factory=dict)
    single_feature: dict = field(default_factory=dict)
    risk: str = "unknown"
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def classify_risk(leak_pct: float, inflation_pts: float) -> str:
    """Plain thresholds, stated in the report so readers can disagree with them.

    A gap between random-split and duplicate-free accuracy only counts when at least 1% of test
    rows have an exact twin. Below that the two splits are nearly the same, so any gap is noise
    (small datasets showed false alarms before this rule: Iris and Wine were rated moderate).
    """
    if leak_pct < 1.0:
        inflation_pts = 0.0
    if leak_pct >= 20 or inflation_pts >= 10:
        return "high"
    if leak_pct >= 5 or inflation_pts >= 3:
        return "moderate"
    return "low"


def audit(
    df: pd.DataFrame,
    target: str,
    features: list[str] | None = None,
    n_splits: int = 10,
    test_size: float = 0.2,
    seed: int = 0,
    near_dup_sample: int = 1500,
    knn_sample: int = 2000,
    fit_models: bool = True,
) -> AuditResult:
    """Run the full audit on a labelled dataframe."""
    features = features or [c for c in df.columns if c != target]
    rng = np.random.default_rng(seed)
    y_codes, classes = pd.factorize(df[target])
    y = np.asarray(y_codes)
    ids = row_ids(df, features)
    n = len(df)

    result = AuditResult(rows=n, features=len(features), classes=len(classes), target=target)
    result.duplicates = duplicate_stats(ids, y)

    leaks = [leak_rate(ids, *random_split(n, rng, test_size)) for _ in range(n_splits)]
    result.n_splits = n_splits
    result.split_leak_rate_pct = 100.0 * float(np.mean(leaks))
    result.split_leak_rate_std = 100.0 * float(np.std(leaks))

    rnd_tr, rnd_te = random_split(n, rng, test_size)
    grp_tr, grp_te = group_split(ids, rng, test_size)
    acc = {
        "majority_baseline_pct": 100.0 * float(np.bincount(y).max() / n),
        "memorizer_random_split_pct": 100.0 * memorizer_accuracy(ids, y, rnd_tr, rnd_te),
        "memorizer_group_split_pct": 100.0 * memorizer_accuracy(ids, y, grp_tr, grp_te),
    }
    result.accuracy = acc

    x = design_matrix(df, features)
    binary = x is not None and bool(np.isin(x, (0, 1)).all())
    if x is not None and fit_models:
        nb_name = "Bernoulli naive Bayes" if binary else "Gaussian naive Bayes"
        result.models.append(
            {
                "name": nb_name,
                "random_split_pct": 100.0 * naive_bayes_accuracy(x, y, rnd_tr, rnd_te, binary),
                "group_split_pct": 100.0 * naive_bayes_accuracy(x, y, grp_tr, grp_te, binary),
            }
        )
        result.models.append(
            {
                "name": "1-nearest-neighbour (can memorize)",
                "random_split_pct": 100.0
                * knn1_accuracy(x, y, rnd_tr, rnd_te, knn_sample, rng, standardize=not binary),
                "group_split_pct": 100.0
                * knn1_accuracy(x, y, grp_tr, grp_te, knn_sample, rng, standardize=not binary),
            }
        )
    if x is not None and binary and near_dup_sample > 0:
        result.near_duplicates = near_duplicate_profile(
            x, ids, grp_tr, grp_te, near_dup_sample, rng
        )
    if x is not None:
        result.single_feature = single_feature_scan(df, features, y)

    drops = [m["random_split_pct"] - m["group_split_pct"] for m in result.models]
    inflation = (
        max(drops)
        if drops
        else (acc["memorizer_random_split_pct"] - acc["memorizer_group_split_pct"])
    )
    result.risk = classify_risk(result.split_leak_rate_pct, inflation)
    if x is None:
        result.notes.append(
            "Free-text features: model, near-duplicate and single-feature checks were skipped."
        )
    return result
