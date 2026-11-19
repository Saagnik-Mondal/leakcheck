"""Leak-aware train/test splitting.

A random split lets identical (and nearly identical) rows land on both sides. `safe_split` keeps
them together: all copies of a feature vector go to the same side, and, if you ask for it, so do
rows that differ by only a few features.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .core import design_matrix, row_ids


@dataclass
class SplitResult:
    train: np.ndarray  # row positions (0-based) in the dataframe
    test: np.ndarray
    groups: int  # number of groups kept together
    largest_group_pct: float  # share of rows in the biggest group
    test_fraction: float  # achieved share of rows in the test set
    near: int  # Hamming distance used to group near-duplicates (0 = exact copies only)


def _near_groups(x: np.ndarray, near: int, chunk: int = 512) -> np.ndarray:
    """Connected components of the graph that links vectors within `near` differing features."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    x = x.astype(np.float32)
    s = x.sum(axis=1)
    rows, cols = [], []
    for start in range(0, len(x), chunk):
        part = x[start : start + chunk]
        dist = s[start : start + chunk, None] + s[None, :] - 2.0 * (part @ x.T)
        r, c = np.nonzero(dist <= near)
        rows.append(r + start)
        cols.append(c)
    r, c = np.concatenate(rows), np.concatenate(cols)
    graph = coo_matrix((np.ones(len(r), dtype=np.int8), (r, c)), shape=(len(x), len(x)))
    return connected_components(graph, directed=False)[1]


def safe_split(
    df: pd.DataFrame,
    target: str,
    features: list[str] | None = None,
    test_size: float = 0.2,
    seed: int = 0,
    near: int = 0,
    max_near_rows: int = 30_000,
) -> SplitResult:
    """Split row positions into train and test with no exact twin across the split.

    near=0   keep all copies of an identical feature vector together
    near=k   also keep together vectors that differ in at most k features (binary features only)

    If near-duplicates link most of the data into one cluster, no leak-free split of the
    requested size exists. The result then reports it (`largest_group_pct`, `test_fraction`)
    and a warning is raised instead of silently returning a leaky split.
    """
    features = features or [c for c in df.columns if c != target]
    ids = row_ids(df, features)
    uniq, first, inverse = np.unique(ids, return_index=True, return_inverse=True)

    if near > 0:
        x = design_matrix(df, features)
        if x is None or not np.isin(x, (0, 1)).all():
            raise ValueError("near-duplicate grouping needs binary (0/1) features")
        if len(uniq) > max_near_rows:
            raise ValueError(
                f"{len(uniq):,} distinct rows is more than max_near_rows={max_near_rows:,}; "
                "pass a sample or raise the limit (cost grows with the square of the rows)"
            )
        vector_group = _near_groups(x[first], near)
    else:
        vector_group = np.arange(len(uniq))

    row_group = vector_group[inverse]
    n = len(df)
    sizes = np.bincount(row_group)
    target_rows = test_size * n
    rng = np.random.default_rng(seed)
    in_test = np.zeros(len(sizes), dtype=bool)
    count = 0
    for g in rng.permutation(len(sizes)):
        if count >= target_rows:
            break
        if count + sizes[g] <= target_rows * 1.1:
            in_test[g] = True
            count += sizes[g]

    is_test = in_test[row_group]
    result = SplitResult(
        train=np.flatnonzero(~is_test),
        test=np.flatnonzero(is_test),
        groups=int(len(sizes)),
        largest_group_pct=100.0 * float(sizes.max() / n),
        test_fraction=float(is_test.mean()),
        near=near,
    )
    if result.test_fraction < 0.5 * test_size:
        warnings.warn(
            f"Could only put {100 * result.test_fraction:.1f}% of rows in the test set "
            f"(wanted {100 * test_size:.0f}%): the biggest group holds {result.largest_group_pct:.1f}% "
            "of the rows. No leak-free split of this size exists at this threshold.",
            stacklevel=2,
        )
    return result
