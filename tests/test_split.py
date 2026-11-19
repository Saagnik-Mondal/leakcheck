import numpy as np
import pandas as pd
import pytest

from leakcheck.core import row_ids
from leakcheck.split import safe_split


def frame(x: np.ndarray) -> pd.DataFrame:
    df = pd.DataFrame(x, columns=[f"f{i}" for i in range(x.shape[1])])
    df["label"] = np.arange(len(df)) % 3
    return df


def random_binary(n=300, d=40, seed=0):
    return np.random.default_rng(seed).integers(0, 2, size=(n, d))


def test_exact_groups_never_straddle_the_split():
    base = random_binary(200)
    df = frame(np.vstack([base] * 4))  # every vector appears 4 times
    r = safe_split(df, "label")
    feats = [c for c in df.columns if c != "label"]
    ids = row_ids(df, feats)
    assert len(set(ids[r.train]) & set(ids[r.test])) == 0
    assert len(r.train) + len(r.test) == len(df)
    assert 0.15 < r.test_fraction < 0.25


def test_split_is_reproducible_and_seed_changes_it():
    df = frame(np.vstack([random_binary(200)] * 3))
    a, b, c = (
        safe_split(df, "label", seed=1),
        safe_split(df, "label", seed=1),
        safe_split(df, "label", seed=2),
    )
    assert np.array_equal(a.test, b.test)
    assert not np.array_equal(a.test, c.test)


def test_near_duplicates_are_kept_together():
    base = random_binary(150, d=60, seed=3)
    twin = base.copy()
    twin[np.arange(150), np.random.default_rng(4).integers(0, 60, 150)] ^= 1  # one feature differs
    x = np.vstack([base, twin])
    df = frame(x)
    r = safe_split(df, "label", near=1)
    side = np.zeros(len(df), dtype=int)
    side[r.test] = 1
    assert (side[:150] == side[150:]).all()  # each pair lands on the same side
    # An exact-only split puts many pairs on opposite sides.
    exact = safe_split(df, "label", near=0)
    side0 = np.zeros(len(df), dtype=int)
    side0[exact.test] = 1
    assert (side0[:150] != side0[150:]).sum() > 20


def test_dense_near_duplicate_graph_triggers_a_warning():
    # A chain where each row differs from the next by one feature forms a single cluster.
    n, d = 120, 120
    x = np.tril(np.ones((n, d), dtype=int))
    df = frame(x)
    with pytest.warns(UserWarning, match="No leak-free split"):
        r = safe_split(df, "label", near=1)
    assert r.largest_group_pct == pytest.approx(100.0)


def test_near_requires_binary_features():
    df = pd.DataFrame({"a": [0.5, 1.5, 2.5], "label": [0, 1, 0]})
    with pytest.raises(ValueError, match="binary"):
        safe_split(df, "label", near=1)


def test_near_refuses_huge_inputs():
    df = frame(random_binary(100))
    with pytest.raises(ValueError, match="max_near_rows"):
        safe_split(df, "label", near=1, max_near_rows=10)
