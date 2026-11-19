import numpy as np
import pandas as pd
import pytest

from leakcheck import audit
from leakcheck.core import (
    compare_splits,
    design_matrix,
    duplicate_stats,
    group_split,
    is_binary,
    knn1_accuracy,
    leak_rate,
    memorizer_accuracy,
    near_duplicate_profile,
    random_split,
    row_ids,
    single_feature_scan,
)
from leakcheck.report import to_markdown


def make_unique(n=400, d=30, classes=4, seed=0):
    """Random binary rows with labels that depend on the features. All rows distinct."""
    rng = np.random.default_rng(seed)
    x = rng.integers(0, 2, size=(n, d))
    y = x[:, :2].dot([1, 2]) % classes
    df = pd.DataFrame(x, columns=[f"f{i}" for i in range(d)])
    df["label"] = y
    return df.drop_duplicates(subset=[c for c in df.columns if c != "label"]).reset_index(drop=True)


def make_duplicated(copies=5, **kw):
    base = make_unique(**kw)
    return pd.concat([base] * copies, ignore_index=True)


def test_row_ids_ignore_the_label_and_match_identical_rows():
    df = pd.DataFrame({"a": [1, 1, 2], "b": [0, 0, 1], "label": [0, 1, 0]})
    ids = row_ids(df, ["a", "b"])
    assert ids[0] == ids[1] and ids[0] != ids[2]


def test_is_binary():
    assert is_binary(pd.DataFrame({"a": [0, 1], "b": [1, 1]}), ["a", "b"])
    assert not is_binary(pd.DataFrame({"a": [0, 2]}), ["a"])
    assert not is_binary(pd.DataFrame({"a": ["x", "y"]}), ["a"])


def test_duplicate_stats_on_known_data():
    # 4 distinct vectors, each repeated 5 times with the same label: 20 rows, 16 duplicates.
    df = make_duplicated(copies=5, n=4, d=20, classes=2).iloc[:20]
    feats = [c for c in df.columns if c != "label"]
    s = duplicate_stats(row_ids(df, feats), df["label"].to_numpy())
    assert s["rows"] == 20 and s["unique_feature_vectors"] == 4
    assert s["duplicate_rows"] == 16 and s["duplicate_pct"] == pytest.approx(80.0)
    assert s["conflicting_vectors"] == 0 and s["accuracy_ceiling_pct"] == pytest.approx(100.0)


def test_ceiling_with_conflicting_labels():
    # One input appears 4 times: labels 0,0,0,1. Best possible accuracy is 3/4.
    df = pd.DataFrame({"a": [1, 1, 1, 1], "label": [0, 0, 0, 1]})
    s = duplicate_stats(row_ids(df, ["a"]), df["label"].to_numpy())
    assert s["conflicting_vectors"] == 1 and s["rows_in_conflict"] == 4
    assert s["accuracy_ceiling_pct"] == pytest.approx(75.0)


def test_group_split_never_shares_a_vector_between_train_and_test():
    df = make_duplicated(copies=5, n=100)
    ids = row_ids(df, [c for c in df.columns if c != "label"])
    tr, te = group_split(ids, np.random.default_rng(0), 0.2)
    assert len(set(ids[tr]) & set(ids[te])) == 0
    assert len(tr) + len(te) == len(df)
    assert leak_rate(ids, tr, te) == 0.0


def test_random_split_leaks_heavily_when_rows_repeat():
    df = make_duplicated(copies=5, n=300)
    ids = row_ids(df, [c for c in df.columns if c != "label"])
    tr, te = random_split(len(df), np.random.default_rng(0), 0.2)
    assert leak_rate(ids, tr, te) > 0.95  # each test row has 4 siblings, almost surely in train


def test_no_leak_when_all_rows_are_distinct():
    df = make_unique(n=500)
    ids = row_ids(df, [c for c in df.columns if c != "label"])
    tr, te = random_split(len(df), np.random.default_rng(0), 0.2)
    assert leak_rate(ids, tr, te) == 0.0


def test_memorizer_scores_high_on_random_split_but_only_prior_on_group_split():
    df = make_duplicated(copies=6, n=400, classes=4)
    feats = [c for c in df.columns if c != "label"]
    ids, y = row_ids(df, feats), df["label"].to_numpy()
    rng = np.random.default_rng(1)
    random_acc = memorizer_accuracy(ids, y, *random_split(len(df), rng, 0.2))
    group_acc = memorizer_accuracy(ids, y, *group_split(ids, rng, 0.2))
    assert random_acc > 0.9
    assert group_acc < 0.5  # nothing to remember: close to guessing the common class


def test_near_duplicate_profile_finds_one_bit_neighbours():
    rng = np.random.default_rng(0)
    base = rng.integers(0, 2, size=(200, 40))
    flipped = base.copy()
    flipped[np.arange(200), rng.integers(0, 40, 200)] ^= 1  # every row differs in exactly 1 bit
    x = np.vstack([base, flipped]).astype(np.float32)
    ids = row_ids(pd.DataFrame(x), list(range(40)))
    train, test = np.arange(200), np.arange(200, 400)
    out = near_duplicate_profile(x, ids, train, test, sample=200, rng=rng)
    assert out["within_1_pct"] == pytest.approx(100.0)
    assert out["median_distance"] == 1.0


def test_single_feature_scan_flags_a_copy_of_the_label():
    df = make_unique(n=400)
    df["leaky"] = df["label"]
    codes = pd.factorize(df["label"])[0]
    out = single_feature_scan(df, [c for c in df.columns if c != "label"], codes)
    assert out["top_features"][0][0] == "leaky"
    assert out["top_features"][0][1] == pytest.approx(100.0)


def test_audit_rates_duplicated_data_high_and_clean_data_low():
    # Labels are random per vector, so nothing can be learned: any accuracy above chance on a
    # random split can only come from memorizing repeated rows.
    base = make_unique(n=600, d=40)
    base["label"] = np.random.default_rng(5).integers(0, 4, len(base))
    dirty = audit(
        pd.concat([base] * 6, ignore_index=True), "label", n_splits=3, near_dup_sample=100
    )
    clean = audit(make_unique(n=600, d=40), "label", n_splits=3, near_dup_sample=100)
    assert dirty.risk == "high" and dirty.split_leak_rate_pct > 90
    assert clean.risk == "low" and clean.split_leak_rate_pct == 0.0
    a = dirty.accuracy
    assert a["memorizer_random_split_pct"] - a["memorizer_group_split_pct"] > 40


def test_audit_handles_text_features_without_crashing():
    # 100 distinct sentences, each appearing 4 times: free text, 75% exact duplicates
    df = pd.DataFrame(
        {
            "text": [f"patient reports symptom number {i % 100}" for i in range(400)],
            "label": [(i % 100) % 3 for i in range(400)],
        }
    )
    r = audit(df, "label", n_splits=2)
    assert r.duplicates["duplicate_pct"] == pytest.approx(75.0)
    assert r.notes  # explains which checks were skipped


def test_design_matrix_codes_categoricals_fills_missing_and_rejects_text():
    df = pd.DataFrame(
        {
            "sex": ["M", "F", "M", "F"],
            "age": [30.0, None, 50.0, 40.0],
            "flag": [True, False, True, True],
        }
    )
    x = design_matrix(df, ["sex", "age", "flag"])
    assert x.shape == (4, 3) and not np.isnan(x).any()
    assert x[1, 1] == 40.0  # median of 30, 40, 50
    texts = pd.DataFrame({"t": [f"sentence number {i}" for i in range(200)]})
    assert design_matrix(texts, ["t"]) is None


def test_knn1_memorizes_repeated_rows_but_not_unseen_ones():
    base = make_unique(n=500, d=40)
    base["label"] = np.random.default_rng(7).integers(0, 4, len(base))  # nothing to learn
    df = pd.concat([base] * 5, ignore_index=True)
    feats = [c for c in df.columns if c != "label"]
    x, ids, y = df[feats].to_numpy(dtype=np.float32), row_ids(df, feats), df["label"].to_numpy()
    rng = np.random.default_rng(0)
    on_random = knn1_accuracy(x, y, *random_split(len(df), rng, 0.2), 300, rng, standardize=False)
    on_group = knn1_accuracy(x, y, *group_split(ids, rng, 0.2), 300, rng, standardize=False)
    assert on_random > 0.95
    assert on_group < 0.5


def test_audit_reports_two_models_and_categorical_columns_work():
    rng = np.random.default_rng(3)
    df = pd.DataFrame(
        {
            "sex": rng.choice(["M", "F"], 400),
            "age": rng.integers(20, 80, 400),
            "lab": rng.normal(size=400),
        }
    )
    df["label"] = (df["lab"] > 0).astype(int)
    r = audit(df, "label", n_splits=2)
    assert len(r.models) == 2
    assert r.single_feature["top_features"][0][0] == "lab"


def test_compare_splits_counts_overlap_and_label_agreement():
    train = pd.DataFrame({"t": ["Fever  and cough", "rash", "headache"], "label": [0, 1, 2]})
    test = pd.DataFrame({"t": ["fever and cough", "rash", "something new"], "label": [0, 2, 1]})
    c = compare_splits(train, test, "label")
    assert c["test_rows_also_in_train"] == 2  # "Fever  and cough" matches after normalizing
    assert c["test_rows_also_in_train_pct"] == pytest.approx(200 / 3)
    assert c["overlap_with_same_label_pct"] == pytest.approx(50.0)  # "rash" has a different label


def test_compare_splits_reports_zero_for_disjoint_files():
    train = pd.DataFrame({"x": [1, 2, 3], "label": [0, 1, 0]})
    test = pd.DataFrame({"x": [4, 5], "label": [1, 0]})
    assert compare_splits(train, test, "label")["test_rows_also_in_train"] == 0


def test_markdown_report_mentions_the_key_numbers():
    md = to_markdown(
        audit(make_duplicated(copies=4, n=300), "label", n_splits=2, near_dup_sample=50)
    )
    for needle in ("Accuracy ceiling", "Duplicate-free split", "Leakage risk", "Near-duplicates"):
        assert needle in md
