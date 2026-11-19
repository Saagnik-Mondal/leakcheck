import numpy as np
import pandas as pd
import pytest

from leakcheck import audit, safe_split
from leakcheck.core import assign_groups, row_ids, union_labels
from leakcheck.report import to_markdown


def patients(n_patients=120, visits=6, d=40, classes=4, seed=0):
    """Each patient has a base symptom vector; every visit flips one random feature.

    No two rows are identical, but all visits of a patient share one label. The label is random
    per patient, so a model can only score well by recognising the patient.
    """
    rng = np.random.default_rng(seed)
    base = rng.integers(0, 2, size=(n_patients, d))
    label = rng.integers(0, classes, size=n_patients)
    rows, y, pid = [], [], []
    for p in range(n_patients):
        for flip in rng.choice(d, size=visits, replace=False):  # distinct flips: no twin visits
            v = base[p].copy()
            v[flip] ^= 1
            rows.append(v)
            y.append(label[p])
            pid.append(f"patient_{p}")
    df = pd.DataFrame(rows, columns=[f"s{i}" for i in range(d)])
    df["disease"] = y
    df["patient_id"] = pid
    return df


def test_union_labels_links_rows_that_share_any_key():
    a = np.array([0, 0, 1, 2, 3])  # rows 0,1 share a value
    b = np.array([7, 8, 8, 9, 9])  # rows 1,2 share; rows 3,4 share
    labels = union_labels([a, b])
    assert labels[0] == labels[1] == labels[2]  # chained through the shared values
    assert labels[3] == labels[4]
    assert labels[0] != labels[3]


def test_assign_groups_never_splits_a_group():
    row_group = np.repeat(np.arange(50), 4)
    is_test = assign_groups(row_group, 0.2, np.random.default_rng(0))
    for g in range(50):
        assert len(set(is_test[row_group == g])) == 1
    assert 0.15 < is_test.mean() < 0.25


def test_audit_without_group_misses_patient_leakage_and_with_group_catches_it():
    df = patients()
    blind = audit(df.drop(columns="patient_id"), "disease", n_splits=3, near_dup_sample=0)
    aware = audit(df, "disease", group="patient_id", n_splits=3, near_dup_sample=0)
    assert blind.split_leak_rate_pct < 5  # visits differ by a feature, so no exact twins
    assert blind.risk == "low"
    gc = aware.group_check
    assert gc["groups"] == 120 and gc["mean_rows_per_group"] == pytest.approx(6.0)
    assert gc["test_rows_with_group_in_train_pct"] > 90
    one_nn = next(m for m in gc["models"] if "nearest" in m["name"])
    assert one_nn["random_split_pct"] - one_nn["group_aware_pct"] > 40
    assert aware.risk == "high"


def test_group_column_is_not_used_as_a_feature():
    df = patients(n_patients=30)
    r = audit(df, "disease", group="patient_id", n_splits=2, near_dup_sample=0)
    assert r.features == 40  # the 40 symptom columns only


def test_safe_split_keeps_every_patient_on_one_side():
    df = patients()
    r = safe_split(df, "disease", group="patient_id", seed=3)
    train_patients = set(df.iloc[r.train]["patient_id"])
    test_patients = set(df.iloc[r.test]["patient_id"])
    assert train_patients.isdisjoint(test_patients)
    assert 0.12 < r.test_fraction < 0.28


def test_safe_split_also_keeps_identical_rows_of_different_patients_together():
    df = patients(n_patients=60, visits=3)
    # Give two different patients one identical visit: they must now end up on the same side.
    a, b = df.index[df.patient_id == "patient_0"][0], df.index[df.patient_id == "patient_1"][0]
    feats = [c for c in df.columns if c.startswith("s")]
    df.loc[b, feats] = df.loc[a, feats].to_numpy()
    for seed in range(8):
        r = safe_split(df, "disease", group="patient_id", seed=seed)
        side = np.zeros(len(df), dtype=int)
        side[r.test] = 1
        p0 = side[df.patient_id == "patient_0"]
        p1 = side[df.patient_id == "patient_1"]
        assert len(set(p0)) == 1 and set(p0) == set(p1)


def test_report_has_a_group_section():
    md = to_markdown(
        audit(patients(60), "disease", group="patient_id", n_splits=2, near_dup_sample=0)
    )
    assert "Group leakage: `patient_id`" in md
    assert "Group-aware split" in md


def test_safe_split_without_group_still_ignores_unrelated_columns():
    df = patients(30, 3)
    r = safe_split(df.drop(columns="patient_id"), "disease")
    assert len(r.train) + len(r.test) == len(df)
    assert not set(r.train) & set(r.test)
    assert row_ids  # imported helper stays available
