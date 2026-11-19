# Does leakcheck cry wolf?

Audits of well-known datasets that ship with scikit-learn. Expected verdict: low risk.

| Dataset | Rows | Features | Exact duplicate rows | Test rows with an exact twin (random split) | Accuracy ceiling | Verdict |
|---|---:|---:|---:|---:|---:|---|
| Iris | 150 | 4 | 1 (0.7%) | 0.8% | 100.0% | low |
| Wine | 178 | 13 | 0 (0.0%) | 0.0% | 100.0% | low |
| Breast cancer (Wisconsin) | 569 | 30 | 0 (0.0%) | 0.0% | 100.0% | low |
| Digits (8x8 images) | 1,797 | 64 | 0 (0.0%) | 0.0% | 100.0% | low |

Verdict rule: high if at least 20% of test rows leak or any model's drop is at least 10 points; moderate if at least 5% or 3 points; otherwise low. A model's drop counts only when at least 1% of test rows leak.

History: the first version of this rule rated Iris and Wine as moderate, because on 150-row datasets the random and duplicate-free splits differ by noise. The 1% condition fixes that.
