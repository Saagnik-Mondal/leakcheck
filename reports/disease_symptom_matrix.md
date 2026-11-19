# Audit: dhivyeshrk/Disease-Symptom-Extensive-Clean

**Leakage risk: High. Many test rows in a random split already appear in training, or a model's score drops sharply without the repeats. Accuracy from a random split can overstate how the model will do on new data.**

246,945 rows, 377 features, 773 classes (label column: `diseases`).

## 1. Duplicates and conflicting labels

| Check | Result |
|---|---:|
| Distinct feature vectors | 169,888 (68.8% of rows) |
| Exact duplicate rows (same features and label as an earlier row) | 57,298 (23.2%) |
| Feature vectors that appear with more than one label | 12,634 |
| Rows that share their features with a differently-labelled row | 39,728 (16.1%) |
| **Accuracy ceiling** (best any classifier can do on this data) | **91.6%** |

The ceiling exists because identical inputs with different labels cannot all be predicted correctly. A reported accuracy above it on the full data is a red flag.

## 2. How much of a random split's test set is already in training?

Over 10 random 80/20 splits, **44.0%** (std 0.12) of test rows have an identical feature vector in the training set.

## 3. Accuracy: random split vs duplicate-free split

In the duplicate-free split, identical rows are kept together, so no test row has an exact twin in training. A large drop means the random-split score was partly memory.

| Model | Random split | Duplicate-free split | Drop |
|---|---:|---:|---:|
| Lookup table (memorizes, learns nothing) | 32.6% | 0.4% | 32.2 pts |
| Bernoulli naive Bayes | 84.3% | 84.5% | -0.2 pts |
| 1-nearest-neighbour (can memorize) | 77.0% | 66.5% | 10.5 pts |
| Always predict the most common class | 0.5% | 0.5% | n/a |

Models that smooth over the data (naive Bayes) can be unaffected by repeated rows, while models that can memorize (nearest neighbour, trees, neural networks) are the ones at risk.

## 4. Near-duplicates

A duplicate-free split is not necessarily leak-free. For 1,500 test rows with no exact twin in training, the nearest training row differs in:

| Features that differ | Share of test rows |
|---|---:|
| at most 1 | 99.0% |
| at most 2 | 100.0% |
| at most 3 | 100.0% |

Median distance to the nearest training row: 1 feature(s).

## 5. Is any single feature suspiciously predictive?

Majority-class baseline: 0.5%. Best single-feature rules:

| Feature | Accuracy using only this feature |
|---|---:|
| back pain | 0.8% |
| knee pain | 0.8% |
| elbow swelling | 0.8% |
| difficulty in swallowing | 0.8% |
| constipation | 0.8% |

---
_Risk rule: high if at least 20% of test rows leak or any model's drop is at least 10 points; moderate if at least 5% or 3 points._
