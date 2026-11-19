# Audit: dux-tecblic/symptom-disease-dataset

**Leakage risk: High. Many test rows in a random split already appear in training, or a model's score drops sharply without the repeats. Accuracy from a random split can overstate how the model will do on new data.**

5,634 rows, 1 features, 866 classes (label column: `label`).

## 1. Duplicates and conflicting labels

| Check | Result |
|---|---:|
| Distinct feature vectors | 1,993 (35.4% of rows) |
| Exact duplicate rows (same features and label as an earlier row) | 3,641 (64.6%) |
| Feature vectors that appear with more than one label | 0 |
| Rows that share their features with a differently-labelled row | 0 (0.0%) |
| **Accuracy ceiling** (best any classifier can do on this data) | **100.0%** |

The ceiling exists because identical inputs with different labels cannot all be predicted correctly. A reported accuracy above it on the full data is a red flag.

## 2. How much of a random split's test set is already in training?

Over 10 random 80/20 splits, **70.0%** (std 1.03) of test rows have an identical feature vector in the training set.

## 3. Accuracy: random split vs duplicate-free split

In the duplicate-free split, identical rows are kept together, so no test row has an exact twin in training. A large drop means the random-split score was partly memory.

| Model | Random split | Duplicate-free split | Drop |
|---|---:|---:|---:|
| Lookup table (memorizes, learns nothing) | 71.2% | 0.6% | 70.6 pts |
| Always predict the most common class | 2.7% | 2.7% | n/a |

Models that smooth over the data (naive Bayes) can be unaffected by repeated rows, while models that can memorize (nearest neighbour, trees, neural networks) are the ones at risk.

_Note: Free-text features: model, near-duplicate and single-feature checks were skipped._

---
_Risk rule: high if at least 20% of test rows leak or any model's drop is at least 10 points; moderate if at least 5% or 3 points._

## Official split: `symptom-disease-train-dataset.csv` vs `symptom-disease-test-dataset.csv`

| Check | Result |
|---|---:|
| Rows in train / test | 5,634 / 1,409 |
| **Test rows that also appear in train** | **980 (69.6%)** |
| ...of those, with the same label | 100.0% |
| Duplicate rows inside train | 64.6% |
| Duplicate rows inside test | 52.5% |

Text is compared after lower-casing and collapsing whitespace.
