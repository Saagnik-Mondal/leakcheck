# Does it matter if a split also keeps near-twins together?

A random sample of 30,000 rows of the public symptom-disease dataset (28,003 distinct rows), split 80/20 three ways.

| Split | Test rows with an exact twin in train | Naive Bayes | 1-nearest-neighbour |
|---|---:|---:|---:|
| Random split | 10.2% | 68.4% | 69.0% |
| Exact copies together | 0.0% | 68.3% | 68.2% |
| Near-twins together (differ by at most 1 symptom) | 0.0% | 80.1% | 69.0% |

## Grouping details

| Split | Groups | Largest group (share of rows) | Test share achieved | Note |
|---|---:|---:|---:|---|
| Exact copies together | 28,003 | 0.0% | 20.0% | ok |
| Near-twins together (differ by at most 1 symptom) | 11,576 | 39.9% | 20.0% | the biggest group stays in training, so the test set excludes that region and is not representative of it: compare accuracies with care |
