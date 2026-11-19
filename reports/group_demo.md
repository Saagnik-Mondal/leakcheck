# Synthetic demonstration: patient-level leakage

Made-up data, not a real dataset: 200 patients, 6 visits each, 5 diagnoses assigned at random per patient. Every row is different from every other row, but visits of one patient look alike. Nothing can be learned across patients, so chance is 20%.

| | Without `group` | With `group="patient_id"` |
|---|---:|---:|
| Test rows with an exact twin in training | 0.0% | 0.0% |
| Test rows whose patient is also in training | not checked | 100.0% |
| 1-nearest-neighbour, random split | 100.0% | 100.0% |
| 1-nearest-neighbour, group-aware split | not available | 23.8% |
| Verdict | low | high |

A 1-nearest-neighbour model scores far above chance on a random split just by recognising patients. The audit without a group column sees no problem, because no rows are identical.
