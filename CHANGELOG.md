# Changelog

## 0.2.0 - 2026-11-19

### Added
- `safe_split`: a leak-aware train/test split that keeps identical rows together, and optionally rows that differ in at most `k` features. It warns when near-duplicates form one giant cluster.
- A study of near-twin grouping on a sample of the public symptom dataset (`reports/near_duplicate_split.md`).
- Audits of clean scikit-learn datasets to check for false alarms (`reports/clean_baselines.md`).
- `scipy` is now a declared dependency.

### Fixed
- False alarms: Iris and Wine were rated "moderate" because on tiny datasets the random and duplicate-free splits differ by noise. A model's accuracy drop now counts only when at least 1% of test rows have an exact twin.
- `safe_split` could choose a giant near-duplicate cluster as the entire test set, which made the stricter split look easier. Groups larger than a quarter of the test set now stay in training.

## 0.1.0 - 2026-11-19

First release: duplicate and conflict analysis with the accuracy ceiling, split-leakage rate, random vs duplicate-free split accuracy for a lookup table, naive Bayes and 1-nearest-neighbour, near-duplicate profile, single-feature scan, official train/test overlap check, a command-line tool, and audits of two public symptom-to-disease datasets.
