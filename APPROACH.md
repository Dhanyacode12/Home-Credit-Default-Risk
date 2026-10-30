# How to approach a new dataset

A reusable checklist. The goal is to know *what to check and why*, not to memorise code.

## 1. Understand the problem first
- [ ] What are we predicting (the target), and why does the business care?
- [ ] What does a mistake cost? (Here: approving a loan that defaults costs more than rejecting a good applicant.)
- [ ] Which metric fits? With ~8% positives, accuracy is misleading, so use ROC-AUC.

## 2. Understand the structure
- [ ] How many rows and columns? What does **one row** represent (applicant, loan, payment)?
- [ ] How many tables? Which key links them?
- [ ] Column types: numbers, categories, dates, IDs.
- [ ] Read the data dictionary: meaning and units of the important columns.

## 3. Check the target
- [ ] Balanced or imbalanced? Any missing values?
- This decides the metric and how you split the data (stratified).

## 4. Check data quality ("does this make sense?")
For each column:
- [ ] **Missing values:** how many, and *why*? Missing can carry meaning (no car, so no car age).
- [ ] **Impossible or odd values:** negative ages, a 1,000-year job, income of 117 million. Check min, max and the most repeated values.
- [ ] **Fake labels:** `XNA`, `Unknown`, `-1`, `999` look like data but mean "not applicable".
- [ ] **Duplicates:** repeated rows or IDs.
- [ ] **Wrong types:** numbers stored as text, dates stored as numbers.
- [ ] **Skew:** do a few huge values dominate?

## 5. Relate features to the target
- [ ] Categories: default rate per category vs the overall rate.
- [ ] Numbers: distribution for defaulters vs non-defaulters.
- [ ] Watch for **leakage**: a feature that predicts suspiciously well, or that would not be known at decision time.

## 6. Look at relationships between features
- [ ] Highly correlated columns carry the same information (e.g. credit amount and goods price).
- [ ] Combinations that make business sense, such as ratios (credit relative to income).

## 7. Decide, and write down why
- [ ] Which columns to fix, drop or keep; how to treat missing values and outliers.
- The output of EDA is a list of decisions with reasons, not just plots.

## 8. Baseline first
- [ ] Build the simplest reasonable model and record its score.
- Every later step must beat it. The score difference is your evidence.

## 9. Improve one thing at a time
- [ ] Add features or tables one by one and check the score after each.
- [ ] If something does not help, say so in the write-up.

## 10. Explain
- [ ] Which features matter and why (SHAP).
- [ ] What the model gets wrong, and what the business should do with it.

## Questions to keep asking
- Does this number make sense in the real world?
- Why might this be missing, repeated or extreme?
- Could this feature be known at the time of the decision?
- What would change my conclusion?

## In this project
| Step | Where |
|---|---|
| 1–3 | `notebooks/01_eda.ipynb`: Overview, Target balance |
| 4 | `01_eda.ipynb`: Missing values, Anomalies, the three scans |
| 5–6 | `01_eda.ipynb`: Default rate by category, numeric features, correlation |
| 8 | `notebooks/02_baseline_model.ipynb` (baseline AUC 0.759) |
| 9–10 | next: feature engineering, more tables, SHAP, write-up |
