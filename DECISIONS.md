# EDA decisions (application_train)

Each decision is backed by a check done in the EDA. Paste the table into a markdown cell in the notebook, or edit it in your own words.

| # | Issue found | Evidence | Decision |
|---|---|---|---|
| 1 | `DAYS_EMPLOYED = 365243` placeholder | 55,374 rows (18%), almost all pensioners; default rate 5.4% vs 8.66% for others | Add flag `DAYS_EMPLOYED_ANOM`, replace the value with NaN |
| 2 | `XNA` in `ORGANIZATION_TYPE` (also 4 rows in `CODE_GENDER`, 2 in `NAME_FAMILY_STATUS`) | 55,374 rows in `ORGANIZATION_TYPE`, same count and same rate (5.4%) as the placeholder group | Treat `XNA` as missing |
| 3 | Extreme incomes (max 117M, median 147k) | A tail of a few very large values; loan sizes look ordinary | Leave for tree models; log or clip for linear models; note as anomaly |
| 4 | Many columns over 50% missing (41 of them) | Mostly housing statistics; but `EXT_SOURCE_1` is 56% missing and a top predictor | Do not drop by threshold; keep columns that carry signal |
| 5 | Missingness vs default | `EXT_SOURCE_3` missing: 9.3% vs 7.8%; `OCCUPATION_TYPE` gap is mostly pensioners (5.4%) | Keep NaN for tree models; add missing flags for linear models |
| 6 | Housing statistics come as `_AVG` / `_MODE` / `_MEDI` triplets | Identical missing counts in each triplet | Consider keeping one per triplet; test against AUC |
| 7 | External scores are strong and mostly independent | Lowest vs highest fifth: 4–5x default rate; mutual correlation 0.11–0.21 | Engineer mean, min and product of the three scores |
| 8 | Credit exceeds goods price at every quartile | Median credit 513k vs goods price 450k | Engineer `AMT_CREDIT / AMT_GOODS_PRICE`, `AMT_CREDIT / AMT_INCOME_TOTAL`, `AMT_ANNUITY / AMT_INCOME_TOTAL` |
| 9 | Age and employment length each matter independently | Within every age band, longer employment lowers default (and vice versa) | Keep both; convert days to years |
| 10 | Tiny categories give unreliable rates | e.g. Maternity leave 40% from 5 people | Merge categories under about 100 applicants into "Other" |
| 11 | Class imbalance | 8.07% positives | Use ROC-AUC, stratified splits |
| 12 | Age is a sensitive attribute | Default rate 11.5% (20-30) down to 4.9% (60-70) | Use as a feature in this project; note in the README that it needs review before real use |
| 13 | Hyperparameter tuning (Optuna, 30 trials, 3-fold search; re-scored on a fresh 5-fold split) | Original 0.7869; old settings + early stopping 0.7893; tuned 0.7914 (fold std ~0.003). Tuned vs old early-stopped: +0.0021, within fold spread | Keep tuned params (`models/best_params.json`: num_leaves 25, max_depth 6, min_child_samples 168, strong L1) since they are no worse; most of the gain comes from more trees at a lower learning rate, not from the search itself |
| 14 | Error-analysis improvements tested with a locked 20% test set (3-fold CV on the other 80%; rules fixed in advance) | External-score features -0.0003 AUC, recent-payment features +0.0007, LightGBM+XGBoost blend +0.0006; fold std about 0.0045. Locked test AUC 0.7922 (95% interval 0.786 to 0.798) | None adopted (feature groups needed to win every fold, blend needed >0.001). Final model stays the notebook 06 LightGBM; more features from the same tables are unlikely to help |
