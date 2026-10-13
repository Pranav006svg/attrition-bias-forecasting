# Prior-evidence retrieval experiment

Evidence = other benchmark trials containing the same normalized DRUG/BIOLOGICAL name (exact match) whose resultsFirstPostDate is strictly before the target trial's start date (asserted). resultsFirstPostDate is used only for this ordering, never as a feature.

Unique normalized drugs: 8,030.

## Coverage

| Split | Experimental arms | Covered (>= 1 prior trial) | Trials (with differential) | All exp. arms covered |
|---|---:|---:|---:|---:|
| all | 21,105 | 19.3% | 15,303 | 20.0% |
| train | 11,354 | 14.4% | 8,263 | 14.9% |
| val | 2,508 | 28.7% | 1,846 | 29.7% |
| test | 7,243 | 23.6% | 5,194 | 24.8% |

Covered share of experimental arms by phase (all splits):

| Phase | Exp. arms | Covered |
|---|---:|---:|
| PHASE3 | 6,802 | 26.3% |
| PHASE2 | 6,257 | 19.4% |
| NA | 4,623 | 3.3% |
| PHASE4 | 1,767 | 37.1% |
| PHASE1 | 560 | 11.4% |
| PHASE2|PHASE3 | 556 | 19.8% |
| PHASE1|PHASE2 | 494 | 15.0% |
| EARLY_PHASE1 | 46 | 34.8% |

## Covered test subset (1,289 trials with a differential whose experimental arms are all covered)

Models fit on train, selected on val. 95% CIs: 1,000 trial-grouped bootstrap resamples; prior_minus_base is the paired difference.

| Subset | Target | Metric | trivial arm-role | flat_lgbm | flat_lgbm + prior | prior − base | n (trials / rows) |
|---|---|---|---|---|---|---|---|
| covered | pairwise_primary | concordance | – | 0.507 [0.481, 0.535] | 0.541 [0.515, 0.568] | 0.035 [0.001, 0.067] | 1,289 / 1,722 |
| covered | pairwise_ae | concordance | 0.626 [0.585, 0.668] | 0.616 [0.575, 0.657] | 0.619 [0.576, 0.659] | 0.003 [-0.023, 0.029] | 1,289 / 1,722 |
| covered | pairwise_efficacy | concordance | 0.613 [0.553, 0.669] | 0.573 [0.510, 0.630] | 0.619 [0.555, 0.673] | 0.047 [0.003, 0.094] | 1,289 / 1,722 |
| covered_reliable | pairwise_primary | concordance | – | 0.504 [0.458, 0.547] | 0.577 [0.537, 0.617] | 0.073 [0.023, 0.124] | 525 / 643 |
| covered_reliable | pairwise_ae | concordance | 0.614 [0.556, 0.674] | 0.605 [0.549, 0.666] | 0.611 [0.557, 0.674] | 0.006 [-0.031, 0.041] | 525 / 643 |
| covered_reliable | pairwise_efficacy | concordance | 0.650 [0.574, 0.724] | 0.574 [0.497, 0.648] | 0.626 [0.550, 0.704] | 0.052 [0.000, 0.107] | 525 / 643 |
| covered_ae_level_ge2pct | ae_gap_sign | sign_accuracy | 0.680 [0.622, 0.734] | 0.658 [0.603, 0.712] | 0.658 [0.600, 0.716] | 0.000 [-0.035, 0.035] | 294 / 294 |
| covered | flag_primary_10_sig | AUROC | – | 0.576 [0.516, 0.633] | 0.629 [0.577, 0.680] | 0.052 [-0.028, 0.135] | 1,289 / 1,289 |
| covered | flag_primary_10_sig | AUPRC | – | 0.114 [0.084, 0.174] | 0.130 [0.094, 0.194] | 0.016 [-0.030, 0.064] | 1,289 / 1,289 |
| covered_reliable | flag_primary_10_sig | AUROC | – | 0.632 [0.525, 0.740] | 0.623 [0.520, 0.721] | -0.009 [-0.169, 0.150] | 508 / 508 |
| covered_reliable | flag_primary_10_sig | AUPRC | – | 0.145 [0.079, 0.281] | 0.140 [0.075, 0.264] | -0.005 [-0.097, 0.082] | 508 / 508 |

## Drift check: temporal vs random trial-grouped split (prevalence, 95% CI)

| Metric | Scheme | train | val | test |
|---|---|---|---|---|
| hidden | temporal | 4.9% [4.4, 5.3] | 3.2% [2.4, 4.0] | 1.7% [1.4, 2.1] |
| hidden | random_grouped | 3.5% [3.1, 3.9] | 3.5% [2.7, 4.4] | 3.9% [3.4, 4.4] |
| y_flag_primary_5_sig | temporal | 11.4% [10.7, 12.1] | 10.5% [9.1, 11.9] | 10.6% [9.8, 11.5] |
| y_flag_primary_5_sig | random_grouped | 11.2% [10.5, 11.8] | 9.7% [8.3, 11.1] | 11.2% [10.4, 12.1] |
| y_flag_primary_10_sig | temporal | 8.4% [7.8, 9.1] | 7.6% [6.4, 8.9] | 8.1% [7.4, 8.9] |
| y_flag_primary_10_sig | random_grouped | 8.4% [7.8, 9.0] | 7.2% [5.9, 8.3] | 8.3% [7.6, 9.1] |
| y_flag_ae_5_sig | temporal | 5.6% [5.1, 6.1] | 3.1% [2.3, 4.0] | 2.2% [1.8, 2.6] |
| y_flag_ae_5_sig | random_grouped | 4.1% [3.7, 4.6] | 3.4% [2.6, 4.3] | 4.5% [3.9, 5.0] |
| y_flag_efficacy_5_sig | temporal | 3.9% [3.5, 4.3] | 1.9% [1.3, 2.5] | 1.3% [0.9, 1.5] |
| y_flag_efficacy_5_sig | random_grouped | 2.8% [2.4, 3.1] | 2.4% [1.8, 3.1] | 2.8% [2.4, 3.3] |

By start year:

| Metric | <=2007 | 2008-10 | 2011-14 | 2015-16 | 2017-19 | >=2020 |
|---|---|---|---|---|---|---|
| hidden | 5.3% | 5.2% | 4.5% | 3.2% | 2.0% | 1.3% |
| y_flag_primary_5_sig | 13.1% | 11.6% | 10.1% | 10.5% | 10.9% | 10.3% |
| y_flag_primary_10_sig | 10.0% | 8.4% | 7.2% | 7.6% | 8.0% | 8.3% |
| y_flag_ae_5_sig | 7.5% | 6.0% | 4.1% | 3.1% | 2.5% | 1.9% |
| y_flag_efficacy_5_sig | 5.6% | 3.7% | 2.9% | 1.9% | 1.6% | 0.9% |
| mean_arm_started | 353.927 | 200.066 | 253.826 | 185.867 | 251.278 | 227.518 |
| median_arm_started | 85.000 | 73.000 | 71.000 | 63.000 | 60.000 | 53.000 |
| mean_primary_rate | 0.205 | 0.181 | 0.168 | 0.173 | 0.171 | 0.148 |
| mean_ae_rate | 0.034 | 0.032 | 0.022 | 0.019 | 0.015 | 0.012 |
| mean_efficacy_rate | 0.035 | 0.024 | 0.022 | 0.016 | 0.013 | 0.006 |
| share_withdrawals_other_unspecified | 17.7% | 25.7% | 12.9% | 12.4% | 12.1% | 14.8% |
| share_withdrawals_ae | 13.9% | 14.4% | 11.8% | 8.5% | 3.2% | 3.6% |
| mean_distinct_primary_categories_per_trial | 3.751 | 3.670 | 3.443 | 3.270 | 3.164 | 2.974 |
| share_placebo_controlled | 49.1% | 55.1% | 49.0% | 47.1% | 45.1% | 44.6% |
| share_phase3 | 44.3% | 37.1% | 36.0% | 33.3% | 29.6% | 25.6% |
| share_reliable | 43.3% | 38.8% | 37.7% | 34.2% | 33.2% | 30.3% |
| n trials | 2,309 | 2,522 | 3,432 | 1,846 | 2,721 | 2,473 |
