# Forecasting arm-level attrition from pre-enrollment protocol data: final results

## Setup

- Benchmark: ClinicalTrials.gov interventional randomized trials with posted results, >= 1 control arm, all participant-flow groups matched to protocol arms (rapidfuzz score >= 80), every arm started >= 10.
- Temporal split by start year: train < 2015, validation 2015-2016, test >= 2017. Trials (arms): train 8,935 (22,041), validation 1,961 (4,810), test 5,424 (13,485).
- Models were fit on train and hyperparameters were selected on validation; no model fitting or tuning used the test split.
- Models: reference class (mean by phase x MeSH branch x arm type with back-off below 20 arms); trial-level LightGBM (one rate per trial); flat per-arm LightGBM on protocol features (flat_lgbm); FINAL = flat_lgbm plus time-respecting prior-evidence features (same normalized drug in earlier trials whose results were posted before the target trial started; missing when no evidence).
- Subsets: all; reliable (arms started >= 100; trials whose pooled control and every experimental arm started >= 100); covered (trials whose experimental arms all have prior evidence: 1,289 test trials); covered_reliable.
- 95% CIs: 1,000 bootstrap resamples of trials (seed 42), percentile intervals. Paired comparisons resample trials jointly for both models; p-values are two-sided Wald tests of the paired difference using its bootstrap standard error (percentile-bootstrap p-values with add-one correction are also given in results/tables/final_paired_tests.csv; with 1,000 resamples their resolution is ~0.002).
- Transparency on test-set use: the test split was also used in earlier exploratory analyses of this project (baselines, arm-feature enrichment, prior-evidence experiment), and the choice of FINAL and of the primary endpoints below was made after seeing those results. In particular, endpoint 3 was first observed in the prior-evidence experiment. These endpoints are therefore pre-specified for this report but not confirmatory; a prospective evaluation on trials posting results after this analysis is needed.

## Pre-specified primary endpoints (test set, unadjusted)

| # | Endpoint | Subset | FINAL | Comparator | Difference (FINAL - comparator) | p | Status |
|---|---|---|---|---|---|---|---|
| 1 | arm_rate_primary MAE | all | 0.110 [0.107, 0.114] | reference_class: 0.128 [0.124, 0.131] | -0.017 [-0.019, -0.015] | 4.9e-69 | robust across sensitivity analyses |
| 2 | composition mean_KL | all | 1.104 [1.084, 1.126] | reference_class: 1.236 [1.216, 1.259] | -0.132 [-0.143, -0.122] | 3.5e-130 | robust across sensitivity analyses |
| 3 | pairwise_primary concordance | covered_reliable | 0.577 [0.537, 0.617] | flat_lgbm: 0.504 [0.458, 0.547] | 0.073 [0.023, 0.124] | 0.005 | exploratory, not robust (fails at m90, m80 random split) |

Lower is better for MAE and KL; higher is better for concordance (0.5 = chance). Primary endpoints are reported without multiplicity adjustment. Status is derived from the sensitivity analyses below (a CI that includes 0 or flips sign in any matched set or split counts as a failure).

## Model comparison (test set)

### Subset: all

| Target | Metric | n (trials / rows) | reference_class | trial_lgbm | flat_lgbm | FINAL |
|---|---|---|---|---|---|---|
| arm_rate_primary | MAE | 5,424 / 13,485 | 0.128 [0.124, 0.131] | 0.112 [0.109, 0.115] | 0.111 [0.108, 0.114] | 0.110 [0.107, 0.114] |
| arm_rate_primary | Spearman | 5,424 / 13,485 | 0.220 [0.195, 0.245] | 0.281 [0.254, 0.308] | 0.295 [0.267, 0.323] | 0.301 [0.274, 0.327] |
| arm_rate_nonendpoint | MAE | 5,424 / 13,485 | 0.107 [0.104, 0.109] | 0.099 [0.097, 0.102] | 0.098 [0.095, 0.101] | 0.098 [0.095, 0.101] |
| arm_rate_nonendpoint | Spearman | 5,424 / 13,485 | 0.169 [0.144, 0.193] | 0.213 [0.186, 0.239] | 0.208 [0.182, 0.235] | 0.210 [0.184, 0.235] |
| composition | mean_KL | 5,367 / 12,192 | 1.236 [1.216, 1.259] | – | 1.103 [1.083, 1.125] | 1.104 [1.084, 1.126] |
| composition | mean_Aitchison | 5,367 / 12,192 | 6.861 [6.838, 6.883] | – | 6.607 [6.581, 6.633] | 6.556 [6.531, 6.583] |
| pairwise_primary | concordance | 5,194 / 8,488 | 0.516 [0.501, 0.531] | n/m† | 0.516 [0.500, 0.533] | 0.514 [0.499, 0.529] |
| pairwise_nonendpoint | concordance | 5,194 / 8,488 | 0.509 [0.493, 0.524] | n/m† | 0.505 [0.492, 0.518] | 0.516 [0.504, 0.528] |
| flag_primary_10_sig | AUROC | 5,194 / 5,194 | 0.524 [0.497, 0.556] | n/m† | 0.562 [0.533, 0.591] | 0.577 [0.549, 0.606] |
| flag_primary_10_sig | AUPRC | 5,194 / 5,194 | 0.093 [0.082, 0.111] | n/m† | 0.108 [0.092, 0.129] | 0.112 [0.097, 0.133] |
| flag_primary_5_sig | AUROC | 5,194 / 5,194 | 0.518 [0.493, 0.546] | n/m† | 0.555 [0.529, 0.581] | 0.579 [0.556, 0.604] |
| flag_primary_5_sig | AUPRC | 5,194 / 5,194 | 0.119 [0.106, 0.137] | n/m† | 0.132 [0.117, 0.151] | 0.138 [0.123, 0.158] |

### Subset: reliable

| Target | Metric | n (trials / rows) | reference_class | trial_lgbm | flat_lgbm | FINAL |
|---|---|---|---|---|---|---|
| arm_rate_primary | MAE | 2,040 / 4,528 | 0.136 [0.131, 0.141] | 0.110 [0.105, 0.115] | 0.109 [0.104, 0.114] | 0.109 [0.104, 0.114] |
| arm_rate_primary | Spearman | 2,040 / 4,528 | 0.297 [0.257, 0.338] | 0.379 [0.337, 0.421] | 0.388 [0.348, 0.431] | 0.397 [0.354, 0.439] |
| arm_rate_nonendpoint | MAE | 2,040 / 4,528 | 0.103 [0.099, 0.106] | 0.093 [0.089, 0.097] | 0.090 [0.086, 0.095] | 0.092 [0.087, 0.096] |
| arm_rate_nonendpoint | Spearman | 2,040 / 4,528 | 0.191 [0.152, 0.230] | 0.238 [0.196, 0.279] | 0.241 [0.197, 0.286] | 0.246 [0.203, 0.290] |
| composition | mean_KL | 2,010 / 4,348 | 1.062 [1.031, 1.097] | – | 0.923 [0.889, 0.959] | 0.920 [0.887, 0.956] |
| composition | mean_Aitchison | 2,010 / 4,348 | 6.731 [6.693, 6.772] | – | 6.386 [6.339, 6.437] | 6.338 [6.293, 6.389] |
| pairwise_primary | concordance | 1,723 / 2,397 | 0.522 [0.497, 0.545] | n/m† | 0.530 [0.505, 0.555] | 0.548 [0.521, 0.574] |
| pairwise_nonendpoint | concordance | 1,723 / 2,397 | 0.512 [0.488, 0.537] | n/m† | 0.521 [0.500, 0.541] | 0.539 [0.520, 0.560] |
| flag_primary_10_sig | AUROC | 1,654 / 1,654 | 0.535 [0.486, 0.583] | n/m† | 0.601 [0.550, 0.651] | 0.588 [0.544, 0.638] |
| flag_primary_10_sig | AUPRC | 1,654 / 1,654 | 0.111 [0.083, 0.153] | n/m† | 0.137 [0.106, 0.191] | 0.129 [0.101, 0.177] |
| flag_primary_5_sig | AUROC | 1,654 / 1,654 | 0.531 [0.490, 0.574] | n/m† | 0.567 [0.526, 0.605] | 0.574 [0.537, 0.613] |
| flag_primary_5_sig | AUPRC | 1,654 / 1,654 | 0.180 [0.152, 0.218] | n/m† | 0.195 [0.166, 0.236] | 0.189 [0.161, 0.228] |

### Subset: covered

| Target | Metric | n (trials / rows) | reference_class | trial_lgbm | flat_lgbm | FINAL |
|---|---|---|---|---|---|---|
| arm_rate_primary | MAE | 1,289 / 2,989 | 0.153 [0.146, 0.162] | 0.128 [0.121, 0.136] | 0.127 [0.121, 0.135] | 0.126 [0.119, 0.133] |
| arm_rate_primary | Spearman | 1,289 / 2,989 | 0.339 [0.291, 0.391] | 0.419 [0.371, 0.467] | 0.419 [0.369, 0.467] | 0.440 [0.393, 0.487] |
| arm_rate_nonendpoint | MAE | 1,289 / 2,989 | 0.110 [0.105, 0.115] | 0.102 [0.096, 0.108] | 0.100 [0.095, 0.106] | 0.100 [0.095, 0.106] |
| arm_rate_nonendpoint | Spearman | 1,289 / 2,989 | 0.163 [0.114, 0.210] | 0.187 [0.135, 0.236] | 0.183 [0.133, 0.232] | 0.195 [0.143, 0.244] |
| composition | mean_KL | 1,284 / 2,772 | 1.310 [1.266, 1.351] | – | 1.113 [1.077, 1.149] | 1.119 [1.082, 1.155] |
| composition | mean_Aitchison | 1,284 / 2,772 | 6.897 [6.850, 6.943] | – | 6.623 [6.568, 6.678] | 6.607 [6.553, 6.661] |
| pairwise_primary | concordance | 1,289 / 1,722 | 0.488 [0.462, 0.518] | n/m† | 0.507 [0.481, 0.535] | 0.541 [0.515, 0.568] |
| pairwise_nonendpoint | concordance | 1,289 / 1,722 | 0.491 [0.462, 0.519] | n/m† | 0.515 [0.494, 0.536] | 0.520 [0.496, 0.545] |
| flag_primary_10_sig | AUROC | 1,289 / 1,289 | 0.530 [0.471, 0.593] | n/m† | 0.576 [0.516, 0.633] | 0.629 [0.577, 0.680] |
| flag_primary_10_sig | AUPRC | 1,289 / 1,289 | 0.095 [0.068, 0.136] | n/m† | 0.114 [0.084, 0.174] | 0.130 [0.094, 0.194] |
| flag_primary_5_sig | AUROC | 1,289 / 1,289 | 0.487 [0.437, 0.542] | n/m† | 0.546 [0.495, 0.596] | 0.596 [0.551, 0.643] |
| flag_primary_5_sig | AUPRC | 1,289 / 1,289 | 0.113 [0.091, 0.148] | n/m† | 0.140 [0.111, 0.192] | 0.152 [0.121, 0.204] |

### Subset: covered_reliable

| Target | Metric | n (trials / rows) | reference_class | trial_lgbm | flat_lgbm | FINAL |
|---|---|---|---|---|---|---|
| arm_rate_primary | MAE | 587 / 1,250 | 0.164 [0.151, 0.176] | 0.119 [0.109, 0.129] | 0.119 [0.109, 0.129] | 0.118 [0.108, 0.127] |
| arm_rate_primary | Spearman | 587 / 1,250 | 0.460 [0.390, 0.525] | 0.584 [0.520, 0.639] | 0.577 [0.512, 0.638] | 0.602 [0.543, 0.658] |
| arm_rate_nonendpoint | MAE | 587 / 1,250 | 0.102 [0.096, 0.108] | 0.091 [0.085, 0.097] | 0.089 [0.082, 0.096] | 0.089 [0.083, 0.096] |
| arm_rate_nonendpoint | Spearman | 587 / 1,250 | 0.185 [0.106, 0.263] | 0.247 [0.170, 0.326] | 0.232 [0.149, 0.310] | 0.250 [0.174, 0.324] |
| composition | mean_KL | 582 / 1,223 | 1.105 [1.048, 1.166] | – | 0.888 [0.835, 0.943] | 0.896 [0.844, 0.951] |
| composition | mean_Aitchison | 582 / 1,223 | 6.730 [6.661, 6.802] | – | 6.356 [6.272, 6.440] | 6.345 [6.261, 6.427] |
| pairwise_primary | concordance | 525 / 643 | 0.498 [0.454, 0.544] | n/m† | 0.504 [0.458, 0.547] | 0.577 [0.537, 0.617] |
| pairwise_nonendpoint | concordance | 525 / 643 | 0.510 [0.468, 0.555] | n/m† | 0.511 [0.479, 0.544] | 0.544 [0.506, 0.581] |
| flag_primary_10_sig | AUROC | 508 / 508 | 0.564 [0.458, 0.668] | n/m† | 0.632 [0.525, 0.740] | 0.623 [0.520, 0.721] |
| flag_primary_10_sig | AUPRC | 508 / 508 | 0.120 [0.058, 0.222] | n/m† | 0.145 [0.079, 0.281] | 0.140 [0.075, 0.264] |
| flag_primary_5_sig | AUROC | 508 / 508 | 0.482 [0.407, 0.554] | n/m† | 0.556 [0.479, 0.637] | 0.568 [0.500, 0.636] |
| flag_primary_5_sig | AUPRC | 508 / 508 | 0.147 [0.102, 0.208] | n/m† | 0.189 [0.132, 0.283] | 0.178 [0.129, 0.267] |

† Not meaningful: trial_lgbm assigns the same rate to every arm of a trial, so its within-trial differential is zero by construction (concordance 0.5, flag AUROC 0.5). Differential targets are compared only against reference_class and flat_lgbm.

## Secondary paired comparisons (Holm-adjusted across the secondary family)

133 secondary tests (24 FINAL-vs-trial_lgbm tests on concordance/flags excluded as not meaningful). Tests with Holm-adjusted p < 0.05:

| Subset | Target | Metric | Comparison | Difference | p raw | p Holm |
|---|---|---|---|---|---|---|
| all | arm_rate_nonendpoint | MAE | FINAL - reference_class | -0.008 [-0.009, -0.007] | 2.3e-61 | 2.9e-59 |
| all | arm_rate_nonendpoint | MAE | FINAL - trial_lgbm | -0.001 [-0.002, -0.001] | 4.2e-05 | 0.005 |
| all | arm_rate_primary | RMSE | FINAL - reference_class | -0.012 [-0.015, -0.009] | 7.8e-13 | 9.1e-11 |
| all | arm_rate_primary | Spearman | FINAL - reference_class | 0.080 [0.057, 0.103] | 2.3e-11 | 2.7e-09 |
| all | composition | mean_Aitchison | FINAL - reference_class | -0.304 [-0.322, -0.288] | 1.1e-271 | 1.5e-269 |
| all | composition | mean_Aitchison | FINAL - flat_lgbm | -0.050 [-0.056, -0.045] | 3.9e-71 | 5.0e-69 |
| all | flag_primary_5_sig | AUROC | FINAL - reference_class | 0.061 [0.029, 0.095] | 4.6e-04 | 0.049 |
| covered | arm_rate_nonendpoint | MAE | FINAL - reference_class | -0.010 [-0.011, -0.008] | 3.3e-23 | 4.0e-21 |
| covered | arm_rate_nonendpoint | MAE | FINAL - trial_lgbm | -0.002 [-0.002, -0.001] | 2.7e-04 | 0.028 |
| covered | arm_rate_primary | MAE | FINAL - reference_class | -0.027 [-0.034, -0.022] | 2.8e-21 | 3.3e-19 |
| covered | arm_rate_primary | RMSE | FINAL - reference_class | -0.026 [-0.037, -0.018] | 7.5e-08 | 8.4e-06 |
| covered | arm_rate_primary | Spearman | FINAL - reference_class | 0.101 [0.059, 0.142] | 2.1e-06 | 2.3e-04 |
| covered | composition | mean_Aitchison | FINAL - reference_class | -0.290 [-0.326, -0.259] | 1.4e-64 | 1.7e-62 |
| covered | composition | mean_KL | FINAL - reference_class | -0.191 [-0.215, -0.169] | 4.6e-60 | 5.9e-58 |
| covered_reliable | arm_rate_nonendpoint | MAE | FINAL - reference_class | -0.012 [-0.016, -0.009] | 3.7e-16 | 4.4e-14 |
| covered_reliable | arm_rate_nonendpoint | RMSE | FINAL - reference_class | -0.006 [-0.010, -0.003] | 2.3e-04 | 0.025 |
| covered_reliable | arm_rate_primary | MAE | FINAL - reference_class | -0.046 [-0.057, -0.036] | 1.5e-18 | 1.7e-16 |
| covered_reliable | arm_rate_primary | RMSE | FINAL - reference_class | -0.054 [-0.071, -0.037] | 2.8e-10 | 3.2e-08 |
| covered_reliable | arm_rate_primary | Spearman | FINAL - reference_class | 0.142 [0.085, 0.198] | 1.1e-06 | 1.2e-04 |
| covered_reliable | composition | mean_Aitchison | FINAL - reference_class | -0.385 [-0.433, -0.334] | 2.0e-50 | 2.5e-48 |
| covered_reliable | composition | mean_KL | FINAL - reference_class | -0.209 [-0.242, -0.177] | 7.4e-37 | 9.2e-35 |
| reliable | arm_rate_nonendpoint | MAE | FINAL - reference_class | -0.011 [-0.013, -0.009] | 1.1e-34 | 1.3e-32 |
| reliable | arm_rate_nonendpoint | RMSE | FINAL - reference_class | -0.004 [-0.006, -0.002] | 4.6e-05 | 0.005 |
| reliable | arm_rate_primary | MAE | FINAL - reference_class | -0.028 [-0.031, -0.024] | 1.3e-40 | 1.7e-38 |
| reliable | arm_rate_primary | RMSE | FINAL - reference_class | -0.026 [-0.033, -0.019] | 5.0e-13 | 5.8e-11 |
| reliable | arm_rate_primary | Spearman | FINAL - reference_class | 0.100 [0.065, 0.135] | 2.6e-08 | 2.9e-06 |
| reliable | composition | mean_Aitchison | FINAL - reference_class | -0.393 [-0.420, -0.364] | 1.8e-160 | 2.3e-158 |
| reliable | composition | mean_Aitchison | FINAL - flat_lgbm | -0.048 [-0.058, -0.038] | 1.7e-21 | 2.1e-19 |
| reliable | composition | mean_KL | FINAL - reference_class | -0.142 [-0.156, -0.127] | 1.9e-76 | 2.5e-74 |

All secondary tests (raw and adjusted p) are in results/tables/final_paired_tests.csv.

## Conformal prediction intervals for the primary arm attrition rate (FINAL, target 90%)

Split-conformal, calibrated on validation arms. Mondrian = absolute-residual quantile per arm-size bin; normalized = single quantile of |y - p| / sqrt(p(1-p)/n). Arm size is the observed number started, used here as a proxy for the planned arm size that is known at design time.

| Method | Arm size (started) | Test arms | Coverage | Mean width |
|---|---|---:|---|---:|
| mondrian | all | 13,485 | 0.909 [0.901, 0.916] | 0.389 |
| mondrian | 10-49 | 5,985 | 0.899 [0.888, 0.910] | 0.364 |
| mondrian | 50-99 | 2,972 | 0.908 [0.890, 0.924] | 0.387 |
| mondrian | 100-299 | 3,161 | 0.923 [0.911, 0.936] | 0.423 |
| mondrian | >=300 | 1,367 | 0.920 [0.897, 0.939] | 0.429 |
| normalized | all | 13,485 | 0.914 [0.906, 0.921] | 0.491 |
| normalized | 10-49 | 5,985 | 0.982 [0.976, 0.987] | 0.653 |
| normalized | 50-99 | 2,972 | 0.940 [0.924, 0.954] | 0.452 |
| normalized | 100-299 | 3,161 | 0.885 [0.870, 0.900] | 0.344 |
| normalized | >=300 | 1,367 | 0.623 [0.583, 0.660] | 0.207 |

## Recruitment-planning simulation (test arms)

Required completers = observed completers (started minus primary-attrition withdrawals). Recruitment target = completers / (1 - assumed attrition), assumed attrition capped at 0.95. Under-recruited = target below the number actually started.

| Strategy | % arms under-recruited | Mean over-recruitment % (arms not under) | Mean shortfall % (arms under) |
|---|---|---|---|
| fixed_20pct | 25.6 [24.6, 26.8] | 15.5 [15.3, 15.7] | 24.9 [23.6, 26.2] |
| reference_class_mean | 31.1 [29.9, 32.3] | 13.2 [13.0, 13.5] | 21.9 [20.8, 23.0] |
| FINAL_point | 39.1 [37.9, 40.4] | 12.2 [11.6, 12.8] | 18.4 [17.5, 19.3] |
| FINAL_conformal_upper_90 | 7.3 [6.6, 7.9] | 56.9 [54.4, 59.3] | 32.0 [29.5, 34.8] |

### Recruitment trade-off frontier

Each strategy is swept over its parameter (fixed attrition 0.05-0.50; reference class + margin 0-0.30; FINAL one-sided Mondrian conformal upper quantile at levels 0.50-0.95). Over-recruitment at matched under-recruitment rates (linear interpolation; paired trial-grouped bootstrap CIs):

| Under-recruited | Metric | Fixed rule | Reference class | FINAL | FINAL - fixed | FINAL - reference |
|---|---|---|---|---|---|---|
| 10% | mean_over_recruitment_pct_not_under | 42.7 [39.8, 45.5] | 41.7 [39.1, 44.6] | 44.2 [41.3, 47.1] | 1.5 [-0.5, 3.4] | 2.6 [0.4, 4.5] |
| 10% | excess_patients_per_1000_completers | 434.7 [360.0, 504.8] | 413.4 [345.9, 480.2] | 469.0 [388.5, 545.6] | 34.3 [2.7, 69.2] | 55.5 [28.5, 87.7] |
| 20% | mean_over_recruitment_pct_not_under | 20.6 [19.7, 21.6] | 21.1 [20.0, 22.2] | 22.9 [21.7, 24.7] | 2.3 [1.4, 3.7] | 1.8 [0.9, 3.2] |
| 20% | excess_patients_per_1000_completers | 195.9 [161.3, 228.1] | 195.2 [161.5, 228.7] | 217.2 [180.7, 258.1] | 21.3 [8.2, 39.8] | 21.9 [11.0, 37.5] |

Full frontier: results/tables/final_recruitment_frontier.csv.

## Interpretability (TreeSHAP, FINAL, test arms)

Share of summed mean |SHAP| by feature family (log-odds scale). Shares are sums over features and grow with family size; prior-evidence features are missing for most test arms.

| Target | design (12) | condition (27) | eligibility_text (300) | arm (13) | prior_evidence (8) |
|---|---:|---:|---:|---:|---:|
| adverse_event | 28.3% | 2.7% | 56.0% | 12.7% | 0.4% |
| arm_rate_primary | 12.4% | 9.6% | 74.7% | 2.2% | 1.1% |
| lack_of_efficacy | 24.6% | 3.5% | 66.3% | 5.1% | 0.6% |
| lost_to_follow_up | 21.3% | 8.0% | 62.2% | 8.1% | 0.3% |
| pre_treatment_exclusion | 3.3% | 1.6% | 94.5% | 0.6% | 0.0% |
| withdrawal_by_subject | 13.4% | 3.0% | 78.7% | 4.4% | 0.5% |

Top-15 features per target: results/tables/final_shap_top15.csv.

## Sensitivity of the primary endpoints

| Matched set | Split | Endpoint | Subset | n trials | Difference (FINAL - comparator) | p |
|---|---|---|---|---:|---|---|
| m80 | temporal | arm_rate_primary MAE (FINAL - reference_class) | all | 5,424 | -0.017 [-0.019, -0.015] | 4.9e-69 |
| m80 | temporal | composition mean_KL (FINAL - reference_class) | all | 5,367 | -0.132 [-0.143, -0.122] | 3.5e-130 |
| m80 | temporal | pairwise_primary concordance (FINAL - flat_lgbm) | covered_reliable | 525 | 0.073 [0.023, 0.124] | 0.005 |
| m70 | temporal | arm_rate_primary MAE (FINAL - reference_class) | all | 5,586 | -0.017 [-0.019, -0.015] | 6.6e-75 |
| m70 | temporal | composition mean_KL (FINAL - reference_class) | all | 5,529 | -0.137 [-0.148, -0.126] | 9.4e-134 |
| m70 | temporal | pairwise_primary concordance (FINAL - flat_lgbm) | covered_reliable | 550 | 0.063 [0.013, 0.114] | 0.017 |
| m90 | temporal | arm_rate_primary MAE (FINAL - reference_class) | all | 5,244 | -0.017 [-0.019, -0.015] | 1.1e-68 |
| m90 | temporal | composition mean_KL (FINAL - reference_class) | all | 5,189 | -0.132 [-0.143, -0.121] | 4.9e-117 |
| m90 | temporal | pairwise_primary concordance (FINAL - flat_lgbm) | covered_reliable | 501 | -0.002 [-0.060, 0.055] | 0.956 |
| m80 | random_grouped | arm_rate_primary MAE (FINAL - reference_class) | all | 5,424 | -0.019 [-0.021, -0.017] | 1.0e-75 |
| m80 | random_grouped | composition mean_KL (FINAL - reference_class) | all | 5,373 | -0.138 [-0.148, -0.128] | 3.7e-156 |
| m80 | random_grouped | pairwise_primary concordance (FINAL - flat_lgbm) | covered_reliable | 432 | 0.025 [-0.031, 0.083] | 0.393 |

The random trial-grouped split ignores calendar time for model fitting (prior evidence remains time-respecting) and is reported only as a sensitivity analysis.
