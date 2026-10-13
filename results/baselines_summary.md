# Baselines (test split, primary benchmark m80)

Train/val/test trials: 8,935 / 1,961 / 5,424; arms: 22,041 / 4,810 / 13,485. Values are point estimates with 95% bootstrap CIs (1,000 trial-grouped resamples, seed 42). Models were fit on train with hyperparameters selected on val; no tuning or selection used test.

Subsets: **all**; **reliable** = arm rates on arms started >= 100, trial targets on trials whose pooled control and every experimental arm started >= 100, pairwise concordance on pairs with both arms >= 100. n = trials / rows (arms, pairs or trials).

# Subset: all

## Arm attrition rate

| Target | Metric | n | global_mean | reference_class | trial_lgbm | flat_lgbm | flat_lgbm_enriched |
|---|---|---|---|---|---|---|---|
| arm_rate_primary | MAE | 5,424 / 13,485 | 0.135 [0.132, 0.138] | 0.128 [0.124, 0.131] | 0.112 [0.109, 0.115] | 0.111 [0.108, 0.114] | 0.111 [0.108, 0.114] |
| arm_rate_primary | RMSE | 5,424 / 13,485 | 0.182 [0.176, 0.189] | 0.172 [0.166, 0.177] | 0.160 [0.154, 0.165] | 0.160 [0.154, 0.166] | 0.159 [0.153, 0.165] |
| arm_rate_primary | Spearman | 5,424 / 13,485 | n/a | 0.220 [0.195, 0.245] | 0.281 [0.254, 0.308] | 0.295 [0.267, 0.323] | 0.295 [0.268, 0.322] |
| arm_rate_nonendpoint | MAE | 5,424 / 13,485 | 0.108 [0.106, 0.111] | 0.107 [0.104, 0.109] | 0.099 [0.097, 0.102] | 0.098 [0.095, 0.101] | 0.098 [0.095, 0.101] |
| arm_rate_nonendpoint | RMSE | 5,424 / 13,485 | 0.145 [0.139, 0.150] | 0.143 [0.138, 0.148] | 0.140 [0.135, 0.146] | 0.141 [0.135, 0.147] | 0.140 [0.134, 0.146] |
| arm_rate_nonendpoint | Spearman | 5,424 / 13,485 | n/a | 0.169 [0.144, 0.193] | 0.213 [0.186, 0.239] | 0.208 [0.182, 0.235] | 0.213 [0.187, 0.240] |

## Within-trial pairwise concordance (experimental vs control arm ordering; 0.5 = chance)

| Target | Metric | n | global_mean | reference_class | trial_lgbm | flat_lgbm | flat_lgbm_enriched |
|---|---|---|---|---|---|---|---|
| pairwise_primary | concordance | 5,194 / 8,488 | 0.500 [0.500, 0.500] | 0.516 [0.501, 0.531] | 0.500 [0.500, 0.500] | 0.516 [0.500, 0.533] | 0.519 [0.503, 0.534] |
| pairwise_nonendpoint | concordance | 5,194 / 8,488 | 0.500 [0.500, 0.500] | 0.509 [0.493, 0.524] | 0.500 [0.500, 0.500] | 0.505 [0.492, 0.518] | 0.514 [0.500, 0.531] |

## Reason composition (arms with >= 1 primary withdrawal; lower is better)

| Target | Metric | n | global_mean | reference_class | trial_lgbm | flat_lgbm | flat_lgbm_enriched |
|---|---|---|---|---|---|---|---|
| composition | mean_KL | 5,367 / 12,192 | 1.330 [1.312, 1.351] | 1.236 [1.216, 1.259] | n/a | 1.103 [1.083, 1.125] | 1.102 [1.082, 1.122] |
| composition | mean_Aitchison | 5,367 / 12,192 | 6.683 [6.662, 6.704] | 6.861 [6.838, 6.883] | n/a | 6.607 [6.581, 6.633] | 6.512 [6.486, 6.539] |

## Trial differential MAE (trials with a computable differential)

| Target | Metric | n | global_mean | reference_class | trial_lgbm | flat_lgbm | flat_lgbm_enriched |
|---|---|---|---|---|---|---|---|
| differential_primary_absmax | MAE | 5,194 / 5,194 | 0.069 [0.066, 0.071] | 0.059 [0.057, 0.062] | 0.069 [0.066, 0.071] | 0.066 [0.063, 0.068] | 0.063 [0.061, 0.065] |
| differential_primary_signed | MAE | 5,194 / 5,194 | 0.064 [0.062, 0.066] | 0.067 [0.065, 0.069] | 0.064 [0.062, 0.066] | 0.064 [0.062, 0.066] | 0.064 [0.062, 0.067] |
| differential_nonendpoint_absmax | MAE | 5,194 / 5,194 | 0.065 [0.063, 0.067] | 0.057 [0.055, 0.060] | 0.065 [0.063, 0.067] | 0.063 [0.061, 0.065] | 0.061 [0.059, 0.063] |
| differential_nonendpoint_signed | MAE | 5,194 / 5,194 | 0.061 [0.059, 0.063] | 0.064 [0.062, 0.066] | 0.061 [0.059, 0.063] | 0.061 [0.059, 0.063] | 0.061 [0.059, 0.063] |
| differential_ae_absmax | MAE | 5,194 / 5,194 | 0.013 [0.013, 0.014] | 0.015 [0.015, 0.016] | n/a | 0.013 [0.013, 0.014] | 0.013 [0.013, 0.014] |
| differential_efficacy_absmax | MAE | 5,194 / 5,194 | 0.006 [0.005, 0.007] | 0.008 [0.007, 0.009] | n/a | 0.006 [0.006, 0.007] | 0.007 [0.006, 0.008] |

## Differential-attrition flags (score = predicted |absmax differential|)

| Target | Metric | n | global_mean | reference_class | trial_lgbm | flat_lgbm | flat_lgbm_enriched |
|---|---|---|---|---|---|---|---|
| flag_primary_5_sig | AUROC | 5,194 / 5,194 | 0.500 [0.500, 0.500] | 0.518 [0.493, 0.546] | 0.500 [0.500, 0.500] | 0.555 [0.529, 0.581] | 0.545 [0.520, 0.570] |
| flag_primary_5_sig | AUPRC | 5,194 / 5,194 | 0.106 [0.098, 0.115] | 0.119 [0.106, 0.137] | 0.106 [0.098, 0.115] | 0.132 [0.117, 0.151] | 0.123 [0.109, 0.140] |
| flag_primary_10_sig | AUROC | 5,194 / 5,194 | 0.500 [0.500, 0.500] | 0.524 [0.497, 0.556] | 0.500 [0.500, 0.500] | 0.562 [0.533, 0.591] | 0.552 [0.524, 0.583] |
| flag_primary_10_sig | AUPRC | 5,194 / 5,194 | 0.081 [0.075, 0.089] | 0.093 [0.082, 0.111] | 0.081 [0.075, 0.089] | 0.108 [0.092, 0.129] | 0.099 [0.087, 0.115] |
| flag_nonendpoint_5_sig | AUROC | 5,194 / 5,194 | 0.500 [0.500, 0.500] | 0.508 [0.480, 0.536] | 0.500 [0.500, 0.500] | 0.544 [0.514, 0.572] | 0.548 [0.520, 0.573] |
| flag_nonendpoint_5_sig | AUPRC | 5,194 / 5,194 | 0.098 [0.089, 0.106] | 0.105 [0.094, 0.121] | 0.098 [0.089, 0.106] | 0.119 [0.106, 0.139] | 0.118 [0.104, 0.136] |
| flag_nonendpoint_10_sig | AUROC | 5,194 / 5,194 | 0.500 [0.500, 0.500] | 0.504 [0.474, 0.536] | 0.500 [0.500, 0.500] | 0.545 [0.514, 0.578] | 0.547 [0.516, 0.576] |
| flag_nonendpoint_10_sig | AUPRC | 5,194 / 5,194 | 0.073 [0.066, 0.081] | 0.077 [0.069, 0.092] | 0.073 [0.066, 0.081] | 0.092 [0.080, 0.110] | 0.089 [0.077, 0.105] |
| flag_ae_5_sig | AUROC | 5,194 / 5,194 | 0.500 [0.500, 0.500] | 0.672 [0.624, 0.715] | n/a | 0.781 [0.743, 0.813] | 0.782 [0.745, 0.815] |
| flag_ae_5_sig | AUPRC | 5,194 / 5,194 | 0.022 [0.018, 0.026] | 0.036 [0.028, 0.049] | n/a | 0.059 [0.045, 0.081] | 0.058 [0.044, 0.077] |
| flag_efficacy_5_sig | AUROC | 5,194 / 5,194 | 0.500 [0.500, 0.500] | 0.705 [0.636, 0.767] | n/a | 0.739 [0.682, 0.793] | 0.772 [0.720, 0.822] |
| flag_efficacy_5_sig | AUPRC | 5,194 / 5,194 | 0.013 [0.009, 0.016] | 0.060 [0.023, 0.121] | n/a | 0.059 [0.027, 0.123] | 0.055 [0.029, 0.118] |

# Subset: reliable

## Arm attrition rate

| Target | Metric | n | global_mean | reference_class | trial_lgbm | flat_lgbm | flat_lgbm_enriched |
|---|---|---|---|---|---|---|---|
| arm_rate_primary | MAE | 2,040 / 4,528 | 0.146 [0.141, 0.152] | 0.136 [0.131, 0.141] | 0.110 [0.105, 0.115] | 0.109 [0.104, 0.114] | 0.110 [0.105, 0.114] |
| arm_rate_primary | RMSE | 2,040 / 4,528 | 0.202 [0.192, 0.211] | 0.185 [0.176, 0.194] | 0.159 [0.151, 0.168] | 0.159 [0.150, 0.167] | 0.158 [0.150, 0.167] |
| arm_rate_primary | Spearman | 2,040 / 4,528 | n/a | 0.297 [0.257, 0.338] | 0.379 [0.337, 0.421] | 0.388 [0.348, 0.431] | 0.393 [0.347, 0.434] |
| arm_rate_nonendpoint | MAE | 2,040 / 4,528 | 0.105 [0.102, 0.109] | 0.103 [0.099, 0.106] | 0.093 [0.089, 0.097] | 0.090 [0.086, 0.095] | 0.091 [0.087, 0.095] |
| arm_rate_nonendpoint | RMSE | 2,040 / 4,528 | 0.137 [0.129, 0.145] | 0.134 [0.127, 0.142] | 0.130 [0.122, 0.139] | 0.130 [0.122, 0.139] | 0.129 [0.121, 0.138] |
| arm_rate_nonendpoint | Spearman | 2,040 / 4,528 | n/a | 0.191 [0.152, 0.230] | 0.238 [0.196, 0.279] | 0.241 [0.197, 0.286] | 0.246 [0.203, 0.290] |

## Within-trial pairwise concordance (experimental vs control arm ordering; 0.5 = chance)

| Target | Metric | n | global_mean | reference_class | trial_lgbm | flat_lgbm | flat_lgbm_enriched |
|---|---|---|---|---|---|---|---|
| pairwise_primary | concordance | 1,723 / 2,397 | 0.500 [0.500, 0.500] | 0.522 [0.497, 0.545] | 0.500 [0.500, 0.500] | 0.530 [0.505, 0.555] | 0.513 [0.489, 0.540] |
| pairwise_nonendpoint | concordance | 1,723 / 2,397 | 0.500 [0.500, 0.500] | 0.512 [0.488, 0.537] | 0.500 [0.500, 0.500] | 0.521 [0.500, 0.541] | 0.534 [0.509, 0.558] |

## Reason composition (arms with >= 1 primary withdrawal; lower is better)

| Target | Metric | n | global_mean | reference_class | trial_lgbm | flat_lgbm | flat_lgbm_enriched |
|---|---|---|---|---|---|---|---|
| composition | mean_KL | 2,010 / 4,348 | 1.186 [1.151, 1.220] | 1.062 [1.031, 1.097] | n/a | 0.923 [0.889, 0.959] | 0.920 [0.888, 0.956] |
| composition | mean_Aitchison | 2,010 / 4,348 | 6.689 [6.656, 6.723] | 6.731 [6.693, 6.772] | n/a | 6.386 [6.339, 6.437] | 6.301 [6.256, 6.351] |

## Trial differential MAE (trials with a computable differential)

| Target | Metric | n | global_mean | reference_class | trial_lgbm | flat_lgbm | flat_lgbm_enriched |
|---|---|---|---|---|---|---|---|
| differential_primary_absmax | MAE | 1,654 / 1,654 | 0.041 [0.038, 0.043] | 0.036 [0.033, 0.039] | 0.041 [0.038, 0.043] | 0.038 [0.035, 0.041] | 0.036 [0.033, 0.039] |
| differential_primary_signed | MAE | 1,654 / 1,654 | 0.038 [0.035, 0.041] | 0.043 [0.040, 0.046] | 0.038 [0.035, 0.041] | 0.039 [0.036, 0.041] | 0.040 [0.037, 0.042] |
| differential_nonendpoint_absmax | MAE | 1,654 / 1,654 | 0.037 [0.034, 0.039] | 0.033 [0.030, 0.035] | 0.037 [0.034, 0.039] | 0.035 [0.032, 0.037] | 0.033 [0.031, 0.036] |
| differential_nonendpoint_signed | MAE | 1,654 / 1,654 | 0.035 [0.032, 0.037] | 0.040 [0.037, 0.042] | 0.035 [0.032, 0.037] | 0.035 [0.032, 0.037] | 0.035 [0.033, 0.038] |
| differential_ae_absmax | MAE | 1,654 / 1,654 | 0.009 [0.008, 0.010] | 0.011 [0.010, 0.012] | n/a | 0.009 [0.008, 0.010] | 0.009 [0.008, 0.010] |
| differential_efficacy_absmax | MAE | 1,654 / 1,654 | 0.006 [0.005, 0.007] | 0.008 [0.007, 0.010] | n/a | 0.007 [0.005, 0.008] | 0.007 [0.006, 0.009] |

## Differential-attrition flags (score = predicted |absmax differential|)

| Target | Metric | n | global_mean | reference_class | trial_lgbm | flat_lgbm | flat_lgbm_enriched |
|---|---|---|---|---|---|---|---|
| flag_primary_5_sig | AUROC | 1,654 / 1,654 | 0.500 [0.500, 0.500] | 0.531 [0.490, 0.574] | 0.500 [0.500, 0.500] | 0.567 [0.526, 0.605] | 0.542 [0.501, 0.580] |
| flag_primary_5_sig | AUPRC | 1,654 / 1,654 | 0.152 [0.135, 0.170] | 0.180 [0.152, 0.218] | 0.152 [0.135, 0.170] | 0.195 [0.166, 0.236] | 0.172 [0.147, 0.205] |
| flag_primary_10_sig | AUROC | 1,654 / 1,654 | 0.500 [0.500, 0.500] | 0.535 [0.486, 0.583] | 0.500 [0.500, 0.500] | 0.601 [0.550, 0.651] | 0.557 [0.504, 0.609] |
| flag_primary_10_sig | AUPRC | 1,654 / 1,654 | 0.086 [0.073, 0.099] | 0.111 [0.083, 0.153] | 0.086 [0.073, 0.099] | 0.137 [0.106, 0.191] | 0.112 [0.087, 0.148] |
| flag_nonendpoint_5_sig | AUROC | 1,654 / 1,654 | 0.500 [0.500, 0.500] | 0.529 [0.488, 0.570] | 0.500 [0.500, 0.500] | 0.564 [0.524, 0.605] | 0.551 [0.511, 0.592] |
| flag_nonendpoint_5_sig | AUPRC | 1,654 / 1,654 | 0.135 [0.120, 0.152] | 0.150 [0.127, 0.183] | 0.135 [0.120, 0.152] | 0.170 [0.142, 0.209] | 0.161 [0.136, 0.198] |
| flag_nonendpoint_10_sig | AUROC | 1,654 / 1,654 | 0.500 [0.500, 0.500] | 0.528 [0.470, 0.583] | 0.500 [0.500, 0.500] | 0.590 [0.538, 0.642] | 0.559 [0.505, 0.612] |
| flag_nonendpoint_10_sig | AUPRC | 1,654 / 1,654 | 0.072 [0.060, 0.083] | 0.084 [0.066, 0.119] | 0.072 [0.060, 0.083] | 0.100 [0.077, 0.138] | 0.088 [0.070, 0.121] |
| flag_ae_5_sig | AUROC | 1,654 / 1,654 | 0.500 [0.500, 0.500] | 0.688 [0.618, 0.757] | n/a | 0.797 [0.751, 0.842] | 0.792 [0.737, 0.839] |
| flag_ae_5_sig | AUPRC | 1,654 / 1,654 | 0.033 [0.025, 0.042] | 0.059 [0.041, 0.088] | n/a | 0.101 [0.066, 0.172] | 0.110 [0.071, 0.173] |
| flag_efficacy_5_sig | AUROC | 1,654 / 1,654 | 0.500 [0.500, 0.500] | 0.677 [0.584, 0.762] | n/a | 0.676 [0.570, 0.771] | 0.717 [0.630, 0.799] |
| flag_efficacy_5_sig | AUPRC | 1,654 / 1,654 | 0.020 [0.013, 0.027] | 0.074 [0.025, 0.164] | n/a | 0.103 [0.033, 0.222] | 0.108 [0.031, 0.216] |

## Effect of arm-feature enrichment (flat_lgbm_enriched minus flat_lgbm, test, point estimates)

| Subset | Target | Metric | flat_lgbm | enriched | change |
|---|---|---|---:|---:|---:|
| all | arm_rate_primary | MAE | 0.111 | 0.111 | +0.001 |
| all | arm_rate_primary | RMSE | 0.160 | 0.159 | -0.001 |
| all | arm_rate_primary | Spearman | 0.295 | 0.295 | -0.000 |
| all | pairwise_primary | concordance | 0.516 | 0.519 | +0.002 |
| all | differential_primary_absmax | MAE | 0.066 | 0.063 | -0.003 |
| all | differential_primary_signed | MAE | 0.064 | 0.064 | +0.001 |
| all | flag_primary_5_sig | AUROC | 0.555 | 0.545 | -0.010 |
| all | flag_primary_5_sig | AUPRC | 0.132 | 0.123 | -0.009 |
| all | flag_primary_10_sig | AUROC | 0.562 | 0.552 | -0.009 |
| all | flag_primary_10_sig | AUPRC | 0.108 | 0.099 | -0.009 |
| all | arm_rate_nonendpoint | MAE | 0.098 | 0.098 | +0.001 |
| all | arm_rate_nonendpoint | RMSE | 0.141 | 0.140 | -0.001 |
| all | arm_rate_nonendpoint | Spearman | 0.208 | 0.213 | +0.005 |
| all | pairwise_nonendpoint | concordance | 0.505 | 0.514 | +0.009 |
| all | differential_nonendpoint_absmax | MAE | 0.063 | 0.061 | -0.002 |
| all | differential_nonendpoint_signed | MAE | 0.061 | 0.061 | +0.000 |
| all | flag_nonendpoint_5_sig | AUROC | 0.544 | 0.548 | +0.005 |
| all | flag_nonendpoint_5_sig | AUPRC | 0.119 | 0.118 | -0.001 |
| all | flag_nonendpoint_10_sig | AUROC | 0.545 | 0.547 | +0.001 |
| all | flag_nonendpoint_10_sig | AUPRC | 0.092 | 0.089 | -0.003 |
| all | differential_ae_absmax | MAE | 0.013 | 0.013 | -0.000 |
| all | flag_ae_5_sig | AUROC | 0.781 | 0.782 | +0.001 |
| all | flag_ae_5_sig | AUPRC | 0.059 | 0.058 | -0.001 |
| all | differential_efficacy_absmax | MAE | 0.006 | 0.007 | +0.000 |
| all | flag_efficacy_5_sig | AUROC | 0.739 | 0.772 | +0.033 |
| all | flag_efficacy_5_sig | AUPRC | 0.059 | 0.055 | -0.004 |
| all | composition | mean_KL | 1.103 | 1.102 | -0.002 |
| all | composition | mean_Aitchison | 6.607 | 6.512 | -0.094 |
| reliable | arm_rate_primary | MAE | 0.109 | 0.110 | +0.000 |
| reliable | arm_rate_primary | RMSE | 0.159 | 0.158 | -0.001 |
| reliable | arm_rate_primary | Spearman | 0.388 | 0.393 | +0.005 |
| reliable | pairwise_primary | concordance | 0.530 | 0.513 | -0.017 |
| reliable | differential_primary_absmax | MAE | 0.038 | 0.036 | -0.002 |
| reliable | differential_primary_signed | MAE | 0.039 | 0.040 | +0.001 |
| reliable | flag_primary_5_sig | AUROC | 0.567 | 0.542 | -0.025 |
| reliable | flag_primary_5_sig | AUPRC | 0.195 | 0.172 | -0.023 |
| reliable | flag_primary_10_sig | AUROC | 0.601 | 0.557 | -0.044 |
| reliable | flag_primary_10_sig | AUPRC | 0.137 | 0.112 | -0.026 |
| reliable | arm_rate_nonendpoint | MAE | 0.090 | 0.091 | +0.001 |
| reliable | arm_rate_nonendpoint | RMSE | 0.130 | 0.129 | -0.001 |
| reliable | arm_rate_nonendpoint | Spearman | 0.241 | 0.246 | +0.005 |
| reliable | pairwise_nonendpoint | concordance | 0.521 | 0.534 | +0.012 |
| reliable | differential_nonendpoint_absmax | MAE | 0.035 | 0.033 | -0.002 |
| reliable | differential_nonendpoint_signed | MAE | 0.035 | 0.035 | +0.000 |
| reliable | flag_nonendpoint_5_sig | AUROC | 0.564 | 0.551 | -0.013 |
| reliable | flag_nonendpoint_5_sig | AUPRC | 0.170 | 0.161 | -0.009 |
| reliable | flag_nonendpoint_10_sig | AUROC | 0.590 | 0.559 | -0.030 |
| reliable | flag_nonendpoint_10_sig | AUPRC | 0.100 | 0.088 | -0.012 |
| reliable | differential_ae_absmax | MAE | 0.009 | 0.009 | -0.000 |
| reliable | flag_ae_5_sig | AUROC | 0.797 | 0.792 | -0.006 |
| reliable | flag_ae_5_sig | AUPRC | 0.101 | 0.110 | +0.008 |
| reliable | differential_efficacy_absmax | MAE | 0.007 | 0.007 | +0.000 |
| reliable | flag_efficacy_5_sig | AUROC | 0.676 | 0.717 | +0.041 |
| reliable | flag_efficacy_5_sig | AUPRC | 0.103 | 0.108 | +0.005 |
| reliable | composition | mean_KL | 0.923 | 0.920 | -0.003 |
| reliable | composition | mean_Aitchison | 6.386 | 6.301 | -0.085 |

Notes: trial_lgbm predicts one rate per trial, so its predicted differential is 0 for every trial (flag AUROC 0.5, concordance 0.5) and it has no composition model. AE/EFFICACY differentials use predicted group rates = predicted primary rate x predicted group proportion (models with a composition head only). Composition KL uses prediction smoothing eps=1e-06; Aitchison uses multiplicative zero replacement delta=0.001.

Caveat on reason-group flags: these flags are largely determined by the trial's overall level of the reason group. On test, the TRUE trial-mean group rate alone ranks the flag with AUROC ae: 0.965, efficacy: 0.990. Their AUROCs therefore mainly reflect predicting reason-group levels, not which arm is higher; direction is measured by pairwise concordance.
