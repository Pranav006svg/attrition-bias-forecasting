# Baselines (test split, primary benchmark m80)

Train/val/test trials: 8,935 / 1,961 / 5,424; arms: 22,041 / 4,810 / 13,485. Values are point estimates with 95% bootstrap CIs (1,000 trial-grouped resamples, seed 42). Models were fit on train with hyperparameters selected on val; test was used once.

## Arm attrition rate

| Target | Metric | global_mean | reference_class | trial_lgbm | flat_lgbm |
|---|---|---|---|---|---|
| arm_rate_primary | MAE | 0.135 [0.132, 0.138] | 0.128 [0.124, 0.131] | 0.112 [0.109, 0.115] | 0.111 [0.108, 0.114] |
| arm_rate_primary | RMSE | 0.182 [0.176, 0.189] | 0.172 [0.166, 0.177] | 0.160 [0.154, 0.165] | 0.160 [0.154, 0.166] |
| arm_rate_primary | Spearman | n/a | 0.220 [0.195, 0.245] | 0.281 [0.254, 0.308] | 0.295 [0.267, 0.323] |
| arm_rate_nonendpoint | MAE | 0.108 [0.106, 0.111] | 0.107 [0.104, 0.109] | 0.099 [0.097, 0.102] | 0.098 [0.095, 0.101] |
| arm_rate_nonendpoint | RMSE | 0.145 [0.139, 0.150] | 0.143 [0.138, 0.148] | 0.140 [0.135, 0.146] | 0.141 [0.135, 0.147] |
| arm_rate_nonendpoint | Spearman | n/a | 0.169 [0.144, 0.193] | 0.213 [0.186, 0.239] | 0.208 [0.182, 0.235] |

## Reason composition (arms with >= 1 primary withdrawal; lower is better)

| Target | Metric | global_mean | reference_class | trial_lgbm | flat_lgbm |
|---|---|---|---|---|---|
| composition | mean_KL | 1.330 [1.312, 1.351] | 1.236 [1.216, 1.259] | n/a | 1.103 [1.083, 1.125] |
| composition | mean_Aitchison | 6.683 [6.662, 6.704] | 6.861 [6.838, 6.883] | n/a | 6.607 [6.581, 6.633] |

## Trial differential (trials with a computable differential)

| Target | Metric | global_mean | reference_class | trial_lgbm | flat_lgbm |
|---|---|---|---|---|---|
| differential_primary_absmax | MAE | 0.069 [0.066, 0.071] | 0.059 [0.057, 0.062] | 0.069 [0.066, 0.071] | 0.066 [0.063, 0.068] |
| differential_primary_signed | MAE | 0.064 [0.062, 0.066] | 0.067 [0.065, 0.069] | 0.064 [0.062, 0.066] | 0.064 [0.062, 0.066] |
| differential_nonendpoint_absmax | MAE | 0.065 [0.063, 0.067] | 0.057 [0.055, 0.060] | 0.065 [0.063, 0.067] | 0.063 [0.061, 0.065] |
| differential_nonendpoint_signed | MAE | 0.061 [0.059, 0.063] | 0.064 [0.062, 0.066] | 0.061 [0.059, 0.063] | 0.061 [0.059, 0.063] |

## Differential-attrition flags (score = predicted |absmax differential|)

| Target | Metric | global_mean | reference_class | trial_lgbm | flat_lgbm |
|---|---|---|---|---|---|
| flag_primary_5_sig | AUROC | 0.500 [0.500, 0.500] | 0.518 [0.493, 0.546] | 0.500 [0.500, 0.500] | 0.555 [0.529, 0.581] |
| flag_primary_5_sig | AUPRC | 0.106 [0.098, 0.115] | 0.119 [0.106, 0.137] | 0.106 [0.098, 0.115] | 0.132 [0.117, 0.151] |
| flag_primary_10_sig | AUROC | 0.500 [0.500, 0.500] | 0.524 [0.497, 0.556] | 0.500 [0.500, 0.500] | 0.562 [0.533, 0.591] |
| flag_primary_10_sig | AUPRC | 0.081 [0.075, 0.089] | 0.093 [0.082, 0.111] | 0.081 [0.075, 0.089] | 0.108 [0.092, 0.129] |
| flag_nonendpoint_5_sig | AUROC | 0.500 [0.500, 0.500] | 0.508 [0.480, 0.536] | 0.500 [0.500, 0.500] | 0.544 [0.514, 0.572] |
| flag_nonendpoint_5_sig | AUPRC | 0.098 [0.089, 0.106] | 0.105 [0.094, 0.121] | 0.098 [0.089, 0.106] | 0.119 [0.106, 0.139] |
| flag_nonendpoint_10_sig | AUROC | 0.500 [0.500, 0.500] | 0.504 [0.474, 0.536] | 0.500 [0.500, 0.500] | 0.545 [0.514, 0.578] |
| flag_nonendpoint_10_sig | AUPRC | 0.073 [0.066, 0.081] | 0.077 [0.069, 0.092] | 0.073 [0.066, 0.081] | 0.092 [0.080, 0.110] |

Notes: trial_lgbm predicts one rate per trial, so its predicted differential is 0 for every trial (flag AUROC undefined -> n/a or 0.5) and it has no composition model. Composition KL uses prediction smoothing eps=1e-06; Aitchison uses multiplicative zero replacement delta=0.001.
