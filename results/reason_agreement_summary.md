# Reason-normalization agreement (human vs pipeline) - blind v2

Sample of 100 raw reason strings (34 rule / 33 llm / 33 knn, frequency-stratified). 95% CIs from 1,000 bootstrap resamples (seed 42); accuracy also with exact Clopper-Pearson intervals, which remain informative when a group scores 100%.

| Group | n | Accuracy (exact Clopper-Pearson CI) | Accuracy (bootstrap CI) | Cohen's kappa | Macro-F1 | Primary-inclusion agreement | Nonendpoint-inclusion agreement |
|---|---:|---|---|---|---|---|---|
| all | 100 | 0.980 [0.930, 0.998] | 0.980 [0.950, 1.000] | 0.976 [0.938, 1.000] | 0.986 [0.956, 1.000] | 0.990 [0.970, 1.000] | 0.990 [0.970, 1.000] |
| rule | 34 | 1.000 [0.897, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |
| llm | 33 | 0.939 [0.798, 0.993] | 0.939 [0.848, 1.000] | 0.922 [0.800, 1.000] | 0.950 [0.827, 1.000] | 0.970 [0.909, 1.000] | 0.970 [0.909, 1.000] |
| knn | 33 | 1.000 [0.894, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |

Dataset-level accuracy (tier accuracies weighted by tier coverage; stratified bootstrap):

- weighted by entries: 0.997 [0.991, 1.000]
- weighted by subjects: 0.981 [0.952, 1.000]

Disagreements: 2 of 100 (see results/tables/reason_agreement_disagreements.csv).

Most frequent (human -> pipeline) confusions:

- PRE_TREATMENT_EXCLUSION -> OTHER_UNSPECIFIED: 1
- SPONSOR_ADMINISTRATIVE -> PRE_TREATMENT_EXCLUSION: 1
