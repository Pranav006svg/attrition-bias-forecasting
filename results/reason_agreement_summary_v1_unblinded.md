# Reason-normalization agreement (human vs pipeline) - UNBLINDED PILOT v1 (annotator had seen the key)

> **UNBLINDED PILOT (v1).** The annotator had seen the pipeline key before labelling, so these numbers are not a blind validation and must be reported only as a pilot. The blind validation is v2 (annotation/validation_sample.csv).

Sample of 100 raw reason strings (34 rule / 33 llm / 33 knn, frequency-stratified). 95% CIs from 1,000 bootstrap resamples (seed 42); accuracy also with exact Clopper-Pearson intervals, which remain informative when a group scores 100%.

| Group | n | Accuracy (exact Clopper-Pearson CI) | Accuracy (bootstrap CI) | Cohen's kappa | Macro-F1 | Primary-inclusion agreement | Nonendpoint-inclusion agreement |
|---|---:|---|---|---|---|---|---|
| all | 100 | 0.940 [0.874, 0.978] | 0.940 [0.890, 0.980] | 0.929 [0.870, 0.977] | 0.931 [0.868, 0.971] | 0.980 [0.950, 1.000] | 0.980 [0.950, 1.000] |
| rule | 34 | 0.971 [0.847, 0.999] | 0.971 [0.882, 1.000] | 0.964 [0.887, 1.000] | 0.961 [0.852, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |
| llm | 33 | 0.939 [0.798, 0.993] | 0.939 [0.848, 1.000] | 0.926 [0.807, 1.000] | 0.874 [0.765, 1.000] | 0.970 [0.909, 1.000] | 0.970 [0.909, 1.000] |
| knn | 33 | 0.909 [0.757, 0.981] | 0.909 [0.818, 1.000] | 0.868 [0.702, 1.000] | 0.909 [0.756, 1.000] | 0.970 [0.909, 1.000] | 0.970 [0.909, 1.000] |

Dataset-level accuracy (tier accuracies weighted by tier coverage; stratified bootstrap):

- weighted by entries: 0.960 [0.901, 0.992]
- weighted by subjects: 0.960 [0.910, 0.998]

Disagreements: 6 of 100 (see results/tables/reason_agreement_disagreements_v1_unblinded.csv).

Most frequent (human -> pipeline) confusions:

- ADVERSE_EVENT -> OTHER_UNSPECIFIED: 2
- SPONSOR_ADMINISTRATIVE -> OTHER_UNSPECIFIED: 2
- ADVERSE_EVENT -> PRE_TREATMENT_EXCLUSION: 1
- WITHDRAWAL_BY_SUBJECT -> ADVERSE_EVENT: 1
