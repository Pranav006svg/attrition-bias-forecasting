# Hidden differential attrition (descriptive, primary set m80, all splits)

Hidden differential = every experimental-vs-control pair has |total primary-attrition diff| < 5 pp, but some pair has |AE diff| >= 5 pp or |EFFICACY diff| >= 5 pp. Reliable pairs: experimental arm and pooled control both started >= 100. 95% CIs: 1,000 bootstrap resamples of trials (seed 42).

| Subset | Trials | Pairs | Total diff < 5 pp | AE diff >= 5 pp | EFFICACY diff >= 5 pp | **Hidden differential** | Hidden, among total < 5 pp |
|---|---:|---:|---|---|---|---|---|
| all | 15,303 | 21,105 | 54.9% [54.1, 55.7] | 12.7% [12.2, 13.2] | 5.7% [5.3, 6.1] | **3.6% [3.3, 3.9]** | 6.6% [6.0, 7.1] |
| placebo_controlled | 7,399 | 11,133 | 51.6% [50.4, 52.7] | 17.4% [16.5, 18.3] | 6.7% [6.1, 7.3] | **4.2% [3.8, 4.7]** | 8.1% [7.3, 9.0] |
| reliable_pairs | 5,736 | 7,380 | 74.2% [73.1, 75.2] | 8.6% [7.9, 9.3] | 5.7% [5.1, 6.3] | **3.8% [3.3, 4.3]** | 5.1% [4.5, 5.8] |
| placebo_controlled_reliable | 2,553 | 3,499 | 71.9% [70.2, 73.6] | 11.2% [10.0, 12.5] | 7.1% [6.1, 8.1] | **4.8% [4.0, 5.6]** | 6.7% [5.6, 7.8] |

Hidden-differential share by split:

| Subset | train | val | test |
|---|---|---|---|
| all | 4.9% | 3.2% | 1.7% |
| placebo_controlled | 5.5% | 2.9% | 2.2% |
| reliable_pairs | 5.1% | 3.2% | 1.5% |
| placebo_controlled_reliable | 6.2% | 3.1% | 2.5% |

AE vs EFFICACY differential, over experimental-control pairs:

| Subset | Pearson r | Spearman rho | Pairs with AE and EFFICACY diffs >= 5 pp in opposite directions |
|---|---|---|---|
| all | -0.128 (p=1.8e-77) | -0.063 (p=8.8e-20) | 1.5% [1.3, 1.7] |
| placebo_controlled | -0.121 (p=1.3e-37) | -0.068 (p=9.3e-13) | 1.8% [1.5, 2.0] |
| reliable_pairs | -0.214 (p=2.8e-77) | -0.105 (p=1.4e-19) | 1.9% [1.6, 2.2] |
| placebo_controlled_reliable | -0.177 (p=3.8e-26) | -0.105 (p=4.1e-10) | 2.2% [1.7, 2.7] |
