# Reproducibility

This document describes how the reported results are produced and how to check that a rerun matches them.

## Environment

- **Python:** 3.11. Pinned direct dependencies are in [requirements.txt](../requirements.txt).
- **Hardware used for the reported results:** Intel Core i9-14900HX (32 logical cores) with 32 GB RAM, running Windows 11. No GPU is required. Sentence embeddings for the nearest-neighbour reason tier run on the CPU.
- **Approximate runtimes on that machine:**
  - benchmark build: ~20 min;
  - start-year ablation: ~30 min;
  - download: depends on network speed (76 gzipped pages, ~580 MB).

## Randomness

- `seed = 42` is used everywhere: NumPy generators, LightGBM, bootstrap resampling, validation-sample drawing, the random trial-grouped split and `torch.manual_seed` for embeddings.
- **Bootstrap CIs:** 1,000 resamples of trials (trial-grouped), percentile intervals. Paired comparisons resample trials jointly for both models.
- **p-values:** two-sided Wald tests using the bootstrap standard error of the paired difference. Percentile-bootstrap p-values (add-one corrected) are also saved in `results/tables/final_paired_tests.csv`.

## Splits

Split definitions and trial IDs are in [`data/processed/splits.json`](../data/processed/splits.json).

| Split | Rule (trial start year) | Trials (m80) | Arms (m80) |
|---|---|---:|---:|
| train | < 2015 | 8,935 | 22,041 |
| validation | 2015–2016 | 1,961 | 4,810 |
| test | ≥ 2017 | 5,424 | 13,485 |

- **Fitting and tuning:** models are fit on train, and hyperparameters and early stopping use validation.
- **Hyperparameter grid:** LightGBM `num_leaves` ∈ {15, 63} × `min_child_samples` ∈ {20, 100} × `objective` ∈ {regression, cross_entropy}. Selected values are in `results/tables/*selected_params.json`.
- **Sensitivity variants:**
  - matched sets m70 and m90 (`data/processed/*_m70.parquet`, `*_m90.parquet`);
  - a random trial-grouped split of m80 that keeps the temporal split sizes. It is generated in `src/eval/final_analysis.py::load_set` with seed 42.
- **Prior evidence** is time-respecting under every split. Only trials whose results were first posted strictly before the target trial's start date are used, and the code asserts this.

## Leakage check

```bash
python -m src.data.leakage_check
```

The check prints `LEAKAGE CHECK PASSED` and exits with code 0 only if all of the following hold:

1. Every feature column in `data/processed/trials*.parquet` and `arms*.parquet` is on the pre-enrollment allowlist. Labels (`y_*`), bookkeeping (`meta_*`) and identifiers are excluded from features by convention.
2. No column name matches the forbidden patterns: enrollment, overall status, why stopped, completion dates, results, started/withdrawn counts, flow and milestones.
3. The fitted `FeatureBuilder` (base and enriched variants) produces no forbidden column when forbidden columns are injected into its input.

The run fails with a list of offending columns if any condition is violated.

## Reason-normalization agreement

- `annotation/validation_sample.csv` holds the 100 human labels from the blind v2 sample.
- The pipeline key (tier and predicted category) was kept out of view during labelling. It lives only in the git-ignored `data/interim/.blind/validation_key.csv`, so `python -m src.eval.agreement` needs that local file.
- The published results are in `results/reason_agreement_summary.md` and `results/tables/reason_agreement*.csv`.
- `python -m src.data.reasons sample` will not overwrite a labelled sample.
- The earlier v1 sample (`annotation/validation_sample_v1_unblinded.csv`) is reported only as an unblinded pilot.

## Expected outputs

After `run_all`, these files are regenerated. Reference values are from the committed results.

| File | Check |
|---|---|
| `results/feasibility_summary.md` | Usable trials: 24,473 |
| `results/tables/dataset_stats.csv` | `overall,trials` = 16,320; `overall,arms` = 40,336 |
| `results/tables/reason_coverage.csv` | Subject share: rule 0.665, llm 0.315, knn 0.020 |
| `results/tables/final_estimates.csv` | `all, arm_rate_primary, MAE, FINAL` = 0.11027892106764998 |
| `results/tables/final_estimates.csv` | `all, composition, mean_KL, FINAL` = 1.1042216989241302 |
| `results/tables/final_conformal.csv` | `mondrian, all` coverage = 0.9088616981831665 |
| `results/final_summary.md` | Primary endpoints, paired tests, conformal, planning, frontier, SHAP, ablation, sensitivity |
| `results/figures/` | F1–F9 and S1 (PNG and PDF) |

**Platform differences.** Results are deterministic on a given platform and library version. LightGBM and BLAS can produce last-digit floating-point differences across operating systems and CPUs. These leave the reported 3-decimal values unchanged, but they can change the last digits of very small p-values.

## Data provenance

- **Raw data:** ClinicalTrials.gov API v2, query `AREA[StudyType]INTERVENTIONAL AND AREA[HasResults]true`, page size 1000. The download records each page's `nextPageToken` and can resume. Because the registry is updated continuously, a fresh download will include trials posted after the reported snapshot (75,748 studies), and counts will differ.
- **Exact replication:** use the committed `data/processed/` tables and start from `python -m src.models.baselines`. The prior-evidence features also need `data/interim/trial_dates.parquet`. `src.data.dates` writes it from the raw download.
