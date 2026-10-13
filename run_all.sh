#!/usr/bin/env bash
# Run every stage of the pipeline in order. Usage: ./run_all.sh  (inside the activated venv)
set -euo pipefail
cd "$(dirname "$0")"
PY="${PYTHON:-python}"
export PYTHONIOENCODING=utf-8

run() { echo; echo "=== $* ==="; "$PY" -m "$@"; }

run src.data.download
run src.data.feasibility
run src.data.extract
run src.data.reasons queue
run src.data.reasons finalize
run src.data.build_benchmark
run src.data.leakage_check
run src.data.dates
if [ -f data/interim/.blind/validation_key.csv ]; then
  run src.eval.agreement
else
  echo; echo "=== src.eval.agreement skipped: blind key not present (see docs/REPRODUCIBILITY.md) ==="
fi
run src.models.baselines
run src.eval.hidden_differential
run src.eval.prior_evidence_experiment
run src.eval.final_analysis
run src.eval.final_cache
run src.eval.recruitment_frontier
run src.eval.shap_analysis
run src.eval.start_year_ablation
run src.eval.final_analysis --summary-only
run src.eval.figures
echo; echo "All stages finished."
