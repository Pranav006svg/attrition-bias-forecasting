@echo off
rem Run every stage of the pipeline in order. Usage: run_all.bat  (inside the activated venv)
setlocal
cd /d "%~dp0"
if "%PYTHON%"=="" set PYTHON=python
set PYTHONIOENCODING=utf-8

call :run src.data.download || goto :fail
call :run src.data.feasibility || goto :fail
call :run src.data.extract || goto :fail
call :run src.data.reasons queue || goto :fail
call :run src.data.reasons finalize || goto :fail
call :run src.data.build_benchmark || goto :fail
call :run src.data.leakage_check || goto :fail
call :run src.data.dates || goto :fail
if exist "data\interim\.blind\validation_key.csv" (
  call :run src.eval.agreement || goto :fail
) else (
  echo === src.eval.agreement skipped: blind key not present, see docs\REPRODUCIBILITY.md ===
)
call :run src.models.baselines || goto :fail
call :run src.eval.hidden_differential || goto :fail
call :run src.eval.prior_evidence_experiment || goto :fail
call :run src.eval.final_analysis || goto :fail
call :run src.eval.final_cache || goto :fail
call :run src.eval.recruitment_frontier || goto :fail
call :run src.eval.shap_analysis || goto :fail
call :run src.eval.start_year_ablation || goto :fail
call :run src.eval.final_analysis --summary-only || goto :fail
call :run src.eval.figures || goto :fail
echo.
echo All stages finished.
exit /b 0

:run
echo.
echo === %* ===
"%PYTHON%" -m %*
exit /b %errorlevel%

:fail
echo Pipeline stopped: the previous stage failed.
exit /b 1
