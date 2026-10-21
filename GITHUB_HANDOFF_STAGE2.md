# Kevin — Stage 2 training handoff

## Completed

- Unified both training runners on the canonical OHLCV snapshot and quarterly
  121/63/63 split.
- Added the eight accepted LLM factors to the existing classification pipeline.
- Compared A/B/C Stage 1 controls with B + 4 and C + 8 factor variants.
- Reused Logistic Regression seed 42 and Random Forest seeds 0, 1 and 2.
- Preserved Abu's prediction schema and reused the shared evaluator/backtester.
- Locked selection on Q3 before evaluating Q4 once.
- Added train-only LR coefficients and RF feature importance for explanation.
- Preserved Taiyu's separate 70/15/15 return-regression experiment unchanged.

## Commands

```bash
python experiments/run_stage2_training.py --phase validation
python experiments/run_stage2_training.py --phase final --confirm-locked-test
python -m unittest discover -s tests -v
python -m unittest discover -s Stage2_Factor_Files/tests -v
```

The included lock is marked `evaluated_once_after_lock`; do not rerun Q4 for
tuning. A new confirmatory iteration needs a fresh held-out period.

## Result

Adding all eight candidates to the 22-feature C baseline reduced Q3 balanced
accuracy from 0.5742 to 0.5545 for Logistic Regression and from 0.5293 to 0.4848
for Random Forest. Q4 changes were positive but remained below the
previous-direction baseline. The factors therefore did not demonstrate a
reliable improvement for 2025.

## Reproducibility

The recorded run used Python 3.12.14, pandas 2.2.3, NumPy 2.3.5 and
scikit-learn 1.8.0. Random Forest values can vary across scikit-learn versions,
so keep `stage2_final_run_info.json` with the results.
