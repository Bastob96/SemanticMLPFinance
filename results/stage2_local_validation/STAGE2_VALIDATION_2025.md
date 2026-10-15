# Stage 2 validation — 2025 quarterly profile

Q1-Q2 was used for training and Q3 for model selection. Q4 was not loaded by
this phase. The only controlled change is the addition of the LLM factors.

## Validation results

| experiment | model | balanced_accuracy_mean | auc_mean | f1_mean | balanced_accuracy_std |
| --- | --- | --- | --- | --- | --- |
| A_stage1 | logistic_regression | 0.5152 | 0.5818 | 0.0588 |  |
| A_stage1 | random_forest | 0.5126 | 0.4754 | 0.1424 | 0.0167 |
| B_stage1 | logistic_regression | 0.5758 | 0.5939 | 0.2632 |  |
| B_stage1 | random_forest | 0.5086 | 0.4690 | 0.2508 | 0.0432 |
| B_stage2 | logistic_regression | 0.4985 | 0.5475 | 0.4286 |  |
| B_stage2 | random_forest | 0.5152 | 0.4761 | 0.3471 | 0.0197 |
| C_stage1 | logistic_regression | 0.5742 | 0.6121 | 0.3000 |  |
| C_stage1 | random_forest | 0.5354 | 0.4660 | 0.2829 | 0.0179 |
| C_stage2 | logistic_regression | 0.5545 | 0.5727 | 0.5758 |  |
| C_stage2 | random_forest | 0.4561 | 0.4586 | 0.3789 | 0.0066 |
| baseline | majority_class | 0.5000 | 0.5000 | 0.6875 |  |
| baseline | previous_direction | 0.5545 | 0.5545 | 0.5758 |  |

## Effect of adding the factors

| ablation | model | delta_balanced_accuracy | delta_auc | delta_f1 |
| --- | --- | --- | --- | --- |
| B | logistic_regression | -0.0773 | -0.0465 | 0.1654 |
| B | random_forest | 0.0066 | 0.0071 | 0.0962 |
| C | logistic_regression | -0.0197 | -0.0394 | 0.2758 |
| C | random_forest | -0.0793 | -0.0074 | 0.0960 |

## LLM-factor explanation fitted on training data only

| feature | logistic_standardized_coefficient | rf_importance_mean | rf_importance_std |
| --- | --- | --- | --- |
| peer_return_dispersion | -0.4405 | 0.0526 | 0.0068 |
| peer_volume_confirmed_aapl_return | 0.2147 | 0.0473 | 0.0056 |
| aapl_to_qqq_normalized_range_ratio | 0.2173 | 0.0392 | 0.0019 |
| lagged_qqq_relative_momentum | 0.4816 | 0.0372 | 0.0026 |
| aapl_to_qqq_relative_realized_risk | 0.2296 | 0.0313 | 0.0038 |
| peer_signed_return_confirmation | -0.1029 | 0.0301 | 0.0004 |
| aapl_peer_consensus_relative_return | -0.2334 | 0.0295 | 0.0026 |
| relative_return_market_risk_interaction | 0.0328 | 0.0258 | 0.0037 |

Logistic coefficients are directional effects on UP log-odds after
standardisation. Random Forest importance is non-directional and is averaged
over seeds 0, 1 and 2. Neither measure proves causality.

## Locked selection

- Experiment: `B_stage1`
- Model: `logistic_regression`
- Seed: `42`
- Validation balanced accuracy: `0.575758`
- Test status: **locked and not evaluated**
