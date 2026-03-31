# PaySim vs Synthetic — Fraud Detection Comparison

## Research Question
> Is the model detecting genuine fraud behavior, or has it overfit to the internal rules of the PaySim simulator?

## Side-by-Side Metrics

| Metric | PaySim | Synthetic | Difference |
|--------|--------|-----------|------------|
| **AUPRC** | 0.998759 | 1.000000 | 0.001241 |
| Precision | 0.982185 | 1.000000 | 0.017815 |
| Recall | 1.000000 | 1.000000 | 0.000000 |
| F1 Score | 0.991013 | 1.000000 | 0.008987 |

**AUPRC Drop: -0.12%**

## Verdict

**GENERALIZES WELL**

The model maintains strong AUPRC on the synthetic dataset (drop of only -0.1%). This suggests the model is detecting genuine fraud signals — specifically the balance-depletion pattern, account prefix structure, and panel behavioral features — rather than PaySim simulator artifacts. The fraud detection pipeline appears robust and likely to generalize beyond PaySim-generated data.

## Temporal Sensitivity (Synthetic)

| train_frac_end | test_width | train_rows | test_rows | test_auprc | test_precision | test_recall | test_f1 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.3000 | 0.1000 | 149897.0000 | 49938.0000 | 0.9999 | 0.9925 | 1.0000 | 0.9963 |
| 0.6000 | 0.1000 | 299762.0000 | 50011.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| 0.9000 | 0.1000 | 449735.0000 | 49933.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

## Confusion Matrix (Synthetic test set)

| | Predicted 0 | Predicted 1 |
|---|---|---|
| **Actual 0** | 99,610 | 0 |
| **Actual 1** | 0 | 316 |

## Model Settings (identical for both runs)

- `n_estimators`: 500
- `max_depth`: 6
- `learning_rate`: 0.08
- `train_frac`: 0.8
- `random_state`: 42
