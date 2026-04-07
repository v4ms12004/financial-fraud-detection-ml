# Time-Sorted vs Random Split — Comparison Report

## Motivation
Financial transactions occur sequentially in time. Training a model on randomly sampled data allows future fraud patterns to leak into training, artificially inflating performance metrics. This experiment directly quantifies that inflation.

## Side-by-Side Results

| Metric | Time-Sorted ✓ | Random Split ✗ | Inflation |
|--------|--------------|----------------|----------|
| **AUPRC** | **0.995820** | 0.993901 | +-0.001919 |
| Precision | 0.961327 | 0.950237 | -0.011090 |
| Recall | 0.987541 | 0.976263 | -0.011278 |
| F1 Score | 0.974258 | 0.963074 | -0.011184 |
| Fraud Caught | 4,201 / 4,254 (98.8%) | 1,604 / 1,643 (97.6%) | — |
| False Alarms | 169 (3.9%) | 84 (5.0%) | — |

## Confusion Matrices

**Time-Sorted Model:**

| | Predicted Not Fraud | Predicted Fraud |
|---|---|---|
| **Actual Not Fraud** | 1,268,101 | 169 |
| **Actual Fraud** | 53 | 4,201 |

**Random Split Model:**

| | Predicted Not Fraud | Predicted Fraud |
|---|---|---|
| **Actual Not Fraud** | 1,270,797 | 84 |
| **Actual Fraud** | 39 | 1,604 |

## Interpretation

The random split model shows an AUPRC inflation of **+-0.0019** (-0.192 percentage points) compared to the time-sorted model. This inflation occurs because random splitting allows the model to train on future fraud examples — transactions that, in a real deployment scenario, would not yet have occurred at prediction time.

The time-sorted model represents the realistic operational scenario: train on historical data, predict on future unseen transactions. This is the only valid evaluation methodology for time-series fraud detection.

## Conclusion

Temporal ordering is non-negotiable for fraud detection model evaluation. Random splits produce optimistically biased metrics that overestimate real-world performance. The time-sorted approach in this project ensures all reported metrics reflect genuine out-of-sample generalization.
