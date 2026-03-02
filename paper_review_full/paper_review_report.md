# MNIST LSTM vs LTC Adversarial Paper Review

## Scope
- Base experiment: clean MNIST performance + FGSM/PGD robustness + transferability.
- Multi-seed protocol: 5 seeds (41, 42, 43, 44, 45).
- Epsilon grid: 0.00, 0.05, 0.10, 0.15, 0.20, 0.30.

## Clean Accuracy Fairness Check
- LSTM clean accuracy (mean across seeds): 97.55%
- LTC clean accuracy (mean across seeds): 97.77%
- Mean clean gap (LTC-LSTM): 0.22 percentage points

## Main Robustness Findings
- FGSM white-box AUC:
  - LSTM: 0.3519 ± 0.0109
  - LTC: 0.3442 ± 0.0212
- PGD white-box AUC:
  - LSTM: 0.1780 ± 0.0070
  - LTC: 0.1784 ± 0.0127
- Transferability remains substantial, but weaker than white-box (see transfer plot).

## Additional Executed Analyses
1. **Targeted PGD (next-class target)**
   - Added white-box and transfer targeted success curves.
   - High epsilon targeted success increases sharply for both models.
2. **Corruption Robustness Suite**
   - Added 3 corruption families: Gaussian noise, Gaussian blur, center occlusion.
   - Robustness trends differ from adversarial trends (especially under blur/occlusion).
3. **Gradient Geometry Diagnostics**
   - Measured input-gradient L2 norms, inter-model gradient cosine similarity, and sign agreement.
   - Used to explain transfer asymmetry.

## Statistical Difference Table (LTC-LSTM, robust accuracy)
| Attack | Epsilon | Mean Diff (LTC-LSTM) | 95% CI |
|---|---:|---:|---:|
| FGSM | 0.00 | 0.22 pp | [0.05, 0.39] pp |
| FGSM | 0.05 | 3.19 pp | [-2.44, 6.45] pp |
| FGSM | 0.10 | 0.93 pp | [-4.08, 4.39] pp |
| FGSM | 0.15 | -2.28 pp | [-3.81, -1.17] pp |
| FGSM | 0.20 | -2.94 pp | [-4.08, -1.86] pp |
| FGSM | 0.30 | -2.14 pp | [-2.99, -1.63] pp |
| PGD | 0.00 | 0.22 pp | [0.05, 0.39] pp |
| PGD | 0.05 | 2.58 pp | [-3.98, 8.18] pp |
| PGD | 0.10 | -1.83 pp | [-3.79, -0.19] pp |
| PGD | 0.15 | -0.67 pp | [-1.09, -0.24] pp |
| PGD | 0.20 | 0.01 pp | [-0.06, 0.11] pp |
| PGD | 0.30 | 0.05 pp | [0.00, 0.13] pp |

## Reproducibility Artifacts
- `seed_results.json`, `aggregate_summary.json`, `robustness_stats.csv`
- `whitebox_mean_std.png`, `transfer_mean_std.png`, `targeted_pgd_success.png`
- `corruption_suite.png`, `gradient_geometry.png`
