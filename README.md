# Liquid Neural Network Adversarial Robustness Benchmark

Benchmark comparing `LTC` vs `LSTM` adversarial robustness, with `MLP` and `CNN` baselines, across:
- MNIST
- Fashion-MNIST
- QMNIST
- CIFAR-10

Includes:
- White-box FGSM and PGD robustness curves
- Transferability matrices
- Standard training vs PGD adversarial training
- Multi-seed aggregation
- Contamination/integrity checks
- Full paper draft with citations and seaborn figures

## Main Outputs
- Benchmark code: [`publishable_benchmark_v2.py`](/Users/j8ck/research/publishable_benchmark_v2.py)
- Final benchmark artifacts: [`publishable_v2_full/`](/Users/j8ck/research/publishable_v2_full)
- Contamination checks: [`contamination_checks.py`](/Users/j8ck/research/contamination_checks.py)
- Contamination report: [`paper_artifacts/contamination_report.md`](/Users/j8ck/research/paper_artifacts/contamination_report.md)
- Paper package: [`paper/`](/Users/j8ck/research/paper)

## Environment
This workspace was run with a local virtual environment at `.venv`.

## Quick Re-run
```bash
.venv/bin/python publishable_benchmark_v2.py \
  --out-dir publishable_v2_full \
  --seeds-standard 41,42,43,44,45 \
  --seeds-adv 41,42,43,44,45
```

## Contamination Check
```bash
.venv/bin/python contamination_checks.py \
  --results-dir publishable_v2_full \
  --out-dir paper_artifacts \
  --seeds-standard 41,42,43,44,45 \
  --seeds-adv 41,42,43,44,45
```
