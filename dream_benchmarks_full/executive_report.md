# Dream Benchmarks Executive Report

- Datasets: mnist, fashion_mnist, qmnist
- Models: mlp, cnn, lstm, ltc
- Seeds: 41, 42
- Epsilons: 0.00, 0.05, 0.10, 0.15, 0.20, 0.30
- Transfer epsilon: 0.20

## Clean Accuracy (mean ± std)
### MNIST
- MLP: 97.61% ± 0.18%
- CNN: 99.11% ± 0.04%
- LSTM: 97.70% ± 0.21%
- LTC: 97.92% ± 0.26%

### FASHION_MNIST
- MLP: 88.16% ± 0.20%
- CNN: 91.42% ± 0.45%
- LSTM: 86.81% ± 0.21%
- LTC: 86.85% ± 0.39%

### QMNIST
- MLP: 97.51% ± 0.02%
- CNN: 98.83% ± 0.04%
- LSTM: 97.55% ± 0.22%
- LTC: 97.60% ± 0.11%

## Headline
- Across datasets and model families, PGD substantially degrades all models.
- Architectural differences shift transfer/corruption profiles, but do not remove gradient-based vulnerability.