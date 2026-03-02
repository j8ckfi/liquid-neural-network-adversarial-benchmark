# Publishable Benchmark V2 Report

- Datasets: mnist, fashion_mnist, qmnist, cifar10
- Models: mlp, cnn, lstm, ltc
- Defenses: standard, adv_pgd
- Seeds (standard): 41, 42, 43, 44, 45
- Seeds (adv): 41, 42, 43, 44, 45

## MNIST
### Clean Accuracy (mean ± std)
- standard:
  - MLP: 97.66% ± 0.10%
  - CNN: 98.69% ± 0.14%
  - LSTM: 97.60% ± 0.38%
  - LTC: 96.88% ± 0.22%
- adv_pgd:
  - MLP: 91.70% ± 0.23%
  - CNN: 97.56% ± 1.12%
  - LSTM: 94.49% ± 0.68%
  - LTC: 94.52% ± 0.31%
### PGD AUC (higher is better)
- standard: MLP=0.1421, CNN=0.3746, LSTM=0.1611, LTC=0.1453
- adv_pgd: MLP=0.5530, CNN=0.7462, LSTM=0.6717, LTC=0.7124

## FASHION_MNIST
### Clean Accuracy (mean ± std)
- standard:
  - MLP: 88.11% ± 0.17%
  - CNN: 89.49% ± 0.55%
  - LSTM: 87.56% ± 0.27%
  - LTC: 85.88% ± 0.11%
- adv_pgd:
  - MLP: 73.50% ± 0.80%
  - CNN: 79.39% ± 1.45%
  - LSTM: 68.15% ± 1.16%
  - LTC: 66.89% ± 0.67%
### PGD AUC (higher is better)
- standard: MLP=0.1207, CNN=0.1022, LSTM=0.1247, LTC=0.1165
- adv_pgd: MLP=0.4745, CNN=0.5751, LSTM=0.4665, LTC=0.4566

## QMNIST
### Clean Accuracy (mean ± std)
- standard:
  - MLP: 97.37% ± 0.18%
  - CNN: 98.37% ± 0.09%
  - LSTM: 97.67% ± 0.27%
  - LTC: 96.69% ± 0.74%
- adv_pgd:
  - MLP: 91.08% ± 0.37%
  - CNN: 96.13% ± 3.73%
  - LSTM: 93.96% ± 0.84%
  - LTC: 93.60% ± 0.62%
### PGD AUC (higher is better)
- standard: MLP=0.1423, CNN=0.3609, LSTM=0.1699, LTC=0.1465
- adv_pgd: MLP=0.5530, CNN=0.7298, LSTM=0.6744, LTC=0.7037

## CIFAR10
### Clean Accuracy (mean ± std)
- standard:
  - MLP: 48.40% ± 0.46%
  - CNN: 68.24% ± 0.65%
  - LSTM: 55.77% ± 0.35%
  - LTC: 49.46% ± 0.96%
- adv_pgd:
  - MLP: 38.39% ± 0.80%
  - CNN: 51.25% ± 1.57%
  - LSTM: 46.46% ± 0.53%
  - LTC: 39.02% ± 0.56%
### PGD AUC (higher is better)
- standard: MLP=0.1576, CNN=0.1294, LSTM=0.1532, LTC=0.1096
- adv_pgd: MLP=0.2664, CNN=0.3318, LSTM=0.3115, LTC=0.2632

## Headline
- Adversarial training improves PGD robustness across all model families, but no architecture becomes robust by itself.
- LTC vs LSTM remains close under PGD after control for clean accuracy and with/without adversarial training.