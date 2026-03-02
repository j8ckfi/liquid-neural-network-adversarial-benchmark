#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from tqdm import tqdm


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(device_name: str) -> torch.device:
    if device_name == "cpu":
        return torch.device("cpu")
    if device_name == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        raise RuntimeError("Requested CUDA, but CUDA is not available.")
    if device_name == "mps":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        raise RuntimeError("Requested MPS, but MPS is not available.")
    if device_name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    raise ValueError(f"Unsupported device '{device_name}'")


class LSTMClassifier(nn.Module):
    def __init__(self, hidden_size: int = 128, num_layers: int = 1, dropout: float = 0.0) -> None:
        super().__init__()
        self.rnn = nn.LSTM(
            input_size=28,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden_size, 10)

    def forward(self, x: Tensor) -> Tensor:
        if x.dim() == 4:
            x = x.squeeze(1)
        out, _ = self.rnn(x)
        return self.head(out[:, -1, :])


class LTCCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.ff_in = nn.Linear(input_size, hidden_size)
        self.ff_rec = nn.Linear(hidden_size, hidden_size, bias=False)
        self.ff_bias = nn.Parameter(torch.zeros(hidden_size))

        self.gate_in = nn.Linear(input_size, hidden_size)
        self.gate_rec = nn.Linear(hidden_size, hidden_size, bias=False)
        self.gate_bias = nn.Parameter(torch.zeros(hidden_size))

        self.tau_in = nn.Linear(input_size, hidden_size)
        self.tau_rec = nn.Linear(hidden_size, hidden_size, bias=False)
        self.tau_bias = nn.Parameter(torch.zeros(hidden_size))

        self.tau_min = 0.5
        self.tau_max = 2.0

    def forward(self, x_t: Tensor, h: Tensor, dt: float = 1.0) -> Tensor:
        candidate = torch.tanh(self.ff_in(x_t) + self.ff_rec(h) + self.ff_bias)
        gate = torch.sigmoid(self.gate_in(x_t) + self.gate_rec(h) + self.gate_bias)
        tau_gate = torch.sigmoid(self.tau_in(x_t) + self.tau_rec(h) + self.tau_bias)
        tau = self.tau_min + (self.tau_max - self.tau_min) * tau_gate
        dh = (-h + gate * candidate) / tau
        return torch.tanh(h + dt * dh)


class LTCClassifier(nn.Module):
    def __init__(self, hidden_size: int = 128) -> None:
        super().__init__()
        self.cell = LTCCell(input_size=28, hidden_size=hidden_size)
        self.head = nn.Linear(hidden_size, 10)

    def forward(self, x: Tensor) -> Tensor:
        if x.dim() == 4:
            x = x.squeeze(1)
        batch_size, seq_len, _ = x.shape
        h = torch.zeros(batch_size, self.cell.hidden_size, device=x.device, dtype=x.dtype)
        for t in range(seq_len):
            h = self.cell(x[:, t, :], h)
        return self.head(h)


@dataclass
class TrainConfig:
    lr: float
    max_epochs: int
    min_epochs: int
    target_val_acc: float


def train_one_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer, device: torch.device) -> float:
    model.train()
    criterion = nn.CrossEntropyLoss()
    running_loss = 0.0
    total = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        if not torch.isfinite(loss):
            continue
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        running_loss += loss.item() * y.size(0)
        total += y.size(0)
    return running_loss / max(total, 1)


@torch.no_grad()
def evaluate_accuracy(model: nn.Module, loader: DataLoader, device: torch.device, max_samples: int | None = None) -> float:
    model.eval()
    correct = 0
    total = 0
    for x, y in loader:
        if max_samples is not None:
            remaining = max_samples - total
            if remaining <= 0:
                break
            if y.size(0) > remaining:
                x = x[:remaining]
                y = y[:remaining]
        x = x.to(device)
        y = y.to(device)
        preds = model(x).argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)
    return correct / max(total, 1)


def fit_model(
    name: str,
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    cfg: TrainConfig,
) -> Dict[str, float]:
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    best_val = 0.0
    best_state = None
    history: Dict[str, float] = {}

    for epoch in range(1, cfg.max_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_acc = evaluate_accuracy(model, val_loader, device)
        history[f"epoch_{epoch}_train_loss"] = train_loss
        history[f"epoch_{epoch}_val_acc"] = val_acc
        if val_acc > best_val:
            best_val = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(f"[{name}] epoch {epoch:02d} train_loss={train_loss:.4f} val_acc={val_acc * 100:.2f}%")
        if epoch >= cfg.min_epochs and val_acc >= cfg.target_val_acc:
            print(f"[{name}] early stop at epoch {epoch} (val_acc reached target {cfg.target_val_acc * 100:.2f}%)")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    history["best_val_acc"] = best_val
    return history


def continue_training_if_needed(
    weaker_name: str,
    weaker_model: nn.Module,
    stronger_val_acc: float,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    lr: float,
    max_extra_epochs: int,
    max_gap: float,
) -> float:
    optimizer = torch.optim.Adam(weaker_model.parameters(), lr=lr * 0.7)
    best_val = evaluate_accuracy(weaker_model, val_loader, device)
    best_state = {k: v.detach().cpu().clone() for k, v in weaker_model.state_dict().items()}

    for epoch in range(1, max_extra_epochs + 1):
        _ = train_one_epoch(weaker_model, train_loader, optimizer, device)
        val_acc = evaluate_accuracy(weaker_model, val_loader, device)
        print(f"[{weaker_name}] fairness_tune epoch {epoch:02d} val_acc={val_acc * 100:.2f}%")
        if val_acc > best_val:
            best_val = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in weaker_model.state_dict().items()}
        if (stronger_val_acc - val_acc) <= max_gap:
            break

    weaker_model.load_state_dict(best_state)
    return best_val


def fgsm_attack(model: nn.Module, x: Tensor, y: Tensor, eps: float) -> Tensor:
    if eps == 0.0:
        return x.detach()
    x_adv = x.detach().clone().requires_grad_(True)
    logits = model(x_adv)
    loss = F.cross_entropy(logits, y)
    grad = torch.autograd.grad(loss, x_adv, only_inputs=True)[0]
    x_adv = x_adv + eps * grad.sign()
    return torch.clamp(x_adv, 0.0, 1.0).detach()


def pgd_attack(model: nn.Module, x: Tensor, y: Tensor, eps: float, alpha: float, steps: int) -> Tensor:
    if eps == 0.0:
        return x.detach()
    x_orig = x.detach()
    x_adv = x_orig + torch.empty_like(x_orig).uniform_(-eps, eps)
    x_adv = torch.clamp(x_adv, 0.0, 1.0)
    for _ in range(steps):
        x_adv.requires_grad_(True)
        logits = model(x_adv)
        loss = F.cross_entropy(logits, y)
        grad = torch.autograd.grad(loss, x_adv, only_inputs=True)[0]
        x_adv = x_adv.detach() + alpha * grad.sign()
        x_adv = torch.max(torch.min(x_adv, x_orig + eps), x_orig - eps)
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    return x_adv.detach()


@torch.no_grad()
def evaluate_on_given_inputs(model: nn.Module, x: Tensor, y: Tensor) -> Tuple[int, int]:
    preds = model(x).argmax(dim=1)
    correct = (preds == y).sum().item()
    return correct, y.size(0)


def evaluate_attack_accuracy(
    source_model: nn.Module,
    target_model: nn.Module,
    loader: DataLoader,
    epsilons: Iterable[float],
    attack_name: str,
    device: torch.device,
    pgd_steps: int,
    pgd_alpha_scale: float,
    max_samples: int | None = None,
) -> Dict[float, float]:
    source_model.eval()
    target_model.eval()
    results: Dict[float, float] = {}

    for eps in epsilons:
        correct = 0
        total = 0
        for x, y in loader:
            if max_samples is not None:
                remaining = max_samples - total
                if remaining <= 0:
                    break
                if y.size(0) > remaining:
                    x = x[:remaining]
                    y = y[:remaining]
            x = x.to(device)
            y = y.to(device)
            if attack_name == "fgsm":
                x_adv = fgsm_attack(source_model, x, y, eps)
            elif attack_name == "pgd":
                alpha = max(eps * pgd_alpha_scale, 1e-4)
                x_adv = pgd_attack(source_model, x, y, eps, alpha=alpha, steps=pgd_steps)
            else:
                raise ValueError(f"Unsupported attack '{attack_name}'")
            batch_correct, batch_total = evaluate_on_given_inputs(target_model, x_adv, y)
            correct += batch_correct
            total += batch_total
        results[eps] = correct / max(total, 1)
        print(f"[{attack_name}] eps={eps:.3f} acc={results[eps] * 100:.2f}%")
    return results


def save_csv(path: Path, rows: List[Dict[str, str]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_curve(path: Path, eps: List[float], lstm_acc: List[float], ltc_acc: List[float], title: str) -> None:
    plt.figure(figsize=(7, 5))
    plt.plot(eps, lstm_acc, marker="o", label="LSTM")
    plt.plot(eps, ltc_acc, marker="s", label="LTC")
    plt.xlabel("Epsilon")
    plt.ylabel("Accuracy")
    plt.ylim(0.0, 1.0)
    plt.grid(True, alpha=0.3)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def plot_transfer(
    path: Path,
    eps: List[float],
    fgsm_lstm_to_ltc: List[float],
    fgsm_ltc_to_lstm: List[float],
    pgd_lstm_to_ltc: List[float],
    pgd_ltc_to_lstm: List[float],
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    axes[0].plot(eps, fgsm_lstm_to_ltc, marker="o", label="LSTM -> LTC")
    axes[0].plot(eps, fgsm_ltc_to_lstm, marker="s", label="LTC -> LSTM")
    axes[0].set_title("FGSM Transfer")
    axes[0].set_xlabel("Epsilon")
    axes[0].set_ylabel("Accuracy")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(eps, pgd_lstm_to_ltc, marker="o", label="LSTM -> LTC")
    axes[1].plot(eps, pgd_ltc_to_lstm, marker="s", label="LTC -> LSTM")
    axes[1].set_title("PGD Transfer")
    axes[1].set_xlabel("Epsilon")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    axes[0].set_ylim(0.0, 1.0)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MNIST LSTM vs LTC adversarial robustness experiment")
    parser.add_argument("--data-dir", type=Path, default=Path("./data"))
    parser.add_argument("--out-dir", type=Path, default=Path("./results"))
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lstm-hidden", type=int, default=128)
    parser.add_argument("--ltc-hidden", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max-epochs", type=int, default=12)
    parser.add_argument("--min-epochs", type=int, default=5)
    parser.add_argument("--target-val-acc", type=float, default=0.975)
    parser.add_argument("--fairness-gap", type=float, default=0.01)
    parser.add_argument("--fairness-extra-epochs", type=int, default=6)
    parser.add_argument("--epsilons", type=str, default="0.0,0.05,0.1,0.15,0.2,0.3")
    parser.add_argument("--pgd-steps", type=int, default=20)
    parser.add_argument("--pgd-alpha-scale", type=float, default=0.25)
    parser.add_argument("--attack-max-samples", type=int, default=10000)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, choices=["auto", "cpu", "cuda", "mps"], default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    device = get_device(args.device)
    print(f"Using device: {device}")

    transform = transforms.ToTensor()
    train_full = datasets.MNIST(root=args.data_dir, train=True, download=True, transform=transform)
    test_set = datasets.MNIST(root=args.data_dir, train=False, download=True, transform=transform)
    train_set, val_set = random_split(
        train_full,
        [55000, 5000],
        generator=torch.Generator().manual_seed(args.seed),
    )

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    cfg = TrainConfig(
        lr=args.lr,
        max_epochs=args.max_epochs,
        min_epochs=args.min_epochs,
        target_val_acc=args.target_val_acc,
    )

    lstm = LSTMClassifier(hidden_size=args.lstm_hidden).to(device)
    ltc = LTCClassifier(hidden_size=args.ltc_hidden).to(device)

    print("\nTraining LSTM...")
    lstm_hist = fit_model("LSTM", lstm, train_loader, val_loader, device, cfg)
    print("\nTraining LTC...")
    ltc_hist = fit_model("LTC", ltc, train_loader, val_loader, device, cfg)

    lstm_val = lstm_hist["best_val_acc"]
    ltc_val = ltc_hist["best_val_acc"]
    gap = abs(lstm_val - ltc_val)
    print(f"\nInitial val accuracy gap: {gap * 100:.2f}%")

    if gap > args.fairness_gap:
        if lstm_val < ltc_val:
            print("\nFairness tuning LSTM to reduce clean accuracy gap...")
            lstm_val = continue_training_if_needed(
                weaker_name="LSTM",
                weaker_model=lstm,
                stronger_val_acc=ltc_val,
                train_loader=train_loader,
                val_loader=val_loader,
                device=device,
                lr=args.lr,
                max_extra_epochs=args.fairness_extra_epochs,
                max_gap=args.fairness_gap,
            )
        else:
            print("\nFairness tuning LTC to reduce clean accuracy gap...")
            ltc_val = continue_training_if_needed(
                weaker_name="LTC",
                weaker_model=ltc,
                stronger_val_acc=lstm_val,
                train_loader=train_loader,
                val_loader=val_loader,
                device=device,
                lr=args.lr,
                max_extra_epochs=args.fairness_extra_epochs,
                max_gap=args.fairness_gap,
            )

    clean_lstm = evaluate_accuracy(lstm, test_loader, device)
    clean_ltc = evaluate_accuracy(ltc, test_loader, device)
    print(f"\nFinal clean test accuracy: LSTM={clean_lstm * 100:.2f}% LTC={clean_ltc * 100:.2f}%")

    epsilons = [float(x.strip()) for x in args.epsilons.split(",") if x.strip()]

    print("\nWhite-box FGSM (each model attacked/evaluated on itself)...")
    fgsm_lstm = evaluate_attack_accuracy(
        source_model=lstm,
        target_model=lstm,
        loader=test_loader,
        epsilons=epsilons,
        attack_name="fgsm",
        device=device,
        pgd_steps=args.pgd_steps,
        pgd_alpha_scale=args.pgd_alpha_scale,
        max_samples=args.attack_max_samples,
    )
    fgsm_ltc = evaluate_attack_accuracy(
        source_model=ltc,
        target_model=ltc,
        loader=test_loader,
        epsilons=epsilons,
        attack_name="fgsm",
        device=device,
        pgd_steps=args.pgd_steps,
        pgd_alpha_scale=args.pgd_alpha_scale,
        max_samples=args.attack_max_samples,
    )

    print("\nWhite-box PGD (each model attacked/evaluated on itself)...")
    pgd_lstm = evaluate_attack_accuracy(
        source_model=lstm,
        target_model=lstm,
        loader=test_loader,
        epsilons=epsilons,
        attack_name="pgd",
        device=device,
        pgd_steps=args.pgd_steps,
        pgd_alpha_scale=args.pgd_alpha_scale,
        max_samples=args.attack_max_samples,
    )
    pgd_ltc = evaluate_attack_accuracy(
        source_model=ltc,
        target_model=ltc,
        loader=test_loader,
        epsilons=epsilons,
        attack_name="pgd",
        device=device,
        pgd_steps=args.pgd_steps,
        pgd_alpha_scale=args.pgd_alpha_scale,
        max_samples=args.attack_max_samples,
    )

    print("\nTransferability: LSTM-crafted examples on LTC and LTC-crafted examples on LSTM...")
    fgsm_lstm_to_ltc = evaluate_attack_accuracy(
        source_model=lstm,
        target_model=ltc,
        loader=test_loader,
        epsilons=epsilons,
        attack_name="fgsm",
        device=device,
        pgd_steps=args.pgd_steps,
        pgd_alpha_scale=args.pgd_alpha_scale,
        max_samples=args.attack_max_samples,
    )
    fgsm_ltc_to_lstm = evaluate_attack_accuracy(
        source_model=ltc,
        target_model=lstm,
        loader=test_loader,
        epsilons=epsilons,
        attack_name="fgsm",
        device=device,
        pgd_steps=args.pgd_steps,
        pgd_alpha_scale=args.pgd_alpha_scale,
        max_samples=args.attack_max_samples,
    )
    pgd_lstm_to_ltc = evaluate_attack_accuracy(
        source_model=lstm,
        target_model=ltc,
        loader=test_loader,
        epsilons=epsilons,
        attack_name="pgd",
        device=device,
        pgd_steps=args.pgd_steps,
        pgd_alpha_scale=args.pgd_alpha_scale,
        max_samples=args.attack_max_samples,
    )
    pgd_ltc_to_lstm = evaluate_attack_accuracy(
        source_model=ltc,
        target_model=lstm,
        loader=test_loader,
        epsilons=epsilons,
        attack_name="pgd",
        device=device,
        pgd_steps=args.pgd_steps,
        pgd_alpha_scale=args.pgd_alpha_scale,
        max_samples=args.attack_max_samples,
    )

    rows: List[Dict[str, str]] = []
    for eps in epsilons:
        rows.append(
            {
                "epsilon": f"{eps:.4f}",
                "clean_lstm_acc": f"{clean_lstm:.6f}",
                "clean_ltc_acc": f"{clean_ltc:.6f}",
                "fgsm_lstm_whitebox_acc": f"{fgsm_lstm[eps]:.6f}",
                "fgsm_ltc_whitebox_acc": f"{fgsm_ltc[eps]:.6f}",
                "pgd_lstm_whitebox_acc": f"{pgd_lstm[eps]:.6f}",
                "pgd_ltc_whitebox_acc": f"{pgd_ltc[eps]:.6f}",
                "fgsm_lstm_to_ltc_acc": f"{fgsm_lstm_to_ltc[eps]:.6f}",
                "fgsm_ltc_to_lstm_acc": f"{fgsm_ltc_to_lstm[eps]:.6f}",
                "pgd_lstm_to_ltc_acc": f"{pgd_lstm_to_ltc[eps]:.6f}",
                "pgd_ltc_to_lstm_acc": f"{pgd_ltc_to_lstm[eps]:.6f}",
            }
        )

    save_csv(args.out_dir / "metrics.csv", rows)

    plot_curve(
        args.out_dir / "fgsm_whitebox_accuracy.png",
        eps=epsilons,
        lstm_acc=[fgsm_lstm[e] for e in epsilons],
        ltc_acc=[fgsm_ltc[e] for e in epsilons],
        title="MNIST White-box FGSM: LSTM vs LTC",
    )
    plot_curve(
        args.out_dir / "pgd_whitebox_accuracy.png",
        eps=epsilons,
        lstm_acc=[pgd_lstm[e] for e in epsilons],
        ltc_acc=[pgd_ltc[e] for e in epsilons],
        title="MNIST White-box PGD: LSTM vs LTC",
    )
    plot_transfer(
        args.out_dir / "transferability_accuracy.png",
        eps=epsilons,
        fgsm_lstm_to_ltc=[fgsm_lstm_to_ltc[e] for e in epsilons],
        fgsm_ltc_to_lstm=[fgsm_ltc_to_lstm[e] for e in epsilons],
        pgd_lstm_to_ltc=[pgd_lstm_to_ltc[e] for e in epsilons],
        pgd_ltc_to_lstm=[pgd_ltc_to_lstm[e] for e in epsilons],
    )

    summary = {
        "clean_test_accuracy": {"lstm": clean_lstm, "ltc": clean_ltc},
        "fgsm_whitebox": {"lstm": fgsm_lstm, "ltc": fgsm_ltc},
        "pgd_whitebox": {"lstm": pgd_lstm, "ltc": pgd_ltc},
        "transfer_fgsm": {"lstm_to_ltc": fgsm_lstm_to_ltc, "ltc_to_lstm": fgsm_ltc_to_lstm},
        "transfer_pgd": {"lstm_to_ltc": pgd_lstm_to_ltc, "ltc_to_lstm": pgd_ltc_to_lstm},
    }
    with (args.out_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    print("\nSaved outputs:")
    print(f"- {(args.out_dir / 'metrics.csv').resolve()}")
    print(f"- {(args.out_dir / 'summary.json').resolve()}")
    print(f"- {(args.out_dir / 'fgsm_whitebox_accuracy.png').resolve()}")
    print(f"- {(args.out_dir / 'pgd_whitebox_accuracy.png').resolve()}")
    print(f"- {(args.out_dir / 'transferability_accuracy.png').resolve()}")


if __name__ == "__main__":
    main()
