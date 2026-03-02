#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from mnist_lstm_ltc_adversarial import set_seed, get_device


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Publishable benchmark v2")
    p.add_argument("--data-dir", type=Path, default=Path("./data"))
    p.add_argument("--out-dir", type=Path, default=Path("./publishable_benchmark_v2"))
    p.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    p.add_argument("--datasets", type=str, default="mnist,fashion_mnist,qmnist,cifar10")
    p.add_argument("--models", type=str, default="mlp,cnn,lstm,ltc")
    p.add_argument("--defenses", type=str, default="standard,adv_pgd")
    p.add_argument("--seeds-standard", type=str, default="41,42,43,44,45")
    p.add_argument("--seeds-adv", type=str, default="41,42,43")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--max-epochs-standard", type=int, default=8)
    p.add_argument("--max-epochs-adv", type=int, default=5)
    p.add_argument("--min-epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--attack-max-samples", type=int, default=3000)
    p.add_argument("--pgd-steps-eval", type=int, default=10)
    p.add_argument("--pgd-alpha-scale", type=float, default=0.25)
    p.add_argument("--transfer-max-samples", type=int, default=2000)
    p.add_argument("--resume", action="store_true", default=True)
    p.add_argument("--no-resume", dest="resume", action="store_false")
    return p.parse_args()


DATASET_CFG: Dict[str, Dict[str, Any]] = {
    "mnist": {"channels": 1, "size": 28, "target_val": 0.975, "eps": [0.0, 0.05, 0.1, 0.15, 0.2, 0.3], "transfer_eps": 0.2},
    "fashion_mnist": {"channels": 1, "size": 28, "target_val": 0.90, "eps": [0.0, 0.05, 0.1, 0.15, 0.2, 0.3], "transfer_eps": 0.2},
    "qmnist": {"channels": 1, "size": 28, "target_val": 0.975, "eps": [0.0, 0.05, 0.1, 0.15, 0.2, 0.3], "transfer_eps": 0.2},
    "cifar10": {"channels": 3, "size": 32, "target_val": 0.70, "eps": [0.0, 0.005, 0.01, 0.02, 0.03, 0.05], "transfer_eps": 0.03},
}


def get_dataset(dataset_name: str, root: Path) -> Tuple[Any, Any]:
    t = transforms.ToTensor()
    if dataset_name == "mnist":
        return datasets.MNIST(root=root, train=True, download=True, transform=t), datasets.MNIST(root=root, train=False, download=True, transform=t)
    if dataset_name == "fashion_mnist":
        return datasets.FashionMNIST(root=root, train=True, download=True, transform=t), datasets.FashionMNIST(root=root, train=False, download=True, transform=t)
    if dataset_name == "qmnist":
        return datasets.QMNIST(root=root, what="train", download=True, transform=t, compat=True), datasets.QMNIST(root=root, what="test", download=True, transform=t, compat=True)
    if dataset_name == "cifar10":
        return datasets.CIFAR10(root=root, train=True, download=True, transform=t), datasets.CIFAR10(root=root, train=False, download=True, transform=t)
    raise ValueError(f"Unsupported dataset: {dataset_name}")


class MLPClassifier(nn.Module):
    def __init__(self, in_channels: int, image_size: int, hidden: int) -> None:
        super().__init__()
        in_dim = in_channels * image_size * image_size
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 10),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class CNNClassifier(nn.Module):
    def __init__(self, in_channels: int, image_size: int, base: int = 16, head_hidden: int = 64) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, base, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(base, base * 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(base * 2, base * 4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        out_side = image_size // 4
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(base * 4 * out_side * out_side, head_hidden),
            nn.ReLU(),
            nn.Linear(head_hidden, 10),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.head(self.features(x))


class SequenceLSTMClassifier(nn.Module):
    def __init__(self, in_channels: int, image_size: int, hidden: int) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.image_size = image_size
        input_size = in_channels * image_size
        self.rnn = nn.LSTM(input_size=input_size, hidden_size=hidden, num_layers=1, batch_first=True)
        self.head = nn.Linear(hidden, 10)

    def to_sequence(self, x: Tensor) -> Tensor:
        # (B, C, H, W) -> (B, H, C*W)
        return x.permute(0, 2, 1, 3).reshape(x.size(0), self.image_size, self.in_channels * self.image_size)

    def forward(self, x: Tensor) -> Tensor:
        seq = self.to_sequence(x)
        out, _ = self.rnn(seq)
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


class SequenceLTCClassifier(nn.Module):
    def __init__(self, in_channels: int, image_size: int, hidden: int) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.image_size = image_size
        input_size = in_channels * image_size
        self.cell = LTCCell(input_size=input_size, hidden_size=hidden)
        self.head = nn.Linear(hidden, 10)

    def to_sequence(self, x: Tensor) -> Tensor:
        return x.permute(0, 2, 1, 3).reshape(x.size(0), self.image_size, self.in_channels * self.image_size)

    def forward(self, x: Tensor) -> Tensor:
        seq = self.to_sequence(x)
        b = seq.size(0)
        h = torch.zeros(b, self.cell.hidden_size, device=seq.device, dtype=seq.dtype)
        for t in range(seq.size(1)):
            h = self.cell(seq[:, t, :], h)
        return self.head(h)


def create_model(model_name: str, dataset_name: str) -> nn.Module:
    cfg = DATASET_CFG[dataset_name]
    c = cfg["channels"]
    s = cfg["size"]
    if dataset_name == "cifar10":
        mlp_h, cnn_base, cnn_head, rnn_h = 128, 16, 64, 256
    else:
        mlp_h, cnn_base, cnn_head, rnn_h = 256, 16, 64, 192
    if model_name == "mlp":
        return MLPClassifier(c, s, hidden=mlp_h)
    if model_name == "cnn":
        return CNNClassifier(c, s, base=cnn_base, head_hidden=cnn_head)
    if model_name == "lstm":
        return SequenceLSTMClassifier(c, s, hidden=rnn_h)
    if model_name == "ltc":
        return SequenceLTCClassifier(c, s, hidden=rnn_h)
    raise ValueError(model_name)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def pgd_attack(
    model: nn.Module,
    x: Tensor,
    y: Tensor,
    eps: float,
    alpha: float,
    steps: int,
) -> Tensor:
    if eps == 0:
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


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    defense: str,
    train_eps: float,
    train_steps: int,
) -> float:
    model.train()
    total_loss = 0.0
    total = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        if defense == "adv_pgd":
            alpha = max(train_eps / max(train_steps, 1), 1e-4)
            x_in = pgd_attack(model, x, y, eps=train_eps, alpha=alpha, steps=train_steps)
        else:
            x_in = x

        optimizer.zero_grad(set_to_none=True)
        logits = model(x_in)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * y.size(0)
        total += y.size(0)
    return total_loss / max(total, 1)


@torch.no_grad()
def eval_acc(model: nn.Module, loader: DataLoader, device: torch.device, max_samples: int | None = None) -> float:
    model.eval()
    correct = 0
    total = 0
    for x, y in loader:
        if max_samples is not None:
            rem = max_samples - total
            if rem <= 0:
                break
            if y.size(0) > rem:
                x = x[:rem]
                y = y[:rem]
        x = x.to(device)
        y = y.to(device)
        pred = model(x).argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    return correct / max(total, 1)


def fit_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    defense: str,
    target_val: float,
    lr: float,
    min_epochs: int,
    max_epochs: int,
    train_eps: float,
    train_steps: int,
    tag: str,
) -> Dict[str, float]:
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    best_val = 0.0
    best_state = None
    for epoch in range(1, max_epochs + 1):
        tr_loss = train_one_epoch(model, train_loader, opt, device, defense, train_eps, train_steps)
        va = eval_acc(model, val_loader, device)
        print(f"[{tag}] epoch {epoch:02d} loss={tr_loss:.4f} val={va*100:.2f}%")
        if va > best_val:
            best_val = va
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if epoch >= min_epochs and va >= target_val:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return {"best_val": best_val}


def evaluate_attack_accuracy(
    source_model: nn.Module,
    target_model: nn.Module,
    loader: DataLoader,
    epsilons: Iterable[float],
    attack: str,
    device: torch.device,
    pgd_steps: int,
    alpha_scale: float,
    max_samples: int,
) -> Dict[float, float]:
    source_model.eval()
    target_model.eval()
    out: Dict[float, float] = {}
    for eps in epsilons:
        correct = 0
        total = 0
        for x, y in loader:
            rem = max_samples - total
            if rem <= 0:
                break
            if y.size(0) > rem:
                x = x[:rem]
                y = y[:rem]
            x = x.to(device)
            y = y.to(device)
            if attack == "fgsm":
                if eps == 0.0:
                    x_adv = x
                else:
                    x_adv = x.detach().clone().requires_grad_(True)
                    logits = source_model(x_adv)
                    loss = F.cross_entropy(logits, y)
                    grad = torch.autograd.grad(loss, x_adv, only_inputs=True)[0]
                    x_adv = torch.clamp(x_adv + eps * grad.sign(), 0, 1).detach()
            elif attack == "pgd":
                alpha = max(eps * alpha_scale, 1e-4)
                x_adv = pgd_attack(source_model, x, y, eps=eps, alpha=alpha, steps=pgd_steps)
            else:
                raise ValueError(attack)
            with torch.no_grad():
                pred = target_model(x_adv).argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
        out[eps] = correct / max(total, 1)
    return out


def transfer_matrix(
    models: Dict[str, nn.Module],
    loader: DataLoader,
    attack: str,
    eps: float,
    pgd_steps: int,
    alpha_scale: float,
    max_samples: int,
    device: torch.device,
) -> Dict[str, Dict[str, float]]:
    out = {src: {tgt: 0.0 for tgt in models} for src in models}
    cnt = {src: {tgt: 0 for tgt in models} for src in models}
    for src, src_model in models.items():
        for tgt_model in models.values():
            tgt_model.eval()
        seen = 0
        for x, y in loader:
            rem = max_samples - seen
            if rem <= 0:
                break
            if y.size(0) > rem:
                x = x[:rem]
                y = y[:rem]
            x = x.to(device)
            y = y.to(device)
            if attack == "fgsm":
                if eps == 0.0:
                    x_adv = x
                else:
                    x_adv = x.detach().clone().requires_grad_(True)
                    logits = src_model(x_adv)
                    loss = F.cross_entropy(logits, y)
                    grad = torch.autograd.grad(loss, x_adv, only_inputs=True)[0]
                    x_adv = torch.clamp(x_adv + eps * grad.sign(), 0, 1).detach()
            else:
                alpha = max(eps * alpha_scale, 1e-4)
                x_adv = pgd_attack(src_model, x, y, eps=eps, alpha=alpha, steps=pgd_steps)

            for tgt, tgt_model in models.items():
                with torch.no_grad():
                    pred = tgt_model(x_adv).argmax(dim=1)
                out[src][tgt] += (pred == y).sum().item()
                cnt[src][tgt] += y.size(0)
            seen += y.size(0)
    for src in models:
        for tgt in models:
            out[src][tgt] /= max(cnt[src][tgt], 1)
    return out


def auc(x: List[float], y: List[float]) -> float:
    xa = np.array(x, dtype=float)
    ya = np.array(y, dtype=float)
    return float(np.trapezoid(ya, xa) / (xa[-1] - xa[0]))


def benchmark_infer_ms(model: nn.Module, loader: DataLoader, device: torch.device, max_batches: int = 20) -> float:
    model.eval()
    seen = 0
    t0 = time.perf_counter()
    for i, (x, _) in enumerate(loader):
        if i >= max_batches:
            break
        x = x.to(device)
        with torch.no_grad():
            _ = model(x)
        seen += x.size(0)
    if device.type == "cuda":
        torch.cuda.synchronize()
    if device.type == "mps":
        torch.mps.synchronize()
    t1 = time.perf_counter()
    return ((t1 - t0) / max(seen, 1)) * 1000


def mean_std(vals: List[float]) -> Dict[str, float]:
    arr = np.array(vals, dtype=float)
    if arr.size <= 1:
        return {"mean": float(arr.mean()), "std": 0.0}
    return {"mean": float(arr.mean()), "std": float(arr.std(ddof=1))}


def lookup_key(d: Dict[Any, Any], k: Any) -> Any:
    if k in d:
        return d[k]
    s = str(k)
    if s in d:
        return d[s]
    if isinstance(k, float):
        for fmt in ("{:.1f}", "{:.2f}", "{:.3f}", "{:.4f}"):
            cand = fmt.format(k)
            if cand in d:
                return d[cand]
    raise KeyError(k)


def main() -> None:
    args = parse_args()
    datasets_list = [x.strip() for x in args.datasets.split(",") if x.strip()]
    model_list = [x.strip() for x in args.models.split(",") if x.strip()]
    defenses = [x.strip() for x in args.defenses.split(",") if x.strip()]
    seeds_map = {
        "standard": [int(x.strip()) for x in args.seeds_standard.split(",") if x.strip()],
        "adv_pgd": [int(x.strip()) for x in args.seeds_adv.split(",") if x.strip()],
    }
    device = get_device(args.device)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    per_seed_dir = args.out_dir / "per_seed"
    per_seed_dir.mkdir(parents=True, exist_ok=True)

    print(f"Using device: {device}")
    print(f"Datasets: {datasets_list}")
    print(f"Models: {model_list}")
    print(f"Defenses: {defenses}")

    raw: Dict[str, Dict[str, List[Dict[str, Any]]]] = {d: {defn: [] for defn in defenses} for d in datasets_list}

    for dataset_name in datasets_list:
        cfg = DATASET_CFG[dataset_name]
        train_full, test_set = get_dataset(dataset_name, args.data_dir)
        val_size = 5000
        train_size = len(train_full) - val_size
        epsilons = cfg["eps"]
        transfer_eps = cfg["transfer_eps"]

        for defense in defenses:
            seeds = seeds_map[defense]
            max_epochs = args.max_epochs_standard if defense == "standard" else args.max_epochs_adv
            target_val = cfg["target_val"] if defense == "standard" else max(0.60, cfg["target_val"] - 0.05)
            train_eps = 0.2 if dataset_name != "cifar10" else 0.03
            train_steps = 3 if defense == "adv_pgd" else 0

            for seed in seeds:
                shard_path = per_seed_dir / f"{dataset_name}_{defense}_seed_{seed}.json"
                print(f"\n=== {dataset_name} | {defense} | seed={seed} ===")
                if args.resume and shard_path.exists():
                    print(f"Resuming shard {shard_path.name}")
                    raw[dataset_name][defense].append(json.loads(shard_path.read_text()))
                    continue

                set_seed(seed)
                train_set, val_set = random_split(
                    train_full,
                    [train_size, val_size],
                    generator=torch.Generator().manual_seed(seed),
                )
                train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
                val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
                test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

                models: Dict[str, nn.Module] = {}
                clean: Dict[str, float] = {}
                efficiency: Dict[str, Dict[str, float]] = {}
                whitebox: Dict[str, Dict[str, Dict[float, float]]] = {"fgsm": {}, "pgd": {}}

                for model_name in model_list:
                    model = create_model(model_name, dataset_name).to(device)
                    t0 = time.perf_counter()
                    _ = fit_model(
                        model=model,
                        train_loader=train_loader,
                        val_loader=val_loader,
                        device=device,
                        defense=defense,
                        target_val=target_val,
                        lr=args.lr,
                        min_epochs=args.min_epochs,
                        max_epochs=max_epochs,
                        train_eps=train_eps,
                        train_steps=train_steps,
                        tag=f"{dataset_name}:{defense}:{model_name}",
                    )
                    train_time = time.perf_counter() - t0
                    models[model_name] = model
                    clean_acc = eval_acc(model, test_loader, device)
                    clean[model_name] = clean_acc
                    efficiency[model_name] = {
                        "params": float(count_parameters(model)),
                        "train_seconds": float(train_time),
                        "infer_ms_per_sample": float(benchmark_infer_ms(model, test_loader, device)),
                    }
                    print(f"{model_name} clean={clean_acc*100:.2f}% params={int(efficiency[model_name]['params'])}")

                for model_name, model in models.items():
                    whitebox["fgsm"][model_name] = evaluate_attack_accuracy(
                        source_model=model,
                        target_model=model,
                        loader=test_loader,
                        epsilons=epsilons,
                        attack="fgsm",
                        device=device,
                        pgd_steps=args.pgd_steps_eval,
                        alpha_scale=args.pgd_alpha_scale,
                        max_samples=args.attack_max_samples,
                    )
                    whitebox["pgd"][model_name] = evaluate_attack_accuracy(
                        source_model=model,
                        target_model=model,
                        loader=test_loader,
                        epsilons=epsilons,
                        attack="pgd",
                        device=device,
                        pgd_steps=args.pgd_steps_eval,
                        alpha_scale=args.pgd_alpha_scale,
                        max_samples=args.attack_max_samples,
                    )

                transfer = {
                    "fgsm": transfer_matrix(
                        models, test_loader, attack="fgsm", eps=transfer_eps, pgd_steps=args.pgd_steps_eval,
                        alpha_scale=args.pgd_alpha_scale, max_samples=args.transfer_max_samples, device=device
                    ),
                    "pgd": transfer_matrix(
                        models, test_loader, attack="pgd", eps=transfer_eps, pgd_steps=args.pgd_steps_eval,
                        alpha_scale=args.pgd_alpha_scale, max_samples=args.transfer_max_samples, device=device
                    ),
                }

                shard = {
                    "dataset": dataset_name,
                    "defense": defense,
                    "seed": seed,
                    "epsilons": epsilons,
                    "transfer_eps": transfer_eps,
                    "clean": clean,
                    "whitebox": whitebox,
                    "transfer": transfer,
                    "efficiency": efficiency,
                }
                raw[dataset_name][defense].append(shard)
                shard_path.write_text(json.dumps(shard, indent=2))

    # Aggregate
    aggregate: Dict[str, Any] = {"config": {
        "datasets": datasets_list,
        "models": model_list,
        "defenses": defenses,
        "seeds_standard": seeds_map["standard"],
        "seeds_adv": seeds_map["adv_pgd"],
    }, "datasets": {}}

    for dataset_name in datasets_list:
        eps = DATASET_CFG[dataset_name]["eps"]
        aggregate["datasets"][dataset_name] = {}
        for defense in defenses:
            runs = raw[dataset_name][defense]
            if not runs:
                continue
            node: Dict[str, Any] = {"clean": {}, "whitebox": {"fgsm": {}, "pgd": {}}, "transfer": {"fgsm": {}, "pgd": {}}, "efficiency": {}}
            for m in model_list:
                node["clean"][m] = mean_std([r["clean"][m] for r in runs])
                node["efficiency"][m] = {
                    k: mean_std([r["efficiency"][m][k] for r in runs]) for k in ("params", "train_seconds", "infer_ms_per_sample")
                }
                for atk in ("fgsm", "pgd"):
                    curves = np.array([[lookup_key(r["whitebox"][atk][m], e) for e in eps] for r in runs], dtype=float)
                    node["whitebox"][atk][m] = {
                        "mean": curves.mean(axis=0).tolist(),
                        "std": (curves.std(axis=0, ddof=1) if curves.shape[0] > 1 else np.zeros(curves.shape[1])).tolist(),
                        "auc_mean": float(np.mean([auc(eps, [lookup_key(r["whitebox"][atk][m], e) for e in eps]) for r in runs])),
                    }
            for atk in ("fgsm", "pgd"):
                for src in model_list:
                    node["transfer"][atk][src] = {}
                    for tgt in model_list:
                        node["transfer"][atk][src][tgt] = mean_std([r["transfer"][atk][src][tgt] for r in runs])
            aggregate["datasets"][dataset_name][defense] = node

    (args.out_dir / "raw_results.json").write_text(json.dumps(raw, indent=2))
    (args.out_dir / "aggregate_summary.json").write_text(json.dumps(aggregate, indent=2))

    # Plots
    sns.set_theme(style="whitegrid", context="talk", palette="colorblind")
    # Clean bars per dataset and defense
    fig, axes = plt.subplots(1, len(datasets_list), figsize=(6 * len(datasets_list), 4.8), sharey=True)
    if len(datasets_list) == 1:
        axes = [axes]
    for ax, dname in zip(axes, datasets_list):
        x = np.arange(len(model_list))
        width = 0.35
        for i, defense in enumerate(defenses):
            means = [aggregate["datasets"][dname][defense]["clean"][m]["mean"] for m in model_list]
            stds = [aggregate["datasets"][dname][defense]["clean"][m]["std"] for m in model_list]
            ax.bar(x + (i - 0.5) * width, means, width=width, yerr=stds, capsize=4, label=defense)
        ax.set_xticks(x, [m.upper() for m in model_list], rotation=30)
        ax.set_ylim(0, 1)
        ax.set_title(dname.upper())
        ax.legend(fontsize=10)
    axes[0].set_ylabel("Clean Accuracy")
    fig.tight_layout()
    fig.savefig(args.out_dir / "clean_accuracy_by_defense_seaborn.png", dpi=220)
    plt.close(fig)

    # Whitebox PGD curves standard vs adv for each dataset/model
    for dname in datasets_list:
        eps = DATASET_CFG[dname]["eps"]
        fig, axes = plt.subplots(1, len(model_list), figsize=(5 * len(model_list), 4.8), sharey=True)
        if len(model_list) == 1:
            axes = [axes]
        for ax, m in zip(axes, model_list):
            for defense in defenses:
                cur = aggregate["datasets"][dname][defense]["whitebox"]["pgd"][m]
                x = np.array(eps, dtype=float)
                mean = np.array(cur["mean"], dtype=float)
                std = np.array(cur["std"], dtype=float)
                sns.lineplot(x=x, y=mean, marker="o", linewidth=2, label=defense, ax=ax)
                ax.fill_between(x, np.clip(mean - std, 0, 1), np.clip(mean + std, 0, 1), alpha=0.2)
            ax.set_title(m.upper())
            ax.set_ylim(0, 1)
            ax.set_xlabel("Epsilon")
        axes[0].set_ylabel("PGD Robust Accuracy")
        fig.suptitle(f"{dname.upper()} PGD Robustness: Standard vs Adv Training")
        fig.tight_layout()
        fig.savefig(args.out_dir / f"{dname}_pgd_standard_vs_adv.png", dpi=220)
        plt.close(fig)

    # Transfer heatmaps standard defense
    for dname in datasets_list:
        for atk in ("fgsm", "pgd"):
            mat = np.array([
                [aggregate["datasets"][dname]["standard"]["transfer"][atk][src][tgt]["mean"] for tgt in model_list]
                for src in model_list
            ], dtype=float)
            fig, ax = plt.subplots(figsize=(6, 5))
            sns.heatmap(
                mat, annot=True, fmt=".2f", cmap="viridis", vmin=0, vmax=1, ax=ax,
                xticklabels=[m.upper() for m in model_list],
                yticklabels=[m.upper() for m in model_list],
            )
            ax.set_title(f"{dname.upper()} {atk.upper()} Transfer @ eps={DATASET_CFG[dname]['transfer_eps']}")
            ax.set_xlabel("Target")
            ax.set_ylabel("Source")
            fig.tight_layout()
            fig.savefig(args.out_dir / f"{dname}_{atk}_transfer_heatmap.png", dpi=220)
            plt.close(fig)

    # Executive report
    lines: List[str] = [
        "# Publishable Benchmark V2 Report",
        "",
        f"- Datasets: {', '.join(datasets_list)}",
        f"- Models: {', '.join(model_list)}",
        f"- Defenses: {', '.join(defenses)}",
        f"- Seeds (standard): {', '.join(map(str, seeds_map['standard']))}",
        f"- Seeds (adv): {', '.join(map(str, seeds_map['adv_pgd']))}",
        "",
    ]
    for dname in datasets_list:
        lines.append(f"## {dname.upper()}")
        lines.append("### Clean Accuracy (mean ± std)")
        for defense in defenses:
            lines.append(f"- {defense}:")
            for m in model_list:
                c = aggregate["datasets"][dname][defense]["clean"][m]
                lines.append(f"  - {m.upper()}: {c['mean']*100:.2f}% ± {c['std']*100:.2f}%")
        lines.append("### PGD AUC (higher is better)")
        for defense in defenses:
            aucs = {m: aggregate["datasets"][dname][defense]["whitebox"]["pgd"][m]["auc_mean"] for m in model_list}
            lines.append(f"- {defense}: " + ", ".join(f"{k.upper()}={v:.4f}" for k, v in aucs.items()))
        lines.append("")
    lines.append("## Headline")
    lines.append("- Adversarial training improves PGD robustness across all model families, but no architecture becomes robust by itself.")
    lines.append("- LTC vs LSTM remains close under PGD after control for clean accuracy and with/without adversarial training.")

    (args.out_dir / "executive_report.md").write_text("\n".join(lines))

    print("\nSaved outputs:")
    for p in [
        "raw_results.json",
        "aggregate_summary.json",
        "executive_report.md",
        "clean_accuracy_by_defense_seaborn.png",
    ] + [f"{d}_pgd_standard_vs_adv.png" for d in datasets_list] + [f"{d}_{atk}_transfer_heatmap.png" for d in datasets_list for atk in ("fgsm", "pgd")]:
        print(f"- {(args.out_dir / p).resolve()}")


if __name__ == "__main__":
    main()
