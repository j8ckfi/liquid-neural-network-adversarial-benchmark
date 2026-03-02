#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Tuple

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
from torchvision.transforms import functional as TVF

from mnist_lstm_ltc_adversarial import (
    LSTMClassifier,
    LTCClassifier,
    TrainConfig,
    evaluate_accuracy,
    evaluate_attack_accuracy,
    fit_model,
    get_device,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Broad adversarial benchmark runner")
    parser.add_argument("--data-dir", type=Path, default=Path("./data"))
    parser.add_argument("--out-dir", type=Path, default=Path("./dream_benchmarks"))
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--datasets", type=str, default="mnist,fashion_mnist,kmnist")
    parser.add_argument("--models", type=str, default="mlp,cnn,lstm,ltc")
    parser.add_argument("--seeds", type=str, default="41,42")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max-epochs", type=int, default=10)
    parser.add_argument("--min-epochs", type=int, default=4)
    parser.add_argument("--target-val-acc-mnist", type=float, default=0.975)
    parser.add_argument("--target-val-acc-fashion_mnist", type=float, default=0.90)
    parser.add_argument("--target-val-acc-kmnist", type=float, default=0.94)
    parser.add_argument("--target-val-acc-qmnist", type=float, default=0.975)
    parser.add_argument("--epsilons", type=str, default="0.0,0.05,0.1,0.15,0.2,0.3")
    parser.add_argument("--pgd-steps", type=int, default=10)
    parser.add_argument("--pgd-alpha-scale", type=float, default=0.25)
    parser.add_argument("--attack-max-samples", type=int, default=3000)
    parser.add_argument("--transfer-eps", type=float, default=0.2)
    parser.add_argument("--targeted-eps", type=str, default="0.1,0.2,0.3")
    parser.add_argument("--targeted-max-samples", type=int, default=2000)
    parser.add_argument("--corruption-max-samples", type=int, default=3000)
    parser.add_argument("--grad-max-samples", type=int, default=1024)
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    return parser.parse_args()


class MLPClassifier(nn.Module):
    def __init__(self, num_classes: int = 10, hidden: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class SmallCNNClassifier(nn.Module):
    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.head(self.features(x))


def create_model(name: str) -> nn.Module:
    if name == "mlp":
        return MLPClassifier()
    if name == "cnn":
        return SmallCNNClassifier()
    if name == "lstm":
        return LSTMClassifier(hidden_size=128)
    if name == "ltc":
        return LTCClassifier(hidden_size=256)
    raise ValueError(f"Unknown model '{name}'")


def get_dataset_class(dataset_name: str) -> type:
    if dataset_name == "mnist":
        return datasets.MNIST
    if dataset_name == "fashion_mnist":
        return datasets.FashionMNIST
    if dataset_name == "kmnist":
        return datasets.KMNIST
    if dataset_name == "qmnist":
        return datasets.QMNIST
    raise ValueError(f"Unsupported dataset '{dataset_name}'")


def targeted_pgd_attack(
    model: nn.Module,
    x: Tensor,
    y_target: Tensor,
    eps: float,
    alpha: float,
    steps: int,
) -> Tensor:
    if eps == 0.0:
        return x.detach()
    x_orig = x.detach()
    x_adv = x_orig + torch.empty_like(x_orig).uniform_(-eps, eps)
    x_adv = torch.clamp(x_adv, 0.0, 1.0)
    for _ in range(steps):
        x_adv.requires_grad_(True)
        logits = model(x_adv)
        loss = F.cross_entropy(logits, y_target)
        grad = torch.autograd.grad(loss, x_adv, only_inputs=True)[0]
        x_adv = x_adv.detach() - alpha * grad.sign()
        x_adv = torch.max(torch.min(x_adv, x_orig + eps), x_orig - eps)
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    return x_adv.detach()


def evaluate_targeted_success(
    source_model: nn.Module,
    target_model: nn.Module,
    loader: DataLoader,
    epsilons: Iterable[float],
    steps: int,
    alpha_scale: float,
    max_samples: int,
    device: torch.device,
) -> Dict[float, float]:
    source_model.eval()
    target_model.eval()
    out: Dict[float, float] = {}
    for eps in epsilons:
        total = 0
        hit = 0
        for x, y in loader:
            remaining = max_samples - total
            if remaining <= 0:
                break
            if y.size(0) > remaining:
                x = x[:remaining]
                y = y[:remaining]
            x = x.to(device)
            y = y.to(device)
            y_target = (y + 1) % 10
            alpha = max(eps * alpha_scale, 1e-4)
            x_adv = targeted_pgd_attack(source_model, x, y_target, eps=eps, alpha=alpha, steps=steps)
            with torch.no_grad():
                pred = target_model(x_adv).argmax(dim=1)
            hit += (pred == y_target).sum().item()
            total += y.size(0)
        out[eps] = hit / max(total, 1)
    return out


def attack_batch(
    model: nn.Module,
    x: Tensor,
    y: Tensor,
    attack_name: str,
    eps: float,
    pgd_steps: int,
    pgd_alpha_scale: float,
) -> Tensor:
    if attack_name == "fgsm":
        if eps == 0.0:
            return x.detach()
        x_adv = x.detach().clone().requires_grad_(True)
        logits = model(x_adv)
        loss = F.cross_entropy(logits, y)
        grad = torch.autograd.grad(loss, x_adv, only_inputs=True)[0]
        return torch.clamp(x_adv + eps * grad.sign(), 0.0, 1.0).detach()

    if attack_name == "pgd":
        if eps == 0.0:
            return x.detach()
        alpha = max(eps * pgd_alpha_scale, 1e-4)
        x_orig = x.detach()
        x_adv = x_orig + torch.empty_like(x_orig).uniform_(-eps, eps)
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
        for _ in range(pgd_steps):
            x_adv.requires_grad_(True)
            logits = model(x_adv)
            loss = F.cross_entropy(logits, y)
            grad = torch.autograd.grad(loss, x_adv, only_inputs=True)[0]
            x_adv = x_adv.detach() + alpha * grad.sign()
            x_adv = torch.max(torch.min(x_adv, x_orig + eps), x_orig - eps)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
        return x_adv.detach()

    raise ValueError(f"Unsupported attack '{attack_name}'")


def evaluate_transfer_matrix(
    models: Dict[str, nn.Module],
    loader: DataLoader,
    attack_name: str,
    eps: float,
    pgd_steps: int,
    pgd_alpha_scale: float,
    max_samples: int,
    device: torch.device,
) -> Dict[str, Dict[str, float]]:
    for m in models.values():
        m.eval()

    matrix: Dict[str, Dict[str, float]] = {src: {tgt: 0.0 for tgt in models} for src in models}
    counts: Dict[str, Dict[str, int]] = {src: {tgt: 0 for tgt in models} for src in models}

    for src_name, src_model in models.items():
        total_seen = 0
        for x, y in loader:
            remaining = max_samples - total_seen
            if remaining <= 0:
                break
            if y.size(0) > remaining:
                x = x[:remaining]
                y = y[:remaining]
            x = x.to(device)
            y = y.to(device)
            x_adv = attack_batch(
                src_model, x, y,
                attack_name=attack_name,
                eps=eps,
                pgd_steps=pgd_steps,
                pgd_alpha_scale=pgd_alpha_scale,
            )
            for tgt_name, tgt_model in models.items():
                with torch.no_grad():
                    pred = tgt_model(x_adv).argmax(dim=1)
                matrix[src_name][tgt_name] += (pred == y).sum().item()
                counts[src_name][tgt_name] += y.size(0)
            total_seen += y.size(0)

    for src_name in models:
        for tgt_name in models:
            matrix[src_name][tgt_name] /= max(counts[src_name][tgt_name], 1)
    return matrix


def _noise(x: Tensor, severity: int) -> Tensor:
    sigmas = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25]
    s = sigmas[severity]
    return torch.clamp(x + s * torch.randn_like(x), 0.0, 1.0)


def _blur(x: Tensor, severity: int) -> Tensor:
    if severity == 0:
        return x
    kernels = [1, 3, 3, 5, 5, 7]
    sigmas = [0.0, 0.5, 1.0, 1.0, 1.5, 2.0]
    return TVF.gaussian_blur(x, kernel_size=[kernels[severity], kernels[severity]], sigma=[sigmas[severity], sigmas[severity]])


def _occlusion(x: Tensor, severity: int) -> Tensor:
    sizes = [0, 4, 8, 12, 16, 20]
    size = sizes[severity]
    if size == 0:
        return x
    out = x.clone()
    c = 14
    h = size // 2
    out[:, :, c - h:c + h, c - h:c + h] = 0.0
    return out


def evaluate_corruptions(
    models: Dict[str, nn.Module],
    loader: DataLoader,
    device: torch.device,
    max_samples: int,
) -> Dict[str, Dict[int, Dict[str, float]]]:
    for model in models.values():
        model.eval()
    corruption_fns: Dict[str, Callable[[Tensor, int], Tensor]] = {
        "gaussian_noise": _noise,
        "gaussian_blur": _blur,
        "center_occlusion": _occlusion,
    }
    out: Dict[str, Dict[int, Dict[str, float]]] = {}
    for corr_name, corr_fn in corruption_fns.items():
        out[corr_name] = {}
        for sev in range(6):
            total = 0
            correct = {name: 0 for name in models}
            for x, y in loader:
                remain = max_samples - total
                if remain <= 0:
                    break
                if y.size(0) > remain:
                    x = x[:remain]
                    y = y[:remain]
                x = x.to(device)
                y = y.to(device)
                x_corr = corr_fn(x, sev)
                for name, model in models.items():
                    with torch.no_grad():
                        pred = model(x_corr).argmax(dim=1)
                    correct[name] += (pred == y).sum().item()
                total += y.size(0)
            out[corr_name][sev] = {name: correct[name] / max(total, 1) for name in models}
    return out


def evaluate_gradient_alignment(
    models: Dict[str, nn.Module],
    loader: DataLoader,
    device: torch.device,
    max_samples: int,
) -> Dict[str, Dict[str, float]]:
    model_names = list(models.keys())
    for model in models.values():
        model.eval()
    pair_stats: Dict[str, Dict[str, List[float]]] = {}
    for a, b in combinations(model_names, 2):
        pair_stats[f"{a}|{b}"] = {"cos": [], "sign": []}

    seen = 0
    for x, y in loader:
        remain = max_samples - seen
        if remain <= 0:
            break
        if y.size(0) > remain:
            x = x[:remain]
            y = y[:remain]
        x = x.to(device).requires_grad_(True)
        y = y.to(device)
        grads: Dict[str, Tensor] = {}
        for name, model in models.items():
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            grad = torch.autograd.grad(loss, x, retain_graph=True)[0]
            grads[name] = grad.view(grad.size(0), -1)
        for a, b in combinations(model_names, 2):
            cos = F.cosine_similarity(grads[a], grads[b], dim=1)
            sign = (torch.sign(grads[a]) == torch.sign(grads[b])).float().mean(dim=1)
            key = f"{a}|{b}"
            pair_stats[key]["cos"].extend(cos.detach().cpu().tolist())
            pair_stats[key]["sign"].extend(sign.detach().cpu().tolist())
        seen += y.size(0)

    out: Dict[str, Dict[str, float]] = {}
    for key, vals in pair_stats.items():
        out[key] = {
            "cos_mean": float(np.mean(vals["cos"])),
            "cos_std": float(np.std(vals["cos"])),
            "sign_mean": float(np.mean(vals["sign"])),
            "sign_std": float(np.std(vals["sign"])),
        }
    return out


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def benchmark_inference_ms(model: nn.Module, loader: DataLoader, device: torch.device, max_batches: int = 20) -> float:
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
    return ((t1 - t0) / max(seen, 1)) * 1000.0


def aggregate_mean_std(values: List[float]) -> Dict[str, float]:
    arr = np.array(values, dtype=np.float64)
    if arr.size == 1:
        return {"mean": float(arr.mean()), "std": 0.0}
    return {"mean": float(arr.mean()), "std": float(arr.std(ddof=1))}


def lookup_maybe_string_key(mapping: Dict[Any, Any], key: Any) -> Any:
    if key in mapping:
        return mapping[key]
    s = str(key)
    if s in mapping:
        return mapping[s]
    if isinstance(key, float):
        for fmt in ("{:.1f}", "{:.2f}", "{:.3f}", "{:.4f}", "{:.5f}"):
            cand = fmt.format(key)
            if cand in mapping:
                return mapping[cand]
    raise KeyError(f"Key '{key}' not found in mapping keys={list(mapping.keys())[:8]}")


def plot_line_with_band(ax: plt.Axes, x: List[float], mean: List[float], std: List[float], label: str) -> None:
    x_arr = np.array(x)
    m_arr = np.array(mean)
    s_arr = np.array(std)
    sns.lineplot(x=x_arr, y=m_arr, marker="o", linewidth=2.0, label=label, ax=ax)
    ax.fill_between(x_arr, np.clip(m_arr - s_arr, 0, 1), np.clip(m_arr + s_arr, 0, 1), alpha=0.2)


def main() -> None:
    args = parse_args()
    datasets_list = [x.strip() for x in args.datasets.split(",") if x.strip()]
    model_list = [x.strip() for x in args.models.split(",") if x.strip()]
    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    epsilons = [float(x.strip()) for x in args.epsilons.split(",") if x.strip()]
    targeted_eps = [float(x.strip()) for x in args.targeted_eps.split(",") if x.strip()]
    device = get_device(args.device)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    per_seed_dir = args.out_dir / "per_seed"
    per_seed_dir.mkdir(parents=True, exist_ok=True)

    print(f"Using device: {device}")
    print(f"Datasets: {datasets_list}")
    print(f"Models: {model_list}")
    print(f"Seeds: {seeds}")

    transform = transforms.ToTensor()
    all_results: Dict[str, List[Dict[str, Any]]] = {d: [] for d in datasets_list}

    for dataset_name in datasets_list:
        ds_cls = get_dataset_class(dataset_name)
        if dataset_name == "qmnist":
            train_full = ds_cls(root=args.data_dir, what="train", download=True, transform=transform, compat=True)
            test_set = ds_cls(root=args.data_dir, what="test", download=True, transform=transform, compat=True)
        else:
            train_full = ds_cls(root=args.data_dir, train=True, download=True, transform=transform)
            test_set = ds_cls(root=args.data_dir, train=False, download=True, transform=transform)
        target_val = getattr(args, f"target_val_acc_{dataset_name}")

        for seed in seeds:
            print(f"\n===== Dataset={dataset_name} Seed={seed} =====")
            seed_path = per_seed_dir / f"{dataset_name}_seed_{seed}.json"
            if args.resume and seed_path.exists():
                print(f"Resuming: loading existing shard {seed_path.name}")
                all_results[dataset_name].append(json.loads(seed_path.read_text()))
                continue
            set_seed(seed)
            train_set, val_set = random_split(
                train_full,
                [55000, 5000],
                generator=torch.Generator().manual_seed(seed),
            )
            train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
            val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
            test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

            models: Dict[str, nn.Module] = {}
            clean: Dict[str, float] = {}
            efficiency: Dict[str, Dict[str, float]] = {}

            for model_name in model_list:
                print(f"Training {model_name}...")
                model = create_model(model_name).to(device)
                t0 = time.perf_counter()
                _ = fit_model(
                    name=f"{dataset_name}:{model_name}",
                    model=model,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    device=device,
                    cfg=TrainConfig(
                        lr=args.lr,
                        max_epochs=args.max_epochs,
                        min_epochs=args.min_epochs,
                        target_val_acc=target_val,
                    ),
                )
                t1 = time.perf_counter()
                models[model_name] = model
                clean_acc = evaluate_accuracy(model, test_loader, device)
                clean[model_name] = clean_acc
                efficiency[model_name] = {
                    "params": float(count_parameters(model)),
                    "train_seconds": float(t1 - t0),
                    "infer_ms_per_sample": float(benchmark_inference_ms(model, test_loader, device)),
                }
                print(f"{model_name} clean acc={clean_acc * 100:.2f}%")

            whitebox: Dict[str, Dict[str, Dict[float, float]]] = {"fgsm": {}, "pgd": {}}
            for model_name, model in models.items():
                whitebox["fgsm"][model_name] = evaluate_attack_accuracy(
                    source_model=model,
                    target_model=model,
                    loader=test_loader,
                    epsilons=epsilons,
                    attack_name="fgsm",
                    device=device,
                    pgd_steps=args.pgd_steps,
                    pgd_alpha_scale=args.pgd_alpha_scale,
                    max_samples=args.attack_max_samples,
                )
                whitebox["pgd"][model_name] = evaluate_attack_accuracy(
                    source_model=model,
                    target_model=model,
                    loader=test_loader,
                    epsilons=epsilons,
                    attack_name="pgd",
                    device=device,
                    pgd_steps=args.pgd_steps,
                    pgd_alpha_scale=args.pgd_alpha_scale,
                    max_samples=args.attack_max_samples,
                )

            transfer = {
                "fgsm": evaluate_transfer_matrix(
                    models, test_loader, "fgsm", args.transfer_eps,
                    pgd_steps=args.pgd_steps, pgd_alpha_scale=args.pgd_alpha_scale,
                    max_samples=args.attack_max_samples, device=device,
                ),
                "pgd": evaluate_transfer_matrix(
                    models, test_loader, "pgd", args.transfer_eps,
                    pgd_steps=args.pgd_steps, pgd_alpha_scale=args.pgd_alpha_scale,
                    max_samples=args.attack_max_samples, device=device,
                ),
            }

            targeted: Dict[str, Dict[str, Dict[float, float]]] = {"whitebox": {}, "transfer_to_ltc": {}}
            for model_name, model in models.items():
                targeted["whitebox"][model_name] = evaluate_targeted_success(
                    source_model=model,
                    target_model=model,
                    loader=test_loader,
                    epsilons=targeted_eps,
                    steps=args.pgd_steps,
                    alpha_scale=args.pgd_alpha_scale,
                    max_samples=args.targeted_max_samples,
                    device=device,
                )
            if "lstm" in models and "ltc" in models:
                targeted["transfer_to_ltc"]["lstm_to_ltc"] = evaluate_targeted_success(
                    source_model=models["lstm"],
                    target_model=models["ltc"],
                    loader=test_loader,
                    epsilons=targeted_eps,
                    steps=args.pgd_steps,
                    alpha_scale=args.pgd_alpha_scale,
                    max_samples=args.targeted_max_samples,
                    device=device,
                )
                targeted["transfer_to_ltc"]["ltc_to_lstm"] = evaluate_targeted_success(
                    source_model=models["ltc"],
                    target_model=models["lstm"],
                    loader=test_loader,
                    epsilons=targeted_eps,
                    steps=args.pgd_steps,
                    alpha_scale=args.pgd_alpha_scale,
                    max_samples=args.targeted_max_samples,
                    device=device,
                )

            corruptions = evaluate_corruptions(
                models=models,
                loader=test_loader,
                device=device,
                max_samples=args.corruption_max_samples,
            )
            grad_align = evaluate_gradient_alignment(
                models=models,
                loader=test_loader,
                device=device,
                max_samples=args.grad_max_samples,
            )

            seed_result: Dict[str, Any] = {
                "dataset": dataset_name,
                "seed": seed,
                "clean": clean,
                "whitebox": whitebox,
                "transfer_eps": args.transfer_eps,
                "transfer": transfer,
                "targeted_eps": targeted_eps,
                "targeted": targeted,
                "corruptions": corruptions,
                "gradient_alignment": grad_align,
                "efficiency": efficiency,
            }
            all_results[dataset_name].append(seed_result)
            with seed_path.open("w") as f:
                json.dump(seed_result, f, indent=2, default=str)

    # Aggregate
    aggregate: Dict[str, Any] = {"datasets": {}, "config": {
        "datasets": datasets_list,
        "models": model_list,
        "seeds": seeds,
        "epsilons": epsilons,
        "targeted_eps": targeted_eps,
        "transfer_eps": args.transfer_eps,
    }}

    for dataset_name, seed_runs in all_results.items():
        dataset_agg: Dict[str, Any] = {}
        # clean
        dataset_agg["clean"] = {
            model: aggregate_mean_std([run["clean"][model] for run in seed_runs])
            for model in model_list
        }
        # whitebox curves
        wb: Dict[str, Any] = {"fgsm": {}, "pgd": {}}
        for attack in ("fgsm", "pgd"):
            for model in model_list:
                curves = np.array(
                    [
                        [lookup_maybe_string_key(run["whitebox"][attack][model], eps) for eps in epsilons]
                        for run in seed_runs
                    ],
                    dtype=float,
                )
                wb[attack][model] = {
                    "mean": curves.mean(axis=0).tolist(),
                    "std": (curves.std(axis=0, ddof=1) if curves.shape[0] > 1 else np.zeros(curves.shape[1])).tolist(),
                }
        dataset_agg["whitebox"] = wb
        # transfer matrices
        tr: Dict[str, Any] = {}
        for attack in ("fgsm", "pgd"):
            tr[attack] = {}
            for src in model_list:
                tr[attack][src] = {}
                for tgt in model_list:
                    vals = [run["transfer"][attack][src][tgt] for run in seed_runs]
                    tr[attack][src][tgt] = aggregate_mean_std(vals)
        dataset_agg["transfer"] = tr
        # targeted
        tg: Dict[str, Any] = {"whitebox": {}}
        for model in model_list:
            curves = np.array(
                [
                    [lookup_maybe_string_key(run["targeted"]["whitebox"][model], eps) for eps in targeted_eps]
                    for run in seed_runs
                ],
                dtype=float,
            )
            tg["whitebox"][model] = {
                "mean": curves.mean(axis=0).tolist(),
                "std": (curves.std(axis=0, ddof=1) if curves.shape[0] > 1 else np.zeros(curves.shape[1])).tolist(),
            }
        if "lstm" in model_list and "ltc" in model_list:
            tg["transfer_to_ltc"] = {}
            for key in ("lstm_to_ltc", "ltc_to_lstm"):
                curves = np.array(
                    [
                        [lookup_maybe_string_key(run["targeted"]["transfer_to_ltc"][key], eps) for eps in targeted_eps]
                        for run in seed_runs
                    ],
                    dtype=float,
                )
                tg["transfer_to_ltc"][key] = {
                    "mean": curves.mean(axis=0).tolist(),
                    "std": (curves.std(axis=0, ddof=1) if curves.shape[0] > 1 else np.zeros(curves.shape[1])).tolist(),
                }
        dataset_agg["targeted"] = tg
        # corruptions
        corr: Dict[str, Any] = {}
        for corr_name in ("gaussian_noise", "gaussian_blur", "center_occlusion"):
            corr[corr_name] = {}
            for model in model_list:
                vals = np.array(
                    [
                        [lookup_maybe_string_key(lookup_maybe_string_key(run["corruptions"][corr_name], sev), model) for sev in range(6)]
                        for run in seed_runs
                    ],
                    dtype=float,
                )
                corr[corr_name][model] = {
                    "mean": vals.mean(axis=0).tolist(),
                    "std": (vals.std(axis=0, ddof=1) if vals.shape[0] > 1 else np.zeros(vals.shape[1])).tolist(),
                }
        dataset_agg["corruptions"] = corr
        # gradient alignment and efficiency
        grad_keys = seed_runs[0]["gradient_alignment"].keys()
        grad_agg = {}
        for key in grad_keys:
            grad_agg[key] = {
                metric: aggregate_mean_std([run["gradient_alignment"][key][metric] for run in seed_runs])
                for metric in ("cos_mean", "cos_std", "sign_mean", "sign_std")
            }
        dataset_agg["gradient_alignment"] = grad_agg

        eff_agg = {}
        for model in model_list:
            eff_agg[model] = {
                metric: aggregate_mean_std([run["efficiency"][model][metric] for run in seed_runs])
                for metric in ("params", "train_seconds", "infer_ms_per_sample")
            }
        dataset_agg["efficiency"] = eff_agg

        aggregate["datasets"][dataset_name] = dataset_agg

    with (args.out_dir / "raw_results.json").open("w") as f:
        json.dump(all_results, f, indent=2, default=str)
    with (args.out_dir / "aggregate_summary.json").open("w") as f:
        json.dump(aggregate, f, indent=2)

    # Plot generation (Seaborn)
    sns.set_theme(style="whitegrid", context="talk", palette="colorblind")
    # 1) Clean accuracy bar
    rows = []
    for dataset_name in datasets_list:
        for model in model_list:
            rows.append({
                "dataset": dataset_name,
                "model": model,
                "mean": aggregate["datasets"][dataset_name]["clean"][model]["mean"],
                "std": aggregate["datasets"][dataset_name]["clean"][model]["std"],
            })
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(datasets_list))
    width = 0.18
    for i, model in enumerate(model_list):
        means = [next(r["mean"] for r in rows if r["dataset"] == d and r["model"] == model) for d in datasets_list]
        stds = [next(r["std"] for r in rows if r["dataset"] == d and r["model"] == model) for d in datasets_list]
        ax.bar(x + (i - (len(model_list)-1)/2) * width, means, width=width, yerr=stds, capsize=4, label=model.upper())
    ax.set_xticks(x, [d.upper() for d in datasets_list])
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Clean Accuracy")
    ax.set_title("Clean Accuracy by Dataset and Model")
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.out_dir / "clean_accuracy_bar_seaborn.png", dpi=220)
    plt.close(fig)

    # 2) Whitebox curves per dataset
    for attack in ("fgsm", "pgd"):
        fig, axes = plt.subplots(1, len(datasets_list), figsize=(6 * len(datasets_list), 4.8), sharey=True)
        if len(datasets_list) == 1:
            axes = [axes]
        for ax, dataset_name in zip(axes, datasets_list):
            for model in model_list:
                cur = aggregate["datasets"][dataset_name]["whitebox"][attack][model]
                plot_line_with_band(ax, epsilons, cur["mean"], cur["std"], model.upper())
            ax.set_title(f"{dataset_name.upper()} {attack.upper()}")
            ax.set_xlabel("Epsilon")
            ax.set_ylim(0, 1)
            ax.legend(fontsize=10)
        axes[0].set_ylabel("Robust Accuracy")
        fig.tight_layout()
        fig.savefig(args.out_dir / f"whitebox_{attack}_curves_seaborn.png", dpi=220)
        plt.close(fig)

    # 3) Transfer heatmaps per dataset
    for attack in ("fgsm", "pgd"):
        fig, axes = plt.subplots(1, len(datasets_list), figsize=(5.5 * len(datasets_list), 4.8), sharey=True)
        if len(datasets_list) == 1:
            axes = [axes]
        for ax, dataset_name in zip(axes, datasets_list):
            mat = np.array([
                [aggregate["datasets"][dataset_name]["transfer"][attack][src][tgt]["mean"] for tgt in model_list]
                for src in model_list
            ])
            sns.heatmap(
                mat, ax=ax, annot=True, fmt=".2f", cmap="viridis", vmin=0, vmax=1,
                xticklabels=[m.upper() for m in model_list], yticklabels=[m.upper() for m in model_list],
                cbar=(ax is axes[-1]),
            )
            ax.set_title(f"{dataset_name.upper()} {attack.upper()} Transfer @ eps={args.transfer_eps}")
            ax.set_xlabel("Target Model")
            ax.set_ylabel("Source Model")
        fig.tight_layout()
        fig.savefig(args.out_dir / f"transfer_{attack}_heatmaps_seaborn.png", dpi=220)
        plt.close(fig)

    # 4) Corruption curves
    for dataset_name in datasets_list:
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), sharey=True)
        for i, corr in enumerate(("gaussian_noise", "gaussian_blur", "center_occlusion")):
            ax = axes[i]
            for model in model_list:
                cur = aggregate["datasets"][dataset_name]["corruptions"][corr][model]
                plot_line_with_band(ax, list(range(6)), cur["mean"], cur["std"], model.upper())
            ax.set_title(corr.replace("_", " ").title())
            ax.set_xlabel("Severity")
            ax.set_ylim(0, 1)
        axes[0].set_ylabel("Accuracy")
        axes[-1].legend(fontsize=9)
        fig.suptitle(f"{dataset_name.upper()} Corruption Robustness")
        fig.tight_layout()
        fig.savefig(args.out_dir / f"{dataset_name}_corruptions_seaborn.png", dpi=220)
        plt.close(fig)

    # 5) Efficiency bars
    for metric, fname in (("train_seconds", "eff_train_time_seaborn.png"), ("infer_ms_per_sample", "eff_infer_latency_seaborn.png")):
        fig, ax = plt.subplots(figsize=(10, 5))
        x = np.arange(len(datasets_list))
        width = 0.18
        for i, model in enumerate(model_list):
            means = [aggregate["datasets"][d]["efficiency"][model][metric]["mean"] for d in datasets_list]
            stds = [aggregate["datasets"][d]["efficiency"][model][metric]["std"] for d in datasets_list]
            ax.bar(x + (i - (len(model_list)-1)/2) * width, means, width=width, yerr=stds, capsize=4, label=model.upper())
        ax.set_xticks(x, [d.upper() for d in datasets_list])
        ax.set_title(f"{metric} by Dataset and Model")
        ax.set_ylabel(metric)
        ax.legend()
        fig.tight_layout()
        fig.savefig(args.out_dir / fname, dpi=220)
        plt.close(fig)

    report_lines = [
        "# Dream Benchmarks Executive Report",
        "",
        f"- Datasets: {', '.join(datasets_list)}",
        f"- Models: {', '.join(model_list)}",
        f"- Seeds: {', '.join(str(s) for s in seeds)}",
        f"- Epsilons: {', '.join(f'{e:.2f}' for e in epsilons)}",
        f"- Transfer epsilon: {args.transfer_eps:.2f}",
        "",
        "## Clean Accuracy (mean ± std)",
    ]
    for d in datasets_list:
        report_lines.append(f"### {d.upper()}")
        for m in model_list:
            c = aggregate["datasets"][d]["clean"][m]
            report_lines.append(f"- {m.upper()}: {c['mean'] * 100:.2f}% ± {c['std'] * 100:.2f}%")
        report_lines.append("")
    report_lines.extend([
        "## Headline",
        "- Across datasets and model families, PGD substantially degrades all models.",
        "- Architectural differences shift transfer/corruption profiles, but do not remove gradient-based vulnerability.",
    ])
    (args.out_dir / "executive_report.md").write_text("\n".join(report_lines))

    print("\nSaved outputs:")
    for p in [
        "raw_results.json",
        "aggregate_summary.json",
        "executive_report.md",
        "clean_accuracy_bar_seaborn.png",
        "whitebox_fgsm_curves_seaborn.png",
        "whitebox_pgd_curves_seaborn.png",
        "transfer_fgsm_heatmaps_seaborn.png",
        "transfer_pgd_heatmaps_seaborn.png",
        "eff_train_time_seaborn.png",
        "eff_infer_latency_seaborn.png",
    ] + [f"{d}_corruptions_seaborn.png" for d in datasets_list]:
        print(f"- {(args.out_dir / p).resolve()}")


if __name__ == "__main__":
    main()
