#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


ROOT = Path(__file__).resolve().parent.parent
AGG_PATH = ROOT / "publishable_v2_full" / "aggregate_summary.json"
KEY_PATH = ROOT / "paper_artifacts" / "key_results.json"
OUT_DIR = ROOT / "paper" / "figures"


EPS = {
    "mnist": np.array([0.0, 0.05, 0.10, 0.15, 0.20, 0.30], dtype=float),
    "fashion_mnist": np.array([0.0, 0.05, 0.10, 0.15, 0.20, 0.30], dtype=float),
    "qmnist": np.array([0.0, 0.05, 0.10, 0.15, 0.20, 0.30], dtype=float),
    "cifar10": np.array([0.0, 0.005, 0.01, 0.02, 0.03, 0.05], dtype=float),
}

DISPLAY = {
    "mnist": "MNIST",
    "fashion_mnist": "Fashion-MNIST",
    "qmnist": "QMNIST",
    "cifar10": "CIFAR-10",
}


def save_ltc_lstm_pgd_curves(agg: dict) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey=True)
    axes = axes.flatten()

    for ax, dname in zip(axes, ["mnist", "fashion_mnist", "qmnist", "cifar10"]):
        e = EPS[dname]
        node = agg["datasets"][dname]
        for defense, ls in [("standard", "--"), ("adv_pgd", "-")]:
            for model, color in [("lstm", "#1f77b4"), ("ltc", "#d62728")]:
                cur = node[defense]["whitebox"]["pgd"][model]
                y = np.array(cur["mean"], dtype=float)
                ystd = np.array(cur["std"], dtype=float)
                label = f"{model.upper()} ({'Adv' if defense == 'adv_pgd' else 'Std'})"
                ax.plot(e, y, linestyle=ls, marker="o", linewidth=2.0, color=color, label=label)
                ax.fill_between(e, np.clip(y - ystd, 0, 1), np.clip(y + ystd, 0, 1), color=color, alpha=0.08)
        ax.set_title(DISPLAY[dname], fontsize=12)
        ax.set_ylim(0.0, 1.0)
        ax.set_xlabel("Perturbation Strength (epsilon)")
        ax.grid(alpha=0.2)

    axes[0].set_ylabel("Robust Accuracy")
    axes[2].set_ylabel("Robust Accuracy")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(OUT_DIR / "fig_ltc_lstm_pgd_curves.png", dpi=300)
    plt.close(fig)


def save_auc_delta_plot(key: dict) -> None:
    datasets = ["mnist", "fashion_mnist", "qmnist", "cifar10"]
    x = np.arange(len(datasets))
    width = 0.35

    std_means = [key["ltc_minus_lstm_auc"][d]["standard"]["mean"] for d in datasets]
    adv_means = [key["ltc_minus_lstm_auc"][d]["adv_pgd"]["mean"] for d in datasets]
    std_lo = [key["ltc_minus_lstm_auc"][d]["standard"]["mean"] - key["ltc_minus_lstm_auc"][d]["standard"]["ci95_low"] for d in datasets]
    std_hi = [key["ltc_minus_lstm_auc"][d]["standard"]["ci95_high"] - key["ltc_minus_lstm_auc"][d]["standard"]["mean"] for d in datasets]
    adv_lo = [key["ltc_minus_lstm_auc"][d]["adv_pgd"]["mean"] - key["ltc_minus_lstm_auc"][d]["adv_pgd"]["ci95_low"] for d in datasets]
    adv_hi = [key["ltc_minus_lstm_auc"][d]["adv_pgd"]["ci95_high"] - key["ltc_minus_lstm_auc"][d]["adv_pgd"]["mean"] for d in datasets]

    fig, ax = plt.subplots(figsize=(10, 4.6))
    ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.7)
    ax.bar(x - width / 2, std_means, width=width, label="Standard", color="#7f7f7f", yerr=[std_lo, std_hi], capsize=4)
    ax.bar(x + width / 2, adv_means, width=width, label="Adv-PGD", color="#2ca02c", yerr=[adv_lo, adv_hi], capsize=4)
    ax.set_xticks(x, [DISPLAY[d] for d in datasets])
    ax.set_ylabel("PGD AUC Delta (LTC - LSTM)")
    ax.set_title("Architecture Effect on PGD Robustness")
    ax.legend(frameon=False, ncol=2, loc="upper right")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_auc_delta_ltc_minus_lstm.png", dpi=300)
    plt.close(fig)


def save_transfer_asymmetry(key: dict) -> None:
    datasets = ["mnist", "fashion_mnist", "qmnist", "cifar10"]
    x = np.arange(len(datasets))
    width = 0.35
    fgsm = [key["transfer_lstm_ltc"][d]["fgsm"]["diff_lstm_to_ltc_minus_reverse"] for d in datasets]
    pgd = [key["transfer_lstm_ltc"][d]["pgd"]["diff_lstm_to_ltc_minus_reverse"] for d in datasets]

    fig, ax = plt.subplots(figsize=(10, 4.6))
    ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.7)
    ax.bar(x - width / 2, fgsm, width=width, color="#9467bd", label="FGSM transfer asymmetry")
    ax.bar(x + width / 2, pgd, width=width, color="#ff7f0e", label="PGD transfer asymmetry")
    ax.set_xticks(x, [DISPLAY[d] for d in datasets])
    ax.set_ylabel("Transfer Difference: LSTM→LTC minus LTC→LSTM")
    ax.set_title("Cross-Architecture Transfer Asymmetry (Standard Training)")
    ax.legend(frameon=False, ncol=2, loc="lower right")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_transfer_asymmetry.png", dpi=300)
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    agg = json.loads(AGG_PATH.read_text())
    key = json.loads(KEY_PATH.read_text())
    save_ltc_lstm_pgd_curves(agg)
    save_auc_delta_plot(key)
    save_transfer_asymmetry(key)


if __name__ == "__main__":
    main()
