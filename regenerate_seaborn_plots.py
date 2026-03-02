#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Regenerate paper review plots with seaborn style")
    parser.add_argument(
        "--aggregate-json",
        type=Path,
        default=Path("/Users/j8ck/research/paper_review_full/aggregate_summary.json"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("/Users/j8ck/research/paper_review_full"),
    )
    return parser.parse_args()


def line_with_band(
    ax: plt.Axes,
    x: Sequence[float],
    mean: Sequence[float],
    std: Sequence[float],
    label: str,
) -> None:
    x_arr = np.asarray(x, dtype=float)
    m_arr = np.asarray(mean, dtype=float)
    s_arr = np.asarray(std, dtype=float)
    sns.lineplot(x=x_arr, y=m_arr, marker="o", linewidth=2.0, label=label, ax=ax)
    ax.fill_between(x_arr, np.clip(m_arr - s_arr, 0, 1), np.clip(m_arr + s_arr, 0, 1), alpha=0.2)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    data = json.loads(args.aggregate_json.read_text())

    sns.set_theme(style="whitegrid", context="talk", palette="colorblind")
    eps = data["epsilons"]

    # White-box
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    line_with_band(
        axes[0], eps,
        data["whitebox"]["fgsm"]["lstm"]["mean"],
        data["whitebox"]["fgsm"]["lstm"]["std"],
        "LSTM",
    )
    line_with_band(
        axes[0], eps,
        data["whitebox"]["fgsm"]["ltc"]["mean"],
        data["whitebox"]["fgsm"]["ltc"]["std"],
        "LTC",
    )
    axes[0].set_title("White-box FGSM")
    axes[0].set_xlabel("Epsilon")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_ylim(0, 1)
    axes[0].legend()

    line_with_band(
        axes[1], eps,
        data["whitebox"]["pgd"]["lstm"]["mean"],
        data["whitebox"]["pgd"]["lstm"]["std"],
        "LSTM",
    )
    line_with_band(
        axes[1], eps,
        data["whitebox"]["pgd"]["ltc"]["mean"],
        data["whitebox"]["pgd"]["ltc"]["std"],
        "LTC",
    )
    axes[1].set_title("White-box PGD")
    axes[1].set_xlabel("Epsilon")
    axes[1].set_ylim(0, 1)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(args.out_dir / "whitebox_mean_std_seaborn.png", dpi=220)
    plt.close(fig)

    # Transfer
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    line_with_band(
        axes[0], eps,
        data["transfer"]["fgsm"]["lstm_to_ltc"]["mean"],
        data["transfer"]["fgsm"]["lstm_to_ltc"]["std"],
        "LSTM->LTC",
    )
    line_with_band(
        axes[0], eps,
        data["transfer"]["fgsm"]["ltc_to_lstm"]["mean"],
        data["transfer"]["fgsm"]["ltc_to_lstm"]["std"],
        "LTC->LSTM",
    )
    axes[0].set_title("Transfer FGSM")
    axes[0].set_xlabel("Epsilon")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_ylim(0, 1)
    axes[0].legend()

    line_with_band(
        axes[1], eps,
        data["transfer"]["pgd"]["lstm_to_ltc"]["mean"],
        data["transfer"]["pgd"]["lstm_to_ltc"]["std"],
        "LSTM->LTC",
    )
    line_with_band(
        axes[1], eps,
        data["transfer"]["pgd"]["ltc_to_lstm"]["mean"],
        data["transfer"]["pgd"]["ltc_to_lstm"]["std"],
        "LTC->LSTM",
    )
    axes[1].set_title("Transfer PGD")
    axes[1].set_xlabel("Epsilon")
    axes[1].set_ylim(0, 1)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(args.out_dir / "transfer_mean_std_seaborn.png", dpi=220)
    plt.close(fig)

    # Targeted PGD success
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    line_with_band(
        axes[0], eps,
        data["targeted_pgd"]["whitebox"]["lstm"]["mean"],
        data["targeted_pgd"]["whitebox"]["lstm"]["std"],
        "LSTM",
    )
    line_with_band(
        axes[0], eps,
        data["targeted_pgd"]["whitebox"]["ltc"]["mean"],
        data["targeted_pgd"]["whitebox"]["ltc"]["std"],
        "LTC",
    )
    axes[0].set_title("Targeted PGD White-box")
    axes[0].set_xlabel("Epsilon")
    axes[0].set_ylabel("Targeted Success")
    axes[0].set_ylim(0, 1)
    axes[0].legend()

    line_with_band(
        axes[1], eps,
        data["targeted_pgd"]["transfer"]["lstm_to_ltc"]["mean"],
        data["targeted_pgd"]["transfer"]["lstm_to_ltc"]["std"],
        "LSTM->LTC",
    )
    line_with_band(
        axes[1], eps,
        data["targeted_pgd"]["transfer"]["ltc_to_lstm"]["mean"],
        data["targeted_pgd"]["transfer"]["ltc_to_lstm"]["std"],
        "LTC->LSTM",
    )
    axes[1].set_title("Targeted PGD Transfer")
    axes[1].set_xlabel("Epsilon")
    axes[1].set_ylim(0, 1)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(args.out_dir / "targeted_pgd_success_seaborn.png", dpi=220)
    plt.close(fig)

    # Corruptions
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    severity = list(range(6))
    for i, corr in enumerate(["gaussian_noise", "gaussian_blur", "center_occlusion"]):
        line_with_band(
            axes[i], severity,
            data["corruption"][corr]["lstm"]["mean"],
            data["corruption"][corr]["lstm"]["std"],
            "LSTM",
        )
        line_with_band(
            axes[i], severity,
            data["corruption"][corr]["ltc"]["mean"],
            data["corruption"][corr]["ltc"]["std"],
            "LTC",
        )
        axes[i].set_title(corr.replace("_", " ").title())
        axes[i].set_xlabel("Severity")
        axes[i].set_ylim(0, 1)
        axes[i].legend()
    axes[0].set_ylabel("Accuracy")
    fig.tight_layout()
    fig.savefig(args.out_dir / "corruption_suite_seaborn.png", dpi=220)
    plt.close(fig)

    # Gradient geometry
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.5))
    grad = data["gradient_geometry"]
    labels = ["LSTM grad L2", "LTC grad L2", "Grad cosine"]
    means = [
        grad["lstm_grad_l2_mean"]["mean"],
        grad["ltc_grad_l2_mean"]["mean"],
        grad["grad_cosine_mean"]["mean"],
    ]
    stds = [
        grad["lstm_grad_l2_mean"]["std"],
        grad["ltc_grad_l2_mean"]["std"],
        grad["grad_cosine_mean"]["std"],
    ]
    for i in range(3):
        sns.barplot(x=[""], y=[means[i]], ax=axes[i], errorbar=None)
        axes[i].errorbar([0], [means[i]], yerr=[stds[i]], color="black", capsize=5)
        axes[i].set_title(labels[i])
        axes[i].set_xlabel("")
        axes[i].set_xticks([])
    fig.tight_layout()
    fig.savefig(args.out_dir / "gradient_geometry_seaborn.png", dpi=220)
    plt.close(fig)

    print("Saved seaborn plots:")
    for name in [
        "whitebox_mean_std_seaborn.png",
        "transfer_mean_std_seaborn.png",
        "targeted_pgd_success_seaborn.png",
        "corruption_suite_seaborn.png",
        "gradient_geometry_seaborn.png",
    ]:
        print(f"- {(args.out_dir / name).resolve()}")


if __name__ == "__main__":
    main()
