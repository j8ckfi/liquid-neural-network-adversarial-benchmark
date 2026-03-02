#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import torch
from torch.utils.data import random_split
from torchvision import datasets


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Dataset and run contamination checks")
    p.add_argument("--data-dir", type=Path, default=Path("./data"))
    p.add_argument("--results-dir", type=Path, default=Path("./publishable_v2_full"))
    p.add_argument("--out-dir", type=Path, default=Path("./paper_artifacts"))
    p.add_argument("--seeds-standard", type=str, default="41,42,43,44,45")
    p.add_argument("--seeds-adv", type=str, default="41,42,43,44,45")
    p.add_argument("--val-size", type=int, default=5000)
    return p.parse_args()


def hash_array(arr: np.ndarray) -> str:
    arr = np.ascontiguousarray(arr)
    return hashlib.blake2b(arr.tobytes(), digest_size=16).hexdigest()


def extract_xy(ds: Any) -> Tuple[np.ndarray, np.ndarray]:
    if hasattr(ds, "data"):
        x = ds.data
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        else:
            x = np.asarray(x)
    else:
        raise RuntimeError(f"Dataset {type(ds).__name__} missing `.data`")

    if hasattr(ds, "targets"):
        y = ds.targets
        if isinstance(y, torch.Tensor):
            y = y.detach().cpu().numpy()
        else:
            y = np.asarray(y)
    elif hasattr(ds, "labels"):
        y = ds.labels
        if isinstance(y, torch.Tensor):
            y = y.detach().cpu().numpy()
        else:
            y = np.asarray(y)
    else:
        raise RuntimeError(f"Dataset {type(ds).__name__} missing labels")

    if y.ndim > 1:
        y = y[:, 0]
    y = y.reshape(-1)
    if x.shape[0] != y.shape[0]:
        raise RuntimeError(f"Mismatched samples: x={x.shape[0]}, y={y.shape[0]}")
    return x, y


def hash_stats(x: np.ndarray) -> Dict[str, Any]:
    hashes = [hash_array(x[i]) for i in range(x.shape[0])]
    unique = set(hashes)
    return {
        "n": int(x.shape[0]),
        "unique_hashes": int(len(unique)),
        "duplicate_count_within_split": int(x.shape[0] - len(unique)),
        "hashes": hashes,
    }


def train_test_overlap(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
) -> Dict[str, Any]:
    train_hashes = [hash_array(x_train[i]) for i in range(x_train.shape[0])]
    test_hashes = [hash_array(x_test[i]) for i in range(x_test.shape[0])]
    train_set = set(train_hashes)
    test_set = set(test_hashes)
    overlap = train_set.intersection(test_set)

    train_label_by_hash: Dict[str, int] = {}
    for i, h in enumerate(train_hashes):
        if h not in train_label_by_hash:
            train_label_by_hash[h] = int(y_train[i])

    same_label = 0
    diff_label = 0
    for i, h in enumerate(test_hashes):
        if h in overlap:
            if train_label_by_hash[h] == int(y_test[i]):
                same_label += 1
            else:
                diff_label += 1

    return {
        "train_n": int(x_train.shape[0]),
        "test_n": int(x_test.shape[0]),
        "train_unique_hashes": int(len(train_set)),
        "test_unique_hashes": int(len(test_set)),
        "train_test_exact_overlap_hashes": int(len(overlap)),
        "overlap_fraction_of_test": float(len(overlap) / max(len(test_set), 1)),
        "overlap_same_label_count": int(same_label),
        "overlap_diff_label_count": int(diff_label),
    }


def split_disjointness(
    ds_train: Any,
    seeds: Iterable[int],
    val_size: int,
) -> Dict[str, Any]:
    n = len(ds_train)
    train_size = n - val_size
    checks: List[Dict[str, Any]] = []
    for seed in seeds:
        train_subset, val_subset = random_split(
            ds_train,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(seed),
        )
        train_idx = set(train_subset.indices)
        val_idx = set(val_subset.indices)
        inter = train_idx.intersection(val_idx)
        checks.append(
            {
                "seed": int(seed),
                "train_len": int(len(train_idx)),
                "val_len": int(len(val_idx)),
                "intersection_size": int(len(inter)),
                "union_size": int(len(train_idx.union(val_idx))),
            }
        )
    all_clean = all(c["intersection_size"] == 0 for c in checks)
    return {
        "total_train_dataset_size": int(n),
        "val_size": int(val_size),
        "all_seed_splits_disjoint": bool(all_clean),
        "per_seed": checks,
    }


def load_dataset_pair(name: str, root: Path) -> Tuple[Any, Any]:
    if name == "mnist":
        return datasets.MNIST(root=root, train=True, download=True), datasets.MNIST(root=root, train=False, download=True)
    if name == "fashion_mnist":
        return datasets.FashionMNIST(root=root, train=True, download=True), datasets.FashionMNIST(root=root, train=False, download=True)
    if name == "qmnist":
        return (
            datasets.QMNIST(root=root, what="train", compat=True, download=True),
            datasets.QMNIST(root=root, what="test", compat=True, download=True),
        )
    if name == "cifar10":
        return datasets.CIFAR10(root=root, train=True, download=True), datasets.CIFAR10(root=root, train=False, download=True)
    raise ValueError(name)


def check_shard_integrity(
    results_dir: Path,
    datasets_list: Iterable[str],
    seeds_standard: Iterable[int],
    seeds_adv: Iterable[int],
) -> Dict[str, Any]:
    per_seed_dir = results_dir / "per_seed"
    missing: List[str] = []
    present: List[str] = []
    for d in datasets_list:
        for s in seeds_standard:
            name = f"{d}_standard_seed_{s}.json"
            (present if (per_seed_dir / name).exists() else missing).append(name)
        for s in seeds_adv:
            name = f"{d}_adv_pgd_seed_{s}.json"
            (present if (per_seed_dir / name).exists() else missing).append(name)

    aggregate_path = results_dir / "aggregate_summary.json"
    raw_path = results_dir / "raw_results.json"
    aggregate_exists = aggregate_path.exists()
    raw_exists = raw_path.exists()

    return {
        "per_seed_dir": str(per_seed_dir.resolve()),
        "expected_shards": int(len(present) + len(missing)),
        "present_shards": int(len(present)),
        "missing_shards": int(len(missing)),
        "missing_list": missing,
        "aggregate_summary_exists": bool(aggregate_exists),
        "raw_results_exists": bool(raw_exists),
    }


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    seeds_standard = [int(x.strip()) for x in args.seeds_standard.split(",") if x.strip()]
    seeds_adv = [int(x.strip()) for x in args.seeds_adv.split(",") if x.strip()]
    all_seeds = sorted(set(seeds_standard + seeds_adv))
    datasets_list = ["mnist", "fashion_mnist", "qmnist", "cifar10"]

    report: Dict[str, Any] = {
        "datasets": {},
        "results_integrity": {},
        "summary": {},
    }

    clean_overlap_all = True
    clean_split_all = True

    for name in datasets_list:
        train_ds, test_ds = load_dataset_pair(name, args.data_dir)
        x_train, y_train = extract_xy(train_ds)
        x_test, y_test = extract_xy(test_ds)

        overlap = train_test_overlap(x_train, y_train, x_test, y_test)
        split = split_disjointness(train_ds, all_seeds, val_size=args.val_size)
        hs_train = hash_stats(x_train)
        hs_test = hash_stats(x_test)

        report["datasets"][name] = {
            "overlap": overlap,
            "split_disjointness": split,
            "within_split_hash_stats": {
                "train": {
                    "n": hs_train["n"],
                    "unique_hashes": hs_train["unique_hashes"],
                    "duplicate_count_within_split": hs_train["duplicate_count_within_split"],
                },
                "test": {
                    "n": hs_test["n"],
                    "unique_hashes": hs_test["unique_hashes"],
                    "duplicate_count_within_split": hs_test["duplicate_count_within_split"],
                },
            },
        }
        clean_overlap_all = clean_overlap_all and (overlap["train_test_exact_overlap_hashes"] == 0)
        clean_split_all = clean_split_all and split["all_seed_splits_disjoint"]

    integrity = check_shard_integrity(args.results_dir, datasets_list, seeds_standard, seeds_adv)
    report["results_integrity"] = integrity

    report["summary"] = {
        "all_datasets_zero_exact_train_test_overlap": bool(clean_overlap_all),
        "all_seeds_train_val_disjoint": bool(clean_split_all),
        "all_expected_result_shards_present": bool(integrity["missing_shards"] == 0),
        "contamination_status": (
            "PASS"
            if (clean_overlap_all and clean_split_all and integrity["missing_shards"] == 0)
            else "FAIL"
        ),
    }

    json_path = args.out_dir / "contamination_report.json"
    md_path = args.out_dir / "contamination_report.md"
    json_path.write_text(json.dumps(report, indent=2))

    lines: List[str] = []
    lines.append("# Contamination Check Report")
    lines.append("")
    lines.append(f"- Status: **{report['summary']['contamination_status']}**")
    lines.append(f"- Zero exact train/test overlap (all datasets): {report['summary']['all_datasets_zero_exact_train_test_overlap']}")
    lines.append(f"- Train/val disjoint for all seeds: {report['summary']['all_seeds_train_val_disjoint']}")
    lines.append(f"- All expected result shards present: {report['summary']['all_expected_result_shards_present']}")
    lines.append("")
    for name in datasets_list:
        d = report["datasets"][name]
        ov = d["overlap"]
        lines.append(f"## {name}")
        lines.append(f"- Train/Test exact overlap hashes: {ov['train_test_exact_overlap_hashes']}")
        lines.append(f"- Train duplicate count (within split): {d['within_split_hash_stats']['train']['duplicate_count_within_split']}")
        lines.append(f"- Test duplicate count (within split): {d['within_split_hash_stats']['test']['duplicate_count_within_split']}")
        lines.append(f"- All seed splits disjoint: {d['split_disjointness']['all_seed_splits_disjoint']}")
        lines.append("")
    lines.append("## Results Integrity")
    lines.append(f"- Expected per-seed shards: {integrity['expected_shards']}")
    lines.append(f"- Present per-seed shards: {integrity['present_shards']}")
    lines.append(f"- Missing per-seed shards: {integrity['missing_shards']}")
    lines.append("")

    md_path.write_text("\n".join(lines))
    print(str(json_path.resolve()))
    print(str(md_path.resolve()))


if __name__ == "__main__":
    main()
