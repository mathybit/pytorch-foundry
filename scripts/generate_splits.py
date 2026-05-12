"""
Generate Train/Valid/Test Splits for a Model
=====

Reads the experiment config for a given model, loads the episode CSV,
splits episode IDs into train/valid/text sets, and writes them as .txt
files. For classification tasks, also computes per-class weights and
writes them to class_weights.json alongside the splits.

Usage:
    python scripts/generate_splits.py --model example_classifier_dnn --exp exp01
    python scripts/generate_splits.py --model example_regressor_dnn --exp exp01

Supported strategies (via --strategy flag):
    random   — pure random split (default)
    stratified — stratified by class (TODO: not yet implemented)
"""

import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path


def _load_config(config_path: Path):
    """Load a config .py module from a file path without import path dependencies."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("exp_config", config_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def strategy_random(
    all_ids: np.ndarray,
    n_train: int,
    n_valid: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split IDs using pure random sampling without replacement."""
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(all_ids))
    train_idx = indices[:n_train]
    valid_idx = indices[n_train:n_train + n_valid]
    test_idx = indices[n_train + n_valid:]
    return all_ids[train_idx], all_ids[valid_idx], all_ids[test_idx]


def strategy_stratified(
    all_ids: np.ndarray,
    labels: np.ndarray,
    n_train: int,
    n_valid: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split IDs preserving class balance in each split.

    TODO: implement stratified sampling — per-class proportional split
    using sklearn.model_selection.train_test_split or manual
    stratified sampling.
    """
    raise NotImplementedError("stratified strategy not yet implemented")


STRATEGIES = {
    "random": strategy_random,
    "stratified": strategy_stratified,
}


def compute_class_weights(labels: np.ndarray) -> list:
    """Compute per-class inverse-frequency weights normalized to mean=1.0.

    For a class with count c_i in a total of N samples with K classes:
        raw_weight_i = N / (K * c_i)
        weight_i = raw_weight_i / mean(raw_weights)
    """
    unique, counts = np.unique(labels, return_counts=True)
    n_classes = len(unique)
    n_samples = len(labels)
    raw_weights = n_samples / (n_classes * counts)
    normalized = (raw_weights / np.mean(raw_weights)).tolist()
    return [round(w, 4) for w in normalized]


def main():
    parser = argparse.ArgumentParser(description="Generate train/valid/test splits for a model")
    parser.add_argument("--model", required=True, help="Model name (e.g. example_classifier_dnn)")
    parser.add_argument("--exp", required=True, help="Experiment name (e.g. exp01)")
    parser.add_argument("--strategy", choices=sorted(STRATEGIES), default="random")
    parser.add_argument("--output-base", type=str, default="data")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    config_path = project_root / "config" / "experiments" / args.model / f"{args.exp}.py"

    if not config_path.exists():
        print(f"Config not found: {config_path}")
        return

    cfg = _load_config(config_path)

    # Derive paths
    base = Path(args.output_base)
    model_dir = base / args.model
    episodes_csv = model_dir / cfg.EPISODES_CSV
    # SPLIT_DIR is absolute from project root
    output_dir = project_root / cfg.SPLIT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    if not episodes_csv.exists():
        print(f"Episode CSV not found: {episodes_csv}")
        return

    # Load episode data
    df = pd.read_csv(episodes_csv)
    ids = df["episode_id"].to_numpy()
    has_labels = False
    labels = None
    if cfg.TASK_TYPE == "classification" and "target" in df.columns:
        labels = df["target"].to_numpy()
        has_labels = True

    # Compute split sizes
    n_total = len(ids)
    n_valid = int(n_total * cfg.VALID_SPLIT_RATIO)
    n_test = int(n_total * cfg.TEST_SPLIT_RATIO)
    n_train = n_total - n_valid - n_test

    # Split integer indices, map back to IDs for output
    rng = np.random.default_rng(cfg.SPLIT_RANDOM_SEED)
    indices = rng.permutation(n_total)
    train_indices = indices[:n_train]
    valid_indices = indices[n_train:n_train + n_valid]
    test_indices = indices[n_train + n_valid:]

    train_ids = ids[train_indices]
    valid_ids = ids[valid_indices]
    test_ids = ids[test_indices]

    # Write split files
    for name, split_ids in [("train", train_ids), ("valid", valid_ids), ("test", test_ids)]:
        path = output_dir / f"{name}.txt"
        with open(path, "w") as f:
            for eid in split_ids:
                f.write(f"{eid}\n")
        print(f"  {name}: {len(split_ids)} episodes -> {path}")

    # Compute and write class weights for classification
    if cfg.TASK_TYPE == "classification" and has_labels:
        train_labels = labels[train_indices]
        class_weights = compute_class_weights(train_labels)
        weights_path = output_dir / "class_weights.json"
        class_names = [str(c) for c in cfg.TARGET_PARAMS["classes"]]
        weight_data = {"weights": class_weights, "classes": class_names}
        with open(weights_path, "w") as f:
            json.dump(weight_data, f, indent=2)
        print(f"  Class weights -> {weights_path}")

    print("Done.")


if __name__ == "__main__":
    main()
