"""
Generate Example Tabular Datasets
=======

Generates synthetic tabular datasets for the classifier and regressor
example models. Each dataset is a single CSV with episode_id columns
for every feature defined in the experiment config, plus a target column.

Uses FEATURE_COLUMNS from each experiment config as the source of truth
for column names. If FEATURE_COLUMNS is absent, falls back to
feature_00 through feature_NN.

Usage:
    python scripts/generate_example_tabular_datasets.py

This script creates:
    data/example_classifier_dnn/datasets/episodes_cls_v1.csv
    data/example_regressor_dnn/datasets/episodes_reg_v1.csv

Directories (raw/, datasets/, splits/) are created automatically.
Splits are NOT created here.
"""

import numpy as np
import pandas as pd
from pathlib import Path

# ======================================================================
# Dataset generation parameters — override here or via the config files.
# ======================================================================
CLASS_IMBALANCE_WEIGHTS = [0.4, 0.2, 0.1, 0.15, 0.15]  # list of floats, one per class; must sum to 1.0; None = balanced
CLASSIFIER_N_SAMPLES = 10000
CLASSIFIER_SEED = 51235
REGRESSOR_N_SAMPLES = 5000
REGRESSOR_SEED = 613562


def _load_config(config_path: Path):
    """Load a config .py module from a file path without import path dependencies."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("exp_config", config_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def generate_classifier_dataset(
    output_dir: Path,
    n_samples: int = 1000,
    n_classes: int = 2,
    feature_names: list = None,
    class_names: list = None,
    class_weights: list = None,
    seed: int = 42,
) -> Path:
    """Generate a classification dataset with class-conditional feature means.

    Each class has a distinct mean vector so that a linear classifier can
    separate them. Features are drawn from Gaussian distributions.

    Writes both a 'class' column (string label) and 'target' column (integer ID).

    Args:
        output_dir: Directory for datasets/episodes_cls_v1.csv.
        n_samples: Total number of samples.
        n_classes: Number of classes.
        feature_names: Column names for features.
        class_names: String labels for each class.
        class_weights: Per-class sampling proportions. Must sum to 1.0.
            If None, samples are uniformly distributed across classes.
        seed: Random seed.
    """
    # Validate class_weights
    if class_weights is not None:
        w_sum = sum(class_weights)
        if abs(w_sum - 1.0) > 1e-6:
            raise ValueError(
                f"class_weights must sum to 1.0, got {w_sum:.6f}."
            )
        if len(class_weights) != n_classes:
            raise ValueError(
                f"class_weights length ({len(class_weights)}) must equal "
                f"n_classes ({n_classes})."
            )
        # Ensure all weights are positive
        for i, w in enumerate(class_weights):
            if w <= 0:
                raise ValueError(
                    f"class_weights[{i}] = {w} must be positive."
                )

    rng = np.random.default_rng(seed)
    n_features = len(feature_names)

    if feature_names is None:
        feature_names = [f"feature_{i:02d}" for i in range(n_features)]

    if class_names is None:
        class_names = [f"class_{chr(65 + (i % 26))}" for i in range(n_classes)]

    class_means = [rng.standard_normal(n_features) for _ in range(n_classes)]
    class_means = [np.array(m) + 0.5 * np.arange(n_features) for m in class_means]

    # Compute samples per class from weights
    if class_weights is not None:
        class_counts = [int(w * n_samples) for w in class_weights]
        # Distribute remaining samples to largest class
        remainder = n_samples - sum(class_counts)
        if remainder > 0:
            max_idx = int(np.argmax(class_weights))
            class_counts[max_idx] += remainder
    else:
        class_counts = [n_samples // n_classes for _ in range(n_classes)]
        # Distribute remainder
        for i in range(n_samples % n_classes):
            class_counts[i] += 1

    rows = []
    for cls_id, n_cls in enumerate(class_counts):
        cls_features = class_means[cls_id] + rng.standard_normal((n_cls, n_features)) * 1.0
        for i in range(n_cls):
            row = {"episode_id": f"ep_{len(rows):06d}"}
            for j, name in enumerate(feature_names):
                row[name] = float(cls_features[i, j])
            row["label"] = class_names[cls_id]
            row["target"] = int(cls_id)
            rows.append(row)

    csv_path = output_dir / "datasets" / "episodes_cls_v1.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)
    print(f"Classifier dataset: {csv_path} ({n_samples} samples, {n_classes} classes)")
    if class_weights is not None:
        counts_str = ", ".join(f"{class_names[c]}={class_counts[c]}" for c in range(n_classes))
        print(f"  Class distribution: {counts_str}")
    return csv_path


def generate_regressor_dataset(
    output_dir: Path,
    n_samples: int = 5000,
    n_features: int = 30,
    feature_names: list = None,
    target_names: list = None,
    seed: int = 42,
) -> Path:
    """Generate a regression dataset with a linear target function.

    target = bias + sum(weights[i] * feature_i) + noise
    """
    rng = np.random.default_rng(seed)

    if feature_names is None:
        feature_names = [f"feature_{i:02d}" for i in range(n_features)]

    if target_names is None:
        target_names = ["target"]

    n_targets = len(target_names)
    weights = rng.standard_normal((len(feature_names), n_targets))
    features = rng.standard_normal((n_samples, len(feature_names)))
    targets = features @ weights + rng.standard_normal((n_samples, n_targets)) * 0.5

    rows = []
    for i in range(n_samples):
        row = {"episode_id": f"ep_{i:06d}"}
        for j, name in enumerate(feature_names):
            row[name] = float(features[i, j])
        for k, tname in enumerate(target_names):
            row[tname] = float(targets[i, k])
        rows.append(row)

    csv_path = output_dir / "datasets" / "episodes_reg_v1.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)
    print(f"Regressor dataset: {csv_path} ({n_samples} samples, noise_std=0.5)")
    return csv_path


def main():
    project_root = Path(__file__).resolve().parent.parent
    base = project_root / "data"

    # Config paths (must exist before running this script)
    cls_config = project_root / "config" / "experiments" / "example_classifier_dnn" / "exp01.py"
    reg_config = project_root / "config" / "experiments" / "example_regressor_dnn" / "exp01.py"

    # --- Classifier dataset ---
    cls_dir = base / "example_classifier_dnn"
    cls_dir.mkdir(parents=True, exist_ok=True)
    cls_cfg = _load_config(cls_config)
    cls_features = getattr(cls_cfg, "FEATURE_COLUMNS", [f"feature_{i:02d}" for i in range(30)])
    cls_target_params = getattr(cls_cfg, "TARGET_PARAMS", {"classes": [f"class_{chr(65 + i)}" for i in range(5)]})
    n_classes = getattr(cls_cfg, "N_CLASSES", len(cls_target_params["classes"]))
    generate_classifier_dataset(
        output_dir=cls_dir,
        n_samples=CLASSIFIER_N_SAMPLES,
        n_classes=n_classes,
        feature_names=list(cls_features),
        class_names=list(cls_target_params["classes"]),
        class_weights=list(CLASS_IMBALANCE_WEIGHTS) if CLASS_IMBALANCE_WEIGHTS is not None else None,
        seed=CLASSIFIER_SEED,
    )

    # --- Regressor dataset ---
    reg_dir = base / "example_regressor_dnn"
    reg_dir.mkdir(parents=True, exist_ok=True)
    reg_cfg = _load_config(reg_config)
    reg_features = getattr(reg_cfg, "FEATURE_COLUMNS", [f"feature_{i:02d}" for i in range(30)])
    reg_target_params = getattr(reg_cfg, "TARGET_PARAMS", {"outputs": ["target"]})
    generate_regressor_dataset(
        output_dir=reg_dir,
        n_samples=REGRESSOR_N_SAMPLES,
        n_features=len(reg_features),
        feature_names=list(reg_features),
        target_names=list(reg_target_params["outputs"]),
        seed=REGRESSOR_SEED,
    )

    # Create split directories (splits are NOT generated here)
    (cls_dir / "splits").mkdir(exist_ok=True)
    (reg_dir / "splits").mkdir(exist_ok=True)

    print("Done. Splits must be created separately.")


if __name__ == "__main__":
    main()
