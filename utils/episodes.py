"""Shared dataset helpers for episode-style CSV datasets."""

import csv
from pathlib import Path
from typing import Dict, List, Tuple

_FEATURE_COL_PREFIX = "feature_"


def feature_cols_from_csv(csv_path: Path) -> List[str]:
    """Extract feature column names (feature_XX) from a CSV header."""
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        return [c for c in reader.fieldnames if c.startswith(_FEATURE_COL_PREFIX)]


def load_csv_to_lookup(csv_path: Path, target_col: str = "target") -> \
        Tuple[List[str], Dict[str, List[float]], Dict[str, int], int]:
    """Load CSV into in-memory structures.

    Returns (feature_col_names, feature_lookup, target_lookup, n_features).
    target_lookup maps eid -> int target. Use target_col to specify the
    target column name in the CSV.
    """
    feature_cols = feature_cols_from_csv(csv_path)
    n_features = len(feature_cols)
    feat_lookup: Dict[str, List[float]] = {}
    tgt_lookup: Dict[str, int] = {}
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            eid = row["episode_id"]
            feat_lookup[eid] = [float(row[c]) for c in feature_cols]
            tgt_lookup[eid] = int(row[target_col])
    return feature_cols, feat_lookup, tgt_lookup, n_features


def load_csv_to_lookup_float(csv_path: Path, target_col: str = "target") -> \
        Tuple[List[str], Dict[str, List[float]], Dict[str, float], int]:
    """Like load_csv_to_lookup but target is float (regression)."""
    feature_cols = feature_cols_from_csv(csv_path)
    n_features = len(feature_cols)
    feat_lookup: Dict[str, List[float]] = {}
    tgt_lookup: Dict[str, float] = {}
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            eid = row["episode_id"]
            feat_lookup[eid] = [float(row[c]) for c in feature_cols]
            tgt_lookup[eid] = float(row[target_col])
    return feature_cols, feat_lookup, tgt_lookup, n_features


def load_ids(split_txt: Path, id_to_data: Dict) -> List[str]:
    """Read episode IDs from a split .txt file, keeping only those present in the CSV."""
    ids: List[str] = []
    with open(split_txt) as f:
        for line in f:
            eid = line.strip()
            if eid and eid in id_to_data:
                ids.append(eid)
    return ids


def csv_to_rows(csv_path: Path) -> Tuple[List[str], List[List[float]]]:
    """Load all feature rows from a CSV. Returns (col_names, rows)."""
    feature_cols = feature_cols_from_csv(csv_path)
    rows: List[List[float]] = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append([float(row[c]) for c in feature_cols])
    return feature_cols, rows
