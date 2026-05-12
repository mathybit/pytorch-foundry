"""Shared data utilities."""

from pathlib import Path
from typing import List


def resolve_csv_path(base_dir: Path, pattern: str) -> Path:
    """Return path to the CSV inside a directory.

    Searches base_dir for the first file matching the given glob pattern.
    """
    candidates = list(base_dir.glob(pattern))
    if not candidates:
        raise FileNotFoundError("No {} files found in {}".format(pattern, base_dir))
    return candidates[0]
