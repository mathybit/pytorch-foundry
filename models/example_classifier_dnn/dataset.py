"""
Example Classifier DNN — Datasets

Three dataset classes for the classifier model:
- TrainDataset: reads episode IDs from a train .txt split, yields (X, y_int) via stream
- TestDataset: reads episode IDs from a test/valid .txt split, yields (X, y_int) via iteration
- InferenceDataset: reads an episode-style CSV directly, yields X-only
"""

from pathlib import Path
from typing import Dict, Union

import torch
from torch.utils.data import IterableDataset

from neural.datasets.iterable import MultiWorkerIterableDataset
from neural.datasets.streaming import MultiWorkerStreamDataset
from neural.datasets.inference import MultiWorkerInferenceDataset
from utils.episodes import (
    load_csv_to_lookup,
    load_ids,
    csv_to_rows
)

_TARGET_COL = "target"


class TrainDataset(MultiWorkerStreamDataset):
    """Training dataset for example_classifier_dnn.

    Uses stream-based iteration with infinite cycling.
    """
    def __init__(self, split_txt: Union[Path, str], csv_path: Union[Path, str], shuffle: bool = False, seed: int = None):
        self.csv_path = Path(csv_path)
        self._feature_cols, self._feat_data, self._tgt_data, self.n_features = load_csv_to_lookup(
            self.csv_path, target_col=_TARGET_COL
        )
        self.episode_ids = load_ids(Path(split_txt), self._feat_data)
        super().__init__(data=self.episode_ids, shuffle=shuffle, seed=seed)

    def preprocess(self, point: int):
        X = torch.tensor(self._feat_data[point], dtype=torch.float32)
        y = torch.tensor(self._tgt_data[point], dtype=torch.long)
        return X, y


class TestDataset(MultiWorkerIterableDataset):
    """Test/validation dataset for example_classifier_dnn."""

    def __init__(self, split_txt: Union[Path, str], csv_path: Union[Path, str]) -> None:
        self.csv_path = Path(csv_path)
        self._feature_cols, self._feat_data, self._tgt_data, self.n_features = load_csv_to_lookup(
            self.csv_path, target_col=_TARGET_COL
        )
        self.episode_ids = load_ids(Path(split_txt), self._feat_data)
        super().__init__(data=self.episode_ids, shuffle=False, seed=None)
    
    def preprocess(self, point: int):
        X = torch.tensor(self._feat_data[point], dtype=torch.float32)
        y = torch.tensor(self._tgt_data[point], dtype=torch.long)
        return X, y


class InferenceDataset(MultiWorkerInferenceDataset):
    """Inference dataset that reads from an episode-style CSV.

    Returns X-only tensors. The CSV should contain the same feature
    columns as the training dataset.

    Args:
        csv_path: Path to an episode CSV with feature columns.
    """

    def __init__(self, csv_path: Union[Path, str]) -> None:
        _, rows = csv_to_rows(Path(csv_path))
        super().__init__(data=list(range(len(rows))))
        self._rows = rows

    def __len__(self) -> int:
        return len(self._rows)

    def preprocess(self, point: int) -> torch.Tensor:
        return torch.tensor(self._rows[point], dtype=torch.float32)


def create_datasets(exp_config) -> Dict[str, IterableDataset]:
    """Build train/test/val datasets from an experiment config.

    Args:
        exp_config: Module with EPISODES_CSV and SPLIT_DIR attributes.

    Returns:
        Dict[str, IterableDataset] with keys "train", "test", "val".
    """
    csv_path = Path(exp_config.EPISODES_CSV)
    split_dir = Path(exp_config.SPLIT_DIR)

    dataset_train = TrainDataset(split_txt=split_dir / "train.txt", csv_path=csv_path, shuffle=True, seed=exp_config.RANDOM_SEED_PYTHON)
    dataset_test = TestDataset(split_txt=split_dir / "test.txt", csv_path=csv_path)
    dataset_valid = TestDataset(split_txt=split_dir / "valid.txt", csv_path=csv_path)

    return {"train": dataset_train, "test": dataset_test, "valid": dataset_valid}
