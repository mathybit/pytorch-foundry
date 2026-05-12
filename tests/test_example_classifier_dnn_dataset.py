"""Tests for models/example_classifier_dnn/dataset.py."""

import csv
from pathlib import Path
import pytest
import sys
import torch

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from models.example_classifier_dnn.dataset import (
    TrainDataset,
    TestDataset,
    InferenceDataset,
    create_datasets,
)


# -- helpers --

def _write_csv(p: Path, n_features: int = 3, n_rows: int = 5) -> str:
    """Write a minimal episode CSV. Returns the filename."""
    cols = ["episode_id"] + [f"feature_{i:02d}" for i in range(n_features)] + ["target"]
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for i in range(n_rows):
            w.writerow({
                "episode_id": f"ep_{i:02d}",
                **{f"feature_{j:02d}": i * n_features + j for j in range(n_features)},
                "target": i % 3,
            })
    return "episodes_cls_v1.csv"


def _write_ids(p: Path, ids):
    with open(p, "w") as f:
        for eid in ids:
            f.write(eid + "\n")


def _make_train_ds(tmp_path: Path) -> TrainDataset:
    csv_path = tmp_path / "episodes_cls_v1.csv"
    _write_csv(csv_path)
    _write_ids(tmp_path / "train.txt", ["ep_00", "ep_01", "ep_02", "ep_03", "ep_04"])
    return TrainDataset(split_txt=tmp_path / "train.txt", csv_path=csv_path, shuffle=False, seed=None)


def _make_test_ds(tmp_path: Path, n_ids: int = 3) -> TestDataset:
    csv_path = tmp_path / "episodes_cls_v1.csv"
    _write_csv(csv_path)
    ids = [f"ep_{i:02d}" for i in range(n_ids)]
    _write_ids(tmp_path / "test.txt", ids)
    return TestDataset(split_txt=tmp_path / "test.txt", csv_path=csv_path)


# -- TrainDataset --

class TestTrainDataset:
    def test_len(self, tmp_path: Path):
        ds = _make_train_ds(tmp_path)
        assert len(ds) == 5

    def test_iteration(self, tmp_path: Path):
        ds = _make_train_ds(tmp_path)
        n = len(ds)
        first = None
        for i, item in enumerate(ds):
            if first is None:
                first = item
            if i == n:
                break
        assert first is not None
        X, y = first
        assert isinstance(X, torch.Tensor)
        assert isinstance(y, torch.Tensor)

    def test_dtypes(self, tmp_path: Path):
        ds = _make_train_ds(tmp_path)
        X, y = ds.preprocess("ep_00")
        assert X.dtype == torch.float32
        assert y.dtype == torch.long

    def test_y_range(self, tmp_path: Path):
        ds = _make_train_ds(tmp_path)
        ys = {y.item() for _, y in [ds.preprocess(f"ep_{i:02d}") for i in range(len(ds))]}
        assert ys == {0, 1, 2}

    def test_streaming(self, tmp_path: Path):
        ds = _make_train_ds(tmp_path)
        stream = ds.get_stream()
        first = [next(stream) for _ in range(5)]
        second = [next(stream) for _ in range(5)]
        assert all([all(first[j][0] == second[j][0]) for j in range(5)])
        assert all([first[j][1] == second[j][1] for j in range(5)])

    def test_multiple_iterations(self, tmp_path: Path):
        ds = _make_train_ds(tmp_path)
        n = len(ds)
        a = []
        b = []
        for i, item in enumerate(ds):
            a.append(item)
            if len(a) == n:
                break
        for i, item in enumerate(ds):
            b.append(item)
            if len(b) == n:
                break
        assert len(a) == len(b) == n
        assert all([all(a[j][0] == b[j][0]) for j in range(5)])
        assert all([a[j][1] == b[j][1] for j in range(5)])

    def test_empty_split(self, tmp_path: Path):
        csv_path = tmp_path / "episodes_cls_v1.csv"
        _write_csv(csv_path)
        _write_ids(tmp_path / "train.txt", [])
        ds = TrainDataset(split_txt=tmp_path / "train.txt", csv_path=csv_path, shuffle=False, seed=None)
        assert len(ds) == 0
        #assert list(ds) == []


# -- TestDataset --

class TestTestDataset:
    def test_len(self, tmp_path: Path):
        ds = _make_test_ds(tmp_path, n_ids=3)
        assert len(ds) == 3

    def test_iteration(self, tmp_path: Path):
        ds = _make_test_ds(tmp_path, n_ids=3)
        n = len(ds)
        first = None
        for i, item in enumerate(ds):
            if first is None:
                first = item
            if i == n:
                break
        assert first is not None
        X, y = first
        assert isinstance(X, torch.Tensor)
        assert isinstance(y, torch.Tensor)

    def test_dtypes(self, tmp_path: Path):
        ds = _make_test_ds(tmp_path, n_ids=3)
        X, y = ds.preprocess("ep_00")
        print(f"X: {X}, y: {y}")
        assert X.dtype == torch.float32
        assert y.dtype == torch.long

    def test_single_pass(self, tmp_path: Path):
        ds = _make_test_ds(tmp_path, n_ids=3)
        a = list(ds)
        b = list(ds)
        assert len(a) == len(b) == 3
        assert all([all(a[j][0] == b[j][0]) for j in range(len(a))])
        assert all([a[j][1] == b[j][1] for j in range(len(b))])

    def test_empty_split(self, tmp_path: Path):
        csv_path = tmp_path / "episodes_cls_v1.csv"
        _write_csv(csv_path)
        _write_ids(tmp_path / "test.txt", [])
        ds = TestDataset(split_txt=tmp_path / "test.txt", csv_path=csv_path)
        assert len(ds) == 0
        assert list(ds) == []


# -- InferenceDataset --

class TestInferenceDataset:
    def _write_inference_csv(self, p: Path, n_rows: int = 4):
        cols = ["episode_id", "feature_00", "feature_01", "feature_02"]
        with open(p, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for i in range(n_rows):
                w.writerow({
                    "episode_id": f"ep_{i:02d}",
                    "feature_00": float(i),
                    "feature_01": float(i * 2),
                    "feature_02": float(i * 3),
                })

    def test_len(self, tmp_path: Path):
        csv_path = tmp_path / "episodes_cls_v1.csv"
        self._write_inference_csv(csv_path)
        ds = InferenceDataset(csv_path=csv_path)
        assert len(ds) == 4

    def test_iteration(self, tmp_path: Path):
        csv_path = tmp_path / "episodes_cls_v1.csv"
        self._write_inference_csv(csv_path)
        ds = InferenceDataset(csv_path=csv_path)
        items = list(ds)
        assert len(items) == 4
        assert all(isinstance(x, torch.Tensor) for x in items)
        assert all(x.dim() == 1 for x in items)

    def test_values(self, tmp_path: Path):
        csv_path = tmp_path / "episodes_cls_v1.csv"
        self._write_inference_csv(csv_path)
        ds = InferenceDataset(csv_path=csv_path)
        rows = list(ds)
        assert rows[0].tolist() == [0.0, 0.0, 0.0]
        assert rows[3].tolist() == [3.0, 6.0, 9.0]

    def test_dtype(self, tmp_path: Path):
        csv_path = tmp_path / "episodes_cls_v1.csv"
        self._write_inference_csv(csv_path)
        ds = InferenceDataset(csv_path=csv_path)
        x = ds.preprocess(0)
        assert x.dtype == torch.float32

    def test_empty_csv(self, tmp_path: Path):
        csv_path = tmp_path / "episodes_cls_v1.csv"
        cols = ["episode_id", "feature_00"]
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
        ds = InferenceDataset(csv_path=csv_path)
        assert len(ds) == 0
        assert list(ds) == []


# -- create_datasets --

class MockConfig:
    EPISODES_CSV = "datasets/episodes_cls_v1.csv"
    SPLIT_DIR = "splits/exp01"
    RANDOM_SEED_PYTHON = 42


class TestCreateDatasets:
    def _make_cfg(self, tmp_path: Path):
        csv_dir = tmp_path / "datasets"
        csv_dir.mkdir()
        _write_csv(csv_dir / "episodes_cls_v1.csv")
        split_dir = tmp_path / "splits" / "exp01"
        split_dir.mkdir(parents=True)
        _write_ids(split_dir / "train.txt", ["ep_00", "ep_01", "ep_02", "ep_03", "ep_04"])
        _write_ids(split_dir / "test.txt", ["ep_00", "ep_01"])
        _write_ids(split_dir / "valid.txt", ["ep_02", "ep_03"])
        cfg = MockConfig()
        cfg.EPISODES_CSV = str(csv_dir / "episodes_cls_v1.csv")
        cfg.SPLIT_DIR = str(split_dir)
        return cfg

    def test_returns_dict(self, tmp_path: Path):
        cfg = self._make_cfg(tmp_path)
        result = create_datasets(cfg)
        assert isinstance(result, dict)

    def test_keys(self, tmp_path: Path):
        cfg = self._make_cfg(tmp_path)
        keys = set(create_datasets(cfg).keys())
        assert keys == {"train", "test", "valid"}

    def test_train_is_stream(self, tmp_path: Path):
        cfg = self._make_cfg(tmp_path)
        result = create_datasets(cfg)
        assert isinstance(result["train"], TrainDataset)

    def test_test_is_iterable(self, tmp_path: Path):
        cfg = self._make_cfg(tmp_path)
        result = create_datasets(cfg)
        assert isinstance(result["test"], TestDataset)

    def test_valid_is_iterable(self, tmp_path: Path):
        cfg = self._make_cfg(tmp_path)
        result = create_datasets(cfg)
        assert isinstance(result["valid"], TestDataset)

    def test_train_iteration(self, tmp_path: Path):
        cfg = self._make_cfg(tmp_path)
        ds = create_datasets(cfg)["train"]
        n = len(ds)
        first = None
        for i, item in enumerate(ds):
            if first is None:
                first = item
            if i == n:
                break
        assert first is not None
        X, y = first
        assert isinstance(X, torch.Tensor)
        assert isinstance(y, torch.Tensor)

    def test_test_iteration(self, tmp_path: Path):
        cfg = self._make_cfg(tmp_path)
        test_items = list(create_datasets(cfg)["test"])
        assert len(test_items) == 2

    def test_valid_iteration(self, tmp_path: Path):
        cfg = self._make_cfg(tmp_path)
        valid_items = list(create_datasets(cfg)["valid"])
        assert len(valid_items) == 2

    def test_train_y_dtype(self, tmp_path: Path):
        cfg = self._make_cfg(tmp_path)
        _, y = create_datasets(cfg)["train"].preprocess('ep_00')
        assert y.dtype == torch.long

    def test_test_y_dtype(self, tmp_path: Path):
        cfg = self._make_cfg(tmp_path)
        _, y = create_datasets(cfg)["test"].preprocess('ep_00')
        assert y.dtype == torch.long
