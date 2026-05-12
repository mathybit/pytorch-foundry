"""Tests for neural.datasets classes."""

from collections.abc import Iterator
from pathlib import Path
import pytest
import sys
import torch

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from neural.datasets.iterable import (
    SingleWorkerIterableDataset,
    ShuffledIterableDataset,
    MultiWorkerIterableDataset,
)
from neural.datasets.streaming import (
    SingleWorkerStreamDataset,
    ShuffledStreamDataset,
    MultiWorkerStreamDataset,
)
from neural.datasets.inference import (
    SingleWorkerInferenceDataset,
    MultiWorkerInferenceDataset,
)


# ---------- Iterable Datasets ---

class TestSingleWorkerIterableDataset:
    def test_iteration(self):
        ds = SingleWorkerIterableDataset([1, 2, 3, 4, 5])
        items = list(ds)
        assert items == [1, 2, 3, 4, 5]

    def test_len(self):
        ds = SingleWorkerIterableDataset([1, 2, 3])
        assert len(ds) == 3

    def test_multiple_iterations(self):
        ds = SingleWorkerIterableDataset([1, 2, 3])
        a = list(ds)
        b = list(ds)
        assert a == b == [1, 2, 3]

    def test_preprocess_skip_none(self):
        class DS(SingleWorkerIterableDataset):
            def preprocess(self, point):
                return None if point == 3 else point
        ds = DS([1, 2, 3, 4, 5])
        items = list(ds)
        assert items == [1, 2, 4, 5]

    def test_empty_data(self):
        ds = SingleWorkerIterableDataset([])
        items = list(ds)
        assert items == []

    def test_get_stream(self):
        ds = SingleWorkerIterableDataset([10, 20])
        stream = ds.get_stream()
        assert isinstance(stream, Iterator)


class TestShuffledIterableDataset:
    def test_no_shuffle(self):
        ds = ShuffledIterableDataset([1, 2, 3, 4, 5], shuffle=False)
        items = list(ds)
        assert items == [1, 2, 3, 4, 5]

    def test_shuffle(self):
        ds = ShuffledIterableDataset(list(range(10)), shuffle=True, seed=42)
        items = list(ds)
        assert items != list(range(10))

    def test_requires_seed_when_shuffled(self):
        with pytest.raises(ValueError, match="seed"):
            ShuffledIterableDataset([1, 2, 3], shuffle=True)

    def test_requires_int_seed(self):
        with pytest.raises(ValueError, match="seed"):
            ShuffledIterableDataset([1, 2, 3], shuffle=True, seed="42")

    def test_reproducible_with_seed(self):
        ds1 = ShuffledIterableDataset([1, 2, 3, 4, 5], shuffle=True, seed=42)
        ds2 = ShuffledIterableDataset([1, 2, 3, 4, 5], shuffle=True, seed=42)
        assert list(ds1) == list(ds2)

    def test_varied_across_iterations(self):
        ds = ShuffledIterableDataset([1, 2, 3, 4, 5], shuffle=True, seed=42)
        first = list(ds)
        second = list(ds)
        assert isinstance(first, list)
        assert isinstance(second, list)
        assert len(first) == len(second) == 5

    def test_seed_mutation(self):
        ds = ShuffledIterableDataset([1, 2, 3], shuffle=True, seed=0)
        list(ds)
        assert ds.seed == 1

    def test_len(self):
        ds = ShuffledIterableDataset([1, 2, 3], shuffle=True, seed=0)
        assert len(ds) == 3

    def test_no_shuffle_without_seed(self):
        ds = ShuffledIterableDataset([1, 2, 3], shuffle=False)
        items = list(ds)
        assert items == [1, 2, 3]

    def test_different_seeds_different_shuffles(self):
        ds1 = ShuffledIterableDataset(list(range(10)), shuffle=True, seed=42)
        ds2 = ShuffledIterableDataset(list(range(10)), shuffle=True, seed=99)
        assert list(ds1) != list(ds2)


class TestMultiWorkerIterableDataset:
    def test_single_worker(self):
        ds = MultiWorkerIterableDataset([1, 2, 3, 4, 5], shuffle=False)
        items = list(ds)
        assert items == [1, 2, 3, 4, 5]

    def test_len(self):
        ds = MultiWorkerIterableDataset([1, 2, 3, 4, 5], shuffle=False)
        assert len(ds) == 5

    def test_requires_seed_when_shuffled(self):
        with pytest.raises(ValueError, match="seed"):
            MultiWorkerIterableDataset([1, 2, 3], shuffle=True)

    def test_shuffle(self):
        ds = MultiWorkerIterableDataset(list(range(10)), shuffle=True, seed=42)
        items = list(ds)
        assert items != list(range(10))

    def test_no_shuffle_without_seed(self):
        ds = MultiWorkerIterableDataset([1, 2, 3], shuffle=False)
        items = list(ds)
        assert items == [1, 2, 3]

    def test_worker_detection(self):
        ds = MultiWorkerIterableDataset([1, 2, 3, 4, 5], shuffle=False)
        ds.__iter__()
        assert ds.multi is False
        assert ds.worker_id == 0
        assert ds.n_workers == 1


# ---------- Streaming Datasets ---

class TestSingleWorkerStreamDataset:
    def test_infinite_stream(self):
        ds = SingleWorkerStreamDataset([(1, 0), (2, 0), (3, 0)])
        stream = ds.get_stream()
        items = [next(stream) for _ in range(7)]
        assert len(items) == 7
        assert items[0] == (torch.tensor([1]), torch.tensor([0]))
        assert items[2] == (torch.tensor([3]), torch.tensor([0]))
        assert items[3] == (torch.tensor([1]), torch.tensor([0]))

    def test_preprocess_returns_tensors(self):
        ds = SingleWorkerStreamDataset([(10, 20)])
        stream = ds.get_stream()
        x, y = next(stream)
        assert isinstance(x, torch.Tensor)
        assert isinstance(y, torch.Tensor)
        assert x.item() == 10
        assert y.item() == 20

    def test_preprocess_scalar_to_list(self):
        ds = SingleWorkerStreamDataset([(5, 6)])
        stream = ds.get_stream()
        x, y = next(stream)
        assert x.item() == 5
        assert y.item() == 6

    def test_infinite_loop(self):
        ds = SingleWorkerStreamDataset([(1, 0)])
        stream = ds.get_stream()
        items = [next(stream) for _ in range(100)]
        assert len(items) == 100
        assert all(x.item() == 1 for x, _ in items)

    def test_len(self):
        ds = SingleWorkerStreamDataset([(1, 0), (2, 0)])
        assert len(ds.data) == 2


class TestShuffledStreamDataset:
    def test_no_shuffle(self):
        ds = ShuffledStreamDataset([(1, 0), (2, 0)], shuffle=False, seed=0)
        stream = ds.get_stream()
        items = [next(stream) for _ in range(5)]
        assert len(items) == 5

    def test_shuffle(self):
        ds = ShuffledStreamDataset([(i, 0) for i in range(10)], shuffle=True, seed=0)
        seen_first = set()
        stream = ds.get_stream()
        for _ in range(100):
            x, _ = next(stream)
            seen_first.add(x.item())
        assert len(seen_first) == 10

    def test_requires_seed_when_shuffled(self):
        with pytest.raises(ValueError, match="seed"):
            ShuffledStreamDataset([(i, 0) for i in range(5)], shuffle=True)

    def test_seed_mutation(self):
        ds = ShuffledStreamDataset([(1, 0), (2, 0)], shuffle=True, seed=0)
        stream = ds.get_stream()
        # Consume 2 items (1 full cycle) to trigger first shuffle
        next(stream)
        next(stream)
        # Consume another full cycle
        next(stream)
        next(stream)
        assert ds.seed == 2

    def test_deterministic_with_seed(self):
        ds1 = ShuffledStreamDataset([(i, 0) for i in range(5)], shuffle=True, seed=42)
        ds2 = ShuffledStreamDataset([(i, 0) for i in range(5)], shuffle=True, seed=42)
        s1 = ds1.get_stream()
        s2 = ds2.get_stream()
        items1 = [next(s1) for _ in range(20)]
        items2 = [next(s2) for _ in range(20)]
        assert len(items1) == len(items2) == 20

    def test_different_seeds_different_shuffles(self):
        ds1 = ShuffledStreamDataset([(i, 0) for i in range(10)], shuffle=True, seed=42)
        ds2 = ShuffledStreamDataset([(i, 0) for i in range(10)], shuffle=True, seed=99)
        s1 = ds1.get_stream()
        s2 = ds2.get_stream()
        items1 = [next(s1) for _ in range(10)]
        items2 = [next(s2) for _ in range(10)]
        assert items1 != items2


class TestMultiWorkerStreamDataset:
    def test_single_worker(self):
        ds = MultiWorkerStreamDataset([(i, 0) for i in range(10)], shuffle=False)
        stream = ds.get_stream()
        items = [next(stream) for _ in range(5)]
        assert len(items) == 5

    def test_multi_detection(self):
        ds = MultiWorkerStreamDataset([(i, 0) for i in range(10)], shuffle=False)
        ds.__iter__()
        assert ds.multi is False
        assert ds.worker_id == 0
        assert ds.n_workers == 1

    def test_len(self):
        ds = MultiWorkerStreamDataset([(i, 0) for i in range(10)], shuffle=False)
        assert len(ds) == 10

    def test_preprocess(self):
        ds = MultiWorkerStreamDataset([(5, 7)], shuffle=False)
        stream = ds.get_stream()
        x, y = next(stream)
        assert isinstance(x, torch.Tensor)
        assert isinstance(y, torch.Tensor)

    def test_infinite_cycling(self):
        ds = MultiWorkerStreamDataset([(i, 0) for i in range(5)], shuffle=False)
        stream = ds.get_stream()
        items = [next(stream) for _ in range(15)]
        assert len(items) == 15

    def test_shuffle_with_seed(self):
        ds = MultiWorkerStreamDataset([(i, 0) for i in range(10)], shuffle=True, seed=42)
        seen_first = set()
        stream = ds.get_stream()
        for _ in range(100):
            x, _ = next(stream)
            seen_first.add(x.item())
        assert len(seen_first) == 10

    def test_no_shuffle_without_seed(self):
        ds = MultiWorkerStreamDataset([(i, 0) for i in range(5)], shuffle=False)
        stream = ds.get_stream()
        items = [next(stream) for _ in range(10)]
        assert len(items) == 10
        assert items[0] == (torch.tensor([0]), torch.tensor([0]))


# ---------- Inference Datasets ---

class TestSingleWorkerInferenceDataset:
    def test_requires_preprocess(self):
        ds = SingleWorkerInferenceDataset([1, 2, 3])
        with pytest.raises(NotImplementedError):
            list(ds)

    def test_custom_preprocess(self):
        class DS(SingleWorkerInferenceDataset):
            def preprocess(self, point):
                return torch.tensor([point * 2])
        ds = DS([1, 2, 3])
        items = list(ds)
        assert len(items) == 3
        assert items[0].item() == 2
        assert items[1].item() == 4
        assert items[2].item() == 6

    def test_len(self):
        ds = SingleWorkerInferenceDataset([1, 2, 3, 4])
        assert len(ds) == 4

    def test_multiple_iterations(self):
        class DS(SingleWorkerInferenceDataset):
            def preprocess(self, point):
                return torch.tensor([point])
        ds = DS([1, 2, 3])
        a = list(ds)
        b = list(ds)
        assert [x.item() for x in a] == [x.item() for x in b] == [1, 2, 3]

    def test_empty_data(self):
        class DS(SingleWorkerInferenceDataset):
            def preprocess(self, point):
                return torch.tensor([point])
        ds = DS([])
        items = list(ds)
        assert items == []

    def test_get_stream(self):
        class DS(SingleWorkerInferenceDataset):
            def preprocess(self, point):
                return torch.tensor([point])
        ds = DS([10, 20])
        stream = ds.get_stream()
        assert isinstance(stream, Iterator)


class TestMultiWorkerInferenceDataset:
    def test_custom_preprocess(self):
        class DS(MultiWorkerInferenceDataset):
            def preprocess(self, point):
                return torch.tensor([point * 3])
        ds = DS([1, 2, 3, 4, 5])
        items = list(ds)
        assert len(items) == 5
        assert items[0].item() == 3
        assert items[4].item() == 15

    def test_len(self):
        ds = MultiWorkerInferenceDataset([1, 2, 3])
        assert len(ds) == 3

    def test_worker_detection_no_dataloader(self):
        ds = MultiWorkerInferenceDataset([1, 2, 3])
        ds.__iter__()
        assert ds.multi is False
        assert ds.worker_id == 0
        assert ds.n_workers == 1

    def test_multiple_iterations(self):
        class DS(MultiWorkerInferenceDataset):
            def preprocess(self, point):
                return torch.tensor([point])
        ds = DS([10, 20, 30])
        a = list(ds)
        b = list(ds)
        assert [x.item() for x in a] == [x.item() for x in b] == [10, 20, 30]

    def test_empty_data(self):
        class DS(MultiWorkerInferenceDataset):
            def preprocess(self, point):
                return torch.tensor([point])
        ds = DS([])
        items = list(ds)
        assert items == []
