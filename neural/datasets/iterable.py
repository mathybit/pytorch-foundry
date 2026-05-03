"""
Iterable Datasets
=================
Single-pass iterable datasets for one-time iteration over fixed data.

Designed for validation and test splits, deterministic evaluation pipelines,
and multi-worker DataLoader integration. Supports optional shuffling with
reproducible progression across iterations.
"""

import random
from typing import List, Any, Optional, Iterator
from torch.utils.data import IterableDataset, get_worker_info


class SingleWorkerIterableDataset(IterableDataset):
    """One-pass iterator over a fixed sequence of samples.

    Iterates through the data exactly once, then stops. Suitable for
    validation and test sets where epoch boundaries do not exist.

    Example::

        ds = SingleWorkerIterableDataset([1, 2, 3])
        for batch in DataLoader(ds, batch_size=2):
            process(batch)  # yields once: [1,2] then [3]
    """

    def __init__(self, data: List[Any]) -> None:
        super().__init__()
        self.data = data

    def __len__(self) -> int:
        return len(self.data)

    def preprocess(self, point: Any) -> Optional[Any]:
        """Override to transform individual samples.

        Return ``None`` to exclude a sample from the stream.
        """
        return point

    def parse_data(self, data: List[Any]) -> Iterator[Any]:
        """Yield preprocessed samples, filtering out ``None`` values."""
        for point in data:
            processed = self.preprocess(point)
            if processed is not None:
                yield processed

    def get_stream(self) -> Iterator[Any]:
        """Return an iterator over the full data sequence."""
        return self.parse_data(self.data)

    def __iter__(self) -> Iterator[Any]:
        return self.get_stream()


class ShuffledIterableDataset(SingleWorkerIterableDataset):
    """Iterable dataset that shuffles data on each iteration.

    When ``shuffle=True``, a deterministic seed is required. The seed
    is incremented after each iteration, so the same sequence of
    shuffles is produced regardless of how many times ``__iter__``
    is called.

    Example::

        ds = ShuffledIterableDataset(data, shuffle=True, seed=31337)
        # First pass: seed=31337, then seed becomes 31338
        # Second pass: seed=31338, then seed becomes 31339
    """

    def __init__(
        self, data: List[Any], shuffle: bool = True, seed: Optional[int] = None
    ) -> None:
        super().__init__(data)
        self.shuffle = shuffle
        if self.shuffle:
            if seed is None or not isinstance(seed, int):
                raise ValueError("seed must be an integer when shuffle=True")
        self.seed = seed

    def get_shuffled_data(self) -> List[Any]:
        """Return the data list, optionally shuffled deterministically.

        Mutates ``self.seed`` by incrementing it when shuffle is enabled,
        so the next call produces a different but reproducible order.
        """
        if not self.shuffle:
            return self.data

        if self.seed is not None:
            random.seed(self.seed)
            self.seed += 1

        shuffled = list(self.data)
        random.shuffle(shuffled)
        return shuffled

    def get_stream(self) -> Iterator[Any]:
        shuffled_data = self.get_shuffled_data()
        return self.parse_data(shuffled_data)

    def __iter__(self) -> Iterator[Any]:
        return self.get_stream()


class MultiWorkerIterableDataset(ShuffledIterableDataset):
    """Iterable dataset with automatic multi-worker data partitioning.

    Detects PyTorch DataLoader worker info in ``__iter__`` and stores
    the result as instance state. ``get_stream`` uses this state to
    slice the data with stride-based partitioning.

    Mirrors the same pattern used by ``MultiWorkerStreamDataset``:
    worker detection happens once at iterator creation time, not
    during stream generation.

    Example::

        ds = MultiWorkerIterableDataset(data, shuffle=True, seed=31337)
        loader = DataLoader(ds, batch_size=32, num_workers=4)
        # Worker 0 gets indices 0,4,8,...
        # Worker 1 gets indices 1,5,9,...
    """

    def __init__(
        self, data: List[Any], shuffle: bool = True, seed: Optional[int] = None
    ) -> None:
        super().__init__(data, shuffle, seed)
        self.multi = False
        self.n_workers = 1
        self.worker_id = 0

    def __len__(self) -> int:
        return len(self.data)

    def get_stream(self) -> Iterator[Any]:
        shuffled_data = self.get_shuffled_data()

        if not self.multi:
            return self.parse_data(shuffled_data)

        return self.parse_data(shuffled_data[self.worker_id::self.n_workers])

    def __iter__(self) -> Iterator[Any]:
        winfo = get_worker_info()
        if winfo is not None:
            self.multi = True
            self.n_workers = winfo.num_workers
            self.worker_id = winfo.id
        else:
            self.multi = False
            self.n_workers = 1
            self.worker_id = 0
        return self.get_stream()
