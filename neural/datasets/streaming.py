"""
Streaming Datasets
=================
Infinite-stream iterable datasets for continuous training loops.

Uses ``while True: yield from`` to cycle through data without buffering
every yielded item in memory (avoiding the RAM leak of ``itertools.cycle``).
Supports deterministic shuffling via a seed and automatic multi-worker
partitioning for distributed DataLoader integration.
"""

import itertools
import random
from typing import List, Any, Tuple, Optional, Iterator
import torch
from torch.utils.data import IterableDataset, get_worker_info


class SingleWorkerStreamDataset(IterableDataset):
    """Infinite stream over a fixed sequence of (x, y) samples.

    Yields preprocessed samples indefinitely. The default ``preprocess``
    method extracts the first two elements of each sample and converts
    them to tensors.

    Example::

        ds = SingleWorkerStreamDataset([(x, y), ...])
        for batch in DataLoader(ds, batch_size=32):
            process(batch)  # runs forever
    """

    def __init__(self, data: List[Any]) -> None:
        super().__init__()
        self.data = data

    def preprocess(self, point: Any) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract and convert a sample point to (x, y) tensors.

        Override to customize parsing logic.

        Args:
            point: Expected to be indexable (e.g. ``(x, y)``). Scalar
                   values are wrapped in a list before tensor conversion.

        Returns:
            (x_tensor, y_tensor)
        """
        x = point[0]
        y = point[1]
        x = [x] if not isinstance(x, (list, tuple)) else x
        y = [y] if not isinstance(y, (list, tuple)) else y
        return torch.tensor(x), torch.tensor(y)

    def parse_data(self, data: List[Any]) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        for point in data:
            yield self.preprocess(point)

    def get_stream(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        """Return an infinite stream.

        Uses ``yield from`` to re-generate each pass from scratch,
        preventing the RAM accumulation caused by ``itertools.cycle``.
        """
        while True:
            yield from self.parse_data(self.data)

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        return self.get_stream()


class ShuffledStreamDataset(SingleWorkerStreamDataset):
    """Streaming dataset that re-shuffles on every cycle.

    When ``shuffle=True``, a seed is required. The seed is incremented
    after each shuffle so the next pass produces a different but
    reproducible ordering.

    Example::

        ds = ShuffledStreamDataset(data, shuffle=True, seed=31337)
        # First cycle: seed=31337, then seed becomes 31338
        # Second cycle: seed=31338, then seed becomes 31339
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
        """Return a deterministically shuffled copy of the data."""
        if not self.shuffle:
            return self.data

        if self.seed is not None:
            random.seed(self.seed)
            self.seed += 1

        return random.sample(self.data, k=len(self.data))

    def get_stream(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        while True:
            shuffled_data = self.get_shuffled_data()
            yield from self.parse_data(shuffled_data)


class MultiWorkerStreamDataset(ShuffledStreamDataset):
    """Streaming dataset with automatic multi-worker partitioning.

    Detects PyTorch DataLoader worker info on ``__iter__`` call and
    assigns each worker a disjoint strided slice of the data.

    Example::

        ds = MultiWorkerStreamDataset(data, shuffle=True, seed=31337)
        loader = DataLoader(ds, batch_size=32, num_workers=4)
        # Worker 0: indices 0,4,8,...
        # Worker 1: indices 1,5,9,...
    """

    def __init__(
        self, data: List[Any], shuffle: bool = True, seed: Optional[int] = None
    ) -> None:
        super().__init__(data, shuffle, seed)

        # We need these variables set to something. They will be properly initialized in __iter__,
        # once worker info is available.
        self.multi = False
        self.n_workers = 1
        self.worker_id = 0

    def parse_data(self, data: List[Any]) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        if not self.multi:
            data_iterator = data
        else:
            data_iterator = itertools.islice(data, self.worker_id, None, self.n_workers)

        yield from map(self.preprocess, data_iterator)

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
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

    def __len__(self) -> int:
        return len(self.data)
