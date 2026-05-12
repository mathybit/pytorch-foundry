"""
Inference Datasets
=======

Single-pass iterable datasets for inference (X-only) with optional
multi-worker support. Designed for deployment and offline batch inference.
"""

import itertools
import random
from typing import List, Any, Optional, Iterator
import torch
from torch.utils.data import IterableDataset, get_worker_info


class SingleWorkerInferenceDataset(IterableDataset):
    """Inference dataset returning only X (no target).

    Iterates through the data exactly once, then stops. Suitable for
    prediction pipelines where no ground truth is available.

    Example::

        ds = SingleWorkerInferenceDataset([file_path1, file_path2, ...])
        for batch in DataLoader(ds, batch_size=32):
            outputs = model(batch)  # batch is a tensor X
    """

    def __init__(self, data: List[Any]) -> None:
        super().__init__()
        self.data = data

    def __len__(self) -> int:
        return len(self.data)

    def preprocess(self, point: Any) -> torch.Tensor:
        """Override to load and convert a sample.

        Return a tensor X. Override this method in your model's dataset.py
        to implement file loading, normalization, and tensor conversion.

        Args:
            point: Raw sample data (e.g. file path, bytes, metadata dict).

        Returns:
            X tensor.
        """
        raise NotImplementedError("InferenceDataset.preprocess() must be implemented")

    def parse_data(self, data: List[Any]) -> Iterator[torch.Tensor]:
        """Yield preprocessed samples."""
        for point in data:
            yield self.preprocess(point)

    def get_stream(self) -> Iterator[torch.Tensor]:
        """Return an iterator over the full data sequence."""
        return self.parse_data(self.data)

    def __iter__(self) -> Iterator[torch.Tensor]:
        return self.get_stream()


class MultiWorkerInferenceDataset(SingleWorkerInferenceDataset):
    """Inference dataset with automatic multi-worker partitioning.

    Detects PyTorch DataLoader worker info on ``__iter__`` call and
    assigns each worker a disjoint strided slice of the data.

    Example::

        ds = MultiWorkerInferenceDataset(data, n_workers=4)
        loader = DataLoader(ds, batch_size=32, num_workers=4)
        # Worker 0: indices 0,4,8,...
        # Worker 1: indices 1,5,9,...
    """

    def __init__(self, data: List[Any]) -> None:
        super().__init__(data)
        self.multi = False
        self.n_workers = 1
        self.worker_id = 0

    def parse_data(self, data: List[Any]) -> Iterator[torch.Tensor]:
        if not self.multi:
            data_iterator = data
        else:
            data_iterator = itertools.islice(data, self.worker_id, None, self.n_workers)
        yield from map(self.preprocess, data_iterator)

    def __len__(self) -> int:
        return len(self.data)

    def __iter__(self) -> Iterator[torch.Tensor]:
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
