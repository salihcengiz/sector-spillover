"""Pencereleme: modelin göreceği (lookback, N, F) girdi pencerelerini üretir.

Bu modül torch'a bağımlı değildir; numpy dizileri döndürür. Tensöre çevirme işi
eğitim katmanında yapılır. Böylece aynı veri hattı farklı modellerle kullanılabilir
ve torch olmadan test edilebilir.
"""
from __future__ import annotations

from typing import Iterator, Optional, Tuple

import numpy as np


class WindowDataset:
    """Örnek indeksleri üzerinde kayan pencere veri kümesi.

    Bir örnek (t için):
      x : X[t-lookback+1 … t]  boyut (lookback, N, F)
      y : y[t]                 boyut (N,)
    """

    def __init__(self, X: np.ndarray, y: np.ndarray, indices: np.ndarray, lookback: int):
        self.X = np.ascontiguousarray(X, dtype=np.float32)
        self.y = np.ascontiguousarray(y, dtype=np.float32)
        self.indices = np.asarray(indices, dtype=np.int64)
        self.lookback = int(lookback)
        if self.indices.size and self.indices.min() - self.lookback + 1 < 0:
            raise ValueError("Bazı örneklerin girdi penceresi verinin başından önceye taşıyor.")
        if self.indices.size and not np.isfinite(self.y[self.indices]).all():
            raise ValueError("Bazı örneklerin hedefi tanımsız (NaN).")

    def __len__(self) -> int:
        return len(self.indices)

    def get(self, positions) -> Tuple[np.ndarray, np.ndarray]:
        """positions: bu veri kümesi içindeki sıra numaraları → (B, L, N, F), (B, N)."""
        t = self.indices[np.asarray(positions, dtype=np.int64)]
        offsets = t[:, None] - self.lookback + 1 + np.arange(self.lookback)
        return self.X[offsets], self.y[t]

    def batches(self, batch_size: int, shuffle: bool = False,
                rng: Optional[np.random.Generator] = None) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Mini-batch üretir. shuffle=True ise sıra rng ile karıştırılır (tekrarlanabilirlik için rng verin)."""
        order = np.arange(len(self))
        if shuffle:
            order = (rng or np.random.default_rng()).permutation(order)
        for start in range(0, len(order), batch_size):
            yield self.get(order[start:start + batch_size])
