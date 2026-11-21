"""Klasik LSTM modelleri (Aşama 1).

- CrossSectorLSTM (ana model): tüm sektörlerin geçmişini birlikte görür.
      (B, L, N, F) → (B, L, N·F) → LSTM → son gizli durum → dropout → Linear(hidden → N)
- OwnSectorLSTM (ablasyon): her sektör yalnızca kendi geçmişini görür. Aynı LSTM, ağırlıkları
  paylaşılarak her sektöre ayrı uygulanır; sektörü ayırt etmek için küçük bir sektör gömmesi eklenir.
      (B, L, N, F) → (B·N, L, F + E) → LSTM → son gizli durum → dropout → Linear(hidden → 1) → (B, N)
  Yapı gereği bir sektörün girdisi başka bir sektörün tahminini etkileyemez.

İkisinin farkı, "diğer sektörlerin geçmişi riski tahmin etmeye yardım ediyor mu?" sorusunu
cevaplar (Granger nedenselliğinin doğrusal olmayan bir karşılığı).
"""
from __future__ import annotations

import torch
from torch import nn

from .registry import register_model


def _lstm(input_size: int, hidden: int, layers: int, dropout: float) -> nn.LSTM:
    return nn.LSTM(input_size, hidden, num_layers=layers, batch_first=True,
                   dropout=dropout if layers > 1 else 0.0)


@register_model("cross_lstm")
class CrossSectorLSTM(nn.Module):
    def __init__(self, n_sectors: int, n_features: int, hidden: int = 64, layers: int = 1,
                 dropout: float = 0.2):
        super().__init__()
        self.n_sectors, self.n_features = n_sectors, n_features
        self.lstm = _lstm(n_sectors * n_features, hidden, layers, dropout)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, n_sectors)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, N, F = x.shape
        out, _ = self.lstm(x.reshape(B, L, N * F))
        return self.head(self.dropout(out[:, -1]))


@register_model("own_lstm")
class OwnSectorLSTM(nn.Module):
    def __init__(self, n_sectors: int, n_features: int, hidden: int = 64, layers: int = 1,
                 dropout: float = 0.2, embed_dim: int = 8):
        super().__init__()
        self.n_sectors, self.n_features = n_sectors, n_features
        self.embed = nn.Embedding(n_sectors, embed_dim)
        self.lstm = _lstm(n_features + embed_dim, hidden, layers, dropout)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, N, F = x.shape
        seq = x.permute(0, 2, 1, 3).reshape(B * N, L, F)                       # sektör sektör diziler
        emb = self.embed(torch.arange(N, device=x.device)).repeat(B, 1)          # (B·N, E)
        seq = torch.cat([seq, emb[:, None, :].expand(-1, L, -1)], dim=-1)
        out, _ = self.lstm(seq)
        return self.head(self.dropout(out[:, -1])).reshape(B, N)
