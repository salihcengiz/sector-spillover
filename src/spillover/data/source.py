"""Kaynak katmanı: temizlenmiş Fama–French 49 sektör günlük getiri dosyasını okur.

Bu katman yalnızca okuma ve doğrulama yapar. Özellik üretimi ve frekans dönüşümü
`views.py` içindeki görünüm sınıflarında yapılır.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

N_SECTORS = 49


def load_returns(path, start=None, end=None, allow_missing: bool = False) -> pd.DataFrame:
    """Günlük sektör getirilerini (yüzde cinsinden) DataFrame olarak döndürür.

    Satırlar işlem günleri (DatetimeIndex), sütunlar veri setindeki sırasıyla 49 sektördür.
    Sektör sütunlarına dokunulmaz; birleştirme veya yeniden adlandırma yapılmaz.

    Args:
        path: Temiz CSV dosyasının yolu (ilk sütun `Date`, dosyada UTF-8 BOM olabilir).
        start, end: Dahil edilecek ilk ve son tarih (örn. "1985-01-01"). None ise sınır yok.
        allow_missing: False ise seçilen dönemde eksik hücre varsa hata verir.
            Erken dönemlerde bazı sektörler yoktur; keşif analizinde True verilebilir.
    """
    df = pd.read_csv(path, encoding="utf-8-sig", parse_dates=["Date"], index_col="Date")
    if start is not None:
        df = df.loc[pd.Timestamp(start):]
    if end is not None:
        df = df.loc[:pd.Timestamp(end)]

    if df.shape[1] != N_SECTORS:
        raise ValueError(f"{N_SECTORS} sektör sütunu bekleniyordu, {df.shape[1]} bulundu.")
    if df.empty:
        raise ValueError("Seçilen tarih aralığında hiç gözlem yok.")
    if not df.index.is_monotonic_increasing:
        raise ValueError("Tarihler artan sırada değil.")
    if df.index.has_duplicates:
        raise ValueError("Aynı tarih birden fazla kez geçiyor.")

    df = df.astype("float64")
    n_missing = int(df.isna().sum().sum())
    if n_missing and not allow_missing:
        first_bad = df.index[df.isna().any(axis=1)][0].date()
        raise ValueError(
            f"Seçilen dönemde {n_missing} eksik hücre var (ilk: {first_bad}). "
            "Başlangıç tarihini ileri alın veya allow_missing=True kullanın."
        )
    if not np.isfinite(df.to_numpy()[~df.isna().to_numpy()]).all():
        raise ValueError("Sonsuz (inf) değer bulundu.")
    return df


_HEADER_LINE = re.compile(r"^\s*(\d+)\s+(\S+)\s+(.+?)\s*$")


def load_sector_names(path) -> dict:
    """`Siccodes49.txt` dosyasından {kısa ad: tam ad} sözlüğünü dosyadaki sırayla döndürür.

    Örnek: {"Agric": "Agriculture", ..., "Fin": "Trading", "Other": "Almost Nothing"}
    SIC aralığı satırları ("0100-0199 ...") başlık deseniyle eşleşmediği için atlanır.
    """
    names = {}
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        match = _HEADER_LINE.match(line)
        if match:
            names[match.group(2)] = match.group(3)
    if len(names) != N_SECTORS:
        raise ValueError(f"{N_SECTORS} sektör adı bekleniyordu, {len(names)} bulundu.")
    return names
