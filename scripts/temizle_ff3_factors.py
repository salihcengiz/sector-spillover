#!/usr/bin/env python3
"""Fama-French 3 Factors daily: clean raw CSV/ZIP into one daily table.

Usage:
  python3 temizle_ff3_factors.py ~/Desktop/F-F_Research_Data_Factors_daily.csv
  python3 temizle_ff3_factors.py ~/Downloads/F-F_Research_Data_Factors_daily_CSV.zip

Output: ff3_daily_clean.csv next to the input file.
Values remain in percentage points (e.g. 1.25 means 1.25%).
"""
import csv
import io
import re
import sys
import zipfile
from pathlib import Path

import pandas as pd

COLUMNS = ["Mkt-RF", "SMB", "HML", "RF"]


def read_raw(path: Path) -> str:
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as archive:
            files = [n for n in archive.namelist() if n.lower().endswith(".csv")]
            if not files:
                raise ValueError("ZIP içinde CSV dosyası bulunamadı.")
            raw = archive.read(files[0])
    else:
        raw = path.read_bytes()
    try:
        return raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        return raw.decode("latin-1")


def clean_daily_factors(text: str) -> pd.DataFrame:
    rows = list(csv.reader(io.StringIO(text)))
    header_row = None
    for i, row in enumerate(rows):
        normalized = [cell.strip() for cell in row]
        if len(normalized) >= 5 and normalized[1:5] == COLUMNS:
            header_row = i
            break
    if header_row is None:
        raise ValueError("'Mkt-RF, SMB, HML, RF' başlığı bulunamadı.")

    values = []
    for row in rows[header_row + 1:]:
        if not row:
            if values:
                break
            continue
        date_text = row[0].strip()
        if re.fullmatch(r"\d{8}", date_text) and len(row) >= 5:
            values.append([date_text] + [s.strip() for s in row[1:5]])
        elif values:
            break

    if not values:
        raise ValueError("Başlıktan sonra günlük YYYYMMDD veri satırı bulunamadı.")

    df = pd.DataFrame(values, columns=["Date"] + COLUMNS)
    df["Date"] = pd.to_datetime(df["Date"], format="%Y%m%d", errors="raise")
    for col in COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if df["Date"].duplicated().any():
        raise ValueError("Birden fazla aynı gün kaydı var; orijinal dosyayı kontrol edin.")
    return df.sort_values("Date").reset_index(drop=True)


def main():
    if len(sys.argv) != 2:
        print("Kullanım: python3 temizle_ff3_factors.py DOSYA.csv (veya DOSYA.zip)")
        sys.exit(1)
    input_file = Path(sys.argv[1]).expanduser()
    if not input_file.is_file():
        raise FileNotFoundError(input_file)
    df = clean_daily_factors(read_raw(input_file))
    output_file = input_file.with_name("ff3_daily_clean.csv")
    df.to_csv(output_file, index=False, date_format="%Y-%m-%d")
    print(f"Kaydedildi: {output_file}")
    print(f"Satır sayısı: {len(df)}, tarih aralığı: {df.Date.min().date()} – {df.Date.max().date()}")
    print("Boş değer sayısı:\n", df.isna().sum())
    print(df.head().to_string(index=False))
    print("NOT: Getiriler halen yüzde birimindedir. 1.25 = %1,25.")


if __name__ == "__main__":
    main()
