#!/usr/bin/env python3
"""Kenneth French 49 Industry Portfolios günlük CSV/ZIP dosyasının
ilk (value-weighted) günlük getiri tablosunu standart CSV'ye dönüştürür.
Yalnızca Python 3 standart kütüphanesini kullanır.

Kullanım:
    python3 temizle_fama_french.py ~/Desktop/49_Industry_Portfolios_Daily.csv
veya
    python3 temizle_fama_french.py ~/Downloads/49_Industry_Portfolios_daily_CSV.zip
"""
import argparse
import csv
import io
import re
import sys
import zipfile
from datetime import datetime
from pathlib import Path


def load_csv_text(path):
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as z:
            names = [n for n in z.namelist() if n.lower().endswith('.csv')]
            if not names:
                raise ValueError('ZIP içinde CSV bulunamadı.')
            raw = z.read(names[0])
    else:
        raw = path.read_bytes()
    return raw.decode('utf-8-sig', errors='replace')


def is_daily_date(cell):
    return bool(re.fullmatch(r'\d{8}', cell.strip()))


def parse_first_value_weighted_table(text):
    reader = list(csv.reader(io.StringIO(text)))
    marker = next((i for i, row in enumerate(reader)
                   if any('average value weighted returns' in c.lower()
                          and 'daily' in c.lower() for c in row)), None)
    if marker is None:
        raise ValueError('"Average Value Weighted Returns -- Daily" başlığı bulunamadı. Dosyanın günlük 49 sektör CSV olduğunu kontrol edin.')

    header_at = None
    for i in range(marker + 1, len(reader) - 1):
        row = reader[i]
        if len(row) == 50 and not row[0].strip() and all(x.strip() for x in row[1:]):
            if len(reader[i + 1]) == 50 and is_daily_date(reader[i + 1][0]):
                header_at = i
                break
        # Önceki tablo bloklarıyla karşılaşılırsa başlık aranır; veri başlamadan kesilmez.
    if header_at is None:
        raise ValueError('49 sektör sütun başlığı ve ardından günlük tarih satırı bulunamadı. Dosya formatı değişmiş olabilir.')

    header = ['Date'] + [h.strip() for h in reader[header_at][1:]]
    out_rows = []
    for row in reader[header_at + 1:]:
        if not row or not is_daily_date(row[0]):
            if out_rows:
                break
            continue
        if len(row) != 50:
            raise ValueError(f'{row[0]} tarihli satırda 49 yerine {len(row) - 1} sektör var.')
        day = datetime.strptime(row[0].strip(), '%Y%m%d').date().isoformat()
        values = []
        for v in row[1:]:
            v = v.strip()
            if not v or v in ('-99.99', '-999', '-99.9900', '-999.00'):
                values.append('')
            else:
                values.append(v)
        out_rows.append([day] + values)

    if not out_rows:
        raise ValueError('Günlük veri satırları bulunamadı.')
    return header, out_rows


def main():
    parser = argparse.ArgumentParser(description='Fama-French 49 sektör günlük getirilerini standart CSV yapar.')
    parser.add_argument('input', help='İndirilen CSV veya ZIP dosyasının yolu')
    args = parser.parse_args()
    source = Path(args.input).expanduser().resolve()
    if not source.is_file():
        sys.exit(f'Dosya bulunamadı: {source}')

    try:
        text = load_csv_text(source)
        header, rows = parse_first_value_weighted_table(text)
    except Exception as exc:
        sys.exit(f'Hata: {exc}')

    target = source.parent / '49_sektor_gunluk_getiri_temiz.csv'
    with target.open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    print(f'BAŞARILI: {target}')
    print(f'Gözlem sayısı: {len(rows):,}')
    print(f'Sektör sayısı: {len(header) - 1}')
    print(f'Tarih aralığı: {rows[0][0]} - {rows[-1][0]}')
    print('Not: Getiriler orijinal dosyadaki gibi yüzde cinsinden tutulmuştur (örn. 2.50 = %2,50).')
    print('Dosyayı Tad, Excel veya pandas ile açabilirsiniz.')


if __name__ == '__main__':
    main()
