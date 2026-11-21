# Sektörler Arası Şok Yayılımı

Bir sektörde başlayan piyasa şokunun hangi koşullarda başka sektörlere yayıldığını ve sonraki dönemlerde hangi sektörlerde risk artışıyla birlikte görüldüğünü inceleyen bitirme projesi. Önce klasik modeller (LSTM), sonra GNN / Temporal GNN kullanılır.

## Veri

[Kenneth R. French Data Library](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html), "49 Industry Portfolios [Daily]".

- Getiriler değer ağırlıklı, yüzde cinsinden ve 1926-07-01 → 2026-09-30 aralığını kapsıyor.
- Sektörler veri setindeki 49 haliyle kullanılır.
- Sektör tanımları `data/processed/Siccodes49.txt` dosyasında.

## Dizin yapısı

```
src/spillover/     Python paketi (yalnızca kod)
data/raw/          indirilen ham dosyalar (git'e girmez)
data/processed/    temizlenmiş veri; kodun okuduğu dosyalar
scripts/           veri temizleme ve deney çalıştırma betikleri
configs/           deney ayarları (YAML)
notebooks/         veri keşfi ve sonuç notebook'ları
tests/             birim ve sızıntı testleri
outputs/           deney çıktıları (git'e girmez)
```

## Colab'da kurulum

```bash
git clone <depo-adresi>
cd <depo-klasörü>
pip install -r requirements.txt
pip install -e .
```

## Ham veriden temiz dosya üretmek

```bash
python scripts/temizle_fama_french.py data/raw/49_Industry_Portfolios_daily_CSV.zip
python scripts/temizle_ff3_factors.py data/raw/F-F_Research_Data_Factors_daily_CSV.zip
```

Not: Betikler çıktıyı girdi dosyasının yanına, yani `data/raw/` klasörüne yazar. Üretilen dosyayı `data/processed/` klasörüne taşıman gerekir.
