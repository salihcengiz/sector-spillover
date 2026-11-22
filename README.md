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

## Colab'da çalıştırma

Hazır akış `notebooks/00_colab_kurulum.ipynb` notebook'unda. Colab'da *File → Open notebook → GitHub* menüsünden bu depoyu açıp notebook'u seçmen yeterli. Notebook şu adımları sırayla yapar: klon → kurulum → testler → hızlı deneme → tam koşular → sonuçları Drive'a kopyalama.

Elle çalıştırmak istersen:

```bash
git clone https://github.com/salihcengiz/sector-spillover.git
cd sector-spillover
pip install -e .          # Colab'da torch/numpy/pandas vb. hazır; requirements.txt kurulmaz
python -m pytest -q
python scripts/run_experiment.py --config configs/lstm_monthly_b.yaml --quick
python scripts/run_experiment.py --config configs/lstm_monthly_b.yaml
```

## Deneyler

| Config | Görünüm | Bölme |
|---|---|---|
| `configs/lstm_monthly_b.yaml` | aylık | B: train ≤ 2010, val 2011–2016, test 2017 → 2026-09; embargo 1 ay |
| `configs/lstm_daily_crisis.yaml` | günlük | COVID, SVB ve son 1 yıl için ayrı katmanlar; her katmanda yalnızca öncesindeki veriyle eğitim; embargo 22 gün |
| `configs/lstm_daily_b.yaml` | günlük | B (ön analizle karşılaştırma için) |
| `configs/lstm_monthly_a.yaml` | aylık | A: walk-forward, test 2017 → 2026-09, her yıl yeniden eğitim |
| `configs/ghar_monthly_b.yaml` / `ghar_monthly_a.yaml` | aylık | GHAR graf modeli (Zhang vd., 2025): GLASSO / korelasyon / boş graf × hocanın özellikleri / HAR; B ve A |

Seçenekler: `--quick` (tek seed, 2 epoch), `--folds covid svb` (yalnızca seçilen katmanlar), `--baselines-only` (yalnızca kıyas modelleri; torch gerekmez). Çıktılar `outputs/runs/<zaman>_<ad>/` altına yazılır.

## Ham veriden temiz dosya üretmek

```bash
python scripts/temizle_fama_french.py data/raw/49_Industry_Portfolios_daily_CSV.zip
python scripts/temizle_ff3_factors.py data/raw/F-F_Research_Data_Factors_daily_CSV.zip
```

Not: Betikler çıktıyı girdi dosyasının yanına, yani `data/raw/` klasörüne yazar. Üretilen dosyayı `data/processed/` klasörüne taşıman gerekir.
