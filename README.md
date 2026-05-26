# ⚽ Futbolcu Oyun Stili Analizi

Futbolcuların maç istatistiklerini **Autoencoder (V1)** ve **Variational Autoencoder (V2)** sinir ağları aracılığıyla analiz ederek oyun stillerini tespit eden ve benzer oyuncuları öneren yapay zeka projesi.

---

## Proje Yapısı

```
futbolcu-analiz/
│
├── src/                        ← Tüm kaynak kodu
│   ├── config.py               ← TEK konfigürasyon noktası (features, yollar, renkler)
│   ├── data/
│   │   ├── loader.py           ← CSV okuma, kaleci/süre filtreleme
│   │   └── preprocessor.py     ← RobustScaler, eksik değer, train/val/test bölme
│   ├── models/
│   │   ├── autoencoder.py      ← FootballAutoencoder (V1, latent_dim=12)
│   │   └── vae.py              ← FootballVAE + vae_loss (V2, latent_dim=16)
│   ├── training/
│   │   ├── trainer.py          ← Ortak eğitim döngüsü, early stopping, checkpoint
│   │   └── clustering.py       ← Ensemble (KMeans+Aggl+GMM) ve GMM kümeleme
│   ├── analysis/
│   │   ├── similarity.py       ← Hibrit benzerlik (Cosine 0.6 + Euclidean 0.4)
│   │   └── visualization.py    ← Plotly UMAP, radar, karşılaştırma grafikleri
│   └── app/
│       └── components.py       ← Streamlit UI bileşenleri (kartlar, stiller)
│
├── scripts/
│   ├── train_v1.py             ← V1 (Autoencoder) eğitim giriş noktası
│   └── train_v2.py             ← V2 (VAE + GMM) eğitim giriş noktası
│
├── artifacts/                  ← Model ağırlıkları ve çıktı CSV'leri (sabit dizin)
│   ├── v1/
│   │   ├── modeller/best_autoencoder.pth
│   │   ├── latent_features.csv
│   │   └── tsne_umap_koordinatlari.csv
│   └── v2/
│       ├── modeller/best_vae.pth
│       ├── latent_features.csv
│       └── tsne_umap_koordinatlari.csv
│
├── veriseti/
│   ├── futbolcular.csv         ← Ham oyuncu istatistikleri
│   └── temiz_veri.csv          ← Ön-işlenmiş feature matrisi
│
├── archive/                    ← Eski tarih damgalı çıktılar (arşiv)
├── app.py                      ← Streamlit giriş noktası
└── requirements.txt
```

---

## Kurulum

```bash
pip install -r requirements.txt
```

---

## Kullanım

### Uygulamayı Başlat

```bash
streamlit run app.py
```

### Model Yeniden Eğitimi

```bash
# V1 (Autoencoder + Ensemble Kümeleme)
python scripts/train_v1.py

# V2 (VAE + GMM Kümeleme)
python scripts/train_v2.py
```

> **Not:** Yeniden eğitim çıktıları otomatik olarak `artifacts/v1/` ve `artifacts/v2/` dizinlerine yazılır. Eski çıktıların üzerine yazar.

---

## Model Mimarisi

### V1 — FootballAutoencoder
- **Encoder:** 20 → 256 → 128 → 64 → 32 → **12** (latent)
- **Kümeleme:** KMeans + Agglomerative + GMM ensemble oylama
- **Metrik ağırlığı:** Silhouette 0.5 + Davies-Bouldin 0.3 + Calinski-Harabasz 0.2

### V2 — FootballVAE
- **Encoder:** 20 → 256 → 128 → 64 → (μ, σ²) → **16** (latent)
- **Loss:** Huber (rekon) + β·KL-Divergence (β-Annealing, warmup=50)
- **Kümeleme:** Gaussian Mixture Model, BIC 0.4 + Silhouette 0.4 + DB 0.2

### Benzerlik Hesabı
```
Hibrit Skor = 0.6 × Cosine Similarity + 0.4 × Euclidean Similarity
```

---

## Özellikler (20 Feature)

| Grup | Özellikler |
|------|-----------|
| Hücum | Gls, Ast, xG, xAG, npxG, Sh/90 |
| Pas & Yaratıcılık | Cmp%, PrgP, KP, PPA, SCA90 |
| Defans | Tkl, TklW, Int, Clr |
| Top Taşıma | PrgC, PrgR, Succ%, Carries, Touches |

---

## Oyun Stili Kümeleri (V2)

| Küme | İsim |
|------|------|
| 0 | Oyun Kurucular (Playmakers) |
| 1 | Pasör Stoperler (Ball-Playing CBs) |
| 2 | Dengeli/Klasik Savunmacılar |
| 3 | Dinamik Kanatlar & 10 Numaralar |
| 4 | İlerici Oyun Kurucular |
| 5 | Fırsatçı / Pivot Forvetler |
| 6 | Saf Bitiriciler (Pure Goalscorers) |
| 7 | Yok Ediciler & Dinamolar |



## Model V3 - SOM (Self-Organizing Map)

Bu projeye üçüncü model olarak SOM (Self-Organizing Map / Kohonen Haritası) eklenmiştir. SOM, gözetimsiz öğrenme yaklaşımıyla çalışan bir yapay sinir ağı modelidir. Oyuncuların 20 temel performans metriğine göre iki boyutlu bir harita üzerinde konumlandırılmasını sağlar.

### Kullanılan Özellikler

Model V3, diğer modellerle aynı 20 futbolcu istatistiğini kullanır:

- Hücum: Gls, Ast, xG, xAG, npxG, Sh/90
- Pas & Oyun Kurma: Cmp%, PrgP, KP, PPA, SCA90
- Defans: Tkl, TklW, Int, Clr
- Top Taşıma & Hareket: PrgC, PrgR, Succ%, Carries, Touches

### Ön İşleme

SOM modeli eğitilmeden önce veri setinde aşağıdaki filtreleme işlemleri uygulanmıştır:

- Kaleciler veri setinden çıkarılmıştır.
- Çok az süre alan oyuncular elenmiştir.
- En az 5 maçlık süreye karşılık gelen `90s >= 5` koşulu uygulanmıştır.
- Eksik değerler uygun şekilde doldurulmuştur.
- Sayısal özellikler ölçeklendirilmiştir.

Bu işlemler sonucunda model, 1798 futbolcu üzerinde eğitilmiştir.

### Model Yapısı

Model V3 için 12x12 boyutunda SOM haritası kullanılmıştır. Bu haritada her futbolcu, oyun stiline göre bir SOM bölgesine yerleştirilmiştir. Aynı veya komşu bölgelerde bulunan oyuncuların oyun stillerinin daha benzer olduğu kabul edilmiştir.

### Parametre Karşılaştırması

SOM modeli için farklı harita boyutları denenmiştir:

| Harita Boyutu | Kullanılan Hücre | Quantization Error | Topographic Error | Silhouette Score |
|---|---:|---:|---:|---:|
| 8x8 | 64 / 64 | 1.6596 | 0.0962 | 0.0701 |
| 10x10 | 100 / 100 | 1.5595 | 0.1652 | 0.0570 |
| 12x12 | 144 / 144 | 1.4811 | 0.1952 | 0.0522 |
| 15x15 | 225 / 225 | 1.3803 | 0.2230 | 0.0485 |

15x15 harita en düşük Quantization Error değerini üretmesine rağmen hücre başına düşen oyuncu sayısının bazı bölgelerde çok azalması ve Topographic Error değerinin yükselmesi nedeniyle fazla parçalanma riski taşımaktadır. 8x8 harita daha düşük Topographic Error üretmesine rağmen bazı hücrelerde oyuncu yoğunluğu artmıştır. Bu nedenle temsil gücü ve dengeli dağılım açısından 12x12 SOM haritası tercih edilmiştir.

### Streamlit Arayüz Entegrasyonu

Model V3, Streamlit arayüzüne entegre edilmiştir. Tekli oyuncu analizinde artık üç model birlikte görüntülenmektedir:

- Model V1: Autoencoder
- Model V2: VAE + GMM
- Model V3: SOM

Model V3 ekranında oyuncunun SOM bölgesi, en yakın veri seti oyuncusu ve SOM haritasına göre benzer oyuncular listelenmektedir.

### Üretilen Dosyalar

Model V3 eğitimi sonucunda aşağıdaki dosyalar oluşturulmuştur:

```text
artifacts/v3/modeller/som_model.pkl
artifacts/v3/modeller/som_scaler.pkl
artifacts/v3/som_koordinatlari.csv
artifacts/v3/som_benzer_oyuncular.csv
artifacts/v3/som_metrikleri.json
artifacts/v3/som_parametre_karsilastirma.csv
artifacts/v3/som_parametre_karsilastirma.json