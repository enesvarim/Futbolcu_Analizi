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
