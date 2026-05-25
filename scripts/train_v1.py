"""
scripts/train_v1.py
====================
V1 (Autoencoder + Ensemble Kümeleme) eğitim giriş noktası.

Kullanım:
    python scripts/train_v1.py

Çıktılar: artifacts/v1/ dizinine yazılır.
"""
import json
import logging
import sys
from pathlib import Path

# Proje kökünü sys.path'e ekle (script olarak çalıştırılabilmesi için)
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances

try:
    import umap.umap_ as umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False

from src.config import (
    V1_DIR, FEATURES, FEATURE_GROUPS, TRAINING, AE_LATENT_DIM,
    CLUSTER_K_RANGE,
)
from src.data.loader import load_dataset
from src.data.preprocessor import prepare_data
from src.models.autoencoder import build_autoencoder
from src.training.trainer import train_autoencoder
from src.training.clustering import (
    find_best_k_ensemble,
    run_ensemble_clustering,
    cluster_summary,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_DIR   = V1_DIR / "loglar"
PLOT_DIR  = V1_DIR / "grafikler"
MODEL_DIR = V1_DIR / "modeller"
for d in [LOG_DIR, PLOT_DIR, MODEL_DIR]:
    d.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "sistem_logu.txt", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("TrainV1")

log.info("=" * 65)
log.info("FUTBOLCU ANALİZ — V1 (Autoencoder) Eğitimi")
log.info("=" * 65)

# ---------------------------------------------------------------------------
# 1. Veri Yükleme & Ön-İşleme
# ---------------------------------------------------------------------------
log.info("Veri yükleniyor...")
players, features_df = load_dataset()
log.info(f"Oyuncu: {len(players)} | Feature: {len(FEATURES)}")

X_clean, X_scaled, scaler, X_train, X_val, X_test = prepare_data(features_df)
log.info(f"Split → Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

# ---------------------------------------------------------------------------
# 2. Model Oluşturma & Eğitim
# ---------------------------------------------------------------------------
log.info(f"Autoencoder oluşturuluyor (latent_dim={AE_LATENT_DIM})...")
model = build_autoencoder()
param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
log.info(f"Parametre sayısı: {param_count:,}")

log.info("Eğitim başlıyor...")
history = train_autoencoder(
    model=model,
    X_train=X_train,
    X_val=X_val,
    X_test=X_test,
    save_path=MODEL_DIR / "best_autoencoder.pth",
)

# ---------------------------------------------------------------------------
# 3. Latent Vektörlerin Çıkarılması
# ---------------------------------------------------------------------------
log.info("Latent vektörler çıkarılıyor...")
model.eval()
with torch.no_grad():
    full_tensor    = torch.FloatTensor(X_scaled)
    latent_vectors, _ = model(full_tensor)
latent_vectors = latent_vectors.numpy()
log.info(f"Latent boyutu: {latent_vectors.shape}")

# ---------------------------------------------------------------------------
# 4. Kümeleme
# ---------------------------------------------------------------------------
log.info("En iyi k aranıyor...")
best_k, cluster_metrics = find_best_k_ensemble(latent_vectors, CLUSTER_K_RANGE)
history["best_k"]         = best_k
history["cluster_metrics"] = cluster_metrics

log.info(f"Ensemble kümeleme uygulanıyor (k={best_k})...")
clusters, gmm_probs, gmm_final = run_ensemble_clustering(latent_vectors, best_k)
players["Cluster"] = clusters

for cid, cnt in pd.Series(clusters).value_counts().sort_index().items():
    log.info(f"  Cluster {cid}: {cnt} oyuncu")

# ---------------------------------------------------------------------------
# 5. Boyut İndirgeme (t-SNE + UMAP)
# ---------------------------------------------------------------------------
log.info("t-SNE hesaplanıyor...")
pca_pre   = PCA(n_components=min(AE_LATENT_DIM, latent_vectors.shape[1]))
lat_pca   = pca_pre.fit_transform(latent_vectors)
tsne      = TSNE(n_components=2, perplexity=40, max_iter=2000,
                 learning_rate="auto", init="pca", random_state=42)
latent_2d = tsne.fit_transform(lat_pca)
log.info("t-SNE tamamlandı.")

if UMAP_AVAILABLE:
    log.info("UMAP hesaplanıyor...")
    reducer     = umap.UMAP(n_components=2, n_neighbors=20, min_dist=0.1,
                            metric="euclidean", random_state=42)
    latent_umap = reducer.fit_transform(latent_vectors)
    log.info("UMAP tamamlandı.")
else:
    log.warning("umap-learn kurulu değil, UMAP yerine t-SNE kullanılacak.")
    latent_umap = latent_2d

# ---------------------------------------------------------------------------
# 6. Çıktıları Kaydet
# ---------------------------------------------------------------------------
log.info("Çıktılar kaydediliyor...")

# Latent features
latent_out = pd.DataFrame(
    latent_vectors, columns=[f"L{i+1}" for i in range(AE_LATENT_DIM)]
)
latent_out["Cluster"]        = clusters
latent_out["GMM_Confidence"] = gmm_probs.max(axis=1)
latent_out["Player"]         = players["Player"].values
latent_out.to_csv(V1_DIR / "latent_features.csv", index=False)

# UMAP koordinatları
pd.DataFrame({
    "Player":  players["Player"].values,
    "Cluster": clusters,
    "tSNE_1":  latent_2d[:, 0], "tSNE_2": latent_2d[:, 1],
    "UMAP_1":  latent_umap[:, 0], "UMAP_2": latent_umap[:, 1],
}).to_csv(V1_DIR / "tsne_umap_koordinatlari.csv", index=False)

# Cluster ortalamaları
cluster_means = cluster_summary(features_df, clusters, FEATURES)
cluster_means.round(4).to_csv(V1_DIR / "cluster_feature_ortalamalar.csv")

# Eğitim geçmişi
with open(LOG_DIR / "egitim_gecmisi.json", "w", encoding="utf-8") as f:
    json.dump(history, f, ensure_ascii=False, indent=2)

log.info("=" * 65)
log.info("V1 EĞİTİMİ TAMAMLANDI")
log.info(f"Çıktılar: {V1_DIR.resolve()}")
log.info("=" * 65)
