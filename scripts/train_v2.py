"""
scripts/train_v2.py
====================
V2 (VAE + GMM Kümeleme) eğitim giriş noktası.

Kullanım:
    python scripts/train_v2.py

Çıktılar: artifacts/v2/ dizinine yazılır.
"""
import json
import logging
import sys
from pathlib import Path
import joblib

# Windows terminali UTF-8'e zorla (μ, Türkçe karakterler için)
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

try:
    import umap.umap_ as umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False

from src.config import (
    V2_DIR, FEATURES, VAE_LATENT_DIM,
    CLUSTER_K_RANGE, VAE_TRAINING,
)
from src.data.loader import load_dataset
from src.data.preprocessor import prepare_data
from src.models.vae import build_vae, vae_loss
from src.training.trainer import train_vae
from src.training.clustering import (
    find_best_k_gmm,
    run_gmm_clustering,
    cluster_summary,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_DIR   = V2_DIR / "loglar"
PLOT_DIR  = V2_DIR / "grafikler"
MODEL_DIR = V2_DIR / "modeller"
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
log = logging.getLogger("TrainV2")

log.info("=" * 65)
log.info("FUTBOLCU ANALİZ — V2 (VAE + GMM) Eğitimi")
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
log.info(f"VAE oluşturuluyor (latent_dim={VAE_LATENT_DIM})...")
model = build_vae()
param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
log.info(f"Parametre sayısı: {param_count:,}")

log.info("VAE eğitimi başlıyor (β-Annealing)...")
history = train_vae(
    model=model,
    loss_fn=vae_loss,
    X_train=X_train,
    X_val=X_val,
    X_test=X_test,
    save_path=MODEL_DIR / "best_vae.pth",
    beta_start=VAE_TRAINING["beta_start"],
    beta_end=VAE_TRAINING["beta_end"],
    beta_warmup=VAE_TRAINING["beta_warmup"],
)

# ---------------------------------------------------------------------------
# 3. Latent μ Vektörlerinin Çıkarılması
# ---------------------------------------------------------------------------
log.info("Latent μ vektörleri çıkarılıyor...")
model.eval()
full_tensor = torch.FloatTensor(X_scaled)
with torch.no_grad():
    h          = model.encoder(full_tensor)
    mu_vectors = model.fc_mu(h).numpy()
log.info(f"Latent boyutu: {mu_vectors.shape}")

# Reconstruction hataları
with torch.no_grad():
    recon_all, _, _ = model(full_tensor)
per_sample_recon = ((recon_all - full_tensor) ** 2).mean(dim=1).detach().numpy()

# ---------------------------------------------------------------------------
# 4. GMM Kümeleme
# ---------------------------------------------------------------------------
log.info("GMM ile optimal k aranıyor...")
best_k, cluster_metrics, gmm_models = find_best_k_gmm(mu_vectors, CLUSTER_K_RANGE)
history["best_k"]          = best_k
history["cluster_metrics"] = cluster_metrics

log.info(f"Nihai GMM kümeleme uygulanıyor (k={best_k})...")
clusters, gmm_probs, gmm_final = run_gmm_clustering(mu_vectors, gmm_models, best_k)
players = players.copy()          # defragment — PerformanceWarning onlemi
players["Cluster"] = clusters

for cid, cnt in pd.Series(clusters).value_counts().sort_index().items():
    log.info(f"  Cluster {cid}: {cnt} oyuncu")

# ---------------------------------------------------------------------------
# 5. Boyut İndirgeme (t-SNE + UMAP)
# ---------------------------------------------------------------------------
log.info("t-SNE hesaplanıyor...")
pca_pre   = PCA(n_components=min(VAE_LATENT_DIM, mu_vectors.shape[1]))
lat_pca   = pca_pre.fit_transform(mu_vectors)
tsne      = TSNE(n_components=2, perplexity=40, max_iter=2000,
                 learning_rate="auto", init="pca", random_state=42)
latent_2d = tsne.fit_transform(lat_pca)
log.info("t-SNE tamamlandı.")

if UMAP_AVAILABLE:
    log.info("UMAP hesaplanıyor...")
    reducer     = umap.UMAP(n_components=2, n_neighbors=20, min_dist=0.1,
                            metric="euclidean", random_state=42)
    latent_umap = reducer.fit_transform(mu_vectors)
    log.info("UMAP tamamlandı.")
    #  UMAP modelini kaydet — yeni oyuncuları haritaya eklemek için
    joblib.dump(reducer, MODEL_DIR / "umap_model.pkl")
    log.info(f"UMAP modeli kaydedildi: {MODEL_DIR / 'umap_model.pkl'}")
else:
    log.warning("umap-learn kurulu değil, UMAP yerine t-SNE kullanılacak.")
    latent_umap = latent_2d

# ---------------------------------------------------------------------------
# 6. Çıktıları Kaydet
# ---------------------------------------------------------------------------
log.info("Çıktılar kaydediliyor...")

# Latent features (μ vektörleri)
latent_out = pd.DataFrame(
    mu_vectors, columns=[f"mu_{i+1}" for i in range(VAE_LATENT_DIM)]
)
latent_out["Cluster"]        = clusters
latent_out["GMM_Confidence"] = gmm_probs.max(axis=1)
latent_out["Recon_Error"]    = per_sample_recon
latent_out["Player"]         = players["Player"].values
latent_out.to_csv(V2_DIR / "latent_features.csv", index=False)

# UMAP koordinatları
pd.DataFrame({
    "Player":  players["Player"].values,
    "Cluster": clusters,
    "tSNE_1":  latent_2d[:, 0], "tSNE_2": latent_2d[:, 1],
    "UMAP_1":  latent_umap[:, 0], "UMAP_2": latent_umap[:, 1],
}).to_csv(V2_DIR / "tsne_umap_koordinatlari.csv", index=False)

# GMM olasılıkları
gmm_prob_df = pd.DataFrame(gmm_probs, columns=[f"P_C{i}" for i in range(best_k)])
gmm_prob_df["Player"]  = players["Player"].values
gmm_prob_df["Cluster"] = clusters
gmm_prob_df.to_csv(V2_DIR / "gmm_olasiliklar.csv", index=False)

# Cluster ortalamaları
cluster_means = cluster_summary(features_df, clusters, FEATURES)
cluster_means.round(4).to_csv(V2_DIR / "cluster_feature_ortalamalar.csv")

# Model son kopyası
torch.save(model.state_dict(), MODEL_DIR / "football_vae_final.pth")

# GMM modelini kaydet — inference sırasında confidence hesabı için
joblib.dump(gmm_final, MODEL_DIR / "gmm_model.pkl")
log.info(f"GMM modeli kaydedildi: {MODEL_DIR / 'gmm_model.pkl'}")

# Eğitim geçmişi
# BUG 1 FIX: Gerçek ortalama Silhouette skoru (np.mean([0]) her zaman 0 veriyordu)
history["avg_silhouette"]       = round(float(np.mean(cluster_metrics["silhouette"])), 4)
history["anomaly_threshold"]    = round(float(np.percentile(per_sample_recon, 95)), 6)
with open(LOG_DIR / "egitim_gecmisi.json", "w", encoding="utf-8") as f:
    json.dump(history, f, ensure_ascii=False, indent=2)

log.info("=" * 65)
log.info("V2 EĞİTİMİ TAMAMLANDI")
log.info(f"Çıktılar: {V2_DIR.resolve()}")
log.info("=" * 65)
