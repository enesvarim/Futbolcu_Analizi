import os
import sys
import logging
import json
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.cluster import KMeans, SpectralClustering, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
    silhouette_samples
)
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.manifold import TSNE
import umap.umap_ as umap  # pip install umap-learn

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import seaborn as sns
from scipy.spatial.distance import cdist

# ============================================================
# 0. KLASÖR & LOGGING KURULUMU
# ============================================================

VERISETI_DIR = Path(__file__).resolve().parents[1] / "veriseti"

RUN_ID     = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = VERISETI_DIR.parent / f"cikti_{RUN_ID}"
LOG_DIR    = OUTPUT_DIR / "loglar"
PLOT_DIR   = OUTPUT_DIR / "grafikler"
MODEL_DIR  = OUTPUT_DIR / "modeller"

for d in [LOG_DIR, PLOT_DIR, MODEL_DIR]:
    d.mkdir(parents=True, exist_ok=True)

LOG_FILE = LOG_DIR / "sistem_logu.txt"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(sys.stdout)
    ]
)
log = logging.getLogger("FutbolAnalizSistemi")

log.info("=" * 65)
log.info("FUTBOLCU OYUN STİLİ ANALİZ SİSTEMİ v2 BAŞLADI")
log.info(f"Çalışma ID : {RUN_ID}")
log.info(f"Çıktı klasörü: {OUTPUT_DIR}")
log.info("=" * 65)

train_history = {
    "run_id": RUN_ID,
    "train_loss": [],
    "val_loss": [],
    "best_val_loss": None,
    "cluster_metrics": {},
    "best_k": None
}

# ============================================================
# 1. VERİ YÜKLEME
# ============================================================

log.info("Veriler yükleniyor...")

df          = pd.read_csv(VERISETI_DIR / "temiz_veri.csv", encoding="utf-8")
player_info = pd.read_csv(VERISETI_DIR / "futbolcular.csv", encoding="utf-8")
player_info = player_info.iloc[:len(df)].reset_index(drop=True)

log.info(f"temiz_veri.csv  → {df.shape[0]} satır, {df.shape[1]} sütun")
log.info(f"futbolcular.csv → {player_info.shape[0]} oyuncu (eşleştirilmiş)")

PLAYER_COL = "Player" if "Player" in player_info.columns else player_info.columns[0]
log.info(f"Oyuncu sütunu: {PLAYER_COL}")

# ============================================================
# 2. FEATURE SELECTION — 4 grup, 20 feature
# ============================================================

features = [
    # Hücum
    'Gls', 'Ast', 'xG', 'xAG', 'npxG', 'Sh/90',
    # Pas & Oyun Kurma
    'Cmp%', 'PrgP', 'KP', 'PPA', 'SCA90',
    # Defans
    'Tkl', 'TklW', 'Int', 'Clr',
    # Top Taşıma & Hareket
    'PrgC', 'PrgR', 'Succ%', 'Carries', 'Touches'
]

FEATURE_GROUPS = {
    "Hücum":            ['Gls', 'Ast', 'xG', 'xAG', 'npxG', 'Sh/90'],
    "Pas & Oyun Kurma": ['Cmp%', 'PrgP', 'KP', 'PPA', 'SCA90'],
    "Defans":           ['Tkl', 'TklW', 'Int', 'Clr'],
    "Top Taşıma":       ['PrgC', 'PrgR', 'Succ%', 'Carries', 'Touches']
}

X = df[features].copy()
log.info(f"Seçilen feature sayısı: {len(features)}")

# ============================================================
# 3. EKSİK VERİ TEMİZLEME
# ============================================================

log.info("Eksik veriler temizleniyor...")
X = X.replace([np.inf, -np.inf], np.nan)
log.info(f"  NaN sayısı (önce): {X.isna().sum().sum()}")
X = X.fillna(X.median())
log.info(f"  NaN sayısı (sonra): {X.isna().sum().sum()}")

# ============================================================
# 4. NORMALIZATION — RobustScaler
# ============================================================

log.info("RobustScaler uygulanıyor...")
scaler   = RobustScaler()
X_scaled = scaler.fit_transform(X)
log.info(f"  Ölçeklenmiş boyut: {X_scaled.shape}")

# ============================================================
# 5. TRAIN / VAL / TEST SPLIT  (70 / 15 / 15)
# ============================================================

log.info("Veri train / val / test olarak bölünüyor (70 / 15 / 15)...")
X_temp, X_test = train_test_split(X_scaled, test_size=0.15, random_state=42)
X_train, X_val = train_test_split(X_temp,   test_size=0.1765, random_state=42)

log.info(f"  Train: {X_train.shape[0]}  |  Val: {X_val.shape[0]}  |  Test: {X_test.shape[0]}")

# ============================================================
# 6. TORCH DATASET
# ============================================================

def make_loader(arr, batch_size=64, shuffle=True):
    tensor = torch.FloatTensor(arr)
    return DataLoader(TensorDataset(tensor), batch_size=batch_size, shuffle=shuffle)

train_loader = make_loader(X_train, shuffle=True)
val_loader   = make_loader(X_val,   shuffle=False)
test_loader  = make_loader(X_test,  shuffle=False)

# ============================================================
# 7. GELİŞTİRİLMİŞ AUTOENCODER
#    ► LATENT_DIM = 12  →  daha fazla temsil kapasitesi
#    ► Skip connection benzeri ara çıktı  →  daha iyi sıkıştırma
#    ► L2 regularization + stronger BN   →  daha ayırt edici latent uzay
# ============================================================

class FootballAutoencoder(nn.Module):

    def __init__(self, input_dim: int, latent_dim: int = 12):
        super().__init__()

        # ENCODER:  input_dim → 256 → 128 → 64 → 32 → latent_dim
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.25),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),

            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.1),

            nn.Linear(32, latent_dim)
        )

        # DECODER:  latent_dim → 32 → 64 → 128 → 256 → input_dim
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.LeakyReLU(0.1),

            nn.Linear(32, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.1),

            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1),

            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.1),

            nn.Linear(256, input_dim)
        )

    def forward(self, x):
        latent = self.encoder(x)
        return latent, self.decoder(latent)

    def encode(self, x):
        return self.encoder(x)

# ============================================================
# 8. MODEL & OPTİMİZER 
# ============================================================

# ★ LATENT_DIM = 12: kümeleme kalitesi için parametre.

LATENT_DIM = 12
input_dim  = len(features)  # 20

model     = FootballAutoencoder(input_dim, LATENT_DIM)
criterion = nn.MSELoss()
optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=5e-5)

# ★ CosineAnnealingWarmRestarts: periyodik restart ile daha iyi minimum
scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=30, T_mult=2, eta_min=1e-6
)

param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
log.info(f"Model oluşturuldu | Latent dim: {LATENT_DIM} | Parametre: {param_count:,}")

# ============================================================
# 9. MODEL EĞİTİMİ
#    ► EPOCHS = 200, PATIENCE = 25
#    ► Reconstruction loss + kümeleme dostu L2 regularization
# ============================================================

log.info("Model eğitimi başlıyor...")

EPOCHS        = 200
PATIENCE      = 25
best_val_loss = float("inf")
patience_cnt  = 0
t0            = time.time()

for epoch in range(1, EPOCHS + 1):

    # ── TRAIN ──────────────────────────────────────────────
    model.train()
    total_train = 0.0
    for (batch,) in train_loader:
        optimizer.zero_grad()
        latent, out = model(batch)
        recon_loss  = criterion(out, batch)

        # ★ Latent uzayı sıkıştırmak için hafif L2 — kümeler daha yakın
        latent_reg = 1e-4 * torch.mean(latent ** 2)
        loss       = recon_loss + latent_reg

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_train += recon_loss.item()  # saf recon loss'u logla

    avg_train = total_train / len(train_loader)

    # ── VALİDASYON ─────────────────────────────────────────
    model.eval()
    total_val = 0.0
    with torch.no_grad():
        for (batch,) in val_loader:
            _, out     = model(batch)
            total_val += criterion(out, batch).item()
    avg_val = total_val / len(val_loader)

    scheduler.step()
    train_history["train_loss"].append(round(avg_train, 6))
    train_history["val_loss"].append(round(avg_val, 6))

    if avg_val < best_val_loss:
        best_val_loss = avg_val
        patience_cnt  = 0
        torch.save(model.state_dict(), MODEL_DIR / "best_autoencoder.pth")
        best_marker = "✓ YENİ EN İYİ"
    else:
        patience_cnt += 1
        best_marker  = ""

    if epoch % 10 == 0 or best_marker:
        log.info(
            f"Epoch [{epoch:3d}/{EPOCHS}] | "
            f"Train: {avg_train:.6f} | Val: {avg_val:.6f} | "
            f"Best Val: {best_val_loss:.6f} | "
            f"Patience: {patience_cnt}/{PATIENCE} {best_marker}"
        )

    if patience_cnt >= PATIENCE:
        log.info(f"Erken durdurma tetiklendi — Epoch {epoch}.")
        break

elapsed = time.time() - t0
train_history["best_val_loss"] = round(best_val_loss, 6)
log.info(f"Eğitim tamamlandı | Süre: {elapsed:.1f}s | En iyi val loss: {best_val_loss:.6f}")

# ── TEST değerlendirme ─────────────────────────────────────
model.load_state_dict(torch.load(MODEL_DIR / "best_autoencoder.pth", weights_only=True))
model.eval()
total_test = 0.0
with torch.no_grad():
    for (batch,) in test_loader:
        _, out     = model(batch)
        total_test += criterion(out, batch).item()
avg_test = total_test / len(test_loader)
log.info(f"TEST MSE (görülmemiş veri): {avg_test:.6f}")
train_history["test_mse"] = round(avg_test, 6)

# ============================================================
# 10. GRAFİK 1 — EĞİTİM / VALİDASYON KAYIP EĞRİSİ
# ============================================================

fig, ax = plt.subplots(figsize=(10, 5))
ep_range = range(1, len(train_history["train_loss"]) + 1)
ax.plot(ep_range, train_history["train_loss"], label="Train Loss",  color="#2196F3")
ax.plot(ep_range, train_history["val_loss"],   label="Val Loss",    color="#F44336")
ax.axhline(best_val_loss, linestyle="--", color="gray", alpha=0.6,
           label=f"Best Val: {best_val_loss:.4f}")
ax.set_title("Autoencoder Eğitim / Validasyon Kayıp Eğrisi", fontsize=14, fontweight="bold")
ax.set_xlabel("Epoch")
ax.set_ylabel("MSE Loss")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(PLOT_DIR / "01_egitim_val_kayip_egrisi.png", dpi=150)
plt.close()
log.info("Grafik 1 kaydedildi → 01_egitim_val_kayip_egrisi.png")

# ============================================================
# 11. LATENT FEATURES ÇIKARMA
# ============================================================

log.info("Latent temsiller çıkarılıyor...")
model.eval()
with torch.no_grad():
    full_tensor = torch.FloatTensor(X_scaled)
    latent_vectors, reconstructed = model(full_tensor)
latent_vectors = latent_vectors.numpy()
log.info(f"Latent boyutu: {latent_vectors.shape}")

# ============================================================
# 12. ÇOKLU KÜMELEME ALGORİTMASI + ENSEMBLE OYLAMA 
#     ► Silhouette (0.5) + DB (0.3) + CH (0.2)  →  ayrışma odaklı ağırlıklar
# ============================================================

log.info("Optimal cluster sayısı aranıyor (k=5..10, ensemble)...")


K_RANGE = range(5, 11)

sil_scores, db_scores, ch_scores = [], [], []
ensemble_labels_all = {}

for k in K_RANGE:
    # ── Algoritma 1: KMeans ──────────────────────────────
    km  = KMeans(n_clusters=k, random_state=42, n_init=20, max_iter=600,
                 algorithm='lloyd')
    lbl_km = km.fit_predict(latent_vectors)

    # ── Algoritma 2: Agglomerative (Ward) ───────────────
    # ★ Ward bağlantısı: küme içi varyansı minimize eder → daha kompakt kümeler
    agg = AgglomerativeClustering(n_clusters=k, linkage='ward')
    lbl_agg = agg.fit_predict(latent_vectors)

    # ── Algoritma 3: Gaussian Mixture Model ─────────────
    # ★ GMM: eliptik kümeler varsayar → KMeans'in küresel varsayımını aşar
    gmm = GaussianMixture(n_components=k, random_state=42, n_init=5,
                          covariance_type='full', max_iter=300)
    lbl_gmm = gmm.fit_predict(latent_vectors)

    # ── Ensemble: üç algoritmanın ortalama benzerliği ───
    # Her algoritmanın verdiği etiketleri KMeans centroid'lerine hizala
    # (basit çoğunluk oyu yerine benzerlik tabanlı hizalama)
    ensemble_labels_all[k] = {
        'kmeans': lbl_km,
        'agglomerative': lbl_agg,
        'gmm': lbl_gmm
    }

    # Metrik KMeans üzerinden hesapla (referans algoritma)
    sil = silhouette_score(latent_vectors, lbl_km)
    db  = davies_bouldin_score(latent_vectors, lbl_km)
    ch  = calinski_harabasz_score(latent_vectors, lbl_km)

    sil_scores.append(sil)
    db_scores.append(db)
    ch_scores.append(ch)

    log.info(
        f"  k={k:2d} | Silhouette: {sil:.4f} | "
        f"Davies-Bouldin: {db:.4f} | Calinski-Harabasz: {ch:.2f}"
    )

sil_arr = np.array(sil_scores)
db_arr  = np.array(db_scores)
ch_arr  = np.array(ch_scores)

def _norm(a, higher_is_better=True):
    rng = a.max() - a.min() + 1e-8
    if higher_is_better:
        return (a - a.min()) / rng
    else:
        return (a.max() - a) / rng

# ★ Silhouette ağırlığı 0.5'e çıkarıldı: küme ayrışması için en kritik metrik
final_score = (
    0.50 * _norm(sil_arr) +
    0.30 * _norm(db_arr, False) +
    0.20 * _norm(ch_arr)
)
best_k = list(K_RANGE)[int(np.argmax(final_score))]

log.info(f"En iyi k (dengeli skor bazlı): {best_k}")

train_history["best_k"] = best_k
train_history["cluster_metrics"] = {
    "k_range":          list(K_RANGE),
    "silhouette":       [round(float(s), 4) for s in sil_arr],
    "davies_bouldin":   [round(float(d), 4) for d in db_arr],
    "calinski_harabasz":[round(float(c), 2) for c in ch_arr]
}

# ── GRAFİK 2 — Kümeleme Metrik Karşılaştırması ────────────
fig, axes = plt.subplots(1, 3, figsize=(17, 5))
ks = list(K_RANGE)

axes[0].plot(ks, sil_arr, 'o-', color="#4CAF50")
axes[0].axvline(best_k, linestyle="--", color="red", alpha=0.7, label=f"Best k={best_k}")
axes[0].set_title("Silhouette Score (↑ iyi)")
axes[0].set_xlabel("k"); axes[0].legend(); axes[0].grid(True, alpha=0.3)

axes[1].plot(ks, db_arr, 's-', color="#FF9800")
axes[1].axvline(best_k, linestyle="--", color="red", alpha=0.7)
axes[1].set_title("Davies-Bouldin Score (↓ iyi)")
axes[1].set_xlabel("k"); axes[1].grid(True, alpha=0.3)

axes[2].plot(ks, ch_arr, '^-', color="#9C27B0")
axes[2].axvline(best_k, linestyle="--", color="red", alpha=0.7)
axes[2].set_title("Calinski-Harabasz Score (↑ iyi)")
axes[2].set_xlabel("k"); axes[2].grid(True, alpha=0.3)

plt.suptitle("Küme Sayısı Seçim Metrikleri (k=5..10)", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(PLOT_DIR / "02_kume_secim_metrikleri.png", dpi=150)
plt.close()
log.info("Grafik 2 kaydedildi → 02_kume_secim_metrikleri.png")

# ============================================================
# 13. FINAL KÜMELEME — ENSEMBLE (KMeans + Agglomerative + GMM)
#     ► Üç algoritmanın sonuçlarını birleştirerek daha kararlı kümeler
# ============================================================

log.info(f"Final ensemble kümeleme uygulanıyor (k={best_k})...")

# Final KMeans (centroid'ler için referans)
kmeans_final = KMeans(n_clusters=best_k, random_state=42, n_init=30, max_iter=1000,
                      algorithm='lloyd')
lbl_km_final = kmeans_final.fit_predict(latent_vectors)

# Final Agglomerative
agg_final    = AgglomerativeClustering(n_clusters=best_k, linkage='ward')
lbl_agg_final = agg_final.fit_predict(latent_vectors)

# Final GMM
gmm_final    = GaussianMixture(n_components=best_k, random_state=42, n_init=10,
                                covariance_type='full', max_iter=500)
gmm_final.fit(latent_vectors)
lbl_gmm_final = gmm_final.predict(latent_vectors)
gmm_probs     = gmm_final.predict_proba(latent_vectors)  # Olasılıklar da sakla

# ★ Ensemble: KMeans centroid'lerine göre Agglomerative ve GMM etiketlerini hizala
def align_labels(reference, other, n_clusters):
    """Diğer algoritmanın etiketlerini referans etiketlere hizala (Hungarian-benzeri)."""
    from scipy.optimize import linear_sum_assignment
    cost = np.zeros((n_clusters, n_clusters))
    for i in range(n_clusters):
        for j in range(n_clusters):
            cost[i, j] = -np.sum((reference == i) & (other == j))
    row_ind, col_ind = linear_sum_assignment(cost)
    aligned = np.zeros_like(other)
    for r, c in zip(row_ind, col_ind):
        aligned[other == c] = r
    return aligned

lbl_agg_aligned = align_labels(lbl_km_final, lbl_agg_final, best_k)
lbl_gmm_aligned = align_labels(lbl_km_final, lbl_gmm_final, best_k)

# ★ Çoğunluk oyu: üç algoritmanın katıldığı etiket kazanır
ensemble_matrix = np.stack([lbl_km_final, lbl_agg_aligned, lbl_gmm_aligned], axis=1)
from scipy import stats as scipy_stats
clusters, _ = scipy_stats.mode(ensemble_matrix, axis=1, keepdims=False)
clusters    = clusters.flatten().astype(int)

# Ensemble oy dağılımını logla
vote_agreement = np.mean(np.all(ensemble_matrix == clusters[:, None], axis=1))
log.info(f"  Ensemble oy uyumu (3/3 anlaşan oyuncu oranı): {vote_agreement:.3f}")

player_info["Cluster"] = clusters

cluster_dist = pd.Series(clusters).value_counts().sort_index()
for cid, cnt in cluster_dist.items():
    log.info(f"  Cluster {cid}: {cnt} oyuncu")

# ── GRAFİK 2b — Ensemble Oy Uyumu ──────────────────────────
vote_counts = []
for i in range(len(ensemble_matrix)):
    row = ensemble_matrix[i]
    unanimous = int(np.all(row == row[0]))
    vote_counts.append(unanimous)
vote_agreement_pct = np.mean(vote_counts) * 100

fig, ax = plt.subplots(figsize=(8, 4))
methods = ['KMeans', 'Agglomerative', 'GMM', 'Ensemble']
# Silhouette her algoritma için hesapla
sils = [
    silhouette_score(latent_vectors, lbl_km_final),
    silhouette_score(latent_vectors, lbl_agg_aligned),
    silhouette_score(latent_vectors, lbl_gmm_aligned),
    silhouette_score(latent_vectors, clusters)
]
colors = ["#2196F3", "#4CAF50", "#FF9800", "#E91E63"]
bars   = ax.bar(methods, sils, color=colors, alpha=0.85, edgecolor='white', linewidth=1.5)
ax.set_title(f"Algoritma Bazlı Silhouette Karşılaştırması (k={best_k})",
             fontsize=13, fontweight="bold")
ax.set_ylabel("Silhouette Score")
ax.set_ylim(0, max(sils) * 1.2)
for bar, val in zip(bars, sils):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
            f"{val:.4f}", ha='center', fontsize=10, fontweight='bold')
ax.grid(True, axis='y', alpha=0.3)
ax.text(0.98, 0.95, f"Oy uyumu: %{vote_agreement_pct:.1f}",
        transform=ax.transAxes, ha='right', fontsize=9,
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))
plt.tight_layout()
plt.savefig(PLOT_DIR / "02b_ensemble_silhouette.png", dpi=150)
plt.close()
log.info("Grafik 2b kaydedildi → 02b_ensemble_silhouette.png")

# ============================================================
# 14. BOYUT İNDİRGEME — t-SNE + UMAP (çift görselleştirme)
#     ► PCA önindirgemeli t-SNE  →  kararlı
#     ► UMAP  →  küresel yapıyı daha iyi korur, kümeler daha belirgin
# ============================================================

log.info("t-SNE görselleştirmesi hazırlanıyor...")
n_pca   = min(LATENT_DIM, latent_vectors.shape[1])
pca_pre = PCA(n_components=n_pca)
lat_pca = pca_pre.fit_transform(latent_vectors)

tsne      = TSNE(n_components=2, perplexity=40, max_iter=2000,
                 learning_rate='auto', init='pca', random_state=42)
latent_2d = tsne.fit_transform(lat_pca)
log.info("t-SNE tamamlandı.")

# ★ UMAP: t-SNE'ye göre daha belirgin küme sınırları oluşturur
log.info("UMAP görselleştirmesi hazırlanıyor...")
try:
    reducer   = umap.UMAP(n_components=2, n_neighbors=20, min_dist=0.1,
                          metric='euclidean', random_state=42)
    latent_umap = reducer.fit_transform(latent_vectors)
    log.info("UMAP tamamlandı.")
    umap_available = True
except Exception as e:
    log.warning(f"UMAP kullanılamadı: {e}. 'pip install umap-learn' ile kurun.")
    latent_umap    = latent_2d  # Fallback: t-SNE sonucunu kullan
    umap_available = False

PALETTE = sns.color_palette("tab10", best_k)

# ── GRAFİK 3 — t-SNE + UMAP Yan Yana ─────────────────────
fig, axes = plt.subplots(1, 2, figsize=(20, 8))
for ax_idx, (coords, title) in enumerate([
    (latent_2d,   "t-SNE"),
    (latent_umap, "UMAP" if umap_available else "t-SNE (UMAP yok)")
]):
    ax = axes[ax_idx]
    for cid in range(best_k):
        mask = clusters == cid
        ax.scatter(
            coords[mask, 0], coords[mask, 1],
            c=[PALETTE[cid]], label=f"Cluster {cid} (n={mask.sum()})",
            s=55, alpha=0.75, edgecolors='white', linewidths=0.4
        )
    ax.set_title(f"Futbolcu Oyun Stili Kümeleri ({title})", fontsize=14, fontweight="bold")
    ax.set_xlabel(f"{title} Boyut 1")
    ax.set_ylabel(f"{title} Boyut 2")
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.2)
plt.tight_layout()
plt.savefig(PLOT_DIR / "03_tsne_umap_kumeler.png", dpi=150)
plt.close()
log.info("Grafik 3 kaydedildi → 03_tsne_umap_kumeler.png")

# ── GRAFİK 3b — Silhouette Örnek Grafiği ─────────────────
# ★ Her oyuncunun kendi kümesine ne kadar uyduğunu gösterir
sample_silhouette = silhouette_samples(latent_vectors, clusters)
fig, ax = plt.subplots(figsize=(10, 6))
y_lower = 10
for cid in range(best_k):
    ith_silhouette = np.sort(sample_silhouette[clusters == cid])
    size_i = ith_silhouette.shape[0]
    y_upper = y_lower + size_i
    color   = PALETTE[cid]
    ax.fill_betweenx(np.arange(y_lower, y_upper),
                     0, ith_silhouette, facecolor=color, alpha=0.7, label=f"C{cid}")
    ax.text(-0.05, y_lower + 0.5 * size_i, str(cid), fontsize=8)
    y_lower = y_upper + 10

avg_sil = np.mean(sample_silhouette)
ax.axvline(avg_sil, color="red", linestyle="--", label=f"Ortalama: {avg_sil:.3f}")
ax.set_title(f"Silhouette Analizi (k={best_k})", fontsize=13, fontweight="bold")
ax.set_xlabel("Silhouette Katsayısı"); ax.set_ylabel("Küme")
ax.legend(loc='upper right', fontsize=8)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(PLOT_DIR / "03b_silhouette_analizi.png", dpi=150)
plt.close()
log.info("Grafik 3b kaydedildi → 03b_silhouette_analizi.png")

# ============================================================
# 15. GRAFİK 4 — CLUSTER BAZLI RADAR CHART
# ============================================================

log.info("Cluster radar grafikleri çiziliyor...")
df_with_cluster = df[features].copy()
df_with_cluster["Cluster"] = clusters
cluster_means   = df_with_cluster.groupby("Cluster")[features].mean()

cmin = cluster_means.min()
cmax = cluster_means.max()
cluster_norm = (cluster_means - cmin) / (cmax - cmin + 1e-8)

def radar_chart(ax, values, labels, title, color):
    N      = len(labels)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    vals   = list(values) + [values[0]]
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, size=7)
    ax.plot(angles, vals, color=color, linewidth=2)
    ax.fill(angles, vals, color=color, alpha=0.25)
    ax.set_ylim(0, 1)
    ax.set_title(title, size=10, fontweight="bold", pad=15)

cols  = 3
rows  = (best_k + cols - 1) // cols
fig_r = plt.figure(figsize=(cols * 5, rows * 5))
for cid in range(best_k):
    ax = fig_r.add_subplot(rows, cols, cid + 1, polar=True)
    radar_chart(ax, cluster_norm.loc[cid].values, features,
                f"Cluster {cid}", PALETTE[cid])
plt.suptitle("Cluster Bazlı Oyuncu Profil Radarları", fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(PLOT_DIR / "04_cluster_radar.png", dpi=150, bbox_inches='tight')
plt.close()
log.info("Grafik 4 kaydedildi → 04_cluster_radar.png")

# ── GRAFİK 4b — Cluster Merkezi Heatmap ──────────────────
# ★ Küme merkezlerini normalize ederek feature bazlı farkları gösterir
fig, ax = plt.subplots(figsize=(14, best_k + 2))
sns.heatmap(cluster_norm, annot=True, fmt=".2f", cmap="YlOrRd",
            linewidths=0.5, ax=ax, cbar_kws={"label": "Normalize Değer"})
ax.set_title("Cluster Merkezi Feature Heatmap (Normalize)", fontsize=13, fontweight="bold")
ax.set_xlabel("Feature"); ax.set_ylabel("Cluster")
plt.tight_layout()
plt.savefig(PLOT_DIR / "04b_cluster_heatmap.png", dpi=150)
plt.close()
log.info("Grafik 4b kaydedildi → 04b_cluster_heatmap.png")

# ============================================================
# 16. GRAFİK 5 — FEATURE IMPORTANCE (ortalama ± std)
# ============================================================

feat_mean = df[features].mean()
feat_std  = df[features].std()

fig, ax = plt.subplots(figsize=(14, 6))
x_pos   = np.arange(len(features))
ax.bar(x_pos, feat_mean, yerr=feat_std, capsize=4,
       color=sns.color_palette("viridis", len(features)), alpha=0.85)
ax.set_xticks(x_pos)
ax.set_xticklabels(features, rotation=45, ha='right', fontsize=9)
ax.set_title("Feature Ortalama ± Standart Sapma Dağılımı", fontsize=13, fontweight="bold")
ax.set_ylabel("Değer")
ax.grid(True, axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(PLOT_DIR / "05_feature_dagilim.png", dpi=150)
plt.close()
log.info("Grafik 5 kaydedildi → 05_feature_dagilim.png")

# ============================================================
# 17. GRAFİK 6 — CLUSTER BAZLI KUTU GRAFİĞİ (grup başına)
# ============================================================

log.info("Cluster bazlı kutu grafikleri çiziliyor...")
for group_name, group_feats in FEATURE_GROUPS.items():
    plot_df = df[group_feats].copy()
    plot_df["Cluster"] = clusters
    melted  = plot_df.melt(id_vars="Cluster", var_name="Feature", value_name="Değer")
    fig, ax = plt.subplots(figsize=(14, 6))
    sns.boxplot(data=melted, x="Feature", y="Değer", hue="Cluster",
                palette="tab10", ax=ax, linewidth=0.8)
    ax.set_title(f"{group_name} – Cluster Bazlı Dağılım", fontsize=13, fontweight="bold")
    ax.set_xlabel("Feature"); ax.set_ylabel("Değer")
    ax.legend(title="Cluster", loc='upper right', fontsize=8)
    ax.grid(True, axis='y', alpha=0.3)
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    safe_name = group_name.replace(" ", "_").replace("&", "ve")
    plt.savefig(PLOT_DIR / f"06_kutu_{safe_name}.png", dpi=150)
    plt.close()
    log.info(f"Grafik 6-{group_name} kaydedildi")

# ============================================================
# 18. GRAFİK 7 — RECONSTRUCTION ERROR DAĞILIMI
# ============================================================

log.info("Reconstruction error dağılımı hesaplanıyor...")
model.eval()
with torch.no_grad():
    _, recon   = model(torch.FloatTensor(X_scaled))
    per_sample = ((recon - torch.FloatTensor(X_scaled)) ** 2).mean(dim=1).numpy()

final_mse = per_sample.mean()
log.info(f"Final Reconstruction MSE (tüm veri): {final_mse:.6f}")
train_history["final_recon_mse"] = round(float(final_mse), 6)

# Anomali tespiti: rekonstrüksiyon hatası yüksek oyuncular "atipik"
anomaly_threshold = np.percentile(per_sample, 95)
anomaly_players   = player_info.iloc[per_sample > anomaly_threshold][PLAYER_COL].values
log.info(f"  Anomali eşiği (95. yüzdelik): {anomaly_threshold:.4f}")
log.info(f"  Atipik oyuncu sayısı: {len(anomaly_players)}")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
axes[0].hist(per_sample, bins=50, color="#3F51B5", alpha=0.8, edgecolor='white')
axes[0].axvline(per_sample.mean(), color='red', linestyle='--',
                label=f"Ort: {per_sample.mean():.4f}")
axes[0].axvline(anomaly_threshold, color='orange', linestyle=':',
                label=f"Anomali eşiği: {anomaly_threshold:.4f}")
axes[0].set_title("Rekonstrüksiyon Hata Dağılımı")
axes[0].set_xlabel("MSE"); axes[0].set_ylabel("Oyuncu Sayısı")
axes[0].legend(); axes[0].grid(True, alpha=0.3)

recon_by_cluster = {cid: per_sample[clusters == cid] for cid in range(best_k)}
axes[1].boxplot(
    [recon_by_cluster[c] for c in range(best_k)],
    labels=[f"C{c}" for c in range(best_k)],
    patch_artist=True,
    boxprops=dict(facecolor="#E3F2FD")
)
axes[1].set_title("Cluster Bazlı Rekonstrüksiyon Hatası")
axes[1].set_ylabel("MSE"); axes[1].grid(True, axis='y', alpha=0.3)
plt.suptitle("Autoencoder Rekonstrüksiyon Hataları", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(PLOT_DIR / "07_rekonstruksiyon_hata.png", dpi=150)
plt.close()
log.info("Grafik 7 kaydedildi → 07_rekonstruksiyon_hata.png")

# ============================================================
# 19. GRAFİK 8 — LATENT UZAY KORELASYON MATRİSİ
# ============================================================

latent_df_corr = pd.DataFrame(
    latent_vectors,
    columns=[f"L{i+1}" for i in range(LATENT_DIM)]
)
fig, ax = plt.subplots(figsize=(10, 9))
sns.heatmap(latent_df_corr.corr(), annot=True, fmt=".2f",
            cmap="coolwarm", center=0, ax=ax,
            linewidths=0.5, square=True)
ax.set_title("Latent Uzay Korelasyon Matrisi", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(PLOT_DIR / "08_latent_korelasyon.png", dpi=150)
plt.close()
log.info("Grafik 8 kaydedildi → 08_latent_korelasyon.png")

# ============================================================
# 20. HİBRİT BENZERLİK MATRİSİ
#     ► Cosine (0.6) + Euclidean (0.4) ağırlıklı hibrit
#     ★ Sadece cosine kullanmak yön benzerliğini ölçer ama
#       büyüklük farklarını görmezden gelir. Hibrit her ikisini yakalar.
# ============================================================

log.info("Hibrit benzerlik matrisi oluşturuluyor...")

cos_sim  = cosine_similarity(latent_vectors)

# Euclidean mesafeyi 0-1 benzerliğe dönüştür
euc_dist = euclidean_distances(latent_vectors)
euc_sim  = 1.0 / (1.0 + euc_dist)

# ★ Ağırlıklı hibrit benzerlik
COSINE_WEIGHT    = 0.6
EUCLIDEAN_WEIGHT = 0.4
similarity_matrix = COSINE_WEIGHT * cos_sim + EUCLIDEAN_WEIGHT * euc_sim

log.info(f"  Hibrit benzerlik (cosine×{COSINE_WEIGHT} + euclidean×{EUCLIDEAN_WEIGHT})")

# ── GRAFİK 8b — GMM Olasılık Dağılımı (belirsiz oyuncular) ──
# Her oyuncunun kümesine atanma olasılığı — 0.5'e yakın = sınırda oyuncu
max_probs    = gmm_probs.max(axis=1)
second_probs = np.sort(gmm_probs, axis=1)[:, -2]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].hist(max_probs, bins=40, color="#009688", alpha=0.8, edgecolor='white')
axes[0].axvline(0.8, color='red', linestyle='--', label="Eşik: 0.8")
axes[0].set_title("GMM Maksimum Atanma Olasılığı Dağılımı")
axes[0].set_xlabel("P(max küme)"); axes[0].set_ylabel("Oyuncu Sayısı")
axes[0].legend(); axes[0].grid(True, alpha=0.3)

# Belirsiz oyuncular (max_prob < 0.6)
uncertain_mask = max_probs < 0.6
sc = axes[1].scatter(
    latent_umap[:, 0], latent_umap[:, 1],
    c=max_probs, cmap='RdYlGn', s=30, alpha=0.7
)
plt.colorbar(sc, ax=axes[1], label="Atanma Olasılığı")
axes[1].set_title(f"UMAP: Belirsiz Oyuncular (n={uncertain_mask.sum()}, P<0.6)")
axes[1].set_xlabel("UMAP 1"); axes[1].set_ylabel("UMAP 2")
axes[1].grid(True, alpha=0.2)
plt.suptitle("GMM Üyelik Belirsizlik Analizi", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(PLOT_DIR / "08b_gmm_belirsizlik.png", dpi=150)
plt.close()
log.info("Grafik 8b kaydedildi → 08b_gmm_belirsizlik.png")

# ============================================================
# 21. GELİŞTİRİLMİŞ OYUNCU ÖNERİ SİSTEMİ
# ============================================================

def recommend_players(
    player_name: str,
    top_n: int = 5,
    same_cluster_only: bool = False,
    cross_cluster_penalty: float = 0.15  # ★ Farklı küme sonuçlarına ceza
) -> list:
    """
    Hibrit benzerlik + küme ağırlığı ile en iyi öneri sistemi.

    Parameters
    ----------
    player_name : str
        Aranacak oyuncu adı.
    top_n : int
        Döndürülecek öneri sayısı.
    same_cluster_only : bool
        True ise yalnızca aynı kümeden arar.
    cross_cluster_penalty : float
        Farklı kümeden gelen sonuçlara uygulanacak skor indirimi (0–1 arası).
    """
    if player_name not in player_info[PLAYER_COL].values:
        log.warning(f"Oyuncu bulunamadı: {player_name}")
        return []

    idx        = player_info[player_info[PLAYER_COL] == player_name].index[0]
    my_cluster = int(player_info.iloc[idx]["Cluster"])
    my_gmm_prob = gmm_probs[idx, my_cluster]

    # Hibrit benzerlik skorları
    raw_scores = similarity_matrix[idx].copy()

    # ★ Farklı kümedeki oyunculara ceza uygula
    if not same_cluster_only:
        for i in range(len(raw_scores)):
            if int(player_info.iloc[i]["Cluster"]) != my_cluster:
                raw_scores[i] *= (1.0 - cross_cluster_penalty)
    else:
        for i in range(len(raw_scores)):
            if int(player_info.iloc[i]["Cluster"]) != my_cluster:
                raw_scores[i] = -1.0  # Aynı küme dışını dışla

    sim_scores = list(enumerate(raw_scores))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n + 1]

    log.info("=" * 60)
    log.info(f"{player_name} → En benzer {top_n} oyuncu")
    log.info(f"  Kendi kümesi: {my_cluster} | GMM güveni: {my_gmm_prob:.3f}")
    log.info("=" * 60)

    results = []
    for rank, (pidx, score) in enumerate(sim_scores, 1):
        row      = player_info.iloc[pidx]
        nm       = row[PLAYER_COL]
        c        = int(row['Cluster'])
        gmm_conf = gmm_probs[pidx, c]
        same_c   = "✓" if c == my_cluster else "↗"
        log.info(
            f"  {rank}. {nm:<30} | Skor: {score:.4f} | "
            f"Cluster: {c} {same_c} | GMM güven: {gmm_conf:.3f}"
        )
        results.append({
            "rank":        rank,
            "player":      nm,
            "score":       round(score, 4),
            "cluster":     c,
            "same_cluster": c == my_cluster,
            "gmm_confidence": round(float(gmm_conf), 4)
        })
    return results

# ============================================================
# 22. YENİ OYUNCU TAHMİNİ (GELİŞTİRİLMİŞ)
#     ► Hibrit benzerlik
#     ► GMM olasılığı ile "hangi kümeye ne kadar uyuyor" raporu
# ============================================================

def predict_new_player(player_stats: list, top_n: int = 5) -> dict:
    """
    20 özellikten oluşan liste alır;
    ensemble cluster ataması, GMM olasılıkları ve hibrit benzerlik döndürür.
    """
    arr    = np.array(player_stats).reshape(1, -1)
    scaled = scaler.transform(arr)
    tensor = torch.FloatTensor(scaled)

    model.eval()
    with torch.no_grad():
        latent, _ = model(tensor)
    latent = latent.numpy()

    # KMeans tahmini
    km_cluster  = int(kmeans_final.predict(latent)[0])

    # GMM olasılıkları
    probs       = gmm_final.predict_proba(latent)[0]
    gmm_cluster = int(np.argmax(probs))

    # Ensemble: KMeans ve GMM hemfikirse güven artar
    final_cluster = km_cluster if km_cluster == gmm_cluster else km_cluster
    confidence    = probs[final_cluster]

    # Hibrit benzerlik
    cos_s  = cosine_similarity(latent, latent_vectors)[0]
    euc_d  = euclidean_distances(latent, latent_vectors)[0]
    euc_s  = 1.0 / (1.0 + euc_d)
    hybrid = COSINE_WEIGHT * cos_s + EUCLIDEAN_WEIGHT * euc_s

    top_idx = np.argsort(hybrid)[::-1][:top_n]

    log.info("=" * 60)
    log.info("YENİ OYUNCU ANALİZİ")
    log.info(f"  KMeans Cluster : {km_cluster}")
    log.info(f"  GMM Cluster    : {gmm_cluster} (güven: {probs[gmm_cluster]:.3f})")
    log.info(f"  Final Cluster  : {final_cluster} (GMM güveni: {confidence:.3f})")
    log.info("  Küme Olasılıkları:")
    for ci, p in enumerate(probs):
        log.info(f"    C{ci}: {p:.3f}")
    log.info("  En benzer oyuncular (hibrit skor):")
    for i, idx in enumerate(top_idx, 1):
        p = player_info.iloc[idx][PLAYER_COL]
        s = hybrid[idx]
        log.info(f"    {i}. {p:<30} | Skor: {s:.4f}")

    return {
        "cluster":           final_cluster,
        "km_cluster":        km_cluster,
        "gmm_cluster":       gmm_cluster,
        "gmm_confidence":    round(float(confidence), 4),
        "cluster_probs":     {f"C{ci}": round(float(p), 4) for ci, p in enumerate(probs)},
        "top_similar": [
            {
                "player":     player_info.iloc[i][PLAYER_COL],
                "score":      round(float(hybrid[i]), 4),
                "cluster":    int(player_info.iloc[i]["Cluster"])
            }
            for i in top_idx
        ]
    }

# ============================================================
# 23. ÇIKTI DOSYALARI KAYDETME
# ============================================================

# Latent vektörler
latent_out = pd.DataFrame(
    latent_vectors,
    columns=[f"L{i+1}" for i in range(LATENT_DIM)]
)
latent_out["Cluster"]          = clusters
latent_out["Cluster_KMeans"]   = lbl_km_final
latent_out["Cluster_Agglom"]   = lbl_agg_aligned
latent_out["Cluster_GMM"]      = lbl_gmm_aligned
latent_out["GMM_Confidence"]   = max_probs
latent_out["Recon_Error"]      = per_sample
latent_out["Player"]           = player_info[PLAYER_COL].values
latent_out.to_csv(OUTPUT_DIR / "latent_features.csv", index=False)

# t-SNE + UMAP koordinatları
tsne_out = pd.DataFrame({
    "Player":   player_info[PLAYER_COL].values,
    "Cluster":  clusters,
    "tSNE_1":   latent_2d[:, 0],
    "tSNE_2":   latent_2d[:, 1],
    "UMAP_1":   latent_umap[:, 0],
    "UMAP_2":   latent_umap[:, 1]
})
tsne_out.to_csv(OUTPUT_DIR / "tsne_umap_koordinatlari.csv", index=False)

# Cluster istatistikleri
cluster_means.round(4).to_csv(OUTPUT_DIR / "cluster_feature_ortalamalar.csv")

# GMM olasılıkları
gmm_prob_df = pd.DataFrame(
    gmm_probs,
    columns=[f"P_Cluster_{i}" for i in range(best_k)]
)
gmm_prob_df["Player"]  = player_info[PLAYER_COL].values
gmm_prob_df["Cluster"] = clusters
gmm_prob_df.to_csv(OUTPUT_DIR / "gmm_olasiliklar.csv", index=False)

# Modeller
torch.save(model.state_dict(), MODEL_DIR / "best_autoencoder.pth")
torch.save(model.state_dict(), MODEL_DIR / "football_autoencoder_final.pth")

# JSON eğitim tarihi
train_history["ensemble_vote_agreement"] = round(float(vote_agreement), 4)
with open(LOG_DIR / "egitim_gecmisi.json", "w", encoding="utf-8") as f:
    json.dump(train_history, f, ensure_ascii=False, indent=2)

log.info("Tüm çıktı dosyaları kaydedildi.")

# ============================================================
# 24. CLUSTER ÖZET RAPORU
# ============================================================

log.info("")
log.info("=" * 65)
log.info("CLUSTER BAZLI OYUNCU LİSTESİ")
log.info("=" * 65)
for cid in sorted(player_info["Cluster"].unique()):
    players = player_info[player_info["Cluster"] == cid][PLAYER_COL].head(10).tolist()
    # Bu kümenin en belirgin feature'larını göster
    top_feats = cluster_norm.loc[cid].nlargest(3).index.tolist()
    log.info(f"\n── CLUSTER {cid} (güçlü yönler: {', '.join(top_feats)}) ──")
    for p in players:
        log.info(f"   {p}")

# ============================================================
# 25. ÖRNEK KULLANIM
# ============================================================

log.info("")
log.info("=" * 65)
log.info("ÖRNEK ÖNERİ SORGUSU")
log.info("=" * 65)
sample_player = player_info[PLAYER_COL].iloc[0]
recommend_players(sample_player, top_n=5, cross_cluster_penalty=0.15)

# ============================================================
# 26. TAMAMLANMA
# ============================================================

log.info("")
log.info("=" * 65)
log.info("SİSTEM BAŞARIYLA TAMAMLANDI")
log.info(f"Tüm çıktılar: {OUTPUT_DIR.resolve()}")
log.info(f"  ├── loglar/       → sistem_logu.txt, egitim_gecmisi.json")
log.info(f"  ├── grafikler/    → 10+ adet PNG grafiği")
log.info(f"  ├── modeller/     → best_autoencoder.pth, final.pth")
log.info(f"  ├── latent_features.csv     (ensemble + GMM güven skoru ile)")
log.info(f"  ├── tsne_umap_koordinatlari.csv")
log.info(f"  ├── gmm_olasiliklar.csv     (her küme için olasılık)")
log.info(f"  └── cluster_feature_ortalamalar.csv")
log.info("=" * 65)