import os, sys, logging, json, time
from datetime import datetime
from pathlib import Path

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score, silhouette_samples
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import umap.umap_ as umap
from scipy.spatial.distance import cdist

# ── Klasörler ──────────────────────────────────────────────────
VERISETI_DIR = Path(__file__).resolve().parents[1] / "veriseti"
RUN_ID       = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR   = VERISETI_DIR.parent / f"cikti_v2_{RUN_ID}"
LOG_DIR      = OUTPUT_DIR / "loglar"
PLOT_DIR     = OUTPUT_DIR / "grafikler"
MODEL_DIR    = OUTPUT_DIR / "modeller"
for d in [LOG_DIR, PLOT_DIR, MODEL_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Logging ────────────────────────────────────────────────────
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
log = logging.getLogger("FutbolV2")
log.info("=" * 65)
log.info("FUTBOLCU ANALİZ SİSTEMİ v2 — VAE + Huber + KL + GMM")
log.info(f"Run ID: {RUN_ID}")
log.info("=" * 65)

# ── Eğitim geçmişi ─────────────────────────────────────────────
train_history = {
    "run_id": RUN_ID, "version": "v2",
    "train_loss": [], "val_loss": [],
    "best_val_loss": None, "test_loss": None,
    "cluster_metrics": {}, "best_k": None
}

# ──────────────────────────────────────────────────────────────
# VERİ YÜKLEME
# ──────────────────────────────────────────────────────────────
log.info("Veriler yükleniyor...")
df          = pd.read_csv(VERISETI_DIR / "temiz_veri.csv", encoding="utf-8")
player_info = pd.read_csv(VERISETI_DIR / "futbolcular.csv", encoding="utf-8")
player_info = player_info[~player_info['Pos'].str.contains('GK', na=False)]
player_info = player_info[player_info['90s'] >= 5].reset_index(drop=True)
PLAYER_COL  = "Player" if "Player" in player_info.columns else player_info.columns[0]
log.info(f"temiz_veri.csv  → {df.shape[0]} satır, {df.shape[1]} sütun")
log.info(f"futbolcular.csv → {player_info.shape[0]} oyuncu")

features = [
    'Gls', 'Ast', 'xG', 'xAG', 'npxG', 'Sh/90',
    'Cmp%', 'PrgP', 'KP', 'PPA', 'SCA90',
    'Tkl', 'TklW', 'Int', 'Clr',
    'PrgC', 'PrgR', 'Succ%', 'Carries', 'Touches'
]
FEATURE_GROUPS = {
    "Hücum":            ['Gls', 'Ast', 'xG', 'xAG', 'npxG', 'Sh/90'],
    "Pas & Oyun Kurma": ['Cmp%', 'PrgP', 'KP', 'PPA', 'SCA90'],
    "Defans":           ['Tkl', 'TklW', 'Int', 'Clr'],
    "Top Taşıma":       ['PrgC', 'PrgR', 'Succ%', 'Carries', 'Touches']
}

X = df[features].copy()
X = X.replace([np.inf, -np.inf], np.nan).fillna(X.median())
log.info(f"Feature sayısı: {len(features)} | NaN sonrası: {X.isna().sum().sum()}")

scaler   = RobustScaler()
X_scaled = scaler.fit_transform(X)

X_temp, X_test = train_test_split(X_scaled, test_size=0.15, random_state=42)
X_train, X_val = train_test_split(X_temp,   test_size=0.1765, random_state=42)
log.info(f"Split → Train: {X_train.shape[0]} | Val: {X_val.shape[0]} | Test: {X_test.shape[0]}")

def make_loader(arr, batch_size=64, shuffle=True):
    t = torch.FloatTensor(arr)
    return DataLoader(TensorDataset(t), batch_size=batch_size, shuffle=shuffle)

train_loader = make_loader(X_train, shuffle=True)
val_loader   = make_loader(X_val,   shuffle=False)
test_loader  = make_loader(X_test,  shuffle=False)

# ──────────────────────────────────────────────────────────────
# VAE MODELİ
# Encoder → (μ, log_var) → reparameterize → Decoder
# ──────────────────────────────────────────────────────────────
LATENT_DIM = 16
INPUT_DIM  = len(features)  # 20

class FootballVAE(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int = 16):
        super().__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256), nn.BatchNorm1d(256), nn.LeakyReLU(0.1), nn.Dropout(0.3),
            nn.Linear(256, 128),       nn.BatchNorm1d(128), nn.LeakyReLU(0.1), nn.Dropout(0.25),
            nn.Linear(128, 64),        nn.BatchNorm1d(64),  nn.LeakyReLU(0.1),
        )
        # Latent parametreler
        self.fc_mu      = nn.Linear(64, latent_dim)
        self.fc_log_var = nn.Linear(64, latent_dim)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),  nn.BatchNorm1d(64),  nn.LeakyReLU(0.1),
            nn.Linear(64, 128),         nn.BatchNorm1d(128), nn.LeakyReLU(0.1),
            nn.Linear(128, 256),        nn.BatchNorm1d(256), nn.LeakyReLU(0.1),
            nn.Linear(256, input_dim)
        )

    def reparameterize(self, mu, log_var):
        """z = μ + σ * ε  (ε ~ N(0,1))  — sadece eğitimde stochastic"""
        if self.training:
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu  # inference'ta deterministic → temiz kümeler

    def forward(self, x):
        h       = self.encoder(x)
        mu      = self.fc_mu(h)
        log_var = self.fc_log_var(h)
        z       = self.reparameterize(mu, log_var)
        recon   = self.decoder(z)
        return recon, mu, log_var

    def encode(self, x):
        """Sadece μ döndür — kümeleme için deterministik temsil"""
        with torch.no_grad():
            h  = self.encoder(x)
            mu = self.fc_mu(h)
        return mu


# ──────────────────────────────────────────────────────────────
# VAE LOSS: Huber (Rekon) + β*KL-Divergence
# ──────────────────────────────────────────────────────────────
huber_fn = nn.HuberLoss(delta=1.0)

def vae_loss(recon, x, mu, log_var, beta=1.0):
    """
    Toplam Loss = Huber(rekon, girdi) + β * KL
    KL = -0.5 * Σ(1 + log_var - μ² - exp(log_var))
    """
    recon_loss = huber_fn(recon, x)
    kl_loss    = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
    return recon_loss + beta * kl_loss, recon_loss, kl_loss


# ──────────────────────────────────────────────────────────────
# MODEL + OPTİMİZER
# ──────────────────────────────────────────────────────────────
model     = FootballVAE(INPUT_DIM, LATENT_DIM)
optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=30, T_mult=2, eta_min=1e-6
)
param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
log.info(f"VAE hazır | Latent: {LATENT_DIM} | Parametre: {param_count:,}")

# ──────────────────────────────────────────────────────────────
# EĞİTİM DÖNGÜSÜ — β-Annealing
# β başta 0 (sadece rekon öğren), sonra 1'e çıkar (KL düzenlemesi)
# Bu sayede model önce veriyi öğrenir, sonra latent uzayı düzenler
# ──────────────────────────────────────────────────────────────
log.info("VAE eğitimi başlıyor...")

EPOCHS        = 200
PATIENCE      = 25
BETA_START    = 0.0
BETA_END      = 1.0
BETA_WARMUP   = 50  # 50 epoch boyunca β kademeli artar

best_val_loss = float("inf")
patience_cnt  = 0
t0            = time.time()

for epoch in range(1, EPOCHS + 1):
    # β-Annealing: lineer artış
    beta = min(BETA_END, BETA_START + (BETA_END - BETA_START) * (epoch / BETA_WARMUP))

    # ── Train ──────────────────────────────────────────────────
    model.train()
    total_train = 0.0
    for (batch,) in train_loader:
        optimizer.zero_grad()
        recon, mu, log_var = model(batch)
        loss, recon_l, kl_l = vae_loss(recon, batch, mu, log_var, beta)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_train += loss.item()
    avg_train = total_train / len(train_loader)

    # ── Validation ─────────────────────────────────────────────
    model.eval()
    total_val = 0.0
    with torch.no_grad():
        for (batch,) in val_loader:
            recon, mu, log_var = model(batch)
            loss, _, _ = vae_loss(recon, batch, mu, log_var, beta)
            total_val += loss.item()
    avg_val = total_val / len(val_loader)

    scheduler.step()
    train_history["train_loss"].append(round(avg_train, 6))
    train_history["val_loss"].append(round(avg_val, 6))

    if avg_val < best_val_loss:
        best_val_loss = avg_val
        patience_cnt  = 0
        torch.save(model.state_dict(), MODEL_DIR / "best_vae.pth")
        marker = "✓ YENİ EN İYİ"
    else:
        patience_cnt += 1
        marker = ""

    if epoch % 10 == 0 or marker:
        log.info(
            f"Epoch [{epoch:3d}/{EPOCHS}] β={beta:.3f} | "
            f"Train: {avg_train:.5f} | Val: {avg_val:.5f} | "
            f"Best: {best_val_loss:.5f} | Pat: {patience_cnt}/{PATIENCE} {marker}"
        )

    if patience_cnt >= PATIENCE:
        log.info(f"Erken durdurma — Epoch {epoch}")
        break

elapsed = time.time() - t0
train_history["best_val_loss"] = round(best_val_loss, 6)
log.info(f"Eğitim bitti | Süre: {elapsed:.1f}s | Best val loss: {best_val_loss:.6f}")

# ── Test değerlendirme ─────────────────────────────────────────
model.load_state_dict(torch.load(MODEL_DIR / "best_vae.pth", weights_only=True))
model.eval()
total_test = 0.0
with torch.no_grad():
    for (batch,) in test_loader:
        recon, mu, log_var = model(batch)
        loss, _, _ = vae_loss(recon, batch, mu, log_var, beta=1.0)
        total_test += loss.item()
avg_test = total_test / len(test_loader)
train_history["test_loss"] = round(avg_test, 6)
log.info(f"TEST Loss (görülmemiş veri): {avg_test:.6f}")

# ── Grafik 1: Eğitim/Val Kayıp Eğrisi ────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
ep_range = range(1, len(train_history["train_loss"]) + 1)
ax.plot(ep_range, train_history["train_loss"], label="Train Loss", color="#2196F3")
ax.plot(ep_range, train_history["val_loss"],   label="Val Loss",   color="#F44336")
ax.axhline(best_val_loss, linestyle="--", color="gray", alpha=0.6,
           label=f"Best Val: {best_val_loss:.4f}")
ax.set_title("VAE Eğitim / Validasyon Kayıp Eğrisi (Huber + β·KL)", fontsize=13, fontweight="bold")
ax.set_xlabel("Epoch"); ax.set_ylabel("Loss"); ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(PLOT_DIR / "01_egitim_kayip_egrisi.png", dpi=150)
plt.close()
log.info("Grafik 1 kaydedildi → 01_egitim_kayip_egrisi.png")

# ──────────────────────────────────────────────────────────────
# LATENT TEMSIL ÇIKARMA (μ vektörleri)
# VAE'de kümeleme için μ kullanırız (deterministic, gürültüsüz)
# ──────────────────────────────────────────────────────────────
log.info("Latent μ vektörleri çıkarılıyor...")
model.eval()
full_tensor = torch.FloatTensor(X_scaled)
with torch.no_grad():
    h          = model.encoder(full_tensor)
    mu_vectors = model.fc_mu(h).numpy()
    lv_vectors = model.fc_log_var(h).numpy()

recon_all, _, _ = model(full_tensor), None, None
with torch.no_grad():
    recon_all, mu_t, lv_t = model(full_tensor)
per_sample_recon = ((recon_all - full_tensor) ** 2).mean(dim=1).detach().numpy()

log.info(f"Latent boyutu: {mu_vectors.shape}")

# ──────────────────────────────────────────────────────────────
# GMM KÜMELEME — BIC + Silhouette ile en iyi k seçimi
# BIC: GMM'e özel, overfitting'i cezalandıran metrik (düşük = iyi)
# Silhouette: küme ayrışma kalitesi (yüksek = iyi)
# ──────────────────────────────────────────────────────────────
log.info("Optimal k aranıyor (GMM + BIC + Silhouette, k=5..10)...")

K_RANGE    = range(5, 11)
bic_scores = []
sil_scores = []
db_scores  = []
ch_scores  = []
gmm_models = {}

for k in K_RANGE:
    gmm = GaussianMixture(
        n_components=k, covariance_type='full',
        random_state=42, n_init=10, max_iter=500
    )
    gmm.fit(mu_vectors)
    labels = gmm.predict(mu_vectors)

    bic = gmm.bic(mu_vectors)
    sil = silhouette_score(mu_vectors, labels)
    db  = davies_bouldin_score(mu_vectors, labels)
    ch  = calinski_harabasz_score(mu_vectors, labels)

    bic_scores.append(bic)
    sil_scores.append(sil)
    db_scores.append(db)
    ch_scores.append(ch)
    gmm_models[k] = gmm

    log.info(
        f"  k={k:2d} | BIC: {bic:.1f} | Silhouette: {sil:.4f} | "
        f"DB: {db:.4f} | CH: {ch:.2f}"
    )

# Normalize + ağırlıklı skor (BIC %40, Silhouette %40, DB %20)
def _norm(a, higher_is_better=True):
    rng = a.max() - a.min() + 1e-8
    return (a - a.min()) / rng if higher_is_better else (a.max() - a) / rng

bic_arr = np.array(bic_scores)
sil_arr = np.array(sil_scores)
db_arr  = np.array(db_scores)
ch_arr  = np.array(ch_scores)

final_score = (
    0.40 * _norm(bic_arr, False) +   # BIC düşük = iyi
    0.40 * _norm(sil_arr) +          # Silhouette yüksek = iyi
    0.20 * _norm(db_arr, False)      # DB düşük = iyi
)
best_k  = list(K_RANGE)[int(np.argmax(final_score))]
log.info(f"En iyi k (BIC+Sil+DB dengeli): {best_k}")
train_history["best_k"] = best_k

# Final GMM
gmm_final    = gmm_models[best_k]
gmm_probs    = gmm_final.predict_proba(mu_vectors)
clusters     = gmm_final.predict(mu_vectors)
max_probs    = gmm_probs.max(axis=1)

player_info["Cluster"] = clusters
train_history["cluster_metrics"] = {
    "k_range":          list(K_RANGE),
    "bic":              [round(float(b), 2) for b in bic_arr],
    "silhouette":       [round(float(s), 4) for s in sil_arr],
    "davies_bouldin":   [round(float(d), 4) for d in db_arr],
    "calinski_harabasz":[round(float(c), 2) for c in ch_arr],
}

cluster_dist = pd.Series(clusters).value_counts().sort_index()
for cid, cnt in cluster_dist.items():
    log.info(f"  Cluster {cid}: {cnt} oyuncu")

# ── Grafik 2: Küme Seçim Metrikleri ──────────────────────────
ks = list(K_RANGE)
fig, axes = plt.subplots(1, 3, figsize=(17, 5))
axes[0].plot(ks, bic_arr, 'o-', color="#9C27B0")
axes[0].axvline(best_k, linestyle="--", color="red", alpha=0.7, label=f"Best k={best_k}")
axes[0].set_title("BIC Score (↓ iyi) — GMM'e Özel"); axes[0].set_xlabel("k")
axes[0].legend(); axes[0].grid(True, alpha=0.3)

axes[1].plot(ks, sil_arr, 's-', color="#4CAF50")
axes[1].axvline(best_k, linestyle="--", color="red", alpha=0.7)
axes[1].set_title("Silhouette Score (↑ iyi)"); axes[1].set_xlabel("k")
axes[1].grid(True, alpha=0.3)

axes[2].plot(ks, db_arr, '^-', color="#FF9800")
axes[2].axvline(best_k, linestyle="--", color="red", alpha=0.7)
axes[2].set_title("Davies-Bouldin (↓ iyi)"); axes[2].set_xlabel("k")
axes[2].grid(True, alpha=0.3)

plt.suptitle(f"GMM Küme Sayısı Seçim Metrikleri (k=5..10) — Best k={best_k}",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(PLOT_DIR / "02_kume_secim_metrikleri.png", dpi=150)
plt.close()
log.info("Grafik 2 kaydedildi → 02_kume_secim_metrikleri.png")

# ──────────────────────────────────────────────────────────────
# GÖRSELLEŞTİRME — t-SNE + UMAP
# ──────────────────────────────────────────────────────────────
log.info("t-SNE + UMAP görselleştirmesi...")
pca_pre   = PCA(n_components=min(LATENT_DIM, mu_vectors.shape[1]))
lat_pca   = pca_pre.fit_transform(mu_vectors)
tsne      = TSNE(n_components=2, perplexity=40, max_iter=2000,
                 learning_rate='auto', init='pca', random_state=42)
latent_2d = tsne.fit_transform(lat_pca)
log.info("t-SNE tamamlandı.")

try:
    reducer     = umap.UMAP(n_components=2, n_neighbors=20, min_dist=0.1,
                            metric='euclidean', random_state=42)
    latent_umap = reducer.fit_transform(mu_vectors)
    umap_ok     = True
    log.info("UMAP tamamlandı.")
except Exception as e:
    log.warning(f"UMAP kullanılamadı: {e}")
    latent_umap = latent_2d
    umap_ok     = False

PALETTE = sns.color_palette("tab10", best_k)

# Grafik 3: t-SNE + UMAP yan yana
fig, axes = plt.subplots(1, 2, figsize=(20, 8))
for ax_i, (coords, title) in enumerate([
    (latent_2d,   "t-SNE"),
    (latent_umap, "UMAP" if umap_ok else "t-SNE (UMAP yok)")
]):
    ax = axes[ax_i]
    for cid in range(best_k):
        mask = clusters == cid
        ax.scatter(coords[mask, 0], coords[mask, 1], c=[PALETTE[cid]],
                   label=f"C{cid} (n={mask.sum()})", s=55, alpha=0.75,
                   edgecolors='white', linewidths=0.4)
    ax.set_title(f"VAE Latent Uzay — {title}", fontsize=14, fontweight="bold")
    ax.set_xlabel(f"{title} Dim 1"); ax.set_ylabel(f"{title} Dim 2")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.2)
plt.tight_layout()
plt.savefig(PLOT_DIR / "03_tsne_umap_kumeler.png", dpi=150)
plt.close()
log.info("Grafik 3 kaydedildi → 03_tsne_umap_kumeler.png")

# Grafik 4: GMM Üyelik Güveni (max_prob dağılımı)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].hist(max_probs, bins=40, color="#009688", alpha=0.8, edgecolor='white')
axes[0].axvline(0.8, color='red', linestyle='--', label="Eşik: 0.8")
axes[0].set_title("GMM Maksimum Üyelik Olasılığı Dağılımı")
axes[0].set_xlabel("P(max küme)"); axes[0].set_ylabel("Oyuncu Sayısı")
axes[0].legend(); axes[0].grid(True, alpha=0.3)

sc = axes[1].scatter(latent_umap[:, 0], latent_umap[:, 1],
                     c=max_probs, cmap='RdYlGn', s=30, alpha=0.7)
plt.colorbar(sc, ax=axes[1], label="Üyelik Güveni")
axes[1].set_title(f"UMAP — GMM Güven Haritası (k={best_k})")
axes[1].set_xlabel("UMAP 1"); axes[1].set_ylabel("UMAP 2")
axes[1].grid(True, alpha=0.2)
plt.suptitle("GMM Üyelik Güveni Analizi", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(PLOT_DIR / "04_gmm_guven_haritasi.png", dpi=150)
plt.close()
log.info("Grafik 4 kaydedildi → 04_gmm_guven_haritasi.png")

# Grafik 5: Silhouette Analizi
sample_sil = silhouette_samples(mu_vectors, clusters)
fig, ax    = plt.subplots(figsize=(10, 6))
y_lower    = 10
for cid in range(best_k):
    vals    = np.sort(sample_sil[clusters == cid])
    y_upper = y_lower + len(vals)
    ax.fill_betweenx(np.arange(y_lower, y_upper), 0, vals,
                     facecolor=PALETTE[cid], alpha=0.7)
    ax.text(-0.05, y_lower + 0.5 * len(vals), str(cid), fontsize=8)
    y_lower = y_upper + 10
avg_sil = np.mean(sample_sil)
ax.axvline(avg_sil, color="red", linestyle="--", label=f"Ort: {avg_sil:.3f}")
ax.set_title(f"Silhouette Analizi — VAE+GMM (k={best_k})", fontsize=13, fontweight="bold")
ax.set_xlabel("Silhouette Katsayısı"); ax.set_ylabel("Küme")
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(PLOT_DIR / "05_silhouette_analizi.png", dpi=150)
plt.close()
log.info("Grafik 5 kaydedildi → 05_silhouette_analizi.png")

# Grafik 6: Cluster Radar
df_with_cluster  = df[features].copy()
df_with_cluster["Cluster"] = clusters
cluster_means    = df_with_cluster.groupby("Cluster")[features].mean()
cmin, cmax       = cluster_means.min(), cluster_means.max()
cluster_norm     = (cluster_means - cmin) / (cmax - cmin + 1e-8)

def radar_chart(ax, values, labels, title, color):
    N      = len(labels)
    angles = [n / float(N) * 2 * np.pi for n in range(N)] + [0]
    vals   = list(values) + [values[0]]
    ax.set_theta_offset(np.pi / 2); ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1]); ax.set_xticklabels(labels, size=7)
    ax.plot(angles, vals, color=color, linewidth=2)
    ax.fill(angles, vals, color=color, alpha=0.25)
    ax.set_ylim(0, 1); ax.set_title(title, size=10, fontweight="bold", pad=15)

cols  = 3; rows = (best_k + cols - 1) // cols
fig_r = plt.figure(figsize=(cols * 5, rows * 5))
for cid in range(best_k):
    ax = fig_r.add_subplot(rows, cols, cid + 1, polar=True)
    radar_chart(ax, cluster_norm.loc[cid].values, features, f"Cluster {cid}", PALETTE[cid])
plt.suptitle("VAE+GMM Cluster Profil Radarları", fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(PLOT_DIR / "06_cluster_radar.png", dpi=150, bbox_inches='tight')
plt.close()
log.info("Grafik 6 kaydedildi → 06_cluster_radar.png")

# Grafik 7: Rekonstrüksiyon Hatası
anomaly_thr = np.percentile(per_sample_recon, 95)
fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(per_sample_recon, bins=50, color="#3F51B5", alpha=0.8, edgecolor='white')
ax.axvline(per_sample_recon.mean(), color='red', linestyle='--',
           label=f"Ort: {per_sample_recon.mean():.4f}")
ax.axvline(anomaly_thr, color='orange', linestyle=':',
           label=f"Anomali eşiği (95p): {anomaly_thr:.4f}")
ax.set_title("Rekonstrüksiyon Hata Dağılımı (VAE)", fontsize=13, fontweight="bold")
ax.set_xlabel("MSE"); ax.set_ylabel("Oyuncu Sayısı")
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(PLOT_DIR / "07_rekonstruksiyon_hata.png", dpi=150)
plt.close()
log.info("Grafik 7 kaydedildi → 07_rekonstruksiyon_hata.png")

# ──────────────────────────────────────────────────────────────
# BENZERLİK MATRİSİ (Cosine × 0.6 + Euclidean × 0.4)
# ──────────────────────────────────────────────────────────────
cos_sim  = cosine_similarity(mu_vectors)
euc_dist = euclidean_distances(mu_vectors)
euc_sim  = 1.0 / (1.0 + euc_dist)
COSINE_W = 0.6; EUCL_W = 0.4
sim_matrix = COSINE_W * cos_sim + EUCL_W * euc_sim

# ──────────────────────────────────────────────────────────────
# OYUNCU ÖNERİ SİSTEMİ
# ──────────────────────────────────────────────────────────────
def recommend_players(player_name: str, top_n: int = 5,
                      same_cluster_only: bool = False,
                      cross_penalty: float = 0.15) -> list:
    if player_name not in player_info[PLAYER_COL].values:
        log.warning(f"Oyuncu bulunamadı: {player_name}"); return []
    idx        = player_info[player_info[PLAYER_COL] == player_name].index[0]
    my_cluster = int(player_info.iloc[idx]["Cluster"])
    scores     = sim_matrix[idx].copy()
    for i in range(len(scores)):
        if int(player_info.iloc[i]["Cluster"]) != my_cluster:
            scores[i] = -1.0 if same_cluster_only else scores[i] * (1.0 - cross_penalty)
    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[1:top_n + 1]
    log.info("=" * 60)
    log.info(f"{player_name} → En benzer {top_n} oyuncu (Cluster: {my_cluster})")
    results = []
    for rank, (pidx, score) in enumerate(ranked, 1):
        row = player_info.iloc[pidx]
        nm  = row[PLAYER_COL]; c = int(row['Cluster'])
        log.info(f"  {rank}. {nm:<30} | Skor: {score:.4f} | Cluster: {c}")
        results.append({"rank": rank, "player": nm, "score": round(score, 4),
                        "cluster": c, "same_cluster": c == my_cluster,
                        "gmm_confidence": round(float(gmm_probs[pidx, c]), 4)})
    return results

def predict_new_player(player_stats: list, top_n: int = 5) -> dict:
    arr    = np.array(player_stats).reshape(1, -1)
    scaled = scaler.transform(arr)
    tensor = torch.FloatTensor(scaled)
    model.eval()
    with torch.no_grad():
        h  = model.encoder(tensor)
        mu = model.fc_mu(h).numpy()
    probs       = gmm_final.predict_proba(mu)[0]
    cluster     = int(np.argmax(probs))
    confidence  = probs[cluster]
    cos_s       = cosine_similarity(mu, mu_vectors)[0]
    euc_d       = euclidean_distances(mu, mu_vectors)[0]
    hybrid      = COSINE_W * cos_s + EUCL_W * (1.0 / (1.0 + euc_d))
    top_idx     = np.argsort(hybrid)[::-1][:top_n]
    log.info("=" * 60)
    log.info(f"YENİ OYUNCU → Cluster: {cluster} | GMM Güven: {confidence:.3f}")
    return {
        "cluster": cluster, "gmm_confidence": round(float(confidence), 4),
        "cluster_probs": {f"C{i}": round(float(p), 4) for i, p in enumerate(probs)},
        "top_similar": [{"player": player_info.iloc[i][PLAYER_COL],
                         "score": round(float(hybrid[i]), 4),
                         "cluster": int(player_info.iloc[i]["Cluster"])}
                        for i in top_idx]
    }

# ──────────────────────────────────────────────────────────────
# ÇIKTI DOSYALARI
# ──────────────────────────────────────────────────────────────
# Latent vektörler
latent_out = pd.DataFrame(mu_vectors, columns=[f"mu_{i+1}" for i in range(LATENT_DIM)])
latent_out["Cluster"]        = clusters
latent_out["GMM_Confidence"] = max_probs
latent_out["Recon_Error"]    = per_sample_recon
latent_out["Player"]         = player_info[PLAYER_COL].values
latent_out.to_csv(OUTPUT_DIR / "latent_features.csv", index=False)

# t-SNE / UMAP koordinatları
pd.DataFrame({
    "Player": player_info[PLAYER_COL].values, "Cluster": clusters,
    "tSNE_1": latent_2d[:, 0], "tSNE_2": latent_2d[:, 1],
    "UMAP_1": latent_umap[:, 0], "UMAP_2": latent_umap[:, 1]
}).to_csv(OUTPUT_DIR / "tsne_umap_koordinatlari.csv", index=False)

# GMM olasılıkları
gmm_prob_df = pd.DataFrame(gmm_probs, columns=[f"P_C{i}" for i in range(best_k)])
gmm_prob_df["Player"]  = player_info[PLAYER_COL].values
gmm_prob_df["Cluster"] = clusters
gmm_prob_df.to_csv(OUTPUT_DIR / "gmm_olasiliklar.csv", index=False)

# Cluster ortalamalar
cluster_means.round(4).to_csv(OUTPUT_DIR / "cluster_feature_ortalamalar.csv")

# Model kaydet
torch.save(model.state_dict(), MODEL_DIR / "best_vae.pth")
torch.save(model.state_dict(), MODEL_DIR / "football_vae_final.pth")

# JSON eğitim tarihi
train_history["avg_silhouette"] = round(float(avg_sil), 4)
train_history["anomaly_threshold"] = round(float(anomaly_thr), 6)
with open(LOG_DIR / "egitim_gecmisi.json", "w", encoding="utf-8") as f:
    json.dump(train_history, f, ensure_ascii=False, indent=2)

log.info("Tüm çıktı dosyaları kaydedildi.")

# ──────────────────────────────────────────────────────────────
# CLUSTER ÖZET RAPORU
# ──────────────────────────────────────────────────────────────
log.info(""); log.info("=" * 65)
log.info("CLUSTER BAZLI OYUNCU LİSTESİ")
log.info("=" * 65)
for cid in sorted(player_info["Cluster"].unique()):
    players   = player_info[player_info["Cluster"] == cid][PLAYER_COL].head(10).tolist()
    top_feats = cluster_norm.loc[cid].nlargest(3).index.tolist()
    log.info(f"\n── CLUSTER {cid} (güçlü: {', '.join(top_feats)}) ──")
    for p in players:
        log.info(f"   {p}")

# Örnek öneri
sample_player = player_info[PLAYER_COL].iloc[0]
recommend_players(sample_player, top_n=5)

# ──────────────────────────────────────────────────────────────
# TAMAMLANMA
# ──────────────────────────────────────────────────────────────
log.info(""); log.info("=" * 65)
log.info("SİSTEM V2 BAŞARIYLA TAMAMLANDI")
log.info(f"Tüm çıktılar: {OUTPUT_DIR.resolve()}")
log.info(f"  ├── loglar/       → sistem_logu.txt, egitim_gecmisi.json")
log.info(f"  ├── grafikler/    → 7 adet PNG grafiği")
log.info(f"  ├── modeller/     → best_vae.pth, football_vae_final.pth")
log.info(f"  ├── latent_features.csv   (μ vektörleri + GMM güven)")
log.info(f"  ├── tsne_umap_koordinatlari.csv")
log.info(f"  ├── gmm_olasiliklar.csv")
log.info(f"  └── cluster_feature_ortalamalar.csv")
log.info("=" * 65)
