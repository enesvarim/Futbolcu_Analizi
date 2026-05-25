"""
src/training/clustering.py
===========================
Kümeleme algoritmaları, metrik hesaplama ve ensemble oylama.
V1 (Autoencoder) ve V2 (VAE) tarafından ortaklaşa kullanılır.
"""
import logging
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
    silhouette_samples,
)

from src.config import CLUSTER_K_RANGE

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Metrik normalizasyonu
# ---------------------------------------------------------------------------
def _norm(arr: np.ndarray, higher_is_better: bool = True) -> np.ndarray:
    rng = arr.max() - arr.min() + 1e-8
    return (arr - arr.min()) / rng if higher_is_better else (arr.max() - arr) / rng


# ---------------------------------------------------------------------------
# Etiket hizalama (Hungarian-benzeri)
# ---------------------------------------------------------------------------
def align_labels(reference: np.ndarray, other: np.ndarray, n_clusters: int) -> np.ndarray:
    """
    Farklı bir algoritmanın etiketlerini referans etiketlere hizalar.
    Ensemble oylama öncesi çağrılır.
    """
    cost = np.zeros((n_clusters, n_clusters))
    for i in range(n_clusters):
        for j in range(n_clusters):
            cost[i, j] = -np.sum((reference == i) & (other == j))
    row_ind, col_ind = linear_sum_assignment(cost)
    aligned = np.zeros_like(other)
    for r, c in zip(row_ind, col_ind):
        aligned[other == c] = r
    return aligned


# ---------------------------------------------------------------------------
# V1: Ensemble Kümeleme (KMeans + Agglomerative + GMM + çoğunluk oyu)
# ---------------------------------------------------------------------------
def find_best_k_ensemble(
    latent_vectors: np.ndarray,
    k_range=CLUSTER_K_RANGE,
) -> tuple[int, dict]:
    """
    Silhouette (0.5) + Davies-Bouldin (0.3) + Calinski-Harabasz (0.2) ağırlıklı
    skorla en iyi k sayısını belirler.

    Returns
    -------
    best_k  : int
    metrics : dict  — tüm metrik dizileri
    """
    sil_scores, db_scores, ch_scores = [], [], []

    for k in k_range:
        km      = KMeans(n_clusters=k, random_state=42, n_init=20, max_iter=600)
        lbl_km  = km.fit_predict(latent_vectors)
        sil     = silhouette_score(latent_vectors, lbl_km)
        db      = davies_bouldin_score(latent_vectors, lbl_km)
        ch      = calinski_harabasz_score(latent_vectors, lbl_km)
        sil_scores.append(sil)
        db_scores.append(db)
        ch_scores.append(ch)
        log.info(f"  k={k:2d} | Silhouette: {sil:.4f} | DB: {db:.4f} | CH: {ch:.2f}")

    sil_arr = np.array(sil_scores)
    db_arr  = np.array(db_scores)
    ch_arr  = np.array(ch_scores)

    final_score = 0.50 * _norm(sil_arr) + 0.30 * _norm(db_arr, False) + 0.20 * _norm(ch_arr)
    best_k = list(k_range)[int(np.argmax(final_score))]
    log.info(f"En iyi k (ensemble skor bazlı): {best_k}")

    return best_k, {
        "k_range":          list(k_range),
        "silhouette":       [round(float(s), 4) for s in sil_arr],
        "davies_bouldin":   [round(float(d), 4) for d in db_arr],
        "calinski_harabasz":[round(float(c), 2) for c in ch_arr],
    }


def run_ensemble_clustering(
    latent_vectors: np.ndarray,
    k: int,
) -> tuple[np.ndarray, np.ndarray, GaussianMixture]:
    """
    KMeans + Agglomerative + GMM ensemble ile nihai küme atamalarını üretir.

    Returns
    -------
    clusters     : np.ndarray  — nihai ensemble etiketleri
    gmm_probs    : np.ndarray  — GMM üyelik olasılıkları [N, k]
    gmm_final    : GaussianMixture
    """
    km      = KMeans(n_clusters=k, random_state=42, n_init=30, max_iter=1000)
    lbl_km  = km.fit_predict(latent_vectors)

    agg     = AgglomerativeClustering(n_clusters=k, linkage="ward")
    lbl_agg = agg.fit_predict(latent_vectors)

    gmm_final = GaussianMixture(n_components=k, random_state=42, n_init=10,
                                covariance_type="full", max_iter=500)
    gmm_final.fit(latent_vectors)
    lbl_gmm   = gmm_final.predict(latent_vectors)
    gmm_probs = gmm_final.predict_proba(latent_vectors)

    lbl_agg_aligned = align_labels(lbl_km, lbl_agg, k)
    lbl_gmm_aligned = align_labels(lbl_km, lbl_gmm, k)

    ensemble_matrix = np.stack([lbl_km, lbl_agg_aligned, lbl_gmm_aligned], axis=1)
    clusters, _     = scipy_stats.mode(ensemble_matrix, axis=1, keepdims=False)
    clusters        = clusters.flatten().astype(int)

    vote_agree = np.mean(np.all(ensemble_matrix == clusters[:, None], axis=1))
    log.info(f"Ensemble oy uyumu (3/3): {vote_agree:.3f}")

    return clusters, gmm_probs, gmm_final


# ---------------------------------------------------------------------------
# V2: GMM Kümeleme (BIC + Silhouette ile k seçimi)
# ---------------------------------------------------------------------------
def find_best_k_gmm(
    mu_vectors: np.ndarray,
    k_range=CLUSTER_K_RANGE,
) -> tuple[int, dict, dict[int, GaussianMixture]]:
    """
    BIC (0.4) + Silhouette (0.4) + Davies-Bouldin (0.2) ağırlıklı skor
    ile en iyi GMM k sayısını belirler.

    Returns
    -------
    best_k     : int
    metrics    : dict
    gmm_models : dict[k, GaussianMixture]
    """
    bic_scores, sil_scores, db_scores, ch_scores = [], [], [], []
    gmm_models: dict[int, GaussianMixture] = {}

    for k in k_range:
        gmm    = GaussianMixture(n_components=k, covariance_type="full",
                                  random_state=42, n_init=10, max_iter=500)
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
        log.info(f"  k={k:2d} | BIC: {bic:.1f} | Silhouette: {sil:.4f} | DB: {db:.4f}")

    bic_arr = np.array(bic_scores)
    sil_arr = np.array(sil_scores)
    db_arr  = np.array(db_scores)
    ch_arr  = np.array(ch_scores)

    final_score = (
        0.40 * _norm(bic_arr, False) +
        0.40 * _norm(sil_arr) +
        0.20 * _norm(db_arr, False)
    )
    best_k = list(k_range)[int(np.argmax(final_score))]
    log.info(f"En iyi k (BIC+Sil+DB dengeli): {best_k}")

    metrics = {
        "k_range":          list(k_range),
        "bic":              [round(float(b), 2) for b in bic_arr],
        "silhouette":       [round(float(s), 4) for s in sil_arr],
        "davies_bouldin":   [round(float(d), 4) for d in db_arr],
        "calinski_harabasz":[round(float(c), 2) for c in ch_arr],
    }
    return best_k, metrics, gmm_models


def run_gmm_clustering(
    mu_vectors: np.ndarray,
    gmm_models: dict[int, GaussianMixture],
    best_k: int,
) -> tuple[np.ndarray, np.ndarray, GaussianMixture]:
    """
    Seçilen k ile nihai GMM küme atamalarını üretir.

    Returns
    -------
    clusters  : np.ndarray
    gmm_probs : np.ndarray  [N, best_k]
    gmm_final : GaussianMixture
    """
    gmm_final = gmm_models[best_k]
    clusters  = gmm_final.predict(mu_vectors)
    gmm_probs = gmm_final.predict_proba(mu_vectors)
    return clusters, gmm_probs, gmm_final


# ---------------------------------------------------------------------------
# Küme Özeti
# ---------------------------------------------------------------------------
def cluster_summary(
    features_df: pd.DataFrame,
    clusters: np.ndarray,
    features: list[str],
) -> pd.DataFrame:
    """
    Her kümenin feature ortalamalarını hesaplar.

    Returns
    -------
    cluster_means : pd.DataFrame  [n_clusters × n_features]
    """
    df = features_df[features].copy()
    df["Cluster"] = clusters
    return df.groupby("Cluster")[features].mean()
