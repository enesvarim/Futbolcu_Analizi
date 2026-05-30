"""
src/analysis/similarity.py
===========================
Hibrit benzerlik hesabı ve oyuncu öneri sistemi.
Cosine (0.6) + Euclidean (0.4) ağırlıklı hibrit skor.
"""
import logging
from typing import Optional

import joblib
import numpy as np
import pandas as pd
import torch
from scipy.stats import mode as _scipy_mode
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.preprocessing import RobustScaler

from src.config import COSINE_WEIGHT, EUCLIDEAN_WEIGHT, CLUSTER_NAMES, CLUSTER_NAMES_V1, FEATURES, V2_GMM

log = logging.getLogger(__name__)


def compute_hybrid_similarity(
    query_vec: np.ndarray,
    latent_matrix: np.ndarray,
    cosine_w: float    = COSINE_WEIGHT,
    euclidean_w: float = EUCLIDEAN_WEIGHT,
) -> np.ndarray:
    """
    Sorgu vektörü ile tüm latent matris arasındaki hibrit benzerliği hesaplar.

    Parameters
    ----------
    query_vec     : np.ndarray  [1, latent_dim] veya [latent_dim]
    latent_matrix : np.ndarray  [N, latent_dim]

    Returns
    -------
    hybrid_scores : np.ndarray  [N]  — 0-1 arası hibrit benzerlik skoru
    """
    q = query_vec.reshape(1, -1)
    cos_sim  = cosine_similarity(q, latent_matrix)[0]
    
    # Cosine değerini [-1, 1] aralığından [0, 1] aralığına ölçekle (Negatif çıkmasını önlemek için!)
    cos_sim_scaled = (cos_sim + 1.0) / 2.0
    
    euc_dist = euclidean_distances(q, latent_matrix)[0]
    euc_sim  = 1.0 / (1.0 + euc_dist)
    
    return cosine_w * cos_sim_scaled + euclidean_w * euc_sim


def predict_and_find_similar(
    stats_array: np.ndarray,
    model: torch.nn.Module,
    scaler: RobustScaler,
    latent_matrix: np.ndarray,
    latent_df: pd.DataFrame,
    version: str = "v1",
    top_n: int   = 5,
) -> tuple[int, pd.DataFrame, np.ndarray, Optional[float]]:
    """
    Oyuncu istatistiklerini latent uzaya gönderir ve en benzer oyuncuları bulur.

    Parameters
    ----------
    stats_array    : Ham (ölçeklenmemiş) istatistik dizisi [n_features]
    model          : Eğitilmiş model (encode() metodu olmalı)
    scaler         : Fit edilmiş RobustScaler
    latent_matrix  : Tüm oyuncuların latent vektörleri [N, latent_dim]
    latent_df      : Player ve Cluster sütunlarını içeren DataFrame
    version        : "v1" veya "v2" — çıktı formatını etkiler
    top_n          : Döndürülecek benzer oyuncu sayısı

    Returns
    -------
    predicted_cluster : int
    similar_df        : pd.DataFrame  [Oyuncu, Oyun Stili, Benzerlik (%)]
    latent_vec        : np.ndarray    [latent_dim]
    confidence_pct    : float | None  — GMM güven skoru (0-100), sadece v2 için
    """
    scaled  = scaler.transform(stats_array.reshape(1, -1))
    tensor  = torch.FloatTensor(scaled)

    with torch.no_grad():
        latent_vec = model.encode(tensor).numpy()

    hybrid_scores = compute_hybrid_similarity(latent_vec, latent_matrix)
    top_indices   = np.argsort(hybrid_scores)[::-1][:top_n]


    # Bu KNN sınıflandırmasının standart yaklaşımı — kenarda kalan oyuncular için daha doğru sonuç verir.
    neighbor_clusters = [int(latent_df.iloc[i]["Cluster"]) for i in top_indices]
    predicted_cluster = int(_scipy_mode(neighbor_clusters, keepdims=False).mode)

    rows = []
    for idx in top_indices:
        c_id  = int(latent_df.iloc[idx]["Cluster"])
        if version == "v2" and c_id in CLUSTER_NAMES:
            style = f"Küme {c_id} ({CLUSTER_NAMES[c_id]})"
        elif version == "v1" and c_id in CLUSTER_NAMES_V1:
            style = f"Küme {c_id} ({CLUSTER_NAMES_V1[c_id]})"
        else:
            style = str(c_id)
        rows.append({
            "Oyuncu":            latent_df.iloc[idx]["Player"],
            "Oyun Stili (Küme)": style,
            "Benzerlik (%)":     round(hybrid_scores[idx] * 100, 1),
        })

    # GMM Güven Skoru — sadece V2 için
    confidence_pct: Optional[float] = None
    if version == "v2" and V2_GMM.exists():
        try:
            gmm = joblib.load(V2_GMM)
            proba = gmm.predict_proba(latent_vec)          # [1, n_clusters]
            confidence_pct = float(proba.max(axis=1)[0]) * 100  # 0-100 arası
        except Exception as e:
            log.warning(f"GMM confidence hesaplanamadı: {e}")

    return predicted_cluster, pd.DataFrame(rows), latent_vec[0], confidence_pct


def compare_players(
    stats_a: np.ndarray,
    stats_b: np.ndarray,
    model: torch.nn.Module,
    scaler: RobustScaler,
    latent_matrix: np.ndarray,
    latent_df: pd.DataFrame,
) -> dict:
    """
    İki oyuncuyu V2 latent uzayında karşılaştırır.

    Returns
    -------
    dict içeriği:
        cluster_a, cluster_b : int
        latent_a, latent_b   : np.ndarray
        similarity_pct       : int  — 0-100 arası benzerlik yüzdesi
    """
    def _get_info(stats):
        cluster, _, latent, _ = predict_and_find_similar(
            stats, model, scaler, latent_matrix, latent_df, version="v2"
        )
        return cluster, latent

    c_a, la = _get_info(stats_a)
    c_b, lb = _get_info(stats_b)

    # Hibrit benzerlik hesabı (Negatif çıkmayı önlemek için Cosine [0,1] aralığına ölçeklenmiştir)
    q_a = la.reshape(1, -1)
    q_b = lb.reshape(1, -1)
    cos_sim  = cosine_similarity(q_a, q_b)[0][0]
    cos_sim_scaled = (cos_sim + 1.0) / 2.0
    
    euc_dist = euclidean_distances(q_a, q_b)[0][0]
    euc_sim  = 1.0 / (1.0 + euc_dist)

    hybrid_score = COSINE_WEIGHT * cos_sim_scaled + EUCLIDEAN_WEIGHT * euc_sim
    sim_pct = int(round(hybrid_score * 100))

    return {
        "cluster_a":      c_a,
        "cluster_b":      c_b,
        "latent_a":       la,
        "latent_b":       lb,
        "similarity_pct": sim_pct,
    }
