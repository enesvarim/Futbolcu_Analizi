"""
src/analysis/similarity.py
===========================
Hibrit benzerlik hesabı ve oyuncu öneri sistemi.
Cosine (0.6) + Euclidean (0.4) ağırlıklı hibrit skor.
"""
import numpy as np
import pandas as pd
import torch
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.preprocessing import RobustScaler

from src.config import COSINE_WEIGHT, EUCLIDEAN_WEIGHT, CLUSTER_NAMES, CLUSTER_NAMES_V1, FEATURES


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
    euc_dist = euclidean_distances(q, latent_matrix)[0]
    euc_sim  = 1.0 / (1.0 + euc_dist)
    return cosine_w * cos_sim + euclidean_w * euc_sim


def predict_and_find_similar(
    stats_array: np.ndarray,
    model: torch.nn.Module,
    scaler: RobustScaler,
    latent_matrix: np.ndarray,
    latent_df: pd.DataFrame,
    version: str = "v1",
    top_n: int   = 5,
) -> tuple[int, pd.DataFrame, np.ndarray]:
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
    """
    scaled  = scaler.transform(stats_array.reshape(1, -1))
    tensor  = torch.FloatTensor(scaled)

    with torch.no_grad():
        latent_vec = model.encode(tensor).numpy()

    hybrid_scores = compute_hybrid_similarity(latent_vec, latent_matrix)
    top_indices   = np.argsort(hybrid_scores)[::-1][:top_n]

    predicted_cluster = int(latent_df.iloc[top_indices[0]]["Cluster"])

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

    return predicted_cluster, pd.DataFrame(rows), latent_vec[0]


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
    def _get(stats):
        _, _, latent = predict_and_find_similar(
            stats, model, scaler, latent_matrix, latent_df, version="v2"
        )
        return latent

    la = _get(stats_a)
    lb = _get(stats_b)

    cos_sim     = cosine_similarity(la.reshape(1, -1), lb.reshape(1, -1))[0][0]
    sim_pct     = int(((cos_sim + 1) / 2) * 100)

    return {
        "latent_a":       la,
        "latent_b":       lb,
        "similarity_pct": sim_pct,
    }
