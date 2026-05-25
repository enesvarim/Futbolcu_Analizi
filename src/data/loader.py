"""
src/data/loader.py
==================
Ham CSV dosyalarını okur ve temel filtreleri uygular.
Kaleciler ve az oynayan oyuncular bu aşamada elenir.
"""
from pathlib import Path

import pandas as pd

from src.config import VERI_DIR, EXCLUDE_POS, MIN_90S, FEATURES


def load_raw_players(veri_dir: Path = VERI_DIR) -> pd.DataFrame:
    """
    futbolcular.csv'yi okur, kalecileri ve az oynayan oyuncuları filtreler.

    Returns
    -------
    pd.DataFrame
        Filtrelenmiş oyuncu bilgi tablosu (Player, Pos, Age, vb.).
    """
    df = pd.read_csv(veri_dir / "futbolcular.csv", encoding="utf-8")
    df = df[~df["Pos"].str.contains(EXCLUDE_POS, na=False)]
    df = df[df["90s"] >= MIN_90S].reset_index(drop=True)
    return df


def load_clean_features(veri_dir: Path = VERI_DIR) -> pd.DataFrame:
    """
    temiz_veri.csv'yi okur (normalize/temizlenmiş feature matrisi).

    Returns
    -------
    pd.DataFrame
        Sadece özellik sütunlarını içeren DataFrame.
    """
    df = pd.read_csv(veri_dir / "temiz_veri.csv", encoding="utf-8")
    return df


def load_dataset(veri_dir: Path = VERI_DIR) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Ham oyuncu bilgisini ve temiz feature matrisini birlikte yükler.

    Returns
    -------
    players : pd.DataFrame
        Filtrelenmiş oyuncu bilgisi.
    features_df : pd.DataFrame
        İlgili feature sütunlarını içeren temiz veri.
    """
    players = load_raw_players(veri_dir)
    features_df = load_clean_features(veri_dir)

    # Boyut uyuşmazlığı kontrolü
    if len(players) != len(features_df):
        raise ValueError(
            f"Oyuncu sayısı ({len(players)}) ile feature satır sayısı "
            f"({len(features_df)}) eşleşmiyor. "
            "Veri setini ve filtreleri kontrol edin."
        )

    return players, features_df
