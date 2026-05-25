"""
src/data/preprocessor.py
=========================
Feature ön-işleme: eksik değer doldurma, aykırı değer kırpma, ölçekleme.
RobustScaler nesnesi eğitimden uygulamaya taşınır.
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split

from src.config import FEATURES, TRAINING


def clean_features(df: pd.DataFrame, features: list[str] = FEATURES) -> pd.DataFrame:
    """
    İlgili feature sütunlarını seçer, sonsuz/NaN değerleri temizler.

    Parameters
    ----------
    df : pd.DataFrame
        Ham ya da temizlenmiş veri.
    features : list[str]
        Kullanılacak feature isimleri.

    Returns
    -------
    pd.DataFrame
        Temizlenmiş feature matrisi.
    """
    X = df[features].copy()
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median())
    return X


def fit_scaler(X: pd.DataFrame) -> tuple[np.ndarray, RobustScaler]:
    """
    RobustScaler'ı fit eder ve dönüştürülmüş numpy dizisini döner.

    Returns
    -------
    X_scaled : np.ndarray
    scaler   : RobustScaler  (serialize/deserialize için saklanmalı)
    """
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X.values)
    return X_scaled, scaler


def split_data(
    X_scaled: np.ndarray,
    test_size: float = TRAINING["test_size"],
    val_size: float  = TRAINING["val_size"],
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    70 / 15 / 15 train/val/test bölümlemesi uygular.

    Returns
    -------
    X_train, X_val, X_test : np.ndarray
    """
    X_temp, X_test = train_test_split(X_scaled, test_size=test_size,
                                       random_state=random_state)
    X_train, X_val = train_test_split(X_temp, test_size=val_size,
                                       random_state=random_state)
    return X_train, X_val, X_test


def prepare_data(
    df: pd.DataFrame,
    features: list[str] = FEATURES,
) -> tuple[pd.DataFrame, np.ndarray, RobustScaler, np.ndarray, np.ndarray, np.ndarray]:
    """
    Tam ön-işleme pipeline: temizle → ölçekle → böl.

    Returns
    -------
    X_clean  : pd.DataFrame  — ölçeklenmemiş temiz veri (radar grafiği için)
    X_scaled : np.ndarray    — ölçeklenmiş tüm veri (latent çıkarma için)
    scaler   : RobustScaler
    X_train, X_val, X_test : np.ndarray
    """
    X_clean = clean_features(df, features)
    X_scaled, scaler = fit_scaler(X_clean)
    X_train, X_val, X_test = split_data(X_scaled)
    return X_clean, X_scaled, scaler, X_train, X_val, X_test
