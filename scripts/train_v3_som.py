import os
import json
import joblib
import numpy as np
import pandas as pd

from minisom import MiniSom
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity


# ==========================================================
# MODEL V3 - SOM / SELF ORGANIZING MAP
# Futbolcu oyun stili analizi için üçüncü model
# ==========================================================


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

POSSIBLE_DATA_PATHS = [
    os.path.join(BASE_DIR, "veriseti", "futbolcular.csv"),
    os.path.join(BASE_DIR, "veriseti", "temiz_veri.csv"),
]

V3_DIR = os.path.join(BASE_DIR, "artifacts", "v3")
MODEL_DIR = os.path.join(V3_DIR, "modeller")

os.makedirs(V3_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


FEATURES = [
    "Gls", "Ast", "xG", "xAG", "npxG", "Sh/90",
    "Cmp%", "PrgP", "KP", "PPA", "SCA90",
    "Tkl", "TklW", "Int", "Clr",
    "PrgC", "PrgR", "Succ%", "Carries", "Touches",
]


def find_data_file():
    for path in POSSIBLE_DATA_PATHS:
        if os.path.exists(path):
            return path

    raise FileNotFoundError(
        "Veri dosyası bulunamadı. veriseti/temiz_veri.csv veya veriseti/futbolcular.csv dosyasını kontrol et."
    )


def find_player_column(df):
    possible_columns = [
        "Player", "Oyuncu", "Name", "player", "player_name",
        "Oyuncu Adı", "oyuncu", "Futbolcu"
    ]

    for col in possible_columns:
        if col in df.columns:
            return col

    raise ValueError(
        "Oyuncu adı sütunu bulunamadı. CSV içinde Player, Oyuncu, Name veya benzeri bir sütun olmalı."
    )


def prepare_dataset(df):
    player_col = find_player_column(df)

    # Kalecileri çıkarıyoruz
    if "Pos" in df.columns:
        df = df[~df["Pos"].astype(str).str.contains("GK", na=False)].copy()

    # Çok az süre almış oyuncuları çıkarıyoruz
    # Veri setinde 90s sütunu varsa en az 5 maçlık süre şartı uygulanır.
    if "90s" in df.columns:
        df["90s"] = pd.to_numeric(df["90s"], errors="coerce")
        df = df[df["90s"] >= 5].copy()

    # Alternatif olarak Min sütunu varsa en az 450 dakika şartı uygulanır.
    elif "Min" in df.columns:
        df["Min"] = pd.to_numeric(df["Min"], errors="coerce")
        df = df[df["Min"] >= 450].copy()

    missing_features = [feature for feature in FEATURES if feature not in df.columns]

    if missing_features:
        raise ValueError(f"Veri setinde eksik özellikler var: {missing_features}")

    df_model = df[[player_col] + FEATURES].copy()

    for feature in FEATURES:
        df_model[feature] = pd.to_numeric(df_model[feature], errors="coerce")

    for feature in FEATURES:
        df_model[feature] = df_model[feature].fillna(df_model[feature].median())

    df_model = df_model.drop_duplicates(subset=[player_col]).reset_index(drop=True)

    X = df_model[FEATURES].values

    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)

    return df_model, X_scaled, scaler, player_col


def train_som(X_scaled):
    som = MiniSom(
        x=12,
        y=12,
        input_len=X_scaled.shape[1],
        sigma=1.5,
        learning_rate=0.5,
        neighborhood_function="gaussian",
        random_seed=42
    )

    som.random_weights_init(X_scaled)

    print("SOM modeli eğitiliyor...")
    som.train_random(X_scaled, num_iteration=5000)
    print("SOM eğitimi tamamlandı.")

    return som


def add_som_coordinates(df_model, X_scaled, som):
    coordinates = np.array([som.winner(x) for x in X_scaled])

    df_result = df_model.copy()
    df_result["som_x"] = coordinates[:, 0]
    df_result["som_y"] = coordinates[:, 1]
    df_result["som_cluster"] = (
        df_result["som_x"].astype(str) + "-" + df_result["som_y"].astype(str)
    )

    return df_result


def calculate_metrics(X_scaled, df_result, som):
    metrics = {}

    metrics["quantization_error"] = float(som.quantization_error(X_scaled))

    try:
        metrics["topographic_error"] = float(som.topographic_error(X_scaled))
    except Exception:
        metrics["topographic_error"] = None

    labels = df_result["som_cluster"].astype("category").cat.codes

    try:
        if len(set(labels)) > 1:
            metrics["silhouette_score"] = float(silhouette_score(X_scaled, labels))
        else:
            metrics["silhouette_score"] = None
    except Exception:
        metrics["silhouette_score"] = None

    return metrics


def calculate_similar_players(df_result, X_scaled, player_col):
    similarity_matrix = cosine_similarity(X_scaled)

    rows = []

    for i, player_name in enumerate(df_result[player_col]):
        similarities = similarity_matrix[i]

        similar_indices = np.argsort(similarities)[::-1]
        similar_indices = [idx for idx in similar_indices if idx != i][:5]

        for rank, idx in enumerate(similar_indices, start=1):
            rows.append({
                "oyuncu": player_name,
                "benzer_oyuncu": df_result.iloc[idx][player_col],
                "sira": rank,
                "benzerlik_yuzde": round(float(similarities[idx] * 100), 2),
                "som_cluster": df_result.iloc[i]["som_cluster"],
                "benzer_oyuncu_som_cluster": df_result.iloc[idx]["som_cluster"]
            })

    return pd.DataFrame(rows)


def save_outputs(som, scaler, df_result, similar_df, metrics):
    joblib.dump(som, os.path.join(MODEL_DIR, "som_model.pkl"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, "som_scaler.pkl"))

    df_result.to_csv(
        os.path.join(V3_DIR, "som_koordinatlari.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    similar_df.to_csv(
        os.path.join(V3_DIR, "som_benzer_oyuncular.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    with open(os.path.join(V3_DIR, "som_metrikleri.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4, ensure_ascii=False)

    print("\nÇıktılar artifacts/v3 klasörüne kaydedildi.")


def main():
    print("Model V3 - SOM eğitimi başlatılıyor...")

    data_path = find_data_file()
    print(f"Kullanılan veri dosyası: {data_path}")

    df = pd.read_csv(data_path)
    print(f"Veri seti boyutu: {df.shape}")

    df_model, X_scaled, scaler, player_col = prepare_dataset(df)
    print(f"Model için kullanılan oyuncu sayısı: {len(df_model)}")
    print(f"Oyuncu adı sütunu: {player_col}")

    som = train_som(X_scaled)

    df_result = add_som_coordinates(df_model, X_scaled, som)

    metrics = calculate_metrics(X_scaled, df_result, som)

    print("\nModel V3 metrikleri:")
    for key, value in metrics.items():
        print(f"{key}: {value}")

    similar_df = calculate_similar_players(df_result, X_scaled, player_col)

    save_outputs(som, scaler, df_result, similar_df, metrics)

    print("\nModel V3 - SOM başarıyla tamamlandı.")


if __name__ == "__main__":
    main()