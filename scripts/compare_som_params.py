import os
import json
import numpy as np
import pandas as pd

from minisom import MiniSom
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import silhouette_score


# ==========================================================
# SOM HİPERPARAMETRE KARŞILAŞTIRMASI
# Amaç: 8x8, 10x10, 12x12, 15x15 SOM haritalarını karşılaştırmak
# ==========================================================


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(BASE_DIR, "veriseti", "futbolcular.csv")

OUTPUT_DIR = os.path.join(BASE_DIR, "artifacts", "v3")
os.makedirs(OUTPUT_DIR, exist_ok=True)


FEATURES = [
    "Gls", "Ast", "xG", "xAG", "npxG", "Sh/90",
    "Cmp%", "PrgP", "KP", "PPA", "SCA90",
    "Tkl", "TklW", "Int", "Clr",
    "PrgC", "PrgR", "Succ%", "Carries", "Touches",
]


def find_player_column(df):
    possible_columns = [
        "Player", "Oyuncu", "Name", "player", "player_name",
        "Oyuncu Adı", "oyuncu", "Futbolcu"
    ]

    for col in possible_columns:
        if col in df.columns:
            return col

    raise ValueError("Oyuncu adı sütunu bulunamadı.")


def prepare_dataset(df):
    player_col = find_player_column(df)

    # Kalecileri çıkarıyoruz
    if "Pos" in df.columns:
        df = df[~df["Pos"].astype(str).str.contains("GK", na=False)].copy()

    # Çok az süre almış oyuncuları çıkarıyoruz
    if "90s" in df.columns:
        df["90s"] = pd.to_numeric(df["90s"], errors="coerce")
        df = df[df["90s"] >= 5].copy()

    elif "Min" in df.columns:
        df["Min"] = pd.to_numeric(df["Min"], errors="coerce")
        df = df[df["Min"] >= 450].copy()

    missing_features = [feature for feature in FEATURES if feature not in df.columns]

    if missing_features:
        raise ValueError(f"Eksik özellikler var: {missing_features}")

    df_model = df[[player_col] + FEATURES].copy()

    for feature in FEATURES:
        df_model[feature] = pd.to_numeric(df_model[feature], errors="coerce")

    for feature in FEATURES:
        df_model[feature] = df_model[feature].fillna(df_model[feature].median())

    df_model = df_model.drop_duplicates(subset=[player_col]).reset_index(drop=True)

    X = df_model[FEATURES].values

    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)

    return df_model, X_scaled


def evaluate_som(X_scaled, grid_size, iterations=5000):
    som = MiniSom(
        x=grid_size,
        y=grid_size,
        input_len=X_scaled.shape[1],
        sigma=1.5,
        learning_rate=0.5,
        neighborhood_function="gaussian",
        random_seed=42
    )

    som.random_weights_init(X_scaled)
    som.train_random(X_scaled, num_iteration=iterations)

    coordinates = np.array([som.winner(x) for x in X_scaled])

    cluster_labels_text = [
        f"{coord[0]}-{coord[1]}" for coord in coordinates
    ]

    labels = pd.Series(cluster_labels_text).astype("category").cat.codes

    quantization_error = float(som.quantization_error(X_scaled))

    try:
        topographic_error = float(som.topographic_error(X_scaled))
    except Exception:
        topographic_error = None

    try:
        if len(set(labels)) > 1:
            silhouette = float(silhouette_score(X_scaled, labels))
        else:
            silhouette = None
    except Exception:
        silhouette = None

    cluster_counts = pd.Series(cluster_labels_text).value_counts()

    result = {
        "grid_size": f"{grid_size}x{grid_size}",
        "total_cells": grid_size * grid_size,
        "used_cells": int(cluster_counts.shape[0]),
        "min_players_per_cell": int(cluster_counts.min()),
        "max_players_per_cell": int(cluster_counts.max()),
        "mean_players_per_cell": float(cluster_counts.mean()),
        "std_players_per_cell": float(cluster_counts.std()),
        "quantization_error": quantization_error,
        "topographic_error": topographic_error,
        "silhouette_score": silhouette
    }

    return result


def main():
    print("SOM hiperparametre karşılaştırması başlatılıyor...")

    df = pd.read_csv(DATA_PATH)
    print(f"Veri seti boyutu: {df.shape}")

    df_model, X_scaled = prepare_dataset(df)
    print(f"Filtrelenmiş oyuncu sayısı: {len(df_model)}")

    grid_sizes = [8, 10, 12, 15]

    results = []

    for grid_size in grid_sizes:
        print(f"\n{grid_size}x{grid_size} SOM eğitiliyor...")
        result = evaluate_som(X_scaled, grid_size)
        results.append(result)

        print(f"{grid_size}x{grid_size} tamamlandı.")
        print(f"Quantization Error: {result['quantization_error']}")
        print(f"Topographic Error: {result['topographic_error']}")
        print(f"Silhouette Score: {result['silhouette_score']}")
        print(f"Kullanılan hücre: {result['used_cells']} / {result['total_cells']}")
        print(f"Hücre başına oyuncu min/max: {result['min_players_per_cell']} / {result['max_players_per_cell']}")

    results_df = pd.DataFrame(results)

    csv_path = os.path.join(OUTPUT_DIR, "som_parametre_karsilastirma.csv")
    json_path = os.path.join(OUTPUT_DIR, "som_parametre_karsilastirma.json")

    results_df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    print("\nKarşılaştırma tamamlandı.")
    print(f"CSV çıktı: {csv_path}")
    print(f"JSON çıktı: {json_path}")

    print("\nSonuç tablosu:")
    print(results_df)


if __name__ == "__main__":
    main()