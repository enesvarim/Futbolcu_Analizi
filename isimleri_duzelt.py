import pandas as pd
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
VERI_DIR = ROOT_DIR / "veriseti"
V1_DIR = ROOT_DIR / "cikti_4"
V2_DIR = ROOT_DIR / "cikti_v2_20260517_204823"

# 1. Doğru isim listesini çıkar
df_raw = pd.read_csv(VERI_DIR / "futbolcular.csv")
df_filtered = df_raw[~df_raw['Pos'].str.contains('GK', na=False)]
df_filtered = df_filtered[df_filtered['90s'] >= 5].reset_index(drop=True)
correct_names = df_filtered['Player']
print(f"Doğru isim sayısı: {len(correct_names)}")

# 2. V1 Latent Features Düzeltme
v1_lf_path = V1_DIR / "latent_features.csv"
if v1_lf_path.exists():
    df_v1 = pd.read_csv(v1_lf_path)
    if len(df_v1) == len(correct_names):
        df_v1['Player'] = correct_names
        df_v1.to_csv(v1_lf_path, index=False)
        print("V1 latent_features.csv düzeltildi.")
    else:
        print(f"V1 boyutu uyuşmuyor: {len(df_v1)} vs {len(correct_names)}")

# 3. V2 Latent Features Düzeltme
v2_lf_path = V2_DIR / "latent_features.csv"
if v2_lf_path.exists():
    df_v2 = pd.read_csv(v2_lf_path)
    if len(df_v2) == len(correct_names):
        df_v2['Player'] = correct_names
        df_v2.to_csv(v2_lf_path, index=False)
        print("V2 latent_features.csv düzeltildi.")
    else:
        print(f"V2 boyutu uyuşmuyor: {len(df_v2)} vs {len(correct_names)}")

# 4. V2 UMAP Koordinatları Düzeltme
v2_coords_path = V2_DIR / "tsne_umap_koordinatlari.csv"
if v2_coords_path.exists():
    df_coords = pd.read_csv(v2_coords_path)
    if len(df_coords) == len(correct_names):
        df_coords['Player'] = correct_names
        df_coords.to_csv(v2_coords_path, index=False)
        print("V2 tsne_umap_koordinatlari.csv düzeltildi.")
    else:
        print(f"V2 coords boyutu uyuşmuyor: {len(df_coords)} vs {len(correct_names)}")

print("Tüm işlemler tamamlandı!")
