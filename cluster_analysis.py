import pandas as pd
import numpy as np

# Verileri Yükle
df_features = pd.read_csv("veriseti/temiz_veri.csv")
df_players = pd.read_csv("veriseti/futbolcular.csv")
df_players = df_players[~df_players['Pos'].str.contains('GK', na=False)]
df_players = df_players[df_players['90s'] >= 5].reset_index(drop=True)

df_latent = pd.read_csv("cikti_v2_20260517_204823/latent_features.csv")

# Sütunları birleştir
df = pd.concat([df_players[['Player', 'Pos']], df_latent[['Cluster']], df_features], axis=1)

# Tüm veri seti ortalaması
global_mean = df_features.mean()

clusters = sorted(df['Cluster'].unique())

for c in clusters:
    subset = df[df['Cluster'] == c]
    print(f"\n{'='*50}")
    print(f"CLUSTER {c} | Toplam Oyuncu: {len(subset)}")
    
    # Küme Ortalaması
    cluster_mean = subset.drop(columns=['Player', 'Pos', 'Cluster']).mean()
    
    # Hangi özelliklerde global ortalamadan çok daha yüksekler? (Ratio)
    ratio = cluster_mean / (global_mean + 1e-6)
    top_features = ratio.sort_values(ascending=False).head(4)
    
    print("Öne Çıkan İstatistikler (Genel Ortalamaya Göre Fark):")
    for feat, val in top_features.items():
        print(f"  - {feat}: {val:.2f}x daha yüksek")
        
    print("Örnek 5 Oyuncu:")
    print("  " + ", ".join(subset['Player'].head(10).tolist()))
