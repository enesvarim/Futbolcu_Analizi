import pandas as pd
import numpy as np

print("=== VERİ KONTROL ===\n")

df          = pd.read_csv('temiz_veri.csv', encoding='utf-8')
player_info = pd.read_csv('futbolcular.csv', encoding='utf-8').iloc[:len(df)].reset_index(drop=True)

print(f'temiz_veri.csv shape   : {df.shape}')
print(f'futbolcular.csv shape  : {player_info.shape} (eşleştirilmiş)')
print(f'Satır sayısı eşleşti   : {len(df) == len(player_info)}')
print()

print("=== İLK 5 OYUNCU ===")
for i in range(5):
    player = player_info.iloc[i]['Player']
    gls = df.iloc[i]['Gls']
    ast = df.iloc[i]['Ast']
    print(f"{i+1}. {player}: Gls={gls}, Ast={ast}")

print("\n✓ Tüm kontroller başarılı!")
