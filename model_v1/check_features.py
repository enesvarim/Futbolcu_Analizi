import pandas as pd

# Veri yükle
clean_df = pd.read_csv('temiz_veri.csv', encoding='utf-8')
player_df = pd.read_csv('futbolcular.csv', encoding='utf-8')

print("=== TEMİZ VERİ (temiz_veri.csv) ===")
print(f"Şekil: {clean_df.shape}")
print(f"Sütunlar: {list(clean_df.columns)}\n")

print("=== FUTBOLCULAR (futbolcular.csv) ===")
print(f"Şekil: {player_df.shape}")
print(f"İlk 30 sütun: {list(player_df.columns[:30])}\n")

# Model beklediği feature'ları kontrol et
features = [
    'Gls', 'Ast', 'xG', 'xAG', 'npxG', 'Sh/90',
    'Cmp%', 'PrgP', 'KP', 'PPA', 'SCA90',
    'Tkl', 'TklW', 'Int', 'Clr',
    'PrgC', 'PrgR', 'Succ%', 'Carries', 'Touches'
]

print("=== MODEL İÇİN GEREKL İ FEATURE'LAR ===")
missing = []
for feat in features:
    in_clean = feat in clean_df.columns
    in_player = feat in player_df.columns
    status = "✓" if (in_clean or in_player) else "✗ EKSIK"
    print(f"  {feat}: {status}")
    if not (in_clean or in_player):
        missing.append(feat)

if missing:
    print(f"\nEKSİK FEATURE'LAR: {missing}")

# Hangi CSV'nin doğru olduğunu belirle
print("\n=== ÇÖZÜM ===")
if set(features).issubset(clean_df.columns):
    print("✓ temiz_veri.csv tüm feature'ları içeriyor - KENDİ BAŞINA YETERLI")
elif set(features).issubset(player_df.columns):
    print("✗ futbolcular.csv UYUMSUZ - temiz_veri.csv kullanılmalı")
else:
    print("⚠ Her iki dosya da eksik feature içeriyor")
