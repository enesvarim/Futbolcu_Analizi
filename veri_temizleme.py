
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os

DOSYA = "futbolcular.csv"

# veriyi oku
try:
    df = pd.read_csv(DOSYA)
except FileNotFoundError:
    print(f"'{DOSYA}' bulunamadi, dosya adini kontrol et.")
    exit()

print(f"Ham veri: {df.shape[0]} satir, {df.shape[1]} sutun")
rapor = [f"Ham veri: {df.shape[0]} satir, {df.shape[1]} sutun"]

# tekrar eden _stats_ sutunlarini at
tekrar_sutunlar = [col for col in df.columns if '_stats_' in col]
df.drop(columns=tekrar_sutunlar, inplace=True)
print(f"{len(tekrar_sutunlar)} tekrar sutun silindi")
rapor.append(f"Silinen tekrar sutun: {len(tekrar_sutunlar)}")

# kalecileri cikar
once = len(df)
df = df[~df['Pos'].str.contains('GK', na=False)]
sonra = len(df)
print(f"{once - sonra} kaleci cikarildi, {sonra} oyuncu kaldi")
rapor.append(f"Cikarilan kaleci: {once - sonra}")

# az oynayanlari cikar (90s < 5)
once = len(df)
df = df[df['90s'] >= 5]
sonra = len(df)
print(f"{once - sonra} az oynayan cikarildi, {sonra} oyuncu kaldi")
rapor.append(f"Az oynayan cikarilan: {once - sonra}")

# kullanilacak ozellikler
KIMLIK_SUTUNLARI = ['Player', 'Pos', 'Age']

OZELLIKLER = [
    'Gls', 'Ast', 'xG', 'xAG', 'Sh/90',       # hucum
    'PrgP', 'KP', 'PPA', 'SCA90',                # pas & yaraticilik
    'Tkl', 'TklW', 'Int', 'Clr',                  # savunma
    'Fld', 'Recov', 'Fls'                          # top tasima & pozisyon
]

# eksik sutun kontrolu
eksik = [col for col in OZELLIKLER if col not in df.columns]
if eksik:
    print(f"Bulunamayan sutunlar: {eksik}, listeden cikartiliyor")
    OZELLIKLER = [col for col in OZELLIKLER if col in df.columns]
    rapor.append(f"Bulunamayan sutunlar: {eksik}")

print(f"Ozellik sayisi: {len(OZELLIKLER)}")

df_kimlik = df[KIMLIK_SUTUNLARI].reset_index(drop=True)
df_model = df[OZELLIKLER].copy().reset_index(drop=True)

# sayisal tipe cevir
for col in OZELLIKLER:
    df_model[col] = pd.to_numeric(df_model[col], errors='coerce')

# eksik degerleri medyan ile doldur
eksik_sayisi = df_model.isnull().sum().sum()
print(f"Eksik hucre: {eksik_sayisi} (medyan ile dolduruluyor)")
rapor.append(f"Doldurulan eksik hucre: {eksik_sayisi}")

for col in OZELLIKLER:
    medyan = df_model[col].median()
    df_model[col].fillna(medyan, inplace=True)

# aykiri degerleri kirp (%1-%99 winsorize)
for col in OZELLIKLER:
    alt = df_model[col].quantile(0.01)
    ust = df_model[col].quantile(0.99)
    df_model[col] = df_model[col].clip(lower=alt, upper=ust)

# minmax normalizasyon (0-1)
scaler = MinMaxScaler()
df_normalize = pd.DataFrame(
    scaler.fit_transform(df_model),
    columns=OZELLIKLER
)

# kaydet
df_normalize.to_csv("temiz_veri3.csv", index=False)
df_kimlik.to_csv("oyuncu_bilgi3.csv", index=False)

rapor.append(f"Son veri: {df_normalize.shape[0]} oyuncu, {df_normalize.shape[1]} ozellik")
with open("temizleme_raporu3.txt", "w", encoding="utf-8") as f:
    f.write("FUTBOLCU VERI TEMIZLEME RAPORU\n")
    f.write("=" * 40 + "\n")
    for satir in rapor:
        f.write(satir + "\n")

# ozet
print(f"\nTemizleme bitti: {df_normalize.shape[0]} oyuncu, {df_normalize.shape[1]} ozellik")
print(f"Eksik deger kaldi mi: {df_normalize.isnull().sum().sum()}")
print("\nIlk 5 satir (normalize):")
print(df_normalize.head())
print("\nOyuncu bilgisi (ilk 5):")
print(df_kimlik.head())
