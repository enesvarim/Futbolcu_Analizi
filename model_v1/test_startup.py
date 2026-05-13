import os
import sys
import logging
import json
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import numpy as np

# Çalışma dizinini ayarla
from pathlib import Path

VERISETI_DIR = Path(__file__).resolve().parents[1] / "veriseti"
os.chdir(VERISETI_DIR)

# Outputs klasörünü oluştur
RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = VERISETI_DIR.parent / f"cikti_{RUN_ID}"
LOG_DIR    = OUTPUT_DIR / "loglar"
PLOT_DIR   = OUTPUT_DIR / "grafikler"
MODEL_DIR  = OUTPUT_DIR / "modeller"

for d in [LOG_DIR, PLOT_DIR, MODEL_DIR]:
    d.mkdir(parents=True, exist_ok=True)

LOG_FILE = LOG_DIR / "sistem_logu.txt"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(sys.stdout)
    ]
)
log = logging.getLogger("FutbolAnalizSistemi")

log.info("=" * 65)
log.info("FUTBOLCU OYUN STİLİ ANALİZ SİSTEMİ BAŞLADI")
log.info(f"Çalışma ID : {RUN_ID}")
log.info(f"Çıktı klasörü: {OUTPUT_DIR}")
log.info("=" * 65)

# Veri yükle
log.info("Veriler yükleniyor...")
df          = pd.read_csv(VERISETI_DIR / "temiz_veri.csv", encoding="utf-8")
player_info = pd.read_csv(VERISETI_DIR / "futbolcular.csv", encoding="utf-8")
player_info = player_info.iloc[:len(df)].reset_index(drop=True)

log.info(f"temiz_veri.csv  → {df.shape[0]} satır, {df.shape[1]} sütun")
log.info(f"futbolcular.csv → {player_info.shape[0]} oyuncu (eşleştirilmiş)")

PLAYER_COL = "Player" if "Player" in player_info.columns else player_info.columns[0]
log.info(f"Oyuncu sütunu: {PLAYER_COL}")

# Feature seçimi
features = [
    'Gls', 'Ast', 'xG', 'xAG', 'npxG', 'Sh/90',
    'Cmp%', 'PrgP', 'KP', 'PPA', 'SCA90',
    'Tkl', 'TklW', 'Int', 'Clr',
    'PrgC', 'PrgR', 'Succ%', 'Carries', 'Touches'
]

X = df[features].copy()
log.info(f"Seçilen feature sayısı: {len(features)}")

log.info("\n✓ BAŞLANGIÇ BAŞARILANDI")
log.info(f"✓ Klasörler oluşturuldu: {OUTPUT_DIR}")
log.info(f"✓ Veri yüklendi: {X.shape}")
log.info(f"✓ Oyuncular eşleştirildi: {len(player_info)}")
log.info("=" * 65)

print(f"\n✓ SİSTEM BAŞARILI!\nÇıktılar: {OUTPUT_DIR}")
