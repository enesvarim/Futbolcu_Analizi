"""
src/config.py
=============
Projenin TEK konfigürasyon noktası.
Tüm sabitler, yollar, model parametreleri ve UI tanımları buradan import edilir.
"""
from pathlib import Path

# ---------------------------------------------------------------------------
# Proje Kök Dizini
# ---------------------------------------------------------------------------
ROOT_DIR     = Path(__file__).resolve().parents[1]
VERI_DIR     = ROOT_DIR / "veriseti"
ARTIFACT_DIR = ROOT_DIR / "artifacts"

# Sabit artifact yolları (hardcode klasör adı yok)
V1_DIR   = ARTIFACT_DIR / "v1"
V2_DIR   = ARTIFACT_DIR / "v2"
V1_MODEL = V1_DIR / "modeller" / "best_autoencoder.pth"
V2_MODEL = V2_DIR / "modeller" / "best_vae.pth"

# ---------------------------------------------------------------------------
# Feature Tanımları  (projedeki TEK tanım — diğer dosyalar buradan import eder)
# ---------------------------------------------------------------------------
FEATURES: list[str] = [
    # Hücum
    "Gls", "Ast", "xG", "xAG", "npxG", "Sh/90",
    # Pas & Oyun Kurma
    "Cmp%", "PrgP", "KP", "PPA", "SCA90",
    # Defans
    "Tkl", "TklW", "Int", "Clr",
    # Top Taşıma & Hareket
    "PrgC", "PrgR", "Succ%", "Carries", "Touches",
]

FEATURE_GROUPS: dict[str, list[str]] = {
    "Hücum":            ["Gls", "Ast", "xG", "xAG", "npxG", "Sh/90"],
    "Pas & Oyun Kurma": ["Cmp%", "PrgP", "KP", "PPA", "SCA90"],
    "Defans":           ["Tkl", "TklW", "Int", "Clr"],
    "Top Taşıma":       ["PrgC", "PrgR", "Succ%", "Carries", "Touches"],
}

FEATURE_LABELS: dict[str, str] = {
    "Gls":     "Gol",
    "Ast":     "Asist",
    "xG":      "Gol Beklentisi (xG)",
    "xAG":     "Asist Beklentisi (xAG)",
    "npxG":    "Penaltısız xG",
    "Sh/90":   "Maç Başı Şut",
    "Cmp%":    "Pas İsabet Oranı (%)",
    "PrgP":    "İleri Yönlü Pas",
    "KP":      "Kilit Pas",
    "PPA":     "Ceza Sahasına Pas",
    "SCA90":   "Şut Yaratma Aksiyonu",
    "Tkl":     "Top Çalma (Tackle)",
    "TklW":    "Kazanılan Top Çalma",
    "Int":     "Pas Arası (Intercept)",
    "Clr":     "Uzaklaştırma",
    "PrgC":    "İleri Top Taşıma",
    "PrgR":    "İleri Pas Alma",
    "Succ%":   "Başarılı Çalım (%)",
    "Carries": "Topla Çıkış (Carry)",
    "Touches": "Topla Buluşma",
}

# ---------------------------------------------------------------------------
# Model Parametreleri
# ---------------------------------------------------------------------------
AE_LATENT_DIM  = 12   # Autoencoder latent boyutu
VAE_LATENT_DIM = 16   # VAE latent boyutu

TRAINING = {
    "epochs":       200,
    "patience":     25,
    "batch_size":   64,
    "lr":           3e-4,
    "weight_decay": 1e-4,
    "test_size":    0.15,
    "val_size":     0.1765,   # train_temp'in yüzdesi → toplam %15 val
}

VAE_TRAINING = {
    **{k: v for k, v in TRAINING.items()},
    "beta_start":  0.0,
    "beta_end":    1.0,
    "beta_warmup": 50,   # beta annealing epoch sayısı
}

# ---------------------------------------------------------------------------
# Kümeleme Parametreleri
# ---------------------------------------------------------------------------
CLUSTER_K_RANGE = range(5, 11)

# ---------------------------------------------------------------------------
# Benzerlik Ağırlıkları
# ---------------------------------------------------------------------------
COSINE_WEIGHT    = 0.6
EUCLIDEAN_WEIGHT = 0.4

# ---------------------------------------------------------------------------
# Küme İsimleri & Renkleri
# ---------------------------------------------------------------------------
CLUSTER_NAMES_V1: dict[int, str] = {
    0: "Savunma Sigortaları & Pasör Stoperler",
    1: "Yaratıcı & Skorer Hücumcular",
    2: "Dinamolar & İki Yönlü Orta Sahalar",
    3: "Direkt & Fırsatçı Hücumcular",
    4: "Çalışkan & Yardımcı Rol Oyuncuları",
}

CLUSTER_NAMES_V2: dict[int, str] = {
    0: "Oyun Kurucular (Playmakers)",
    1: "Pasör Stoperler (Ball-Playing CBs)",
    2: "Dengeli/Klasik Savunmacılar",
    3: "Dinamik Kanatlar & 10 Numaralar",
    4: "İlerici Oyun Kurucular",
    5: "Fırsatçı / Pivot Forvetler",
    6: "Saf Bitiriciler (Pure Goalscorers)",
    7: "Yok Ediciler & Dinamolar",
}

# Geriye dönük uyumluluk için CLUSTER_NAMES, CLUSTER_NAMES_V2'ye eşittir
CLUSTER_NAMES = CLUSTER_NAMES_V2

CLUSTER_COLORS: dict[int, str] = {
    0:  "#E41A1C",  # Kırmızı
    1:  "#377EB8",  # Mavi
    2:  "#4DAF4A",  # Yeşil
    3:  "#984EA3",  # Mor
    4:  "#FF7F00",  # Turuncu
    5:  "#F781BF",  # Pembe
    6:  "#A65628",  # Kahverengi
    7:  "#17BECF",  # Turkuaz
    8:  "#999999",
    9:  "#66C2A5",
    10: "#FC8D62",
    11: "#8DA0CB",
}

# ---------------------------------------------------------------------------
# Veri Filtreleri
# ---------------------------------------------------------------------------
MIN_90S = 5          # Minimum 90 dakikalık maç sayısı
EXCLUDE_POS = "GK"   # Filtre: kalecileri çıkar
