import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import RobustScaler
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go

# ---------------------------------------------------------
# 1. AYARLAR VE SABİTLER
# ---------------------------------------------------------
st.set_page_config(page_title="Futbolcu Stil Analizi", page_icon="⚽", layout="wide")

st.markdown("""
<style>
    .main {background-color: #0e1117;}
    h1, h2, h3 {color: #00d2ff;}
    .stButton>button {
        background: linear-gradient(90deg, #00d2ff 0%, #3a7bd5 100%);
        color: white; border: none; border-radius: 5px;
    }
    .metric-card {
        background-color: #1e2530; padding: 20px; border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3); text-align: center;
        border-left: 5px solid #00d2ff; margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

ROOT_DIR = Path(__file__).resolve().parent
VERI_DIR = ROOT_DIR / "veriseti"

# Çıktı yolları (En güncel olanlar)
V1_DIR = ROOT_DIR / "cikti_4"
V2_DIR = ROOT_DIR / "cikti_v2_20260517_204823"

FEATURES = [
    'Gls', 'Ast', 'xG', 'xAG', 'npxG', 'Sh/90',
    'Cmp%', 'PrgP', 'KP', 'PPA', 'SCA90',
    'Tkl', 'TklW', 'Int', 'Clr',
    'PrgC', 'PrgR', 'Succ%', 'Carries', 'Touches'
]

# ---------------------------------------------------------
# 2. MODEL MİMARİLERİ (Ağırlıkları yüklemek için)
# ---------------------------------------------------------
class FootballAutoencoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int = 12):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256), nn.BatchNorm1d(256), nn.LeakyReLU(0.1), nn.Dropout(0.3),
            nn.Linear(256, 128),       nn.BatchNorm1d(128), nn.LeakyReLU(0.1), nn.Dropout(0.25),
            nn.Linear(128, 64),        nn.BatchNorm1d(64),  nn.LeakyReLU(0.1), nn.Dropout(0.2),
            nn.Linear(64, 32),         nn.BatchNorm1d(32),  nn.LeakyReLU(0.1),
            nn.Linear(32, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),  nn.LeakyReLU(0.1),
            nn.Linear(32, 64),          nn.BatchNorm1d(64),  nn.LeakyReLU(0.1),
            nn.Linear(64, 128),         nn.BatchNorm1d(128), nn.LeakyReLU(0.1),
            nn.Linear(128, 256),        nn.BatchNorm1d(256), nn.LeakyReLU(0.1),
            nn.Linear(256, input_dim)
        )
    def forward(self, x):
        latent = self.encoder(x)
        return latent, self.decoder(latent)
    def encode(self, x):
        return self.encoder(x)

class FootballVAE(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int = 16):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256), nn.BatchNorm1d(256), nn.LeakyReLU(0.1), nn.Dropout(0.3),
            nn.Linear(256, 128),       nn.BatchNorm1d(128), nn.LeakyReLU(0.1), nn.Dropout(0.25),
            nn.Linear(128, 64),        nn.BatchNorm1d(64),  nn.LeakyReLU(0.1)
        )
        self.fc_mu = nn.Linear(64, latent_dim)
        self.fc_log_var = nn.Linear(64, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),  nn.BatchNorm1d(64),  nn.LeakyReLU(0.1),
            nn.Linear(64, 128),         nn.BatchNorm1d(128), nn.LeakyReLU(0.1),
            nn.Linear(128, 256),        nn.BatchNorm1d(256), nn.LeakyReLU(0.1),
            nn.Linear(256, input_dim)
        )
    def reparameterize(self, mu, log_var):
        if self.training:
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu
    def forward(self, x):
        h       = self.encoder(x)
        mu      = self.fc_mu(h)
        log_var = self.fc_log_var(h)
        z       = self.reparameterize(mu, log_var)
        recon   = self.decoder(z)
        return recon, mu, log_var
    def encode(self, x):
        with torch.no_grad():
            h  = self.encoder(x)
            mu = self.fc_mu(h)
        return mu

# ---------------------------------------------------------
# 3. VERİ VE MODEL YÜKLEME (CACHE)
# ---------------------------------------------------------
@st.cache_resource
def load_system():
    # 1. Ham veriyi yükle ve Scaler'ı fit et
    df = pd.read_csv(VERI_DIR / "temiz_veri.csv")
    
    # Oyuncu isimlerini doğru hizalamak için aynı filtreleri uygula
    players_raw = pd.read_csv(VERI_DIR / "futbolcular.csv")
    players = players_raw[~players_raw['Pos'].str.contains('GK', na=False)]
    players = players[players['90s'] >= 5].reset_index(drop=True)
    
    X_raw = df[FEATURES].copy()
    X_raw = X_raw.replace([np.inf, -np.inf], np.nan).fillna(X_raw.median())
    X = X_raw.values
    scaler = RobustScaler()
    scaler.fit(X)
    
    # Ortalama değerleri doğru alabilmek için
    df_raw = X_raw

    # 2. V1 Modeli Yükle
    model_v1 = FootballAutoencoder(len(FEATURES), 12)
    model_v1.load_state_dict(torch.load(V1_DIR / "modeller" / "best_autoencoder.pth", map_location='cpu'))
    model_v1.eval()

    # 3. V2 Modeli Yükle
    model_v2 = FootballVAE(len(FEATURES), 16)
    model_v2.load_state_dict(torch.load(V2_DIR / "modeller" / "best_vae.pth", map_location='cpu'))
    model_v2.eval()

    # 4. Latent Çıktıları Yükle (Benzerlik hesabı için)
    v1_latent_df = pd.read_csv(V1_DIR / "latent_features.csv")
    v2_latent_df = pd.read_csv(V2_DIR / "latent_features.csv")
    
    # 5. UMAP Koordinatlarını Yükle
    v2_coords_df = pd.read_csv(V2_DIR / "tsne_umap_koordinatlari.csv")
    
    # Latent matrisleri numpy'a çevir
    v1_matrix = v1_latent_df[[c for c in v1_latent_df.columns if c.startswith('L')]].values
    v2_matrix = v2_latent_df[[c for c in v2_latent_df.columns if c.startswith('mu_')]].values

    return df, players, scaler, model_v1, model_v2, v1_latent_df, v2_latent_df, v1_matrix, v2_matrix, v2_coords_df

with st.spinner("Modeller ve yapay zeka ağı yükleniyor..."):
    try:
        df_raw, players, scaler, model_v1, model_v2, v1_latent_df, v2_latent_df, v1_matrix, v2_matrix, v2_coords_df = load_system()
        st.toast('Sistem Başarıyla Yüklendi!', icon='✅')
    except Exception as e:
        st.error(f"Sistem yüklenirken hata oluştu: {e}")
        st.stop()

# ---------------------------------------------------------
# 4. YARDIMCI FONKSİYONLAR VE SABİTLER
# ---------------------------------------------------------
CLUSTER_NAMES = {
    0: "Oyun Kurucular (Playmakers)",
    1: "Pasör Stoperler (Ball-Playing CBs)",
    2: "Dengeli/Klasik Savunmacılar",
    3: "Dinamik Kanatlar & 10 Numaralar",
    4: "İlerici Oyun Kurucular",
    5: "Fırsatçı / Pivot Forvetler",
    6: "Saf Bitiriciler (Pure Goalscorers)",
    7: "Yok Ediciler & Dinamolar"
}

def predict_and_find_similar(stats_array, model, latent_matrix, latent_df, version="v1"):
    scaled = scaler.transform(stats_array.reshape(1, -1))
    tensor = torch.FloatTensor(scaled)
    
    with torch.no_grad():
        latent_vec = model.encode(tensor).numpy()
    
    # Hibrit Benzerlik Hesabı (Cosine 0.6 + Euclidean 0.4)
    cos_sim = cosine_similarity(latent_vec, latent_matrix)[0]
    euc_dist = euclidean_distances(latent_vec, latent_matrix)[0]
    euc_sim = 1.0 / (1.0 + euc_dist)
    
    hybrid_score = 0.6 * cos_sim + 0.4 * euc_sim
    
    # En benzer 5 oyuncuyu bul
    top_indices = np.argsort(hybrid_score)[::-1][:5]
    
    # Tahmini küme (En çok benzeyen oyuncunun kümesi)
    predicted_cluster = int(latent_df.iloc[top_indices[0]]["Cluster"])
    
    results = []
    for idx in top_indices:
        c_id = int(latent_df.iloc[idx]["Cluster"])
        c_str = str(c_id)
        if version == "v2" and c_id in CLUSTER_NAMES:
            c_str = f"Küme {c_id} ({CLUSTER_NAMES[c_id]})"
            
        results.append({
            "Oyuncu": latent_df.iloc[idx]["Player"],
            "Oyun Stili (Küme)": c_str,
            "Benzerlik (%)": round(hybrid_score[idx] * 100, 1)
        })
        
    return predicted_cluster, pd.DataFrame(results), latent_vec[0]

# Özelliklerin Türkçe ve anlaşılır isimleri
FEATURE_LABELS = {
    'Gls': 'Gol',
    'Ast': 'Asist',
    'xG': 'Gol Beklentisi (xG)',
    'xAG': 'Asist Beklentisi (xAG)',
    'npxG': 'Penaltısız xG',
    'Sh/90': 'Maç Başı Şut',
    'Cmp%': 'Pas İsabet Oranı (%)',
    'PrgP': 'İleri Yönlü Pas',
    'KP': 'Kilit Pas',
    'PPA': 'Ceza Sahasına Pas',
    'SCA90': 'Şut Yaratma Aksiyonu',
    'Tkl': 'Top Çalma (Tackle)',
    'TklW': 'Kazanılan Top Çalma',
    'Int': 'Pas Arası (Intercept)',
    'Clr': 'Uzaklaştırma',
    'PrgC': 'İleri Top Taşıma',
    'PrgR': 'İleri Pas Alma',
    'Succ%': 'Başarılı Çalım (%)',
    'Carries': 'Topla Çıkış (Carry)',
    'Touches': 'Topla Buluşma'
}

# ---------------------------------------------------------
# 5. ARAYÜZ - SIDEBAR (GİRDİLER)
# ---------------------------------------------------------
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/1070/1070443.png", width=100)
st.sidebar.title("Oyuncu Parametreleri")

st.sidebar.markdown("Mevcut bir oyuncuyu seçerek özelliklerini kopyalayabilir veya kendiniz sıfırdan değer girebilirsiniz.")
selected_player = st.sidebar.selectbox("Hazır Şablon (Opsiyonel)", ["-- Manuel Giriş --"] + players['Player'].tolist())

input_data = {}
if selected_player != "-- Manuel Giriş --":
    idx = players[players['Player'] == selected_player].index[0]
    default_vals = df_raw.iloc[idx]
else:
    default_vals = df_raw.mean() # Ortalama bir oyuncu

for feat in FEATURES:
    label = f"{FEATURE_LABELS.get(feat, feat)} ({feat})"
    input_data[feat] = st.sidebar.number_input(label, value=float(default_vals[feat]), format="%.2f")

# ---------------------------------------------------------
# 6. ARAYÜZ - ANA EKRAN
# ---------------------------------------------------------
st.title("⚡ Yapay Zeka Destekli Futbolcu Stili Analizi")
st.markdown("Girilen istatistiklere göre oyuncunun hangi stile ait olduğunu **Model V1 (Autoencoder)** ve **Model V2 (Variational Autoencoder)** kullanarak analiz edin.")

if st.sidebar.button("🧠 Oyuncuyu Analiz Et", use_container_width=True):
    stats_array = np.array([input_data[f] for f in FEATURES])
    
    col1, col2 = st.columns(2)
    
    # MODEL V1 ANALİZİ
    c1, sim_v1, _ = predict_and_find_similar(stats_array, model_v1, v1_matrix, v1_latent_df, "v1")
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>🤖 Model V1 (Autoencoder)</h3>
            <p style="color:#aaa;">Tahmini Küme</p>
            <h1 style="font-size: 50px; margin:0; color:#00d2ff;">Cluster {c1}</h1>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("#### 🔍 V1 Uzayındaki En Benzer Oyuncular")
        st.dataframe(sim_v1, use_container_width=True, hide_index=True)
        
    # MODEL V2 ANALİZİ
    c2, sim_v2, _ = predict_and_find_similar(stats_array, model_v2, v2_matrix, v2_latent_df, "v2")
    c2_name = CLUSTER_NAMES.get(c2, f"Küme {c2}")
    
    with col2:
        st.markdown(f"""
        <div class="metric-card" style="border-left-color: #ff007f;">
            <h3>🧠 Model V2 (VAE + GMM)</h3>
            <p style="color:#aaa;">Tahmini Oyun Stili</p>
            <h1 style="font-size: 32px; margin:0; color:#ff007f;">{c2_name}</h1>
            <p style="color:#ff007f; margin-top:5px; font-weight:bold;">(Cluster {c2})</p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("#### 🎯 V2 Uzayındaki En Benzer Oyuncular")
        st.dataframe(sim_v2, use_container_width=True, hide_index=True)
        
    st.divider()
    
    # RADAR GRAFİĞİ (GİRDİLERİ GÖRSELLEŞTİR)
    st.markdown("### 📊 Oyuncu Profil Radarı (Girilen Değerler)")
    
    # Girdileri 0-1 arasına normalize et (Görsellik için, max değere bölerek)
    max_vals = df_raw[FEATURES].max()
    norm_inputs = stats_array / max_vals.values
    
    fig = go.Figure(data=go.Scatterpolar(
      r=norm_inputs,
      theta=FEATURES,
      fill='toself',
      line_color='#00d2ff',
      fillcolor='rgba(0, 210, 255, 0.4)'
    ))
    fig.update_layout(
      polar=dict(radialaxis=dict(visible=False, range=[0, 1])),
      showlegend=False,
      paper_bgcolor='rgba(0,0,0,0)',
      plot_bgcolor='rgba(0,0,0,0)',
      font_color="white",
      height=500
    )
    st.plotly_chart(fig, use_container_width=True)
    
    st.divider()
    
    # KÜMELEME HARİTASI (UMAP) ÜZERİNDE GÖSTERİM
    st.markdown("### 🗺️ Oyuncunun V2 Kümeleme Haritasındaki Yeri (UMAP)")
    st.markdown("Aşağıdaki haritada, girdiğiniz özelliklere **en çok benzeyen 5 oyuncunun** (⭐) konumunu ve **kendi oyuncunuzun** (❌) konumunu görüyorsunuz.")
    
    # V2 Koordinat verisini hazırla
    plot_df = v2_coords_df.copy()
    plot_df['Oyun Stili'] = plot_df['Cluster'].apply(lambda c: f"{int(c)} - {CLUSTER_NAMES.get(int(c), 'Bilinmeyen')}")
    
    # Benzer oyuncuların isimlerini al
    similar_players = sim_v2["Oyuncu"].tolist()
    
    # Scatter plot oluştur
    fig_umap = px.scatter(
        plot_df, x="UMAP_1", y="UMAP_2", color="Oyun Stili", hover_name="Player",
        color_discrete_sequence=px.colors.qualitative.Set1,
        title="V2 UMAP Gizli Uzayı",
        opacity=0.6,
    )
    
    # Benzer oyuncuları yıldızla işaretle
    similar_df = plot_df[plot_df['Player'].isin(similar_players)]
    
    fig_umap.add_trace(go.Scatter(
        x=similar_df['UMAP_1'], y=similar_df['UMAP_2'],
        mode='markers+text',
        marker=dict(symbol='star', size=15, color='yellow', line=dict(width=2, color='black')),
        text=similar_df['Player'],
        textposition="top center",
        name="Benzer Oyuncular",
        hoverinfo="text"
    ))
    
    # Kendi oyuncumuzun konumunu bul
    target_umap_x, target_umap_y = None, None
    target_name = "Sizin Oyuncunuz"
    
    if selected_player != "-- Manuel Giriş --":
        target_name = selected_player
        row = plot_df[plot_df['Player'] == selected_player]
        if not row.empty:
            target_umap_x = row.iloc[0]['UMAP_1']
            target_umap_y = row.iloc[0]['UMAP_2']
            
    # Eğer manuel girişse, benzer 5 oyuncunun ortalama konumunu (yaklaşık) al
    if target_umap_x is None or target_umap_y is None:
        target_umap_x = similar_df['UMAP_1'].mean()
        target_umap_y = similar_df['UMAP_2'].mean()
        target_name = "Hedef Oyuncu (Tahmini Konum)"
        
    # Kendi oyuncumuzu haritaya dev bir kırmızı çarpı ile ekle
    fig_umap.add_trace(go.Scatter(
        x=[target_umap_x], y=[target_umap_y],
        mode='markers+text',
        marker=dict(symbol='x', size=20, color='white', line=dict(width=4, color='red')),
        text=[f"<b>{target_name}</b>"],
        textposition="bottom center",
        name="Sizin Oyuncunuz",
        hoverinfo="text"
    ))
    
    fig_umap.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color="white",
        height=600
    )
    st.plotly_chart(fig_umap, use_container_width=True)

else:
    st.info("👈 Analizi başlatmak için sol panelden istatistikleri belirleyip 'Oyuncuyu Analiz Et' butonuna tıklayın.")
