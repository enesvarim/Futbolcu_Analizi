"""
app.py
=======
Futbolcu Oyun Stili Analizi — Streamlit Uygulaması

Başlatma:
    streamlit run app.py
"""
import numpy as np
import pandas as pd
import streamlit as st
import torch

from src.config import (
    FEATURES, FEATURE_LABELS, CLUSTER_NAMES, CLUSTER_NAMES_V1,
    V1_DIR, V2_DIR, V1_MODEL, V2_MODEL,
    AE_LATENT_DIM, VAE_LATENT_DIM,
)
from src.data.loader import load_raw_players, load_clean_features
from src.data.preprocessor import clean_features, fit_scaler
from src.models.autoencoder import load_autoencoder
from src.models.vae import load_vae
from src.analysis.similarity import predict_and_find_similar, compare_players
from src.analysis.visualization import (
    plot_umap, plot_radar, plot_comparison_radar, normalize_for_radar,
)
from src.app.components import (
    apply_global_styles, render_cluster_card, render_similarity_card,
)

# ---------------------------------------------------------------------------
# Sayfa Konfigürasyonu
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Futbolcu Stil Analizi",
    page_icon="⚽",
    layout="wide",
)
apply_global_styles()

# ---------------------------------------------------------------------------
# Sistem Yükleme (Cache)
# ---------------------------------------------------------------------------
@st.cache_resource
def load_system():
    """Modelleri, scaler'ı ve tüm artifact CSV'lerini yükler."""
    # Veri
    players    = load_raw_players()
    raw_df     = load_clean_features()
    X_clean    = clean_features(raw_df, FEATURES)
    _, scaler  = fit_scaler(X_clean)

    # Modeller
    model_v1 = load_autoencoder(V1_MODEL, latent_dim=AE_LATENT_DIM)
    model_v2 = load_vae(V2_MODEL,         latent_dim=VAE_LATENT_DIM)

    # Artifact CSV'leri
    v1_latent_df  = pd.read_csv(V1_DIR / "latent_features.csv")
    v2_latent_df  = pd.read_csv(V2_DIR / "latent_features.csv")
    v1_coords_df  = pd.read_csv(V1_DIR / "tsne_umap_koordinatlari.csv")
    v2_coords_df  = pd.read_csv(V2_DIR / "tsne_umap_koordinatlari.csv")

    # Latent matrisleri
    v1_matrix = v1_latent_df[[c for c in v1_latent_df.columns if c.startswith("L")]].values
    v2_matrix = v2_latent_df[[c for c in v2_latent_df.columns if c.startswith("mu_")]].values

    return (
        players, X_clean, scaler,
        model_v1, model_v2,
        v1_latent_df, v2_latent_df,
        v1_matrix, v2_matrix,
        v1_coords_df, v2_coords_df,
    )


with st.spinner("Modeller ve yapay zeka ağı yükleniyor..."):
    try:
        (
            players, df_raw, scaler,
            model_v1, model_v2,
            v1_latent_df, v2_latent_df,
            v1_matrix, v2_matrix,
            v1_coords_df, v2_coords_df,
        ) = load_system()
        st.toast("Sistem Başarıyla Yüklendi!", icon="✅")
    except Exception as e:
        st.error(f"Sistem yüklenirken hata oluştu: {e}")
        st.stop()

# ---------------------------------------------------------------------------
# Sidebar — Oyuncu Parametreleri
# ---------------------------------------------------------------------------
st.sidebar.title("Oyuncu Parametreleri")
st.sidebar.markdown(
    "Mevcut bir oyuncuyu seçerek özelliklerini kopyalayabilir "
    "veya kendiniz sıfırdan değer girebilirsiniz."
)

selected_player = st.sidebar.selectbox(
    "Hazır Şablon (Opsiyonel)",
    ["-- Manuel Giriş --"] + players["Player"].tolist(),
)

if selected_player != "-- Manuel Giriş --":
    idx          = players[players["Player"] == selected_player].index[0]
    default_vals = df_raw.iloc[idx]
else:
    default_vals = df_raw.mean()

input_data: dict[str, float] = {}
for feat in FEATURES:
    label            = f"{FEATURE_LABELS.get(feat, feat)} ({feat})"
    input_data[feat] = st.sidebar.number_input(
        label, value=float(default_vals[feat]), format="%.2f"
    )

# ---------------------------------------------------------------------------
# Ana Ekran — Sekmeler
# ---------------------------------------------------------------------------
st.title("⚽ Futbolcu Oyun Stili Analizi")
st.markdown(
    "Seçilen istatistiklere göre oyuncunun hangi stile ait olduğunu "
    "**Temel Model (V1)** ve **Gelişmiş Model (V2)** aracılığıyla inceleyebilirsiniz."
)

tab1, tab2 = st.tabs(["Tekli Oyuncu Analizi", "Oyuncu Karşılaştırma"])

# ─────────────────────────────────────────────
# TAB 1: Tekli Oyuncu Analizi
# ─────────────────────────────────────────────
with tab1:
    if st.sidebar.button("Oyuncuyu Analiz Et", use_container_width=True):
        stats = np.array([input_data[f] for f in FEATURES])

        col1, col2 = st.columns(2)

        # Model V1
        c1, sim_v1, _ = predict_and_find_similar(
            stats, model_v1, scaler, v1_matrix, v1_latent_df, version="v1"
        )
        with col1:
            render_cluster_card("Model V1 (Autoencoder)", c1, version="v1")
            st.markdown("#### V1 Uzayındaki Benzer Oyuncular")
            st.dataframe(sim_v1, use_container_width=True, hide_index=True)

        # Model V2
        c2, sim_v2, _ = predict_and_find_similar(
            stats, model_v2, scaler, v2_matrix, v2_latent_df, version="v2"
        )
        with col2:
            render_cluster_card("Model V2 (VAE + GMM)", c2, version="v2")
            st.markdown("#### V2 Uzayındaki Benzer Oyuncular")
            st.dataframe(sim_v2, use_container_width=True, hide_index=True)

        st.divider()

        # Radar Grafiği
        st.markdown("### 📊 Oyuncu Profil Radarı (Girilen Değerler)")
        max_vals    = df_raw[FEATURES].max().values
        norm_inputs = normalize_for_radar(stats, max_vals)
        st.plotly_chart(plot_radar(norm_inputs), use_container_width=True)

        st.divider()

        # UMAP Haritaları
        st.markdown("### 🗺️ Oyuncunun Kümeleme Haritalarındaki Konumu (UMAP)")
        st.markdown(
            "Haritada hedef oyuncu **✕** ile, en çok benzeyen 5 oyuncu **⭐** ile gösterilir."
        )
        st.plotly_chart(
            plot_umap(v1_coords_df, sim_v1, selected_player, "Temel UMAP (V1)", CLUSTER_NAMES_V1),
            use_container_width=True,
        )
        st.markdown("<br>", unsafe_allow_html=True)
        st.plotly_chart(
            plot_umap(v2_coords_df, sim_v2, selected_player,
                      "Gelişmiş UMAP (V2)", CLUSTER_NAMES),
            use_container_width=True,
        )
    else:
        st.info(
            "Analizi başlatmak için sol panelden istatistikleri belirleyip "
            "**'Oyuncuyu Analiz Et'** butonuna tıklayın."
        )

# ─────────────────────────────────────────────
# TAB 2: Oyuncu Karşılaştırma
# ─────────────────────────────────────────────
with tab2:
    st.markdown("### ⚖️ Oyuncu Karşılaştırma")
    st.markdown(
        "Veri setindeki iki oyuncuyu seçerek yapay zeka uzayındaki "
        "benzerliklerini ve istatistiklerini kıyaslayın."
    )

    col_a, col_b = st.columns(2)
    with col_a:
        player_a = st.selectbox("1. Oyuncu", players["Player"].tolist(), index=0)
    with col_b:
        player_b = st.selectbox("2. Oyuncu", players["Player"].tolist(), index=1)

    if st.button("Oyuncuları Karşılaştır", type="primary", use_container_width=True):
        idx_a = players[players["Player"] == player_a].index[0]
        idx_b = players[players["Player"] == player_b].index[0]

        stats_a = np.array([df_raw.iloc[idx_a][f] for f in FEATURES])
        stats_b = np.array([df_raw.iloc[idx_b][f] for f in FEATURES])

        # V1 kümeleri
        c1_a, _, _ = predict_and_find_similar(
            stats_a, model_v1, scaler, v1_matrix, v1_latent_df, version="v1"
        )
        c1_b, _, _ = predict_and_find_similar(
            stats_b, model_v1, scaler, v1_matrix, v1_latent_df, version="v1"
        )

        # V2 kümeleri + benzerlik
        c2_a, _, _ = predict_and_find_similar(
            stats_a, model_v2, scaler, v2_matrix, v2_latent_df, version="v2"
        )
        c2_b, _, _ = predict_and_find_similar(
            stats_b, model_v2, scaler, v2_matrix, v2_latent_df, version="v2"
        )

        result = compare_players(stats_a, stats_b, model_v2, scaler, v2_matrix, v2_latent_df)
        render_similarity_card(result["similarity_pct"])

        c_col1, c_col2 = st.columns(2)
        with c_col1:
            st.markdown(f"#### 🟦 {player_a}")
            st.write(f"**V1 Stili:** {CLUSTER_NAMES_V1.get(c1_a, f'Küme {c1_a}')}")
            st.write(f"**V2 Stili:** {CLUSTER_NAMES.get(c2_a, f'Küme {c2_a}')}")
        with c_col2:
            st.markdown(f"#### 🟥 {player_b}")
            st.write(f"**V1 Stili:** {CLUSTER_NAMES_V1.get(c1_b, f'Küme {c1_b}')}")
            st.write(f"**V2 Stili:** {CLUSTER_NAMES.get(c2_b, f'Küme {c2_b}')}")

        st.divider()

        # Karşılaştırma Radarı
        max_vals = df_raw[FEATURES].max().values
        norm_a   = normalize_for_radar(stats_a, max_vals)
        norm_b   = normalize_for_radar(stats_b, max_vals)
        st.plotly_chart(
            plot_comparison_radar(norm_a, norm_b, player_a, player_b),
            use_container_width=True,
        )
