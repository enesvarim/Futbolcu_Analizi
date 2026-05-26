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
from pathlib import Path

from src.config import (
    FEATURES, FEATURE_LABELS, CLUSTER_NAMES, CLUSTER_NAMES_V1,
    V1_DIR, V2_DIR, V1_MODEL, V2_MODEL,
    V3_SOM_COORDS, V3_SOM_SIMILAR,
    AE_LATENT_DIM, VAE_LATENT_DIM, VERI_DIR, EXCLUDE_POS, MIN_90S
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

    # Model V3 - SOM artifact dosyaları
    v3_som_coords_df = pd.read_csv(V3_SOM_COORDS)
    v3_som_similar_df = pd.read_csv(V3_SOM_SIMILAR)

    # Latent matrisleri
    v1_matrix = v1_latent_df[[c for c in v1_latent_df.columns if c.startswith("L")]].values
    v2_matrix = v2_latent_df[[c for c in v2_latent_df.columns if c.startswith("mu_")]].values

    # Süper Lig 3 Büyükler (2024-25) verisini yükle ve winsorize min-max ölçekle (0-1)
    try:
        superlig_path = VERI_DIR / "superlig_3buyukler_2024-25.csv"
        if superlig_path.exists():
            superlig_df = pd.read_csv(superlig_path, encoding="utf-8")
            # Kalecileri filtrele
            superlig_df = superlig_df[~superlig_df["Pos"].str.contains(EXCLUDE_POS, na=False)].reset_index(drop=True)

            # Ham futbolcular veri setinin min-max sınırlarını bulalım (aynı filtreler ile)
            raw_players = pd.read_csv(VERI_DIR / "futbolcular.csv", encoding="utf-8")
            raw_players = raw_players[~raw_players["Pos"].str.contains(EXCLUDE_POS, na=False)]
            raw_players = raw_players[raw_players["90s"] >= MIN_90S].reset_index(drop=True)

            # Her bir özellik için winsorize ve min-max sınırlarını uygulayalım
            for feat in FEATURES:
                if feat in raw_players.columns:
                    # Ham serinin winsorize sınırlarını (1% - 99%) çıkar
                    feat_series = pd.to_numeric(raw_players[feat], errors='coerce').fillna(raw_players[feat].median())
                    alt = feat_series.quantile(0.01)
                    ust = feat_series.quantile(0.99)
                    
                    # Süper Lig oyuncu serisini temizle ve winsorize kırpması yap
                    sl_series = pd.to_numeric(superlig_df[feat], errors='coerce').fillna(feat_series.median())
                    sl_series = sl_series.clip(lower=alt, upper=ust)
                    
                    # Min-Max Ölçekleme (0 ile 1 arasına çekme)
                    payda = ust - alt if ust != alt else 1.0
                    superlig_df[feat] = (sl_series - alt) / payda
        else:
            superlig_df = pd.DataFrame()
    except Exception as e:
        st.warning(f"Süper Lig verisi yüklenirken hata oluştu: {e}")
        superlig_df = pd.DataFrame()

    return (
        players, X_clean, scaler,
        model_v1, model_v2,
        v1_latent_df, v2_latent_df,
        v1_matrix, v2_matrix,
        v1_coords_df, v2_coords_df,
        v3_som_coords_df, v3_som_similar_df,
        superlig_df,
    )


with st.spinner("Modeller ve yapay zeka ağı yükleniyor..."):
    try:
        (
            players, df_raw, scaler,
            model_v1, model_v2,
            v1_latent_df, v2_latent_df,
            v1_matrix, v2_matrix,
            v1_coords_df, v2_coords_df,
            v3_som_coords_df, v3_som_similar_df,
            superlig_df,
        ) = load_system()
        st.toast("Sistem Başarıyla Yüklendi!", icon="✅")
    except Exception as e:
        st.error(f"Sistem yüklenirken hata oluştu: {e}")
        st.stop()


def predict_som_result(
    stats_array: np.ndarray,
    som_coords_df: pd.DataFrame,
    som_similar_df: pd.DataFrame,
):
    """
    Girilen oyuncu istatistiklerine en yakın SOM oyuncusunu bulur.
    Bu oyuncunun SOM bölgesini ve SOM tabanlı benzer oyuncularını döndürür.
    """
    feature_matrix = som_coords_df[FEATURES].values.astype(float)
    query = stats_array.reshape(1, -1)

    distances = np.linalg.norm(feature_matrix - query, axis=1)
    nearest_idx = int(np.argmin(distances))

    nearest_player = som_coords_df.iloc[nearest_idx]["Player"]
    som_cluster = som_coords_df.iloc[nearest_idx]["som_cluster"]

    similar_rows = som_similar_df[
        som_similar_df["oyuncu"] == nearest_player
    ].copy()

    similar_display = similar_rows[
        ["benzer_oyuncu", "benzerlik_yuzde", "benzer_oyuncu_som_cluster"]
    ].rename(columns={
        "benzer_oyuncu": "Oyuncu",
        "benzerlik_yuzde": "Benzerlik (%)",
        "benzer_oyuncu_som_cluster": "SOM Bölgesi",
    })

    return nearest_player, som_cluster, similar_display

# ---------------------------------------------------------------------------
# Sidebar — Oyuncu Parametreleri
# ---------------------------------------------------------------------------
st.sidebar.title("Oyuncu Parametreleri")
st.sidebar.markdown(
    "Mevcut bir oyuncuyu seçerek özelliklerini kopyalayabilir "
    "veya kendiniz sıfırdan değer girebilirsiniz."
)

# Süper Lig oyuncularının isim listesi
superlig_names = superlig_df["Player"].tolist() if not superlig_df.empty else []
original_names = players["Player"].tolist()

# Seçenekleri birleştirelim
selected_player = st.sidebar.selectbox(
    "Hazır Şablon (Opsiyonel)",
    ["-- Manuel Giriş --"] + [f"⭐ {name}" for name in superlig_names] + original_names,
)

if selected_player != "-- Manuel Giriş --":
    if selected_player.startswith("⭐ "):
        # Süper Lig oyuncusu
        real_name = selected_player[2:]
        idx = superlig_df[superlig_df["Player"] == real_name].index[0]
        default_vals = superlig_df.iloc[idx]
    else:
        # Normal oyuncu
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
    "**Temel Model (V1)**, **Gelişmiş Model (V2)** ve "
    "**SOM Modeli (V3)** aracılığıyla inceleyebilirsiniz."
)

tab1, tab2 = st.tabs(["Tekli Oyuncu Analizi", "Oyuncu Karşılaştırma"])

# ─────────────────────────────────────────────
# TAB 1: Tekli Oyuncu Analizi
# ─────────────────────────────────────────────
with tab1:
    if st.sidebar.button("Oyuncuyu Analiz Et", use_container_width=True):
        stats = np.array([input_data[f] for f in FEATURES])

        col1, col2, col3 = st.columns(3)

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

        # Model V3 - SOM
        nearest_player_v3, som_cluster_v3, sim_v3 = predict_som_result(
            stats,
            v3_som_coords_df,
            v3_som_similar_df,
        )

        with col3:
            st.markdown("### 🧭 Model V3 (SOM)")
            st.info(f"**SOM Bölgesi:** {som_cluster_v3}")
            st.caption(f"En yakın veri seti oyuncusu: {nearest_player_v3}")
            st.markdown("#### SOM Haritasındaki Benzer Oyuncular")
            st.dataframe(sim_v3, use_container_width=True, hide_index=True)    

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
        
        display_target_name = selected_player.replace("⭐ ", "")
        
        st.plotly_chart(
            plot_umap(v1_coords_df, sim_v1, display_target_name, "Temel UMAP (V1)", CLUSTER_NAMES_V1),
            use_container_width=True,
        )
        st.markdown("<br>", unsafe_allow_html=True)
        st.plotly_chart(
            plot_umap(v2_coords_df, sim_v2, display_target_name,
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

    comparison_options = [f"⭐ {name}" for name in superlig_names] + original_names

    col_a, col_b = st.columns(2)
    with col_a:
        player_a = st.selectbox("1. Oyuncu", comparison_options, index=0)
    with col_b:
        player_b = st.selectbox(
            "2. Oyuncu", 
            comparison_options, 
            index=1 if len(comparison_options) > 1 else 0
        )

    if st.button("Oyuncuları Kıyasla", type="primary", use_container_width=True):
        # BUG 5 FIX: IndexError korumalı oyuncu istatistiği yükleme
        def _get_player_stats(p_name: str):
            try:
                if p_name.startswith("⭐ "):
                    real_name = p_name[len("⭐ "):]  # Unicode-safe slice
                    rows = superlig_df[superlig_df["Player"] == real_name]
                    if rows.empty:
                        st.error(f"❌ '{real_name}' Süper Lig verisinde bulunamadı.")
                        st.stop()
                    stats = np.array([rows.iloc[0][f] for f in FEATURES])
                    display_name = real_name
                else:
                    rows = players[players["Player"] == p_name]
                    if rows.empty:
                        st.error(f"❌ '{p_name}' veri setinde bulunamadı.")
                        st.stop()
                    raw_idx = rows.index[0]
                    stats = np.array([df_raw.iloc[raw_idx][f] for f in FEATURES])
                    display_name = p_name
                return stats, display_name
            except Exception as e:
                st.error(f"❌ Oyuncu verisi alınırken hata: {e}")
                st.stop()

        stats_a, name_a = _get_player_stats(player_a)
        stats_b, name_b = _get_player_stats(player_b)

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
            st.markdown(f"#### 🟦 {name_a}")
            st.write(f"**V1 Stili:** {CLUSTER_NAMES_V1.get(c1_a, f'Küme {c1_a}')}")
            st.write(f"**V2 Stili:** {CLUSTER_NAMES.get(c2_a, f'Küme {c2_a}')}")
        with c_col2:
            st.markdown(f"#### 🟥 {name_b}")
            st.write(f"**V1 Stili:** {CLUSTER_NAMES_V1.get(c1_b, f'Küme {c1_b}')}")
            st.write(f"**V2 Stili:** {CLUSTER_NAMES.get(c2_b, f'Küme {c2_b}')}")

        st.divider()

        # Karşılaştırma Radarı
        max_vals = df_raw[FEATURES].max().values
        norm_a   = normalize_for_radar(stats_a, max_vals)
        norm_b   = normalize_for_radar(stats_b, max_vals)
        st.plotly_chart(
            plot_comparison_radar(norm_a, norm_b, name_a, name_b),
            use_container_width=True,
        )
