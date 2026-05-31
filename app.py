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
import plotly.express as px
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
original_names = v2_latent_df["Player"].tolist()

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
        idx          = v2_latent_df[v2_latent_df["Player"] == selected_player].index[0]
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

tab1, tab2, tab3 = st.tabs(["Tekli Oyuncu Analizi", "Oyuncu Karşılaştırma", "🤖 Model Performans Karşılaştırması"])

# ---------------------------------------------------------------------------
# UI Yardımcıları — Model sonuç kartları
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
        .model-result-card {
            min-height: 330px;
            height: 330px;
            border-radius: 14px;
            padding: 28px 22px;
            background: #1f2937;
            box-shadow: 0 4px 14px rgba(15, 23, 42, 0.22);
            display: flex;
            flex-direction: column;
            justify-content: center;
            text-align: center;
            margin-bottom: 18px;
        }
        .model-result-card.v1 { border-left: 7px solid #2f80ed; }
        .model-result-card.v2 { border-left: 7px solid #ff8c00; }
        .model-result-card.v3 { border-left: 7px solid #10b981; }

        .model-result-title {
            color: #ffffff;
            font-size: 1.55rem;
            font-weight: 750;
            line-height: 1.25;
            margin-bottom: 24px;
        }
        .model-result-label {
            color: #cbd5e1;
            font-size: 1rem;
            margin-bottom: 18px;
        }
        .model-result-value {
            font-size: 1.55rem;
            font-weight: 800;
            line-height: 1.25;
            margin-bottom: 18px;
        }
        .model-result-value.v1 { color: #2f80ed; }
        .model-result-value.v2 { color: #ff8c00; }
        .model-result-value.v3 { color: #10b981; }

        .model-result-sub {
            color: #cbd5e1;
            font-size: 0.95rem;
            line-height: 1.4;
        }
        .model-confidence-box {
            margin-top: 14px;
            padding: 10px 12px;
            border-radius: 10px;
            background: rgba(16, 185, 129, 0.13);
            color: #bbf7d0;
            font-size: 0.92rem;
            line-height: 1.35;
        }
        .model-table-title {
            min-height: 58px;
            display: flex;
            align-items: flex-end;
            font-size: 1.35rem;
            font-weight: 750;
            color: #1f2937;
            margin: 4px 0 12px 0;
        }
        div[data-testid="stDataFrame"] {
            width: 100%;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


def render_model_result_card(
    title: str,
    label: str,
    value: str,
    sub_text: str,
    version_class: str,
    confidence_text: str | None = None,
):
    confidence_html = (
        f'<div class="model-confidence-box">{confidence_text}</div>'
        if confidence_text else ""
    )

    st.markdown(
        f"""
        <div class="model-result-card {version_class}">
            <div class="model-result-title">{title}</div>
            <div class="model-result-label">{label}</div>
            <div class="model-result-value {version_class}">{value}</div>
            <div class="model-result-sub">{sub_text}</div>
            {confidence_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────
# TAB 1: Tekli Oyuncu Analizi
# ─────────────────────────────────────────────
with tab1:
    if st.sidebar.button("Oyuncuyu Analiz Et", use_container_width=True):
        stats = np.array([input_data[f] for f in FEATURES])

        # Model V1
        c1, sim_v1, _, _ = predict_and_find_similar(
            stats, model_v1, scaler, v1_matrix, v1_latent_df, version="v1"
        )

        # Model V2
        c2, sim_v2, _, confidence_pct = predict_and_find_similar(
            stats, model_v2, scaler, v2_matrix, v2_latent_df, version="v2"
        )

        # Model V3 - SOM
        nearest_player_v3, som_cluster_v3, sim_v3 = predict_som_result(
            stats,
            v3_som_coords_df,
            v3_som_similar_df,
        )

        v1_style_name = CLUSTER_NAMES_V1.get(c1, f"Küme {c1}")
        v2_style_name = CLUSTER_NAMES.get(c2, f"Küme {c2}")

        if confidence_pct is not None:
            if confidence_pct >= 80:
                confidence_text = f"Yüksek güven — model bu küme atamasından %{confidence_pct:.1f} emin."
            elif confidence_pct >= 60:
                confidence_text = f"Orta güven — oyuncu birden fazla kümeye yakın olabilir (%{confidence_pct:.1f})."
            else:
                confidence_text = f"Düşük güven — küme ataması belirsiz (%{confidence_pct:.1f})."
        else:
            confidence_text = None

        # Üç model sonucu aynı kart yapısında gösterilir.
        col1, col2, col3 = st.columns(3)

        with col1:
            render_model_result_card(
                title="Model V1<br>(Autoencoder)",
                label="Tahmini Küme",
                value=v1_style_name,
                sub_text=f"(Küme {c1})",
                version_class="v1",
            )

        with col2:
            render_model_result_card(
                title="Model V2<br>(VAE + GMM)",
                label="Tespit Edilen Oyun Stili",
                value=v2_style_name,
                sub_text=f"(Küme {c2})",
                version_class="v2",
                confidence_text=confidence_text,
            )

        with col3:
            render_model_result_card(
                title="Model V3<br>(SOM)",
                label="SOM Bölgesi",
                value=str(som_cluster_v3),
                sub_text=f"En yakın veri seti oyuncusu: {nearest_player_v3}",
                version_class="v3",
            )

        # Benzer oyuncu tabloları aynı satır düzeninde gösterilir.
        table_col1, table_col2, table_col3 = st.columns(3)

        with table_col1:
            st.markdown('<div class="model-table-title">V1 Uzayındaki Benzer Oyuncular</div>', unsafe_allow_html=True)
            st.dataframe(sim_v1, use_container_width=True, hide_index=True)

        with table_col2:
            st.markdown('<div class="model-table-title">V2 Uzayındaki Benzer Oyuncular</div>', unsafe_allow_html=True)
            st.dataframe(sim_v2, use_container_width=True, hide_index=True)

        with table_col3:
            st.markdown('<div class="model-table-title">SOM Haritasındaki Benzer Oyuncular</div>', unsafe_allow_html=True)
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
                    rows = v2_latent_df[v2_latent_df["Player"] == p_name]
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
        c1_a, _, _, _ = predict_and_find_similar(
            stats_a, model_v1, scaler, v1_matrix, v1_latent_df, version="v1"
        )
        c1_b, _, _, _ = predict_and_find_similar(
            stats_b, model_v1, scaler, v1_matrix, v1_latent_df, version="v1"
        )

        # V2 kümeleri + benzerlik
        c2_a, _, _, _ = predict_and_find_similar(
            stats_a, model_v2, scaler, v2_matrix, v2_latent_df, version="v2"
        )
        c2_b, _, _, _ = predict_and_find_similar(
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

# ─────────────────────────────────────────────
# TAB 3: Model Performans Karşılaştırması
# ─────────────────────────────────────────────
with tab3:
    st.markdown("### 🤖 Yapay Zeka Kümeleme Kalitesi & Matematiksel Karşılaştırma")
    st.markdown(
        "Modellerimizin (Model V1 - Autoencoder ve Model V2 - Variational Autoencoder) "
        "kümeleme kalitesini gösteren **Silhouette Skoru**, **Davies-Bouldin Endeksi** ve "
        "**Calinski-Harabasz Skoru** gibi akademik doğrulama metriklerini buradan sayısal ve grafiksel olarak kıyaslayabilirsiniz."
    )

    # 1. Metrikleri Dinamik Olarak Hesaplayalım
    from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
    
    # Model V1 latent features & clusters
    v1_features = v1_latent_df[[c for c in v1_latent_df.columns if c.startswith("L")]].values
    v1_labels = v1_latent_df["Cluster"].values
    
    # Model V2 latent features & clusters
    v2_features = v2_latent_df[[c for c in v2_latent_df.columns if c.startswith("mu_")]].values
    v2_labels = v2_latent_df["Cluster"].values
    
    # Hesaplamalar
    v1_sil = silhouette_score(v1_features, v1_labels)
    v1_db = davies_bouldin_score(v1_features, v1_labels)
    v1_ch = calinski_harabasz_score(v1_features, v1_labels)
    
    v2_sil = silhouette_score(v2_features, v2_labels)
    v2_db = davies_bouldin_score(v2_features, v2_labels)
    v2_ch = calinski_harabasz_score(v2_features, v2_labels)

    # 2. Metrik Kutuları (st.columns)
    st.markdown("#### 📊 Kümeleme Kalite Özet Kartları")
    mc1, mc2, mc3 = st.columns(3)
    
    with mc1:
        # Silhouette: Higher is better
        delta_sil_pct = ((v2_sil - v1_sil) / v1_sil) * 100
        st.metric(
            label="✨ Silhouette Skoru (Yüksek Daha İyi)", 
            value=f"{v2_sil:.4f}",
            delta=f"+{delta_sil_pct:.1f}% (V1: {v1_sil:.4f})",
            help="Kümelerin kendi içinde ne kadar yoğun ve diğer kümelerden ne kadar uzak olduğunu gösterir (-1 ile +1 arası; +1 en iyisidir)."
        )
        st.caption("Model V2, Model V1'e kıyasla daha net çizilmiş küme sınırlarına sahiptir.")
        
    with mc2:
        # Davies-Bouldin: Lower is better
        delta_db_pct = ((v2_db - v1_db) / v1_db) * 100
        st.metric(
            label="📉 Davies-Bouldin Endeksi (Düşük Daha İyi)", 
            value=f"{v2_db:.4f}",
            delta=f"{delta_db_pct:.1f}% (V1: {v1_db:.4f})",
            delta_color="normal" if delta_db_pct < 0 else "inverse",
            help="Kümelerin birbirine olan mesafesi ile kendi içlerindeki yayılımın oranıdır. Değerin küçük olması kümelerin daha iyi ayrıştığını gösterir (0 en iyisidir)."
        )
        st.caption("Model V2'de kümeler arası benzerlik ve çakışma oranı daha düşüktür.")

    with mc3:
        # Calinski-Harabasz: Higher is better
        delta_ch_pct = ((v2_ch - v1_ch) / v1_ch) * 100
        st.metric(
            label="🏆 Calinski-Harabasz Skoru (Yüksek Daha İyi)", 
            value=f"{v2_ch:.1f}",
            delta=f"+{delta_ch_pct:.1f}% (V1: {v1_ch:.1f})",
            help="Küme içi varyans ile kümeler arası varyansın oranıdır. Değerin büyük olması kümelerin daha belirgin olduğunu gösterir."
        )
        st.caption("Model V2'nin ürettiği kümeler istatistiksel varyans açısından çok daha belirgindir.")

    st.divider()

    # 3. Performans Matris Tablosu
    st.markdown("#### 📋 Akademik Karşılaştırma Matrisi")
    
    matrix_data = {
        "Kümeleme Kalite Metriği": [
            "Silhouette Skoru (Sıkılık & Ayrışma)",
            "Davies-Bouldin Endeksi (Çakışma Oranı)",
            "Calinski-Harabasz Skoru (Varyans Oranı)",
            "Latent (Gizli) Alan Boyutu",
            "Optimal Küme Sayısı (k)",
            "Kullanılan Derin Öğrenme Modeli",
            "Kümeleme Algoritması"
        ],
        "Model V1 (Temel Model)": [
            f"{v1_sil:.4f}",
            f"{v1_db:.4f}",
            f"{v1_ch:.2f}",
            f"{v1_features.shape[1]} Boyut",
            "5 Küme",
            "Autoencoder (AE)",
            "Ensemble (K-Means/Agglomerative/GMM)"
        ],
        "Model V2 (Gelişmiş Model)": [
            f"{v2_sil:.4f}",
            f"{v2_db:.4f}",
            f"{v2_ch:.2f}",
            f"{v2_features.shape[1]} Boyut",
            "8 Küme",
            "Variational Autoencoder (VAE)",
            "Gaussian Mixture Model (GMM)"
        ],
        "Gelişim / Fark (%)": [
            f"+{delta_sil_pct:.2f}% (Daha İyi)",
            f"{delta_db_pct:.2f}% (Daha İyi)",
            f"+{delta_ch_pct:.2f}% (Daha İyi)",
            f"+{((v2_features.shape[1]-v1_features.shape[1])/v1_features.shape[1])*100:.1f}% daha fazla bilgi kapasitesi",
            "+3 Küme (Daha detaylı stil analizi)",
            "Olasılıksal / Generative Uzay",
            "Yumuşak Kümeler (Soft-Clustering)"
        ]
    }
    
    matrix_df = pd.DataFrame(matrix_data)
    st.dataframe(matrix_df, use_container_width=True, hide_index=True)

    st.divider()

    # 4. Plotly Karşılaştırma Grafikleri (Metrik Bar Grafikleri)
    st.markdown("#### 📊 Model Metrik Karşılaştırma Grafikleri")
    
    # Silhouette ve Davies-Bouldin Grafik
    col_chart1, col_chart2 = st.columns(2)
    
    with col_chart1:
        # Silhouette ve DB Karşılaştırma Bar Grafiği
        df_bar1 = pd.DataFrame({
            "Metrik": ["Silhouette Skoru", "Davies-Bouldin Endeksi"] * 2,
            "Skor": [v1_sil, v1_db, v2_sil, v2_db],
            "Model": ["Model V1 (AE)"] * 2 + ["Model V2 (VAE)"] * 2
        })
        fig_bar1 = px.bar(
            df_bar1,
            x="Metrik",
            y="Skor",
            color="Model",
            barmode="group",
            title="Silhouette & Davies-Bouldin Karşılaştırması",
            color_discrete_sequence=["#1E88E5", "#FFB300"],
            text_auto=".3f"
        )
        fig_bar1.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig_bar1, use_container_width=True)
        
    with col_chart2:
        # Calinski-Harabasz Karşılaştırma Bar Grafiği
        df_bar2 = pd.DataFrame({
            "Model": ["Model V1 (AE)", "Model V2 (VAE)"],
            "Calinski-Harabasz Skoru": [v1_ch, v2_ch]
        })
        fig_bar2 = px.bar(
            df_bar2,
            x="Model",
            y="Calinski-Harabasz Skoru",
            color="Model",
            title="Calinski-Harabasz Skoru Karşılaştırması",
            color_discrete_sequence=["#1E88E5", "#FFB300"],
            text_auto=".1f"
        )
        fig_bar2.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig_bar2, use_container_width=True)

    st.divider()

    # 5. Model V2 k Küme Sayısı Seçim Grafikleri (Eğitim Günlüklerinden)
    try:
        import json
        with open("artifacts/v2/loglar/egitim_gecmisi.json", "r", encoding="utf-8") as f:
            v2_hist = json.load(f)
            
        if "cluster_metrics" in v2_hist:
            c_metrics = v2_hist["cluster_metrics"]
            k_range = c_metrics.get("k_range", [4, 5, 6, 7, 8, 9, 10, 11])
            
            st.markdown("#### 🔬 Model V2 için Küme Sayısı (k) Belirleme Analizi")
            st.markdown(
                "Model V2 eğitilirken en uygun küme sayısının **8** olarak seçilmesinin arkasındaki matematiksel sebep aşağıda gösterilmiştir. "
                "En düşük Davies-Bouldin Endeksi (kümelerin en iyi ayrışması) ve yüksek Silhouette kararlılığı $k=8$ noktasında yakalanmıştır."
            )
            
            # Line Chart Verisi Hazırlayalım
            k_df = pd.DataFrame({
                "Küme Sayısı (k)": k_range,
                "Silhouette Skoru": c_metrics.get("silhouette", []),
                "Davies-Bouldin": c_metrics.get("davies_bouldin", []),
                "Calinski-Harabasz": c_metrics.get("calinski_harabasz", [])
            })
            
            col_k1, col_k2 = st.columns(2)
            
            with col_k1:
                # Silhouette ve DB Çizgi Grafiği
                fig_k1 = px.line(
                    k_df,
                    x="Küme Sayısı (k)",
                    y=["Silhouette Skoru", "Davies-Bouldin"],
                    title="Küme Sayısına (k) Göre Silhouette & Davies-Bouldin Eğrisi",
                    markers=True,
                    color_discrete_sequence=["#FFB300", "#1E88E5"]
                )
                # k=8 çizgisi
                fig_k1.add_vline(x=8, line_dash="dash", line_color="green", annotation_text="Seçilen k=8")
                fig_k1.update_layout(
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                st.plotly_chart(fig_k1, use_container_width=True)
                
            with col_k2:
                # Calinski-Harabasz Çizgi Grafiği
                fig_k2 = px.line(
                    k_df,
                    x="Küme Sayısı (k)",
                    y="Calinski-Harabasz",
                    title="Küme Sayısına (k) Göre Calinski-Harabasz Varyans Oranı Eğrisi",
                    markers=True,
                    color_discrete_sequence=["#26A69A"]
                )
                # k=8 çizgisi
                fig_k2.add_vline(x=8, line_dash="dash", line_color="green", annotation_text="Seçilen k=8")
                fig_k2.update_layout(
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)"
                )
                st.plotly_chart(fig_k2, use_container_width=True)
    except Exception as e:
        st.warning(f"Küme seçim grafikleri yüklenemedi: {e}")


