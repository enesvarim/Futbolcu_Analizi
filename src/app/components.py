"""
src/app/components.py
======================
Yeniden kullanılabilir Streamlit UI bileşenleri.
"""
import streamlit as st

from src.config import CLUSTER_COLORS, CLUSTER_NAMES, CLUSTER_NAMES_V1


def apply_global_styles() -> None:
    """Uygulamanın koyu tema CSS stillerini uygular."""
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
            border-left: 5px solid #00d2ff; margin-bottom: 20px; color: white;
        }
        .metric-card h3 {color: white;}
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        .stDeployButton {display:none;}
    </style>
    """, unsafe_allow_html=True)


def render_cluster_card(
    model_name: str,
    cluster_id: int,
    version: str = "v1",
) -> None:
    """Küme sonucunu renkli kart olarak gösterir."""
    color = CLUSTER_COLORS.get(cluster_id, "#3a7bd5")

    if version == "v2":
        cluster_label = CLUSTER_NAMES.get(cluster_id, f"Küme {cluster_id}")
        subtitle      = f"(Küme {cluster_id})"
        font_size     = "28px"
    elif version == "v1":
        cluster_label = CLUSTER_NAMES_V1.get(cluster_id, f"Küme {cluster_id}")
        subtitle      = f"(Küme {cluster_id})"
        font_size     = "28px"
    else:
        cluster_label = f"Küme {cluster_id}"
        subtitle      = ""
        font_size     = "32px"

    st.markdown(f"""
    <div class="metric-card" style="border-left-color: {color};">
        <h3>{model_name}</h3>
        <p style="color:#aaa;">
            {"Tespit Edilen Oyun Stili" if version == "v2" else "Tahmini Küme"}
        </p>
        <h1 style="font-size: {font_size}; margin:0; color:{color};">
            {cluster_label}
        </h1>
        {f'<p style="color:{color}; margin-top:5px; font-weight:bold;">{subtitle}</p>'
         if subtitle else ""}
    </div>
    """, unsafe_allow_html=True)


def render_similarity_card(sim_pct: int) -> None:
    """İki oyuncu arasındaki benzerlik yüzdesini büyük kart olarak gösterir."""
    st.markdown(f"""
    <div class="metric-card" style="border-left-color: #4DAF4A; margin-top:20px;">
        <h3>Yapay Zeka (V2) Benzerlik Skoru</h3>
        <p style="color:#aaa;">
            İki oyuncunun 16 boyutlu zeka uzayındaki oyun stili benzerliği
        </p>
        <h1 style="font-size: 40px; margin:0; color:#4DAF4A;">%{sim_pct}</h1>
    </div>
    """, unsafe_allow_html=True)
