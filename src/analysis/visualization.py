"""
src/analysis/visualization.py
==============================
Görselleştirme fonksiyonları.
- Plotly: UMAP scatter, radar chart (Streamlit için)
- Matplotlib: eğitim kayıp eğrisi, cluster heatmap (eğitim scriptleri için)
"""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.config import CLUSTER_COLORS, CLUSTER_NAMES, FEATURES


# ---------------------------------------------------------------------------
# Plotly — UMAP Haritası (Streamlit)
# ---------------------------------------------------------------------------
def plot_umap(
    coords_df: pd.DataFrame,
    similar_df: pd.DataFrame,
    selected_player: str,
    title: str,
    cluster_names_dict: dict | None = None,
) -> go.Figure:
    """
    UMAP scatter grafiği üzerinde benzer oyuncuları ve hedef oyuncuyu gösterir.

    Parameters
    ----------
    coords_df          : Player, Cluster, UMAP_1, UMAP_2 sütunlarını içerir.
    similar_df         : 'Oyuncu' sütununu içerir (benzer oyuncular listesi).
    selected_player    : Kullanıcının seçtiği oyuncu adı veya "-- Manuel Giriş --".
    title              : Grafik başlığı.
    cluster_names_dict : Küme ID → isim eşlemesi (V2 için).

    Returns
    -------
    go.Figure
    """
    plot_df   = coords_df.copy()
    color_map = {}

    if cluster_names_dict:
        plot_df["Oyun Stili"] = plot_df["Cluster"].apply(
            lambda c: f"{int(c)} - {cluster_names_dict.get(int(c), 'Bilinmeyen')}"
        )
        for c_id in plot_df["Cluster"].unique():
            name = f"{int(c_id)} - {cluster_names_dict.get(int(c_id), 'Bilinmeyen')}"
            color_map[name] = CLUSTER_COLORS.get(int(c_id), "#ffffff")
    else:
        plot_df["Oyun Stili"] = plot_df["Cluster"].apply(lambda c: f"Küme {int(c)}")
        for c_id in plot_df["Cluster"].unique():
            name = f"Küme {int(c_id)}"
            color_map[name] = CLUSTER_COLORS.get(int(c_id), "#ffffff")

    similar_players = similar_df["Oyuncu"].tolist()

    fig = px.scatter(
        plot_df, x="UMAP_1", y="UMAP_2", color="Oyun Stili",
        hover_name="Player", color_discrete_map=color_map,
        title=title, opacity=0.6,
    )

    similar_rows = plot_df[plot_df["Player"].isin(similar_players)]
    fig.add_trace(go.Scatter(
        x=similar_rows["UMAP_1"], y=similar_rows["UMAP_2"],
        mode="markers+text",
        marker=dict(symbol="star", size=15, color="yellow",
                    line=dict(width=2, color="black")),
        text=similar_rows["Player"], textposition="top center",
        name="Benzer Oyuncular", hoverinfo="text",
    ))

    # Hedef oyuncunun konumu
    target_x, target_y, target_name = None, None, "Sizin Oyuncunuz"
    if selected_player != "-- Manuel Giriş --":
        row = plot_df[plot_df["Player"] == selected_player]
        if not row.empty:
            target_x    = row.iloc[0]["UMAP_1"]
            target_y    = row.iloc[0]["UMAP_2"]
            target_name = selected_player

    if target_x is None:
        target_x    = similar_rows["UMAP_1"].mean()
        target_y    = similar_rows["UMAP_2"].mean()
        target_name = "Hedef Oyuncu (Tahmini)"

    fig.add_trace(go.Scatter(
        x=[target_x], y=[target_y],
        mode="markers+text",
        marker=dict(symbol="x", size=20, color="white",
                    line=dict(width=4, color="red")),
        text=[f"<b>{target_name}</b>"], textposition="bottom center",
        name="Sizin Oyuncunuz", hoverinfo="text",
    ))

    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="white",
        height=600,
        legend=dict(orientation="h", yanchor="bottom", y=-0.3,
                    xanchor="center", x=0.5),
    )
    return fig


# ---------------------------------------------------------------------------
# Plotly — Radar Grafiği (Streamlit)
# ---------------------------------------------------------------------------
def plot_radar(
    norm_values: np.ndarray,
    labels: list[str] = FEATURES,
    name: str = "Oyuncu",
    line_color: str = "#00d2ff",
    fill_color: str = "rgba(0, 210, 255, 0.4)",
) -> go.Figure:
    """Tek oyuncu için radar grafiği."""
    fig = go.Figure(data=go.Scatterpolar(
        r=norm_values, theta=labels, fill="toself",
        line_color=line_color, fillcolor=fill_color, name=name,
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=False, range=[0, 1])),
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="white",
        height=500,
    )
    return fig


def plot_comparison_radar(
    norm_a: np.ndarray,
    norm_b: np.ndarray,
    name_a: str,
    name_b: str,
    labels: list[str] = FEATURES,
) -> go.Figure:
    """İki oyuncuyu aynı radar grafiğinde karşılaştırır."""
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=norm_a, theta=labels, fill="toself", name=name_a,
        line_color="#00d2ff", fillcolor="rgba(0, 210, 255, 0.4)",
    ))
    fig.add_trace(go.Scatterpolar(
        r=norm_b, theta=labels, fill="toself", name=name_b,
        line_color="#ff007f", fillcolor="rgba(255, 0, 127, 0.4)",
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=False, range=[0, 1])),
        showlegend=True,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="white",
        height=500,
    )
    return fig


# ---------------------------------------------------------------------------
# Normalize yardımcısı (radar için)
# ---------------------------------------------------------------------------
def normalize_for_radar(stats_array: np.ndarray, max_vals: np.ndarray) -> np.ndarray:
    """İstatistikleri 0-1 arasına normalize eder (görsel radar için)."""
    return stats_array / (max_vals + 1e-8)
