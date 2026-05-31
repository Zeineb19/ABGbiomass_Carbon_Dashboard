"""
ForestBiomass Intelligence Dashboard
======================================
Loads real prediction CSVs from the /data/ folder (same repo).

Expected folder structure:
  data/
    spectral_xgboost_with_chm_test_predictions.csv
    spectral_xgboost_without_chm_test_predictions.csv
    spectral_catboost_with_chm_test_predictions.csv
    spectral_catboost_without_chm_test_predictions.csv
    spectral_mlp_with_chm_test_predictions.csv
    spectral_mlp_without_chm_test_predictions.csv
    spectral_xgboost_with_chm_full_predictions.csv
    spectral_xgboost_without_chm_full_predictions.csv
    spectral_catboost_with_chm_full_predictions.csv
    spectral_catboost_without_chm_full_predictions.csv
    spectral_mlp_with_chm_full_predictions.csv
    spectral_mlp_without_chm_full_predictions.csv
    spectral_models_summary_metrics.csv
    embedding_mlp_test_predictions.csv
    embedding_xgboost_test_predictions.csv
    embedding_catboost_test_predictions.csv
    embedding_mlp_full_predictions.csv
    embedding_xgboost_full_predictions.csv
    embedding_catboost_full_predictions.csv
    embedding_models_summary_metrics.csv
    merged_tree_level.csv

Run:  streamlit run dashboard.py
"""

import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Forest Biomass Intelligence",
    page_icon="🌲",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

# ─────────────────────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;600;700&display=swap');
:root {
  --bg:#0b1117; --panel:#131c26; --accent1:#2dd4bf; --accent2:#86efac;
  --accent3:#fb923c; --accent4:#a78bfa; --text:#e2e8f0; --muted:#64748b; --border:#1e293b;
}
html,body,[data-testid="stAppViewContainer"]{background-color:var(--bg)!important;color:var(--text)!important;font-family:'DM Sans',sans-serif;}
[data-testid="stSidebar"]{background:var(--panel)!important;border-right:1px solid var(--border);}
[data-testid="stSidebar"] *{color:var(--text)!important;}
h1,h2,h3,h4{font-family:'Space Mono',monospace!important;}
h1{color:var(--accent1)!important;letter-spacing:-1px;}
h2{color:var(--accent2)!important;}
h3{color:var(--accent1)!important;font-size:1rem!important;}
[data-testid="metric-container"]{background:var(--panel)!important;border:1px solid var(--border)!important;border-radius:10px!important;padding:14px 18px!important;}
[data-testid="metric-container"] label{color:var(--muted)!important;font-size:.75rem!important;text-transform:uppercase;letter-spacing:1px;}
[data-testid="metric-container"] [data-testid="stMetricValue"]{color:var(--accent1)!important;font-family:'Space Mono',monospace!important;}
[data-baseweb="tab-list"]{background:var(--panel)!important;border-radius:8px;padding:4px;}
[data-baseweb="tab"]{color:var(--muted)!important;font-family:'Space Mono',monospace!important;font-size:.8rem!important;}
[aria-selected="true"][data-baseweb="tab"]{color:var(--accent1)!important;background:var(--bg)!important;border-radius:6px!important;}
[data-baseweb="select"]>div{background:var(--panel)!important;border-color:var(--border)!important;}
[data-testid="stExpander"]{background:var(--panel)!important;border:1px solid var(--border)!important;border-radius:8px!important;}
[data-testid="stInfo"]{background:rgba(45,212,191,.08)!important;border-left:3px solid var(--accent1)!important;}
hr{border-color:var(--border)!important;}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# PLOTLY BASE LAYOUT
# ─────────────────────────────────────────────────────────────────────────────
PL = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(11,17,23,0.6)",
    font=dict(family="DM Sans", color="#e2e8f0", size=12),
    margin=dict(l=40, r=20, t=50, b=40),
    colorway=["#2dd4bf","#86efac","#fb923c","#a78bfa","#38bdf8","#f472b6"],
)

MODEL_COLORS = {
    "MLP":      "#a78bfa",
    "XGBoost":  "#2dd4bf",
    "CatBoost": "#fb923c",
}

# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADERS  (cached + dtype-compressed for speed)
# ─────────────────────────────────────────────────────────────────────────────

_PRED_DTYPES  = {"lon":"float32","lat":"float32",
                 "y_true":"float32","y_pred":"float32","residual":"float32"}

_LIDAR_KEEP   = [
    "tile","tree_lon","tree_lat","dist_to_px_m","point_count",
    "height_max_m","height_mean_m","height_std_m",
    "height_p25_m","height_p50_m","height_p75_m","height_p90_m","height_p95_m",
    "cv_height","crown_radius_m","crown_r95_m","crown_area_m2",
    "canopy_pts","density_pt_m2",
    "sat_NDVI","sat_EVI","sat_LAI","sat_temp","sat_pr","sat_biomass",
]
_LIDAR_DTYPES = {c:"float32" for c in _LIDAR_KEEP if c != "tile"}
_LIDAR_DTYPES.update({"tile":"int8","point_count":"int32","canopy_pts":"int32"})

@st.cache_data(show_spinner=False)
def _read(path, usecols=None, dtype=None):
    if not os.path.exists(path) or os.path.getsize(path) < 10:
        return None
    try:
        df = pd.read_csv(path, usecols=usecols, dtype=dtype)
        return df if not df.empty else None
    except Exception:
        return None

@st.cache_data(show_spinner=False)
def load_spectral_summary():
    return _read(os.path.join(DATA_DIR, "spectral_models_summary_metrics.csv"))

@st.cache_data(show_spinner=False)
def load_embedding_summary():
    return _read(os.path.join(DATA_DIR, "embedding_models_summary_metrics.csv"))

@st.cache_data(show_spinner=False)
def load_lidar():
    df = _read(os.path.join(DATA_DIR, "merged_tree_level.csv"),
               usecols=_LIDAR_KEEP, dtype=_LIDAR_DTYPES)
    if df is not None:
        df["height_range_m"] = (df["height_p95_m"] - df["height_p25_m"]).astype("float32")
        df["tile_label"]     = df["tile"].map({1:"Tile 1", 2:"Tile 2", 3:"Tile 3"})
    return df

@st.cache_data(show_spinner=False)
def spectral_test(model, variant):
    return _read(os.path.join(DATA_DIR,
                 f"spectral_{model.lower()}_{variant}_test_predictions.csv"),
                 dtype=_PRED_DTYPES)

@st.cache_data(show_spinner=False)
def spectral_full(model, variant):
    df = _read(os.path.join(DATA_DIR,
               f"spectral_{model.lower()}_{variant}_full_predictions.csv"),
               dtype=_PRED_DTYPES)
    if df is not None and len(df) > 8_000:
        df = df.sample(8_000, random_state=42).reset_index(drop=True)
    return df

@st.cache_data(show_spinner=False)
def embedding_test(model):
    return _read(os.path.join(DATA_DIR,
                 f"embedding_{model.lower()}_test_predictions.csv"),
                 dtype={c:"float32" for c in ["lat","lon","y_true","y_pred","residual"]})

@st.cache_data(show_spinner=False)
def embedding_full(model):
    df = _read(os.path.join(DATA_DIR,
               f"embedding_{model.lower()}_full_predictions.csv"),
               dtype={c:"float32" for c in ["lat","lon","y_true","y_pred","residual"]})
    if df is not None and len(df) > 8_000:
        df = df.sample(8_000, random_state=42).reset_index(drop=True)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# SHARED CHART HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def metrics_row(df_pred, label=""):
    yt, yp = df_pred["y_true"], df_pred["y_pred"]
    r2   = r2_score(yt, yp)
    rmse = np.sqrt(mean_squared_error(yt, yp))
    mae  = mean_absolute_error(yt, yp)
    bias = float(np.mean(yp - yt))
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("R²",   f"{r2:.4f}")
    c2.metric("RMSE", f"{rmse:.2f} Mg/ha")
    c3.metric("MAE",  f"{mae:.2f} Mg/ha")
    c4.metric("Bias", f"{bias:+.2f} Mg/ha")

def scatter_obs_pred(df_pred, title, color):
    yt = df_pred["y_true"].values
    yp = df_pred["y_pred"].values
    r2   = r2_score(yt, yp)
    rmse = np.sqrt(mean_squared_error(yt, yp))
    lims = [min(yt.min(), yp.min()) - 3, max(yt.max(), yp.max()) + 3]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=yt, y=yp, mode="markers",
                             marker=dict(size=4, color=color, opacity=0.5), name="Samples"))
    fig.add_trace(go.Scatter(x=lims, y=lims, mode="lines",
                             line=dict(color="#ef4444", dash="dash", width=1.5), name="1:1"))
    fig.update_layout(**PL, title=f"{title} — R²={r2:.3f} | RMSE={rmse:.2f}",
                      xaxis_title="Observed (Mg/ha)", yaxis_title="Predicted (Mg/ha)", height=380)
    return fig

def residual_plot(df_pred, color):
    yt  = df_pred["y_true"].values
    res = df_pred["residual"].values
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=yt, y=res, mode="markers",
                             marker=dict(size=4, color=color, opacity=0.4), name="Residuals"))
    fig.add_hline(y=0, line_color="#ef4444", line_dash="dash")
    fig.update_layout(**PL, title="Residual Plot",
                      xaxis_title="Observed (Mg/ha)", yaxis_title="Residual (Pred − Obs)", height=300)
    return fig

def residual_histogram(df_pred, color):
    res = df_pred["residual"].values
    fig = px.histogram(x=res, nbins=50, color_discrete_sequence=[color],
                       title="Residual Distribution",
                       labels={"x": "Residual (Mg/ha)"})
    fig.add_vline(x=0, line_color="#ef4444", line_dash="dash")
    fig.update_layout(**PL, height=280)
    return fig

def spatial_map(df_full, col="y_pred", title="Predicted Biomass", colorscale="Viridis"):
    fig = go.Figure(go.Scattergl(
        x=df_full["lon"], y=df_full["lat"], mode="markers",
        marker=dict(size=3, color=df_full[col], colorscale=colorscale,
                    showscale=True, colorbar=dict(title="Mg/ha")),
        name=title,
    ))
    fig.update_layout(**PL, title=title,
                      xaxis_title="Longitude", yaxis_title="Latitude", height=420)
    return fig

def comparison_bar_from_summary(df_summary, metric, title, unit=""):
    fig = go.Figure()
    for _, row in df_summary.iterrows():
        label = f"{row['model']} ({row['variant'].replace('_',' ')})"
        color = MODEL_COLORS.get(row["model"], "#64748b")
        fig.add_trace(go.Bar(
            x=[label], y=[row[metric]],
            marker_color=color, name=label,
            text=[f"{row[metric]:.3f}{unit}"], textposition="outside",
        ))
    fig.update_layout(**PL, title=title, height=360,
                      yaxis_title=metric, showlegend=False,
                      bargap=0.3)
    return fig

def chm_delta_table(df_summary):
    """Show Δ per model when CHM is added."""
    rows = []
    for model in df_summary["model"].unique():
        sub = df_summary[df_summary["model"] == model]
        chm    = sub[sub["variant"].str.contains("with", case=False)]
        no_chm = sub[sub["variant"].str.contains("without", case=False)]
        if len(chm) and len(no_chm):
            rows.append({
                "Model":        model,
                "R² w/o CHM":  no_chm["R2"].values[0],
                "R² with CHM": chm["R2"].values[0],
                "ΔR²":         round(chm["R2"].values[0] - no_chm["R2"].values[0], 4),
                "RMSE w/o CHM": no_chm["RMSE"].values[0],
                "RMSE with CHM": chm["RMSE"].values[0],
                "ΔRMSE":        round(chm["RMSE"].values[0] - no_chm["RMSE"].values[0], 4),
            })
    return pd.DataFrame(rows)

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🌲 ForestBiomass\n### Intelligence Dashboard")
    st.markdown("---")
    section = st.radio("Navigation", [
        "🛰 Spectral Models",
        "🔮 Embedding Models",
        "📡 LiDAR Analysis",
    ])
    st.markdown("---")
    st.markdown("""
**Study area:** Veluwe, Netherlands  
**Target:** Above-Ground Biomass (Mg/ha)  
**Source:** ESA CCI AGB · GEDI MU

---
**Spectral:** NDVI · EVI · LAI · Temp · Precip (± CHM)  
**Embeddings:** 64-dim Google Satellite Embeddings  
**LiDAR:** Individual tree segmentation
""")
    st.markdown("---")
    st.caption("Built with Streamlit · Plotly · XGBoost · CatBoost · MLP")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — SPECTRAL MODELS
# ─────────────────────────────────────────────────────────────────────────────
if section == "🛰 Spectral Models":
    st.title("🛰 Spectral Models — Biomass Prediction")
    st.markdown("Real predictions from **XGBoost**, **CatBoost**, and **MLP** trained on "
                "Sentinel-2 spectral indices ± CHM over the Veluwe.")

    summary = load_spectral_summary()

    if summary is None:
        st.error("❌ `spectral_models_summary_metrics.csv` not found in `/data/`. "
                 "Please run the spectral notebook and place output files in `/data/`.")
        st.stop()

    # ── Tabs
    detail_tab, compare_tab, chm_tab, map_tab = st.tabs([
        "📊 Model Detail", "⚖ Model Comparison", "🌿 CHM Impact", "🗺 Spatial Map"
    ])

    with detail_tab:
        c1, c2 = st.columns(2)
        with c1:
            model_sel = st.selectbox("Model", ["MLP", "XGBoost", "CatBoost"], key="sp_model")
        with c2:
            variant_sel = st.selectbox("Variant", ["with_chm", "without_chm"], key="sp_variant")

        df_test = spectral_test(model_sel, variant_sel)

        if df_test is None:
            st.warning(f"⚠ File not found: `spectral_{model_sel.lower()}_{variant_sel}_test_predictions.csv`")
        else:
            color = MODEL_COLORS[model_sel]
            st.markdown(f"#### {model_sel} ({variant_sel.replace('_',' ')}) — Test-Set Performance")
            metrics_row(df_test)

            c1, c2 = st.columns(2)
            with c1:
                st.plotly_chart(
                    scatter_obs_pred(df_test, f"{model_sel} — {variant_sel.replace('_',' ')}", color),
                    use_container_width=True)
            with c2:
                st.plotly_chart(residual_plot(df_test, color), use_container_width=True)

            st.plotly_chart(residual_histogram(df_test, color), use_container_width=True)

            with st.expander("📋 Raw test-set predictions (first 200 rows)"):
                st.dataframe(df_test.head(200), use_container_width=True)

    with compare_tab:
        st.markdown("#### All Models — Side-by-Side Metrics")

        # Summary metrics table
        st.dataframe(
            summary.sort_values("R2", ascending=False)
                   .reset_index(drop=True)
                   .rename(columns={"model":"Model","variant":"Variant"}),
            use_container_width=True
        )

        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(
                comparison_bar_from_summary(summary, "R2", "R² — All Models"),
                use_container_width=True)
            st.plotly_chart(
                comparison_bar_from_summary(summary, "MAE", "MAE — All Models", " Mg/ha"),
                use_container_width=True)
        with c2:
            st.plotly_chart(
                comparison_bar_from_summary(summary, "RMSE", "RMSE — All Models", " Mg/ha"),
                use_container_width=True)
            st.plotly_chart(
                comparison_bar_from_summary(summary, "Bias", "Bias — All Models", " Mg/ha"),
                use_container_width=True)

        # Overlay scatter — all 6 models
        st.markdown("#### Observed vs Predicted — All Models Overlay")
        fig_all = go.Figure()
        for model in ["MLP", "XGBoost", "CatBoost"]:
            for variant in ["with_chm", "without_chm"]:
                df_t = spectral_test(model, variant)
                if df_t is not None:
                    dash = "solid" if "with" in variant else "dot"
                    fig_all.add_trace(go.Scatter(
                        x=df_t["y_true"], y=df_t["y_pred"], mode="markers",
                        marker=dict(size=3, color=MODEL_COLORS[model], opacity=0.3,
                                    symbol="circle" if "with" in variant else "x"),
                        name=f"{model} {variant.replace('_',' ')}",
                    ))
        lims = [0, 260]
        fig_all.add_trace(go.Scatter(x=lims, y=lims, mode="lines",
                                     line=dict(color="#ef4444", dash="dash", width=1.2),
                                     name="1:1", showlegend=True))
        fig_all.update_layout(**PL, height=460, xaxis_title="Observed (Mg/ha)",
                              yaxis_title="Predicted (Mg/ha)",
                              title="All Spectral Models — Obs vs Pred")
        st.plotly_chart(fig_all, use_container_width=True)

    with chm_tab:
        st.markdown("#### Impact of CHM on Model Performance")
        delta_df = chm_delta_table(summary)
        if not delta_df.empty:
            st.dataframe(delta_df.set_index("Model"), use_container_width=True)

            # ΔR² bar chart
            fig_delta = go.Figure()
            for _, row in delta_df.iterrows():
                color = MODEL_COLORS.get(row["Model"], "#64748b")
                fig_delta.add_trace(go.Bar(
                    x=[row["Model"]], y=[row["ΔR²"]],
                    marker_color=color,
                    text=[f"+{row['ΔR²']:.4f}" if row["ΔR²"] >= 0 else f"{row['ΔR²']:.4f}"],
                    textposition="outside", name=row["Model"],
                ))
            fig_delta.add_hline(y=0, line_color="#ef4444", line_dash="dash")
            fig_delta.update_layout(**PL, title="ΔR² when CHM is added (with − without)",
                                    yaxis_title="ΔR²", height=340, showlegend=False)
            st.plotly_chart(fig_delta, use_container_width=True)

            # Side-by-side R² bars per model
            fig_side = go.Figure()
            for _, row in delta_df.iterrows():
                color = MODEL_COLORS.get(row["Model"], "#64748b")
                fig_side.add_trace(go.Bar(
                    name=f"{row['Model']} w/o CHM",
                    x=[row["Model"]], y=[row["R² w/o CHM"]],
                    marker_color=color, opacity=0.45,
                    offsetgroup=row["Model"], base=0,
                ))
                fig_side.add_trace(go.Bar(
                    name=f"{row['Model']} with CHM",
                    x=[row["Model"]], y=[row["R² with CHM"]],
                    marker_color=color, opacity=1.0,
                    offsetgroup=row["Model"],
                ))
            fig_side.update_layout(**PL, barmode="group",
                                   title="R² with vs without CHM — per Model",
                                   yaxis_title="R²", height=360)
            st.plotly_chart(fig_side, use_container_width=True)

    with map_tab:
        st.markdown("#### Spatial Biomass Map")
        cm1, cm2 = st.columns(2)
        with cm1:
            map_model = st.selectbox("Model", ["MLP","XGBoost","CatBoost"], key="map_m")
        with cm2:
            map_variant = st.selectbox("Variant", ["with_chm","without_chm"], key="map_v")

        df_full = spectral_full(map_model, map_variant)

        if df_full is None:
            st.warning(f"⚠ Full prediction file not found for {map_model} {map_variant}. "
                       "Falling back to test-set points.")
            df_full = spectral_test(map_model, map_variant)

        if df_full is not None:
            fig_obs  = spatial_map(df_full, "y_true", "Observed Biomass (ESA CCI AGB)")
            fig_pred = spatial_map(df_full, "y_pred",
                                   f"Predicted — {map_model} ({map_variant.replace('_',' ')})")
            c1, c2 = st.columns(2)
            with c1: st.plotly_chart(fig_obs,  use_container_width=True)
            with c2: st.plotly_chart(fig_pred, use_container_width=True)

            # Difference map
            df_full = df_full.copy()
            df_full["diff"] = df_full["y_pred"] - df_full["y_true"]
            fig_diff = go.Figure(go.Scattergl(
                x=df_full["lon"], y=df_full["lat"], mode="markers",
                marker=dict(size=3, color=df_full["diff"],
                            colorscale="RdBu", cmid=0,
                            showscale=True, colorbar=dict(title="Diff (Mg/ha)")),
            ))
            fig_diff.update_layout(**PL, title="Difference Map (Pred − Obs)",
                                   xaxis_title="Longitude", yaxis_title="Latitude", height=420)
            st.plotly_chart(fig_diff, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — EMBEDDING MODELS
# ─────────────────────────────────────────────────────────────────────────────
elif section == "🔮 Embedding Models":
    st.title("🔮 Embedding Models — Satellite Embeddings")
    st.markdown("Real predictions from **MLP**, **XGBoost**, and **CatBoost** trained on "
                "64-dim **Google Satellite Embeddings** (GEDI MU as target).")

    summary = load_embedding_summary()

    if summary is None:
        st.error("❌ `embedding_models_summary_metrics.csv` not found in `/data/`.")
        st.stop()

    detail_tab, compare_tab, map_tab = st.tabs([
        "📊 Model Detail", "⚖ Model Comparison", "🗺 Spatial Map"
    ])

    with detail_tab:
        model_sel = st.selectbox("Model", ["MLP", "XGBoost", "CatBoost"], key="emb_model")
        df_test   = embedding_test(model_sel)
        color     = MODEL_COLORS[model_sel]

        if df_test is None:
            st.warning(f"⚠ File not found: `embedding_{model_sel.lower()}_test_predictions.csv`")
        else:
            st.markdown(f"#### {model_sel} — Test-Set Performance (Embeddings)")
            metrics_row(df_test)

            c1, c2 = st.columns(2)
            with c1:
                st.plotly_chart(
                    scatter_obs_pred(df_test, f"{model_sel} Embeddings", color),
                    use_container_width=True)
            with c2:
                st.plotly_chart(residual_plot(df_test, color), use_container_width=True)

            st.plotly_chart(residual_histogram(df_test, color), use_container_width=True)

            with st.expander("📋 Raw predictions (first 200 rows)"):
                st.dataframe(df_test.head(200), use_container_width=True)

    with compare_tab:
        st.markdown("#### All Embedding Models — Side-by-Side Metrics")
        st.dataframe(
            summary.sort_values("R2", ascending=False).reset_index(drop=True)
                   .rename(columns={"model":"Model"}),
            use_container_width=True
        )

        c1, c2 = st.columns(2)
        with c1:
            fig_r2 = go.Figure(go.Bar(
                x=summary["model"], y=summary["R2"],
                marker_color=[MODEL_COLORS.get(m,"#64748b") for m in summary["model"]],
                text=[f"{v:.4f}" for v in summary["R2"]], textposition="outside",
            ))
            fig_r2.update_layout(**PL, title="R² — Embedding Models", height=340, yaxis_title="R²")
            st.plotly_chart(fig_r2, use_container_width=True)

            fig_mae = go.Figure(go.Bar(
                x=summary["model"], y=summary["MAE"],
                marker_color=[MODEL_COLORS.get(m,"#64748b") for m in summary["model"]],
                text=[f"{v:.3f}" for v in summary["MAE"]], textposition="outside",
            ))
            fig_mae.update_layout(**PL, title="MAE — Embedding Models", height=340, yaxis_title="MAE (Mg/ha)")
            st.plotly_chart(fig_mae, use_container_width=True)

        with c2:
            fig_rmse = go.Figure(go.Bar(
                x=summary["model"], y=summary["RMSE"],
                marker_color=[MODEL_COLORS.get(m,"#64748b") for m in summary["model"]],
                text=[f"{v:.3f}" for v in summary["RMSE"]], textposition="outside",
            ))
            fig_rmse.update_layout(**PL, title="RMSE — Embedding Models", height=340, yaxis_title="RMSE (Mg/ha)")
            st.plotly_chart(fig_rmse, use_container_width=True)

            fig_bias = go.Figure(go.Bar(
                x=summary["model"], y=summary["Bias"],
                marker_color=[MODEL_COLORS.get(m,"#64748b") for m in summary["model"]],
                text=[f"{v:+.3f}" for v in summary["Bias"]], textposition="outside",
            ))
            fig_bias.add_hline(y=0, line_color="#ef4444", line_dash="dash")
            fig_bias.update_layout(**PL, title="Bias — Embedding Models", height=340, yaxis_title="Bias (Mg/ha)")
            st.plotly_chart(fig_bias, use_container_width=True)

        # Overlay scatter
        st.markdown("#### Observed vs Predicted — All Embedding Models")
        fig_all = go.Figure()
        for model in ["MLP", "XGBoost", "CatBoost"]:
            df_t = embedding_test(model)
            if df_t is not None:
                fig_all.add_trace(go.Scatter(
                    x=df_t["y_true"], y=df_t["y_pred"], mode="markers",
                    marker=dict(size=3, color=MODEL_COLORS[model], opacity=0.3),
                    name=model,
                ))
        lims = [0, 260]
        fig_all.add_trace(go.Scatter(x=lims, y=lims, mode="lines",
                                     line=dict(color="#ef4444", dash="dash", width=1.2),
                                     name="1:1"))
        fig_all.update_layout(**PL, height=440, xaxis_title="Observed GEDI MU (Mg/ha)",
                              yaxis_title="Predicted (Mg/ha)",
                              title="All Embedding Models — Obs vs Pred")
        st.plotly_chart(fig_all, use_container_width=True)

    with map_tab:
        st.markdown("#### Spatial Biomass Map — Embedding Predictions")
        map_model = st.selectbox("Model", ["MLP","XGBoost","CatBoost"], key="emb_map")
        df_full   = embedding_full(map_model)

        if df_full is None:
            st.warning(f"⚠ Full prediction file not found. Falling back to test-set points.")
            df_full = embedding_test(map_model)

        if df_full is not None:
            c1, c2 = st.columns(2)
            with c1:
                st.plotly_chart(
                    spatial_map(df_full, "y_true", "Observed GEDI MU Biomass"),
                    use_container_width=True)
            with c2:
                st.plotly_chart(
                    spatial_map(df_full, "y_pred", f"Predicted — {map_model} (Embeddings)"),
                    use_container_width=True)

            df_full = df_full.copy()
            df_full["diff"] = df_full["y_pred"] - df_full["y_true"]
            fig_diff = go.Figure(go.Scattergl(
                x=df_full["lon"], y=df_full["lat"], mode="markers",
                marker=dict(size=3, color=df_full["diff"],
                            colorscale="RdBu", cmid=0,
                            showscale=True, colorbar=dict(title="Diff (Mg/ha)")),
            ))
            fig_diff.update_layout(**PL, title="Difference Map (Pred − Obs)",
                                   xaxis_title="Longitude", yaxis_title="Latitude", height=420)
            st.plotly_chart(fig_diff, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — LIDAR
# ─────────────────────────────────────────────────────────────────────────────
elif section == "📡 LiDAR Analysis":
    st.title("📡 LiDAR Analysis — Individual Tree Segmentation")
    st.markdown("Real tree-level structural features from LiDAR point-cloud segmentation "
                "over the Veluwe, merged with satellite-derived biomass.")

    df_lid = load_lidar()

    if df_lid is None:
        st.error("❌ `merged_tree_level.csv` not found in `/data/`.")
        st.stop()

    LIDAR_FEATS = [
        "height_max_m","height_mean_m","height_std_m",
        "height_p25_m","height_p50_m","height_p75_m","height_p90_m","height_p95_m",
        "cv_height","height_range_m",
        "crown_radius_m","crown_r95_m","crown_area_m2",
        "canopy_pts","density_pt_m2","point_count",
    ]
    FEAT_LABELS = {
        "height_max_m":   "Max Height (m)",
        "height_mean_m":  "Mean Height (m)",
        "height_std_m":   "Height Std Dev (m)",
        "height_p25_m":   "Height P25 (m)",
        "height_p50_m":   "Height P50 (m)",
        "height_p75_m":   "Height P75 (m)",
        "height_p90_m":   "Height P90 (m)",
        "height_p95_m":   "Height P95 (m)",
        "cv_height":      "CV Height",
        "height_range_m": "Height Range P95−P25 (m)",
        "crown_radius_m": "Crown Radius (m)",
        "crown_r95_m":    "Crown R95 (m)",
        "crown_area_m2":  "Crown Area (m²)",
        "canopy_pts":     "Canopy Point Count",
        "density_pt_m2":  "Point Density (pt/m²)",
        "point_count":    "Total Point Count",
        "sat_biomass":    "Satellite Biomass (Mg/ha)",
        "sat_NDVI":       "NDVI",
        "sat_EVI":        "EVI",
        "sat_LAI":        "LAI",
        "sat_temp":       "Temperature (K)",
        "sat_pr":         "Precipitation",
    }
    TILE_COLORS = {"Tile 1":"#2dd4bf","Tile 2":"#86efac","Tile 3":"#fb923c"}

    st.info("💡 All data is cached after first load — subsequent tab switches are instant.")
    overview_tab, dist_tab, struct_tab, spatial_tab = st.tabs([
        "🌳 Overview", "📊 Distributions", "🌿 Structural Relationships", "🗺 Spatial Map"
    ])

    with overview_tab:
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Trees Segmented",      f"{len(df_lid):,}")
        c2.metric("Mean Max Height (m)",  f"{df_lid['height_max_m'].mean():.1f}")
        c3.metric("Mean Crown Area (m²)", f"{df_lid['crown_area_m2'].mean():.0f}")
        c4.metric("Mean Point Count",     f"{df_lid['point_count'].mean():.0f}")
        c5.metric("Mean Biomass (Mg/ha)", f"{df_lid['sat_biomass'].mean():.1f}")

        st.markdown("---")
        tile_counts = df_lid["tile_label"].value_counts().reset_index()
        tile_counts.columns = ["tile","count"]

        fig_pie = px.pie(tile_counts, names="tile", values="count",
                         color="tile", color_discrete_map=TILE_COLORS,
                         title="Trees per Tile")
        fig_pie.update_layout(**PL, height=340)
        fig_pie.update_traces(textposition="inside", textinfo="percent+label")

        fig_hbox = px.box(df_lid, x="tile_label", y="height_max_m",
                          color="tile_label", color_discrete_map=TILE_COLORS,
                          title="Max Height Distribution by Tile",
                          labels={"tile_label":"Tile","height_max_m":"Max Height (m)"})
        fig_hbox.update_layout(**PL, height=340, showlegend=False)

        c1, c2 = st.columns(2)
        with c1: st.plotly_chart(fig_pie,  use_container_width=True)
        with c2: st.plotly_chart(fig_hbox, use_container_width=True)

        with st.expander("📋 Per-tile Summary Statistics"):
            st.dataframe(
                df_lid.groupby("tile_label")[
                    ["height_max_m","height_mean_m","crown_area_m2","density_pt_m2","sat_biomass"]
                ].describe().round(2),
                use_container_width=True
            )

    with dist_tab:
        st.markdown("#### Feature Distributions")
        all_dist_feats = LIDAR_FEATS + ["sat_biomass","sat_NDVI","sat_EVI","sat_LAI"]
        c1, c2 = st.columns([2,1])
        with c1:
            feat_sel = st.selectbox("Feature", all_dist_feats,
                                    format_func=lambda x: FEAT_LABELS.get(x, x))
        with c2:
            split_tile = st.checkbox("Split by tile", value=True)

        if split_tile:
            fig_hist = px.histogram(df_lid, x=feat_sel, color="tile_label",
                                    nbins=70, barmode="overlay", opacity=0.7,
                                    color_discrete_map=TILE_COLORS,
                                    title=f"Distribution — {FEAT_LABELS.get(feat_sel, feat_sel)}",
                                    labels={feat_sel: FEAT_LABELS.get(feat_sel, feat_sel)})
        else:
            fig_hist = px.histogram(df_lid, x=feat_sel, nbins=70,
                                    color_discrete_sequence=["#2dd4bf"],
                                    title=f"Distribution — {FEAT_LABELS.get(feat_sel, feat_sel)}",
                                    labels={feat_sel: FEAT_LABELS.get(feat_sel, feat_sel)})
        fig_hist.update_layout(**PL, height=360)
        st.plotly_chart(fig_hist, use_container_width=True)

        fig_vio = px.violin(df_lid, y=feat_sel, x="tile_label",
                            color="tile_label", box=True, points=False,
                            color_discrete_map=TILE_COLORS,
                            title=f"{FEAT_LABELS.get(feat_sel, feat_sel)} — by Tile",
                            labels={"tile_label":"Tile", feat_sel: FEAT_LABELS.get(feat_sel, feat_sel)})
        fig_vio.update_layout(**PL, height=340, showlegend=False)
        st.plotly_chart(fig_vio, use_container_width=True)

    with struct_tab:
        st.markdown("#### LiDAR Feature Correlation Matrix")
        corr_cols = LIDAR_FEATS + ["sat_biomass"]
        corr_df = df_lid[corr_cols].rename(columns=FEAT_LABELS).corr().round(3)
        fig_corr = px.imshow(corr_df, text_auto=".2f",
                             color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                             aspect="auto",
                             title="Pearson Correlation — LiDAR Features vs Biomass")
        fig_corr.update_layout(**PL, height=600)
        st.plotly_chart(fig_corr, use_container_width=True)

        st.markdown("---")
        st.markdown("#### Scatter Explorer")
        all_feats = LIDAR_FEATS + ["sat_biomass"]
        cc1, cc2 = st.columns(2)
        with cc1:
            xf = st.selectbox("X axis", all_feats, index=0,
                              format_func=lambda x: FEAT_LABELS.get(x, x), key="lx")
        with cc2:
            yf = st.selectbox("Y axis", all_feats, index=12,
                              format_func=lambda x: FEAT_LABELS.get(x, x), key="ly")

        fig_sc = px.scatter(df_lid.sample(min(5000,len(df_lid)),random_state=42), x=xf, y=yf, color="tile_label",
                            opacity=0.4, color_discrete_map=TILE_COLORS,
                            marginal_x="histogram", marginal_y="box",
                            title=f"{FEAT_LABELS.get(xf,xf)} vs {FEAT_LABELS.get(yf,yf)}",
                            labels={xf: FEAT_LABELS.get(xf,xf), yf: FEAT_LABELS.get(yf,yf),
                                    "tile_label":"Tile"})
        fig_sc.update_traces(marker_size=3, selector=dict(type="scatter"))
        fig_sc.update_layout(**PL, height=520)
        st.plotly_chart(fig_sc, use_container_width=True)

        st.markdown("#### 3D — Height × Crown Area × Biomass")
        sample = df_lid.sample(min(3000, len(df_lid)), random_state=42)
        fig3d = px.scatter_3d(
            sample, x="height_max_m", y="crown_area_m2", z="sat_biomass",
            color="tile_label", opacity=0.5, color_discrete_map=TILE_COLORS,
            labels={"height_max_m":"Max Height (m)", "crown_area_m2":"Crown Area (m²)",
                    "sat_biomass":"Biomass (Mg/ha)", "tile_label":"Tile"},
            title="3D — Tree Height × Crown Area × Satellite Biomass",
        )
        fig3d.update_traces(marker_size=2)
        fig3d.update_layout(**PL, height=540,
                            scene=dict(bgcolor="rgba(11,17,23,0.8)",
                                       xaxis_title="Max Height (m)",
                                       yaxis_title="Crown Area (m²)",
                                       zaxis_title="Biomass (Mg/ha)"))
        st.plotly_chart(fig3d, use_container_width=True)

    with spatial_tab:
        st.markdown("#### Spatial Distribution of Trees")
        map_col = st.selectbox(
            "Colour trees by",
            ["height_max_m","crown_area_m2","density_pt_m2","cv_height","sat_biomass","tile_label"],
            format_func=lambda x: FEAT_LABELS.get(x, x),
        )

        # Use Scattergl (WebGL) for 19k points — much faster than SVG scatter
        _lid_map = df_lid  # already cached, no copy needed
        if map_col == "tile_label":
            fig_map = go.Figure()
            for tname, tcolor in TILE_COLORS.items():
                sub = _lid_map[_lid_map["tile_label"] == tname]
                fig_map.add_trace(go.Scattergl(
                    x=sub["tree_lon"], y=sub["tree_lat"], mode="markers",
                    marker=dict(size=3, color=tcolor, opacity=0.6),
                    name=tname,
                ))
            fig_map.update_layout(**PL, height=520,
                                  title="Tree Locations by Tile",
                                  xaxis_title="Longitude", yaxis_title="Latitude")
        else:
            fig_map = go.Figure(go.Scattergl(
                x=_lid_map["tree_lon"], y=_lid_map["tree_lat"], mode="markers",
                marker=dict(size=3, color=_lid_map[map_col],
                            colorscale="Viridis", opacity=0.6,
                            showscale=True,
                            colorbar=dict(title=FEAT_LABELS.get(map_col, map_col))),
            ))
            fig_map.update_layout(**PL, height=520,
                                  title=f"Tree Locations — {FEAT_LABELS.get(map_col, map_col)}",
                                  xaxis_title="Longitude", yaxis_title="Latitude")
        st.plotly_chart(fig_map, use_container_width=True)

        fig_dist = px.histogram(df_lid, x="dist_to_px_m", color="tile_label",
                                nbins=60, barmode="overlay", opacity=0.7,
                                color_discrete_map=TILE_COLORS,
                                title="Distance: Tree centroid → Satellite pixel (m)",
                                labels={"dist_to_px_m":"Distance (m)","tile_label":"Tile"})
        fig_dist.update_layout(**PL, height=320)
        st.plotly_chart(fig_dist, use_container_width=True)
