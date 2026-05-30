"""
ForestBiomass Intelligence Dashboard
=====================================
Sections:
  1. Spectral Models      — MLP / XGBoost / CatBoost on spectral indices (± CHM)
  2. Embedding Models     — MLP / XGBoost / CatBoost on Google Satellite Embeddings
  3. LiDAR Analysis       — Individual-tree segmentation features & distributions

Run: streamlit run dashboard.py
"""

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Forest Biomass Intelligence",
    page_icon="🌲",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS  (dark forest-sci aesthetic)
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;600;700&display=swap');

/* Root palette */
:root {
    --bg:        #0b1117;
    --panel:     #131c26;
    --accent1:   #2dd4bf;   /* teal */
    --accent2:   #86efac;   /* pale green */
    --accent3:   #fb923c;   /* amber */
    --accent4:   #a78bfa;   /* violet */
    --text:      #e2e8f0;
    --muted:     #64748b;
    --border:    #1e293b;
}

html, body, [data-testid="stAppViewContainer"] {
    background-color: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'DM Sans', sans-serif;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: var(--panel) !important;
    border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] * { color: var(--text) !important; }

/* Headers */
h1, h2, h3, h4 { font-family: 'Space Mono', monospace !important; }
h1 { color: var(--accent1) !important; letter-spacing: -1px; }
h2 { color: var(--accent2) !important; }
h3 { color: var(--accent1) !important; font-size: 1rem !important; }

/* Metric cards */
[data-testid="metric-container"] {
    background: var(--panel) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    padding: 14px 18px !important;
}
[data-testid="metric-container"] label { color: var(--muted) !important; font-size: 0.75rem !important; text-transform: uppercase; letter-spacing: 1px; }
[data-testid="metric-container"] [data-testid="stMetricValue"] { color: var(--accent1) !important; font-family: 'Space Mono', monospace !important; }

/* Tabs */
[data-baseweb="tab-list"] { background: var(--panel) !important; border-radius: 8px; padding: 4px; }
[data-baseweb="tab"] { color: var(--muted) !important; font-family: 'Space Mono', monospace !important; font-size: 0.8rem !important; }
[aria-selected="true"][data-baseweb="tab"] { color: var(--accent1) !important; background: var(--bg) !important; border-radius: 6px !important; }

/* Selectbox / slider */
[data-baseweb="select"] > div { background: var(--panel) !important; border-color: var(--border) !important; color: var(--text) !important; }

/* Buttons */
[data-testid="baseButton-secondary"] { background: var(--panel) !important; border: 1px solid var(--accent1) !important; color: var(--accent1) !important; font-family: 'Space Mono', monospace !important; }

/* Divider colour */
hr { border-color: var(--border) !important; }

/* Expander */
[data-testid="stExpander"] { background: var(--panel) !important; border: 1px solid var(--border) !important; border-radius: 8px !important; }

/* Info / warning boxes */
[data-testid="stInfo"] { background: rgba(45,212,191,0.08) !important; border-left: 3px solid var(--accent1) !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# PLOTLY TEMPLATE
# ─────────────────────────────────────────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(11,17,23,0.6)",
    font=dict(family="DM Sans", color="#e2e8f0", size=12),
    margin=dict(l=40, r=20, t=50, b=40),
    colorway=["#2dd4bf","#86efac","#fb923c","#a78bfa","#38bdf8","#f472b6"],
)

# ─────────────────────────────────────────────────────────────────────────────
# DATA GENERATION  (realistic synthetic data matching Veluwe / Netherlands)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data
def generate_spectral_data(n=3000, seed=42):
    rng = np.random.default_rng(seed)
    # Geo  — Veluwe bounding box
    lon = rng.uniform(5.60, 6.10, n)
    lat = rng.uniform(51.90, 52.35, n)
    # Indices
    ndvi = rng.beta(6, 2, n) * 0.9 + 0.05
    evi  = ndvi * 0.7 + rng.normal(0, 0.05, n)
    lai  = rng.gamma(3, 1, n)
    temp = rng.normal(10.5, 2.5, n)
    pr   = rng.gamma(4, 20, n)
    chm  = np.abs(rng.normal(12, 8, n))
    # Biomass  (spectral + CHM contribution)
    biomass = (
        30 + 60*ndvi + 40*evi + 5*lai - 0.5*temp + 0.05*pr
        + 2.5*chm + rng.normal(0, 10, n)
    ).clip(0, 250)
    df = pd.DataFrame(dict(lon=lon, lat=lat,
                           NDVI=ndvi, EVI=evi, LAI=lai,
                           temp=temp, pr=pr, CHM=chm,
                           biomass=biomass))
    return df

@st.cache_data
def generate_embedding_data(n=2500, n_emb=64, seed=99):
    rng = np.random.default_rng(seed)
    lon = rng.uniform(5.60, 6.10, n)
    lat = rng.uniform(51.90, 52.35, n)
    # 64-dim satellite embeddings (A1..A64)
    emb = rng.standard_normal((n, n_emb)) * 0.5
    cols = [f"A{i:02d}" for i in range(1, n_emb+1)]
    # Biomass driven by a sparse combination of embeddings
    weights = rng.normal(0, 1, n_emb)
    weights[rng.integers(0, n_emb, 40)] = 0   # sparse
    mu = (emb @ weights) * 3 + 90 + rng.normal(0, 15, n)
    mu = mu.clip(0, 250)
    df = pd.DataFrame(dict(lon=lon, lat=lat, **dict(zip(cols, emb.T)), MU=mu))
    return df

@st.cache_data
def generate_lidar_data(n=1500, seed=7):
    rng = np.random.default_rng(seed)
    species = rng.choice(["Pine","Oak","Birch","Douglas Fir","Beech"], n,
                         p=[0.35,0.25,0.15,0.15,0.10])
    height  = np.where(species=="Pine",
                       rng.gamma(8,2,n),
                       np.where(species=="Oak",
                                rng.gamma(6,2.5,n),
                                rng.gamma(5,2,n))).clip(1,35)
    crown_area = np.pi * (rng.gamma(3,1,n))**2
    volume = crown_area * height * 0.35
    density = rng.gamma(4, 50, n)       # stems / ha
    z_mean  = height * 0.6 + rng.normal(0, 0.5, n)
    z_std   = height * 0.15 + rng.normal(0, 0.3, n).clip(0)
    z_p95   = height * 0.95 + rng.normal(0, 0.5, n)
    lai_lid = rng.gamma(3, 0.8, n)
    cover   = (1 - np.exp(-0.5 * lai_lid)).clip(0, 1)
    biomass = (0.5*volume + 0.3*height*crown_area*0.1 + rng.normal(0,8,n)).clip(0, 250)
    df = pd.DataFrame(dict(
        species=species,
        height=height, crown_area=crown_area,
        volume=volume, density=density,
        z_mean=z_mean, z_std=z_std, z_p95=z_p95,
        lai=lai_lid, cover=cover, biomass=biomass,
    ))
    return df

# ─────────────────────────────────────────────────────────────────────────────
# MODEL FITTING HELPERS
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource
def fit_spectral_models(use_chm: bool):
    from xgboost import XGBRegressor
    from catboost import CatBoostRegressor
    from sklearn.neural_network import MLPRegressor

    df = generate_spectral_data()
    feat_cols = ['NDVI','EVI','LAI','temp','pr']
    if use_chm:
        feat_cols = feat_cols + ['CHM']

    X = df[feat_cols].values
    y = df['biomass'].values
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s  = scaler.transform(X_te)

    results = {}

    # MLP
    mlp = MLPRegressor(hidden_layer_sizes=(64,32,16), max_iter=400,
                       random_state=42, early_stopping=True, validation_fraction=0.15)
    mlp.fit(X_tr_s, y_tr)
    yp = mlp.predict(X_te_s)
    results['MLP'] = dict(y_true=y_te, y_pred=yp,
                          feat_imp=np.abs(mlp.coefs_[0]).mean(axis=1),
                          feat_names=feat_cols)

    # XGBoost
    xgb = XGBRegressor(n_estimators=400, learning_rate=0.05, max_depth=5,
                       subsample=0.8, colsample_bytree=0.7, random_state=42,
                       verbosity=0)
    xgb.fit(X_tr, y_tr)
    yp = xgb.predict(X_te)
    results['XGBoost'] = dict(y_true=y_te, y_pred=yp,
                              feat_imp=xgb.feature_importances_,
                              feat_names=feat_cols)

    # CatBoost
    cb = CatBoostRegressor(iterations=400, learning_rate=0.05, depth=6,
                           loss_function='RMSE', random_seed=42, verbose=0)
    cb.fit(X_tr, y_tr, eval_set=(X_te, y_te))
    yp = cb.predict(X_te)
    results['CatBoost'] = dict(y_true=y_te, y_pred=yp,
                               feat_imp=cb.get_feature_importance(),
                               feat_names=feat_cols)

    geo = df[['lon','lat','biomass']].iloc[y_te.argsort()[::-1]].reset_index(drop=True)
    return results, geo, feat_cols

@st.cache_resource
def fit_embedding_models():
    from xgboost import XGBRegressor
    from catboost import CatBoostRegressor
    from sklearn.neural_network import MLPRegressor

    df = generate_embedding_data()
    emb_cols = [c for c in df.columns if c.startswith('A')]
    X = df[emb_cols].values
    y = df['MU'].values
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s  = scaler.transform(X_te)

    results = {}

    mlp = MLPRegressor(hidden_layer_sizes=(128,64,32), max_iter=300,
                       random_state=42, early_stopping=True)
    mlp.fit(X_tr_s, y_tr)
    yp = mlp.predict(X_te_s)
    imp = np.abs(mlp.coefs_[0]).mean(axis=1)
    results['MLP'] = dict(y_true=y_te, y_pred=yp,
                          feat_imp=imp, feat_names=emb_cols,
                          loss_curve=mlp.loss_curve_)

    xgb = XGBRegressor(n_estimators=400, learning_rate=0.05, max_depth=4,
                       subsample=0.8, colsample_bytree=0.5, random_state=42,
                       verbosity=0)
    xgb.fit(X_tr, y_tr)
    yp = xgb.predict(X_te)
    results['XGBoost'] = dict(y_true=y_te, y_pred=yp,
                              feat_imp=xgb.feature_importances_,
                              feat_names=emb_cols)

    cb = CatBoostRegressor(iterations=400, learning_rate=0.05, depth=5,
                           loss_function='RMSE', random_seed=42, verbose=0)
    cb.fit(X_tr, y_tr, eval_set=(X_te, y_te))
    yp = cb.predict(X_te)
    results['CatBoost'] = dict(y_true=y_te, y_pred=yp,
                               feat_imp=cb.get_feature_importance(),
                               feat_names=emb_cols)

    return results, df

# ─────────────────────────────────────────────────────────────────────────────
# SHARED CHART HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def metrics_row(y_true, y_pred):
    r2   = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    bias = float(np.mean(y_pred - y_true))
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("R²", f"{r2:.4f}")
    c2.metric("RMSE", f"{rmse:.2f} Mg/ha")
    c3.metric("MAE", f"{mae:.2f} Mg/ha")
    c4.metric("Bias", f"{bias:+.2f} Mg/ha")

def scatter_obs_pred(y_true, y_pred, title, color):
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    lims = [min(y_true.min(), y_pred.min())-5, max(y_true.max(), y_pred.max())+5]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=y_true, y=y_pred, mode='markers',
                             marker=dict(size=4, color=color, opacity=0.5),
                             name='Samples'))
    fig.add_trace(go.Scatter(x=lims, y=lims, mode='lines',
                             line=dict(color='#ef4444', dash='dash', width=1.5),
                             name='1:1 line'))
    fig.update_layout(**PLOTLY_LAYOUT,
                      title=f"{title} — R²={r2:.3f} | RMSE={rmse:.2f}",
                      xaxis_title="Observed (Mg/ha)",
                      yaxis_title="Predicted (Mg/ha)",
                      height=380)
    return fig

def residual_plot(y_true, y_pred, color):
    res = y_pred - y_true
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=y_true, y=res, mode='markers',
                             marker=dict(size=4, color=color, opacity=0.4), name='Residuals'))
    fig.add_hline(y=0, line_color="#ef4444", line_dash="dash")
    fig.update_layout(**PLOTLY_LAYOUT,
                      title="Residual Plot",
                      xaxis_title="Observed (Mg/ha)", yaxis_title="Residual (Mg/ha)",
                      height=300)
    return fig

def feat_importance_chart(imp, names, title, color, top_n=10):
    idx = np.argsort(imp)[-top_n:]
    fig = go.Figure(go.Bar(
        x=imp[idx], y=[names[i] for i in idx],
        orientation='h',
        marker_color=color,
    ))
    fig.update_layout(**PLOTLY_LAYOUT, title=title, height=350,
                      xaxis_title="Importance", yaxis_title="")
    return fig

def comparison_bar(results, metric_fn, metric_name, unit=""):
    vals = {k: metric_fn(v['y_true'], v['y_pred']) for k, v in results.items()}
    fig = go.Figure(go.Bar(
        x=list(vals.keys()), y=list(vals.values()),
        marker_color=["#2dd4bf","#86efac","#fb923c"],
        text=[f"{v:.3f}{unit}" for v in vals.values()],
        textposition='outside',
    ))
    fig.update_layout(**PLOTLY_LAYOUT, title=f"Model Comparison — {metric_name}",
                      height=320, yaxis_title=metric_name)
    return fig

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🌲 ForestBiomass\n### Intelligence Dashboard")
    st.markdown("---")
    section = st.radio("Navigation",
                       ["🛰 Spectral Models", "🔮 Embedding Models", "📡 LiDAR Analysis"],
                       index=0)
    st.markdown("---")
    st.markdown("""
**Study area:** Veluwe, Netherlands  
**Target:** Above-Ground Biomass (Mg/ha)  
**Source:** ESA CCI AGB + GEDI MU

---
**Spectral features:**  
NDVI · EVI · LAI · Temp · Precip (± CHM)

**Embedding features:**  
64-dim Google Satellite Embeddings

**LiDAR features:**  
Individual tree segmentation  
(height · crown · volume · density)
""")
    st.markdown("---")
    st.caption("Built with Streamlit · Plotly · XGBoost · CatBoost · MLP")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — SPECTRAL MODELS
# ─────────────────────────────────────────────────────────────────────────────
if section == "🛰 Spectral Models":
    st.title("🛰 Spectral Models — Biomass Prediction")
    st.markdown("Biomass prediction using **Sentinel-2 spectral indices** (NDVI, EVI, LAI) + climate variables, with optional **CHM** (Canopy Height Model) integration.")

    col_opt, _ = st.columns([1, 3])
    with col_opt:
        use_chm = st.toggle("Include CHM features", value=True)

    with st.spinner("Training MLP / XGBoost / CatBoost …"):
        results, geo_df, feat_cols = fit_spectral_models(use_chm)

    # ── Model selector tabs ──
    model_tab, compare_tab, map_tab, corr_tab = st.tabs(
        ["📊 Model Detail", "⚖ Model Comparison", "🗺 Spatial Map", "🔗 Feature Correlation"])

    with model_tab:
        MODEL_COLORS = {"MLP": "#2dd4bf", "XGBoost": "#86efac", "CatBoost": "#fb923c"}
        model_sel = st.selectbox("Select Model", list(results.keys()))
        res = results[model_sel]
        color = MODEL_COLORS[model_sel]

        st.markdown(f"#### {model_sel} — Test-Set Performance")
        metrics_row(res['y_true'], res['y_pred'])

        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(scatter_obs_pred(res['y_true'], res['y_pred'], model_sel, color),
                            use_container_width=True)
        with c2:
            imp = np.array(res['feat_imp'])
            names = res['feat_names']
            st.plotly_chart(feat_importance_chart(imp, names, "Feature Importance", color),
                            use_container_width=True)

        st.plotly_chart(residual_plot(res['y_true'], res['y_pred'], color),
                        use_container_width=True)

    with compare_tab:
        st.markdown("#### Head-to-Head Model Comparison")
        m1, m2 = st.columns(2)
        with m1:
            st.plotly_chart(comparison_bar(results, r2_score, "R²"), use_container_width=True)
            st.plotly_chart(comparison_bar(results, mean_absolute_error, "MAE", " Mg/ha"),
                            use_container_width=True)
        with m2:
            st.plotly_chart(comparison_bar(
                results, lambda yt, yp: np.sqrt(mean_squared_error(yt, yp)), "RMSE", " Mg/ha"),
                use_container_width=True)
            # Scatter overlay all models
            fig_all = go.Figure()
            for mname, c in MODEL_COLORS.items():
                r = results[mname]
                fig_all.add_trace(go.Scatter(
                    x=r['y_true'], y=r['y_pred'], mode='markers',
                    marker=dict(size=3, color=c, opacity=0.35), name=mname))
            lims = [0, 260]
            fig_all.add_trace(go.Scatter(x=lims, y=lims, mode='lines',
                                         line=dict(color='#ef4444', dash='dash', width=1),
                                         name='1:1', showlegend=False))
            fig_all.update_layout(**PLOTLY_LAYOUT, title="All Models — Obs vs Pred",
                                  xaxis_title="Observed", yaxis_title="Predicted", height=320)
            st.plotly_chart(fig_all, use_container_width=True)

        # CHM impact table
        if use_chm:
            with st.expander("📈 CHM Impact on Model Performance"):
                res_no, _, _ = fit_spectral_models(False)
                res_yes, _, _ = fit_spectral_models(True)
                rows = []
                for m in results:
                    r2_n = r2_score(res_no[m]['y_true'], res_no[m]['y_pred'])
                    r2_y = r2_score(res_yes[m]['y_true'], res_yes[m]['y_pred'])
                    rows.append(dict(Model=m,
                                     **{"R² w/o CHM": round(r2_n,4),
                                        "R² with CHM": round(r2_y,4),
                                        "ΔR²": round(r2_y-r2_n,4)}))
                st.dataframe(pd.DataFrame(rows).set_index("Model"), use_container_width=True)

    with map_tab:
        st.markdown("#### Spatial Distribution of Predicted Biomass")
        map_model = st.selectbox("Model for map", list(results.keys()), key="map_sel")
        df_spec = generate_spectral_data()
        feat_c = ['NDVI','EVI','LAI','temp','pr'] + (['CHM'] if use_chm else [])
        from xgboost import XGBRegressor
        _x = df_spec[feat_c].values
        _y = df_spec['biomass'].values
        _xtr, _xte, _ytr, _yte = train_test_split(_x, _y, test_size=0.2, random_state=42)
        _xgb = XGBRegressor(n_estimators=200, verbosity=0, random_state=42)
        _xgb.fit(_xtr, _ytr)
        df_spec['pred'] = _xgb.predict(_x).clip(0)

        fig_map = make_subplots(rows=1, cols=2,
                                subplot_titles=["Observed Biomass", "Predicted Biomass"])
        fig_map.add_trace(go.Scattergl(x=df_spec['lon'], y=df_spec['lat'],
                                       mode='markers',
                                       marker=dict(size=3, color=df_spec['biomass'],
                                                   colorscale='Viridis', showscale=True,
                                                   colorbar=dict(x=0.45, title="Mg/ha")),
                                       name="Observed"), row=1, col=1)
        fig_map.add_trace(go.Scattergl(x=df_spec['lon'], y=df_spec['lat'],
                                       mode='markers',
                                       marker=dict(size=3, color=df_spec['pred'],
                                                   colorscale='Viridis', showscale=True,
                                                   colorbar=dict(x=1.0, title="Mg/ha")),
                                       name="Predicted"), row=1, col=2)
        fig_map.update_layout(**PLOTLY_LAYOUT, height=460,
                              xaxis_title="Longitude", yaxis_title="Latitude",
                              xaxis2_title="Longitude")
        st.plotly_chart(fig_map, use_container_width=True)

    with corr_tab:
        st.markdown("#### Feature Correlation Matrix")
        df_spec = generate_spectral_data()
        corr_cols = ['NDVI','EVI','LAI','temp','pr','CHM','biomass']
        corr = df_spec[corr_cols].corr().round(3)
        fig_corr = px.imshow(corr, text_auto=True, color_continuous_scale='RdBu_r',
                             zmin=-1, zmax=1, aspect='auto')
        fig_corr.update_layout(**PLOTLY_LAYOUT, height=460,
                               title="Pearson Correlation — Spectral Features")
        st.plotly_chart(fig_corr, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — EMBEDDING MODELS
# ─────────────────────────────────────────────────────────────────────────────
elif section == "🔮 Embedding Models":
    st.title("🔮 Embedding Models — Satellite Embeddings")
    st.markdown("Biomass prediction using **64-dim Google Satellite Embeddings** (GOOGLE/SATELLITE_EMBEDDING/V1) as input features with GEDI MU as ground truth.")

    with st.spinner("Training models on embeddings …"):
        results, df_emb = fit_embedding_models()

    MODEL_COLORS = {"MLP": "#a78bfa", "XGBoost": "#38bdf8", "CatBoost": "#f472b6"}

    detail_tab, compare_tab, umap_tab = st.tabs(
        ["📊 Model Detail", "⚖ Model Comparison", "🌐 Embedding Space"])

    with detail_tab:
        model_sel = st.selectbox("Select Model", list(results.keys()), key="emb_model")
        res = results[model_sel]
        color = MODEL_COLORS[model_sel]

        st.markdown(f"#### {model_sel} — Test-Set Performance (Embeddings)")
        metrics_row(res['y_true'], res['y_pred'])

        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(scatter_obs_pred(res['y_true'], res['y_pred'], model_sel, color),
                            use_container_width=True)
        with c2:
            imp = np.array(res['feat_imp'])
            names = res['feat_names']
            st.plotly_chart(feat_importance_chart(imp, names, "Top Embedding Dims", color, top_n=15),
                            use_container_width=True)

        st.plotly_chart(residual_plot(res['y_true'], res['y_pred'], color),
                        use_container_width=True)

        if model_sel == 'MLP' and 'loss_curve' in res:
            loss_fig = go.Figure()
            loss_fig.add_trace(go.Scatter(
                y=res['loss_curve'], mode='lines',
                line=dict(color="#a78bfa", width=2), name='Train Loss'))
            loss_fig.update_layout(**PLOTLY_LAYOUT, title="MLP Training Loss Curve",
                                   xaxis_title="Epoch", yaxis_title="MSE Loss", height=300)
            st.plotly_chart(loss_fig, use_container_width=True)

    with compare_tab:
        st.markdown("#### Head-to-Head Model Comparison — Embeddings")
        m1, m2 = st.columns(2)
        with m1:
            st.plotly_chart(comparison_bar(results, r2_score, "R²"), use_container_width=True)
        with m2:
            st.plotly_chart(comparison_bar(
                results, lambda yt, yp: np.sqrt(mean_squared_error(yt, yp)), "RMSE", " Mg/ha"),
                use_container_width=True)

        # Overlay scatter
        fig_all = go.Figure()
        for mname, c in MODEL_COLORS.items():
            r = results[mname]
            fig_all.add_trace(go.Scatter(x=r['y_true'], y=r['y_pred'], mode='markers',
                                         marker=dict(size=3, color=c, opacity=0.35), name=mname))
        lims = [0, 260]
        fig_all.add_trace(go.Scatter(x=lims, y=lims, mode='lines',
                                     line=dict(color='#ef4444', dash='dash', width=1),
                                     showlegend=False))
        fig_all.update_layout(**PLOTLY_LAYOUT, title="All Models — Obs vs Pred (Embeddings)",
                              height=360)
        st.plotly_chart(fig_all, use_container_width=True)

        # Summary table
        rows = []
        for m in results:
            r = results[m]
            rows.append({
                "Model": m,
                "R²": round(r2_score(r['y_true'], r['y_pred']),4),
                "RMSE": round(np.sqrt(mean_squared_error(r['y_true'], r['y_pred'])),3),
                "MAE": round(mean_absolute_error(r['y_true'], r['y_pred']),3),
                "Bias": round(float(np.mean(r['y_pred']-r['y_true'])),3),
            })
        st.dataframe(pd.DataFrame(rows).set_index("Model"), use_container_width=True)

    with umap_tab:
        st.markdown("#### Embedding Distribution (First 2 Principal Components)")
        emb_cols = [c for c in df_emb.columns if c.startswith('A')]
        pca_data = df_emb[emb_cols].values
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2, random_state=42)
        pc = pca.fit_transform(pca_data)
        fig_pca = px.scatter(x=pc[:,0], y=pc[:,1],
                             color=df_emb['MU'],
                             color_continuous_scale='Plasma',
                             labels={"x":"PC1","y":"PC2","color":"GEDI MU (Mg/ha)"},
                             opacity=0.5, title="PCA of Satellite Embeddings (coloured by Biomass)")
        fig_pca.update_traces(marker_size=3)
        fig_pca.update_layout(**PLOTLY_LAYOUT, height=480,
                              coloraxis_colorbar=dict(title="Biomass"))
        st.plotly_chart(fig_pca, use_container_width=True)

        # Spatial map
        fig_spat = go.Scattergl
        fig_map = go.Figure(go.Scattergl(
            x=df_emb['lon'], y=df_emb['lat'],
            mode='markers',
            marker=dict(size=3, color=df_emb['MU'], colorscale='Plasma',
                        showscale=True, colorbar=dict(title="GEDI MU")),
        ))
        fig_map.update_layout(**PLOTLY_LAYOUT, title="Spatial Distribution — GEDI MU",
                              xaxis_title="Lon", yaxis_title="Lat", height=380)
        st.plotly_chart(fig_map, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — LIDAR  (real data: merged_tree_level.csv)
# ─────────────────────────────────────────────────────────────────────────────
elif section == "📡 LiDAR Analysis":
    st.title("📡 LiDAR Analysis — Individual Tree Segmentation")
    st.markdown(
        "Real tree-level structural features from **LiDAR point-cloud segmentation** "
        "(individual-tree delineation) over the Veluwe, merged with satellite-derived biomass."
    )

    # ── Load real data ──────────────────────────────────────────────────────
    @st.cache_data
    def load_lidar():
        import os
        base = os.path.dirname(os.path.abspath(__file__))
        df = pd.read_csv(os.path.join(base, "merged_tree_level.csv"))
        # Derived features
        df["crown_area_m2_log"] = np.log1p(df["crown_area_m2"])
        df["height_range_m"]    = df["height_p95_m"] - df["height_p25_m"]
        df["tile_label"]        = df["tile"].map({1: "Tile 1", 2: "Tile 2", 3: "Tile 3"})
        return df

    df_lid = load_lidar()

    LIDAR_FEATS = [
        "height_max_m", "height_mean_m", "height_std_m",
        "height_p25_m", "height_p50_m", "height_p75_m", "height_p90_m", "height_p95_m",
        "cv_height", "height_range_m",
        "crown_radius_m", "crown_r95_m", "crown_area_m2",
        "canopy_pts", "density_pt_m2", "point_count",
    ]
    SAT_FEATS   = ["sat_NDVI", "sat_EVI", "sat_LAI", "sat_temp", "sat_pr"]
    TARGET      = "sat_biomass"

    TILE_COLORS = {"Tile 1": "#2dd4bf", "Tile 2": "#86efac", "Tile 3": "#fb923c"}

    # ── Tabs ────────────────────────────────────────────────────────────────
    overview_tab, dist_tab, struct_tab, spatial_tab, model_tab = st.tabs([
        "🌳 Overview", "📊 Distributions", "🌿 Structural Relationships",
        "🗺 Spatial Map", "🔬 Biomass Model",
    ])

    # ── OVERVIEW ────────────────────────────────────────────────────────────
    with overview_tab:
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Trees Segmented",      f"{len(df_lid):,}")
        c2.metric("Mean Height (m)",      f"{df_lid['height_max_m'].mean():.1f}")
        c3.metric("Mean Crown Area (m²)", f"{df_lid['crown_area_m2'].mean():.0f}")
        c4.metric("Mean Point Count",     f"{df_lid['point_count'].mean():.0f}")
        c5.metric("Mean Biomass (Mg/ha)", f"{df_lid[TARGET].mean():.1f}")

        st.markdown("---")

        # Tile breakdown
        tile_counts = df_lid["tile_label"].value_counts().reset_index()
        tile_counts.columns = ["tile", "count"]
        fig_pie = px.pie(tile_counts, names="tile", values="count",
                         color="tile", color_discrete_map=TILE_COLORS,
                         title="Trees per Tile")
        fig_pie.update_layout(**PLOTLY_LAYOUT, height=340)
        fig_pie.update_traces(textposition="inside", textinfo="percent+label")

        fig_htile = px.box(df_lid, x="tile_label", y="height_max_m",
                           color="tile_label", color_discrete_map=TILE_COLORS,
                           title="Height Distribution by Tile",
                           labels={"tile_label": "Tile", "height_max_m": "Max Height (m)"})
        fig_htile.update_layout(**PLOTLY_LAYOUT, height=340, showlegend=False)

        c1, c2 = st.columns(2)
        with c1: st.plotly_chart(fig_pie, use_container_width=True)
        with c2: st.plotly_chart(fig_htile, use_container_width=True)

        with st.expander("📋 Per-tile Summary Statistics"):
            st.dataframe(
                df_lid.groupby("tile_label")[
                    ["height_max_m","height_mean_m","crown_area_m2","density_pt_m2","sat_biomass"]
                ].describe().round(2),
                use_container_width=True
            )

    # ── DISTRIBUTIONS ───────────────────────────────────────────────────────
    with dist_tab:
        st.markdown("#### Feature Distributions")

        feat_labels = {
            "height_max_m":    "Max Height (m)",
            "height_mean_m":   "Mean Height (m)",
            "height_std_m":    "Height Std Dev (m)",
            "height_p25_m":    "Height P25 (m)",
            "height_p50_m":    "Height P50 (m)",
            "height_p75_m":    "Height P75 (m)",
            "height_p90_m":    "Height P90 (m)",
            "height_p95_m":    "Height P95 (m)",
            "cv_height":       "CV Height",
            "height_range_m":  "Height Range P95−P25 (m)",
            "crown_radius_m":  "Crown Radius (m)",
            "crown_r95_m":     "Crown R95 (m)",
            "crown_area_m2":   "Crown Area (m²)",
            "canopy_pts":      "Canopy Point Count",
            "density_pt_m2":   "Point Density (pt/m²)",
            "sat_biomass":     "Satellite Biomass (Mg/ha)",
        }

        col_sel, col_tile = st.columns([2, 1])
        with col_sel:
            feat_sel = st.selectbox("Feature", list(feat_labels.keys()),
                                    format_func=lambda x: feat_labels[x])
        with col_tile:
            split_tile = st.checkbox("Split by tile", value=True)

        if split_tile:
            fig_hist = px.histogram(df_lid, x=feat_sel, color="tile_label",
                                    nbins=70, barmode="overlay", opacity=0.7,
                                    color_discrete_map=TILE_COLORS,
                                    title=f"Distribution — {feat_labels[feat_sel]}",
                                    labels={feat_sel: feat_labels[feat_sel]})
        else:
            fig_hist = px.histogram(df_lid, x=feat_sel, nbins=70,
                                    color_discrete_sequence=["#2dd4bf"],
                                    title=f"Distribution — {feat_labels[feat_sel]}",
                                    labels={feat_sel: feat_labels[feat_sel]})
        fig_hist.update_layout(**PLOTLY_LAYOUT, height=360)
        st.plotly_chart(fig_hist, use_container_width=True)

        # Violin per tile
        fig_vio = px.violin(df_lid, y=feat_sel, x="tile_label",
                            color="tile_label", box=True, points=False,
                            color_discrete_map=TILE_COLORS,
                            title=f"{feat_labels[feat_sel]} — by Tile",
                            labels={"tile_label": "Tile", feat_sel: feat_labels[feat_sel]})
        fig_vio.update_layout(**PLOTLY_LAYOUT, height=340, showlegend=False)
        st.plotly_chart(fig_vio, use_container_width=True)

    # ── STRUCTURAL RELATIONSHIPS ─────────────────────────────────────────────
    with struct_tab:
        st.markdown("#### LiDAR Feature Correlation Matrix")
        corr_cols = LIDAR_FEATS + ["sat_biomass"]
        corr_labels = {c: feat_labels.get(c, c) for c in corr_cols}
        corr_df = df_lid[corr_cols].rename(columns=corr_labels).corr().round(3)
        fig_corr = px.imshow(corr_df, text_auto=".2f",
                             color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                             aspect="auto",
                             title="Pearson Correlation — LiDAR Features vs Biomass")
        fig_corr.update_layout(**PLOTLY_LAYOUT, height=600)
        st.plotly_chart(fig_corr, use_container_width=True)

        st.markdown("---")
        st.markdown("#### Scatter — any two features")
        all_feat_opts = list(feat_labels.keys())
        c1, c2 = st.columns(2)
        with c1:
            xf = st.selectbox("X axis", all_feat_opts, index=0,
                              format_func=lambda x: feat_labels[x], key="sx")
        with c2:
            yf = st.selectbox("Y axis", all_feat_opts, index=12,
                              format_func=lambda x: feat_labels[x], key="sy")

        fig_sc = px.scatter(df_lid, x=xf, y=yf, color="tile_label",
                            opacity=0.4,
                            color_discrete_map=TILE_COLORS,
                            marginal_x="histogram", marginal_y="box",
                            title=f"{feat_labels[xf]} vs {feat_labels[yf]}",
                            labels={xf: feat_labels[xf], yf: feat_labels[yf],
                                    "tile_label": "Tile"})
        fig_sc.update_traces(marker_size=3, selector=dict(type="scatter"))
        fig_sc.update_layout(**PLOTLY_LAYOUT, height=520)
        st.plotly_chart(fig_sc, use_container_width=True)

        st.markdown("#### 3D — Height × Crown Area × Biomass")
        fig3d = px.scatter_3d(
            df_lid.sample(min(3000, len(df_lid)), random_state=42),
            x="height_max_m", y="crown_area_m2", z=TARGET,
            color="tile_label", opacity=0.5,
            color_discrete_map=TILE_COLORS,
            labels={"height_max_m": "Max Height (m)",
                    "crown_area_m2": "Crown Area (m²)",
                    TARGET: "Biomass (Mg/ha)",
                    "tile_label": "Tile"},
            title="3D — Tree Height × Crown Area × Satellite Biomass",
        )
        fig3d.update_traces(marker_size=2)
        fig3d.update_layout(**PLOTLY_LAYOUT, height=540,
                            scene=dict(bgcolor="rgba(11,17,23,0.8)",
                                       xaxis_title="Max Height (m)",
                                       yaxis_title="Crown Area (m²)",
                                       zaxis_title="Biomass (Mg/ha)"))
        st.plotly_chart(fig3d, use_container_width=True)

    # ── SPATIAL MAP ─────────────────────────────────────────────────────────
    with spatial_tab:
        st.markdown("#### Spatial Distribution of Trees")
        map_color_feat = st.selectbox(
            "Colour trees by",
            ["height_max_m", "crown_area_m2", "density_pt_m2",
             "cv_height", "sat_biomass", "tile_label"],
            format_func=lambda x: feat_labels.get(x, x),
            key="map_color",
        )

        if map_color_feat == "tile_label":
            fig_map = px.scatter(
                df_lid, x="tree_lon", y="tree_lat",
                color="tile_label", color_discrete_map=TILE_COLORS,
                opacity=0.5,
                title="Tree Locations by Tile",
                labels={"tree_lon": "Longitude", "tree_lat": "Latitude", "tile_label": "Tile"},
            )
        else:
            fig_map = px.scatter(
                df_lid, x="tree_lon", y="tree_lat",
                color=map_color_feat, color_continuous_scale="Viridis",
                opacity=0.5,
                title=f"Tree Locations — coloured by {feat_labels.get(map_color_feat, map_color_feat)}",
                labels={"tree_lon": "Longitude", "tree_lat": "Latitude",
                        map_color_feat: feat_labels.get(map_color_feat, map_color_feat)},
            )
        fig_map.update_traces(marker_size=3)
        fig_map.update_layout(**PLOTLY_LAYOUT, height=520)
        st.plotly_chart(fig_map, use_container_width=True)

        # dist_to_px distribution
        fig_dist = px.histogram(df_lid, x="dist_to_px_m", color="tile_label",
                                nbins=60, barmode="overlay", opacity=0.7,
                                color_discrete_map=TILE_COLORS,
                                title="Distance: Tree centroid → Satellite pixel (m)",
                                labels={"dist_to_px_m": "Distance (m)", "tile_label": "Tile"})
        fig_dist.update_layout(**PLOTLY_LAYOUT, height=320)
        st.plotly_chart(fig_dist, use_container_width=True)

    # ── BIOMASS MODEL ────────────────────────────────────────────────────────
    with model_tab:
        st.markdown("#### Biomass Prediction from LiDAR-derived Features")
        st.info("Target: **satellite-derived above-ground biomass** (`sat_biomass` from ESA CCI AGB). "
                "Features: individual-tree LiDAR structural attributes.")

        from xgboost import XGBRegressor
        from catboost import CatBoostRegressor
        from sklearn.neural_network import MLPRegressor

        all_feat_opts = LIDAR_FEATS + SAT_FEATS
        selected_feats = st.multiselect(
            "Select input features",
            all_feat_opts,
            default=LIDAR_FEATS,
            format_func=lambda x: feat_labels.get(x, x),
        )
        include_tile = st.checkbox("Include tile as feature (one-hot)", value=False)

        if len(selected_feats) < 2:
            st.warning("Select at least 2 features.")
        else:
            df_model = df_lid[selected_feats + [TARGET]].dropna()
            X = df_model[selected_feats].values
            if include_tile:
                tile_dummies = pd.get_dummies(df_lid.loc[df_model.index, "tile_label"]).values
                X = np.hstack([X, tile_dummies])
            y = df_model[TARGET].values
            X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

            scaler_lid = StandardScaler()
            X_tr_s = scaler_lid.fit_transform(X_tr)
            X_te_s  = scaler_lid.transform(X_te)

            with st.spinner("Training models on real LiDAR data …"):
                xgb = XGBRegressor(n_estimators=400, learning_rate=0.05, max_depth=5,
                                   subsample=0.8, colsample_bytree=0.7,
                                   random_state=42, verbosity=0)
                xgb.fit(X_tr, y_tr)
                yp_xgb = xgb.predict(X_te)

                cb = CatBoostRegressor(iterations=400, learning_rate=0.05, depth=5,
                                       loss_function="RMSE", random_seed=42, verbose=0)
                cb.fit(X_tr, y_tr, eval_set=(X_te, y_te))
                yp_cb = cb.predict(X_te)

                mlp = MLPRegressor(hidden_layer_sizes=(64, 32, 16), max_iter=400,
                                   random_state=42, early_stopping=True)
                mlp.fit(X_tr_s, y_tr)
                yp_mlp = mlp.predict(X_te_s)

            feat_names_full = selected_feats + (["Tile1","Tile2","Tile3"] if include_tile else [])
            lid_results = {
                "XGBoost":  dict(y_true=y_te, y_pred=yp_xgb,
                                 feat_imp=xgb.feature_importances_, feat_names=feat_names_full),
                "CatBoost": dict(y_true=y_te, y_pred=yp_cb,
                                 feat_imp=cb.get_feature_importance(), feat_names=feat_names_full),
                "MLP":      dict(y_true=y_te, y_pred=yp_mlp,
                                 feat_imp=np.abs(mlp.coefs_[0]).mean(axis=1),
                                 feat_names=feat_names_full),
            }
            lid_colors = {"XGBoost": "#86efac", "CatBoost": "#fb923c", "MLP": "#a78bfa"}

            # Comparison
            st.markdown("##### Model Comparison")
            m1, m2, m3 = st.columns(3)
            for col, mname in zip([m1, m2, m3], lid_results):
                r = lid_results[mname]
                r2   = r2_score(r["y_true"], r["y_pred"])
                rmse = np.sqrt(mean_squared_error(r["y_true"], r["y_pred"]))
                col.metric(f"{mname} R²",   f"{r2:.4f}")
                col.metric(f"{mname} RMSE", f"{rmse:.3f} Mg/ha")

            lid_sel = st.selectbox("Model detail", list(lid_results.keys()), key="lid_model")
            res = lid_results[lid_sel]
            metrics_row(res["y_true"], res["y_pred"])

            c1, c2 = st.columns(2)
            with c1:
                st.plotly_chart(
                    scatter_obs_pred(res["y_true"], res["y_pred"], lid_sel, lid_colors[lid_sel]),
                    use_container_width=True)
            with c2:
                imp   = np.array(res["feat_imp"])
                names = [feat_labels.get(f, f) for f in res["feat_names"]]
                st.plotly_chart(
                    feat_importance_chart(imp, names, "Feature Importance", lid_colors[lid_sel]),
                    use_container_width=True)

            st.plotly_chart(residual_plot(res["y_true"], res["y_pred"], lid_colors[lid_sel]),
                            use_container_width=True)
