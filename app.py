import warnings
warnings.filterwarnings("ignore", message=".*use_column_width.*")

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os
from PIL import Image
import plotly.graph_objects as go
import plotly.express as px

# ───────────────────────────────────────────────────────
# 1) Page config & sidebar metrics
# ───────────────────────────────────────────────────────
st.set_page_config(page_title="Product Innovation Dashboard", layout="wide")

# Load metrics.json (must exist from your training script)
with open("metrics.json") as mf:
    metrics = json.load(mf)

st.sidebar.header("Model Performance")
st.sidebar.metric("Train RMSE", f"{metrics['train_rmse']:.2f}")
st.sidebar.metric("Val   RMSE", f"{metrics['val_rmse']:.2f}")
st.sidebar.metric("Train MAE", f"{metrics['train_mae']:.2f}")
st.sidebar.metric("Val   MAE", f"{metrics['val_mae']:.2f}")

# ───────────────────────────────────────────────────────
# 2) Load artifacts
# ───────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    # features table
    feats = pd.read_parquet("features.parquet")

    # pick up whichever model file exists
    model_file = "rf_model_tuned.joblib" if os.path.exists("rf_model_tuned.joblib") else "rf_model.joblib"
    model = joblib.load(model_file)

    # screened variants
    cands = pd.read_csv("top_candidates_all.csv")

    # aggregated products
    products = pd.read_parquet("products.parquet")

    return feats, model, cands, products

feats, model, cands, products = load_artifacts()

# ───────────────────────────────────────────────────────
# 3) Sidebar controls
# ───────────────────────────────────────────────────────
st.sidebar.title("Controls")
base_ids = sorted(cands["base_product_id"].unique())
selected = st.sidebar.selectbox("Base Product", base_ids)
top_n    = st.sidebar.slider("Show top N variants", 1, 20, 5)

# ───────────────────────────────────────────────────────
# 4) Main header & metadata
# ───────────────────────────────────────────────────────
st.title("🛠 Product Innovation Dashboard")
meta = products.loc[products["product_id"] == selected].iloc[0]
st.subheader(meta["product_name"])
st.markdown(f"**Category:** {meta['category']}")

# ───────────────────────────────────────────────────────
# 5) Random Forest Feature Importances
# ───────────────────────────────────────────────────────
st.header("Random Forest Feature Importances")
fi_path = os.path.join("images", "feature_importances.png")
if os.path.exists(fi_path):
    st.image(Image.open(fi_path), use_container_width=True)
else:
    st.warning("Feature importances image not found.")

# ───────────────────────────────────────────────────────
# 6) SHAP Global Summary
# ───────────────────────────────────────────────────────
st.header("SHAP Global Summary")
shap_path = os.path.join("images", "shap_summary.png")
if os.path.exists(shap_path):
    st.image(Image.open(shap_path), use_container_width=True)
else:
    st.warning("SHAP summary image not found.")

# ───────────────────────────────────────────────────────
# 7) Top Variants Table
# ───────────────────────────────────────────────────────
st.header(f"Top {top_n} In-Silico Variants for {selected}")
df_sub = (
    cands[cands["base_product_id"] == selected]
    .nlargest(top_n, "predicted_rating")
    .reset_index(drop=True)
)
st.dataframe(df_sub.style.format({"predicted_rating": "{:.3f}"}), height=300)

# ───────────────────────────────────────────────────────
# 8) Predicted Rating Distribution
# ───────────────────────────────────────────────────────
st.header("Predicted Rating Distribution")
df_dist = cands[cands["base_product_id"] == selected].reset_index(drop=True)
fig_dist = px.histogram(
    df_dist,
    x="predicted_rating",
    nbins=10,
    title="All Variants’ Predicted Ratings",
    labels={"predicted_rating": "Predicted Rating"},
)
fig_dist.update_layout(template="plotly_white")
st.plotly_chart(fig_dist, use_container_width=True)

# ───────────────────────────────────────────────────────
# 9) Interactive 6-Month Lifecycle Forecast (best/worst highlighted)
# ───────────────────────────────────────────────────────
st.header("6-Month Lifecycle Forecast")
n_months = 6
months   = np.arange(1, n_months + 1)
ramp     = 1 + (months - 1)/(n_months - 1) if n_months > 1 else np.ones(n_months)

# base-product stats
brate = meta["avg_rating"]
brevs = meta["total_reviews"]

# all variants for this base
variants = cands[cands["base_product_id"] == selected].reset_index(drop=True)

# find best/worst
pr_vals   = variants["predicted_rating"]
best_idx  = pr_vals.idxmax()
worst_idx = pr_vals.idxmin()

fig = go.Figure()
for idx, row in variants.iterrows():
    pr        = row["predicted_rating"]
    base_rate = brevs / 12.0
    boost     = pr / brate if brate > 0 else 1.0
    forecast  = base_rate * boost * ramp

    is_best  = (idx == best_idx)
    is_worst = (idx == worst_idx)

    if is_best or is_worst:
        # bold line for best/worst
        fig.add_trace(go.Scatter(
            x=months,
            y=forecast,
            mode="lines+markers",
            name=f"Var {idx+1}: {pr:.3f}",
            line=dict(width=4, color="firebrick" if is_best else "navy"),
            marker=dict(size=8)
        ))
    else:
        # de‐emphasized for the rest
        fig.add_trace(go.Scatter(
            x=months,
            y=forecast,
            mode="lines+markers",
            name=f"Var {idx+1}: {pr:.3f}",
            line=dict(width=1, color="lightgray", dash="dot"),
            marker=dict(size=4),
            opacity=0.6
        ))

fig.update_layout(
    title=f"6-Month Forecast for {selected} (best & worst highlighted)",
    xaxis_title="Month into Launch",
    yaxis_title="Predicted Reviews",
    legend_title="Variants",
    template="plotly_white",
)
st.plotly_chart(fig, use_container_width=True)