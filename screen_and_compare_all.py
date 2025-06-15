import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

def load_model(path="rf_model.joblib"):
    return joblib.load(path)

def load_features(path="features.parquet"):
    return pd.read_parquet(path)

def generate_variants(base_vec: pd.Series, feature_cols, n_variants=50):
    variants = []
    for _ in range(n_variants):
        v = base_vec.copy()
        # perturb each numeric feature by ±10%
        for col in feature_cols:
            # assume all features are numeric here
            v[col] *= np.random.uniform(0.9, 1.1)
        variants.append(v.values)
    return pd.DataFrame(variants, columns=feature_cols)

def screen_all(
    features_path    ="features.parquet",
    model_path       ="rf_model.joblib",
    out_csv          ="top_candidates_all.csv",
    n_variants       =100,
    top_n            =10
):
    # 1. Load
    df_feat = load_features(features_path)
    model   = load_model(model_path)

    # 2. Prepare predictor matrix columns
    feat_cols = [c for c in df_feat.columns
                 if c not in ["product_id","product_name","about_product","avg_rating"]]

    all_cands = []
    # 3. Loop over every product as base
    for pid in df_feat["product_id"].unique():
        base_row = df_feat[df_feat["product_id"] == pid]
        if base_row.empty:
            continue
        # <<< HERE: get a Series, not a DataFrame >>>>
        base_vec = base_row[feat_cols].iloc[0]

        # 4. Generate & score
        variants = generate_variants(base_vec, feat_cols, n_variants=n_variants)
        preds    = model.predict(variants)
        variants["predicted_rating"]  = preds
        variants["base_product_id"]   = pid

        # 5. Shortlist
        top = variants.nlargest(top_n, "predicted_rating")
        all_cands.append(top)

    # 6. Save all candidates
    result = pd.concat(all_cands, ignore_index=True)
    result.to_csv(out_csv, index=False)
    print(f"→ Saved all {len(result)} variant rows to {out_csv}")
    return result

def compare_all(
    products_parquet  ="products.parquet",
    candidates_csv    ="top_candidates_all.csv",
    n_months          = 6,
    lifetime_months   = 12
):
    # 1. Load
    products   = pd.read_parquet(products_parquet)
    candidates = pd.read_csv(candidates_csv)

    # 2. Merge in avg_rating & total_reviews → total_reviews_x
    merged = candidates.merge(
        products[["product_id","avg_rating","total_reviews"]],
        left_on="base_product_id",
        right_on="product_id",
        how="left"
    )

    # 3. Time axis
    months = np.arange(1, n_months + 1)
    ramp   = 1 + (months - 1)/(n_months - 1) if n_months>1 else np.ones(n_months)

    # 4. Loop per base product
    for pid, group in merged.groupby("base_product_id"):
        plt.figure(figsize=(8,6))
        for idx, row in group.iterrows():
            pr  = row["predicted_rating"]
            br  = row["avg_rating"]
            tr  = row["total_reviews_x"]
            base_rate = tr / lifetime_months if lifetime_months>0 else tr
            boost     = pr / br if br>0 else 1.0
            fc        = base_rate * boost * ramp
            plt.plot(months, fc, marker="o", label=f"Var {idx+1}: {pr:.2f}")

        plt.title(f"Forecast Comparison for {pid}")
        plt.xlabel("Month into Launch")
        plt.ylabel("Predicted Reviews")
        plt.xticks(months)
        plt.legend()
        plt.tight_layout()
        out = f"forecast_comparison_{pid}.png"
        plt.savefig(out)
        plt.close()
        print(f"→ Saved chart for {pid} to {out}")

if __name__ == "__main__":
    # 1) Screen and save variants for all products
    screen_all()
    # 2) Compare forecasts for all base products
    compare_all()