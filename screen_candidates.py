import pandas as pd
import numpy as np
import joblib

def screen_candidates(
    model_path="rf_model.joblib",
    features_parquet="features.parquet",
    base_product_id=None,
    n_variants=100,
    top_n=10
):
    # 1. Load model + features
    model = joblib.load(model_path)
    df    = pd.read_parquet(features_parquet)
    X     = df.drop(columns=["product_id","product_name","about_product","avg_rating"])

    # 2. Choose a “base” vector
    if base_product_id:
        base = df[df["product_id"] == base_product_id]
        if base.empty:
            raise ValueError("Base product ID not found")
        base_vec = base.drop(columns=["product_id","product_name","about_product","avg_rating"]).iloc[0]
    else:
        base_vec = X.sample(1, random_state=42).iloc[0]

    # 3. Generate perturbed variants
    variants = []
    for _ in range(n_variants):
        vec = base_vec.copy()
        # randomly tweak numeric features by ±10%
        for col in vec.index:
            if pd.api.types.is_numeric_dtype(type(vec[col])):
                vec[col] *= np.random.uniform(0.9, 1.1)
        variants.append(vec.values)

    X_var = pd.DataFrame(variants, columns=X.columns)

    # 4. Predict and rank
    preds = model.predict(X_var)
    X_var["predicted_rating"] = preds
    X_var["base_product_id"] = base_product_id
    shortlisted = X_var.sort_values("predicted_rating", ascending=False).head(top_n)
    return shortlisted

if __name__ == "__main__":
    # 1. Choose a real product_id from your features.parquet
    base_id = "B003L62T7W"   

    # 2. Call the function with that ID
    top_hits = screen_candidates(base_product_id=base_id)

    # 3. Inspect and save
    print("Top candidate variants:\n", top_hits)
    top_hits.to_csv("top_candidates.csv", index=False)
    print("Saved shortlist to top_candidates.csv")