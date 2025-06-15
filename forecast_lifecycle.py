import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def forecast_lifecycle_simple(
    products_parquet   = "products.parquet",
    candidates_csv     = "top_candidates.csv",
    n_months           = 6,
    lifetime_months    = 12
):
    # 1. Load product stats and screened variants
    products   = pd.read_parquet(products_parquet)
    candidates = pd.read_csv(candidates_csv)

    # 2. Merge in avg_rating & total_reviews (left’s count becomes total_reviews_x)
    merged = candidates.merge(
        products[["product_id","avg_rating","total_reviews"]],
        left_on  = "base_product_id",
        right_on = "product_id",
        how      = "left"
    )

    # 3. For each variant, compute & plot a simple monthly-review forecast
    for _, row in merged.iterrows():
        pid     = row["base_product_id"]
        prating = row["predicted_rating"]
        brate   = row["avg_rating"]
        brevs   = row["total_reviews_x"]   # use the merged “total_reviews_x”

        # a) Base monthly rate
        base_rate = brevs / lifetime_months if lifetime_months > 0 else brevs

        # b) Boost factor by rating ratio
        boost = prating / brate if brate > 0 else 1.0

        # c) Build months vector and linear ramp
        months = np.arange(1, n_months+1)
        if n_months > 1:
            ramp = 1 + (months - 1) / (n_months - 1)
        else:
            ramp = np.ones(n_months)
        forecast = base_rate * boost * ramp

        # d) Plot & save
        plt.figure(figsize=(6,4))
        plt.plot(months, forecast, marker="o")
        plt.title(f"Forecasted Monthly Reviews\nVariant of {pid}")
        plt.xlabel("Month into Launch")
        plt.ylabel("Predicted Reviews")
        plt.xticks(months)
        plt.tight_layout()
        fname = f"lifecycle_simple_{pid}.png"
        plt.savefig(fname)
        plt.close()
        print(f"→ Saved forecast for {pid} to {fname}")

if __name__ == "__main__":
    forecast_lifecycle_simple()