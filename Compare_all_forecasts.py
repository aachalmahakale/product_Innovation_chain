import pandas as pd
import numpy as np

import plotly.graph_objects as go
import os

def compare_all_forecasts(
    products_parquet   = "products.parquet",
    candidates_csv     = "top_candidates.csv",
    n_months           = 6,
    lifetime_months    = 12,
    output_folder      = "charts_html"
):
    # 1. Load data
    products   = pd.read_parquet(products_parquet)
    candidates = pd.read_csv(candidates_csv)

    # 2. Merge in avg_rating & total_reviews → total_reviews_x
    merged = candidates.merge(
        products[["product_id", "avg_rating", "total_reviews"]],
        left_on  = "base_product_id",
        right_on = "product_id",
        how      = "left"
    )

    # 3. Prepare the time axis and linear ramp
    months = np.arange(1, n_months + 1)
    ramp   = 1 + (months - 1) / (n_months - 1) if n_months > 1 else np.ones(n_months)

    # 4. Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # 5. Loop over each base_product_id
    for pid, group in merged.groupby("base_product_id"):
        fig = go.Figure()
        for idx, row in group.reset_index().iterrows():
            prating = row["predicted_rating"]
            brate   = row["avg_rating"]
            brevs   = row["total_reviews_x"]

            base_rate = brevs / lifetime_months if lifetime_months > 0 else brevs
            boost     = prating / brate if brate > 0 else 1.0
            forecast  = base_rate * boost * ramp

            fig.add_trace(go.Scatter(
                x=months,
                y=forecast,
                mode="lines+markers",
                name=f"Variant {idx+1}: {prating:.2f}"
            ))

        fig.update_layout(
            title=f"Forecast Comparison for Base Product {pid}",
            xaxis_title="Month into Launch",
            yaxis_title="Predicted Reviews",
            legend_title="Variants",
            template="plotly_white"
        )

        out_html = os.path.join(output_folder, f"forecast_comparison_{pid}.html")
        fig.write_html(out_html, include_plotlyjs="cdn")
        print(f"→ Saved interactive chart for {pid} to {out_html}")

if __name__ == "__main__":
    compare_all_forecasts()