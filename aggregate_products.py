import pandas as pd

def aggregate_products(
    in_parquet="cleaned_reviews.parquet",
    out_parquet="products.parquet"
):
    # 1. Load the cleaned-review data
    df = pd.read_parquet(in_parquet)

    # 2. Define which fields to keep as “metadata”
    meta = ["product_id", "product_name", "category", "about_product"]

    # 3. Group & aggregate
    agg = (
        df.groupby(meta)
          .agg(
            avg_rating       = ("rating", "mean"),
            total_reviews    = ("review_id", "count"),
            mean_discount_pct= ("discount_percentage", "mean"),
            min_price        = ("actual_price", "min"),
            max_price        = ("actual_price", "max"),
          )
          .reset_index()
    )

    # 4. Save out the per-product table
    agg.to_parquet(out_parquet, index=False)
    print(f"Aggregated {agg.shape[0]} products → {out_parquet}")

if __name__ == "__main__":
    aggregate_products()