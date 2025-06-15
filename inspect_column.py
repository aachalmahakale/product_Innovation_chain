import pandas as pd

# 1) products.parquet
p = pd.read_parquet("products.parquet")
print("products.parquet columns:", p.columns.tolist())

# 2) top_candidates.csv
c = pd.read_csv("top_candidates.csv")
print("top_candidates.csv columns:", c.columns.tolist())

# 3) merged
m = c.merge(
    p[["product_id","avg_rating","total_reviews"]],
    left_on="base_product_id", right_on="product_id", how="left"
)
print("merged columns:", m.columns.tolist())