import pandas as pd

def clean_data(in_csv="amazon.csv", out_parquet="cleaned_reviews.parquet"):
    # 1. Load raw data
    df = pd.read_csv(in_csv)

    # 2. Drop duplicate reviews
    df = df.drop_duplicates(subset=["review_id"])

    # 3. Drop rows missing key fields
    df = df.dropna(subset=["product_id", "review_content"])

    # 4. Clean price columns
    for col in ["actual_price", "discounted_price"]:
        df[col] = (
            df[col].astype(str)
                  .str.replace(r"[^\d\.]", "", regex=True)
                  .replace("", pd.NA)
                  .astype(float)
        )

    # 5. Clean discount percentage
    df["discount_percentage"] = (
        df["discount_percentage"].astype(str)
                                 .str.replace("%", "")
                                 .replace("", pd.NA)
                                 .astype(float)
    )

    # 6. Numeric ratings
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df = df.dropna(subset=["rating"])

    # 7. Filter valid records
    df = df[df["rating"].between(1, 5) & df["actual_price"].notna()]
    
    # 8. Save cleaned data
    df.reset_index(drop=True).to_parquet(out_parquet, index=False)
    print(f"Saved {df.shape[0]} rows to {out_parquet}")

if __name__ == "__main__":
    clean_data()