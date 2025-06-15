import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

def engineer_features(in_parquet="products.parquet",
                      out_parquet="features.parquet"):
    df = pd.read_parquet(in_parquet)

    # Numeric descriptors
    df["title_length"] = df["product_name"].str.len()
    df["price_range"] = df["max_price"] - df["min_price"]

    # One-hot encode category
    df = pd.get_dummies(df, columns=["category"], drop_first=True)

    # TF-IDF on about_product
    tfidf = TfidfVectorizer(max_features=50, stop_words="english")
    mat = tfidf.fit_transform(df["about_product"].fillna(""))
    tfidf_df = pd.DataFrame(mat.toarray(), columns=tfidf.get_feature_names_out())

    df_final = pd.concat([df.reset_index(drop=True), tfidf_df], axis=1)
    df_final.to_parquet(out_parquet, index=False)
    print(f"Features saved to {out_parquet}. Shape: {df_final.shape}")

if __name__ == "__main__":
    engineer_features()