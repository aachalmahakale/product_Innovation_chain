import pandas as pd
import joblib
import matplotlib.pyplot as plt

def plot_feature_importance(
    model_path="rf_model.joblib",
    features_parquet="features.parquet",
    output_png="feature_importances.png"
):
    # 1. Load the trained model and feature table
    model = joblib.load(model_path)
    df    = pd.read_parquet(features_parquet)

    # 2. Separate predictors (drop non-features)
    X = df.drop(columns=["product_id","product_name","about_product","avg_rating"])

    # 3. Extract and sort importances
    imp   = model.feature_importances_
    names = X.columns
    imp_df = (
        pd.DataFrame({"feature": names, "importance": imp})
          .sort_values("importance", ascending=False)
    )

    # 4. Plot top 20
    top20 = imp_df.head(20).iloc[::-1]  # reverse for horizontal bar
    plt.figure(figsize=(8,6))
    plt.barh(top20["feature"], top20["importance"])
    plt.xlabel("Importance")
    plt.title("Top 20 Feature Importances")
    plt.tight_layout()
    plt.savefig(output_png)
   
    print(f"Saved plot to {output_png}")

if __name__ == "__main__":
    plot_feature_importance()