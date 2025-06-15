import shap
import joblib
import pandas as pd
import matplotlib.pyplot as plt

def explain_model(
    model_path       = "rf_model_tuned.joblib",
    features_parquet = "features.parquet",
    output_path      = "images/shap_summary.png"
):
    print("Loading model and data…")
    model = joblib.load(model_path)
    df    = pd.read_parquet(features_parquet)
    X     = df.drop(columns=[
        "product_id",
        "product_name",
        "about_product",
        "avg_rating"
    ])

    print("Initializing SHAP explainer…")
    explainer = shap.TreeExplainer(model)

    print("Computing SHAP values (this can take a moment)…")
    # disable additivity check here
    shap_values = explainer.shap_values(X, check_additivity=False)

    print("Rendering and saving summary plot…")
    
    # create a larger figure canvas
    plt.figure(figsize=(14, 10))

    # draw the SHAP summary plot
    shap.summary_plot(shap_values, X, show=False)

    # adjust layout and margins so labels aren’t cut off
    plt.subplots_adjust(left=0.35, right=0.95, top=0.95, bottom=0.05)

    # save at higher resolution
    plt.savefig(output_path, dpi=150)
    plt.close()
   

    print(f" SHAP summary saved to {output_path}")

if __name__ == "__main__":
    explain_model()