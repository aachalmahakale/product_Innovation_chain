import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib
import json

def train_and_evaluate(
        in_parquet="features.parquet",
        model_path="rf_model.joblib",
        metrics_path="metrics.json",
        test_size=0.2,
        random_state=42
):
    # 1. Load feature table
    df = pd.read_parquet(in_parquet)
    target_col   = 'avg_rating'
    exclude_cols = ['product_id', 'product_name', 'about_product']
    X = df.drop(columns=exclude_cols + [target_col])
    y = df[target_col]

    # 2. Train/validation split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # 3. Initialize & fit model
    model = RandomForestRegressor(n_estimators=100, random_state=random_state)
    model.fit(X_train, y_train)

    # 4. Predict on both sets
    y_train_pred = model.predict(X_train)
    y_val_pred   = model.predict(X_val)

    # 5. Compute errors
    train_mse = mean_squared_error(y_train, y_train_pred)
    val_mse   = mean_squared_error(y_val,   y_val_pred)
    train_rmse = np.sqrt(train_mse)
    val_rmse   = np.sqrt(val_mse)

    train_mae = mean_absolute_error(y_train, y_train_pred)
    val_mae   = mean_absolute_error(y_val,   y_val_pred)

    print(f"Train RMSE: {train_rmse:.3f}")
    print(f"Train MAE : {train_mae:.3f}")
    print(f"Val   RMSE: {val_rmse:.3f}")
    print(f"Val   MAE : {val_mae:.3f}")

    # 6. Save the model
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")

    # 7. Save metrics for the dashboard
    metrics = {
        "train_rmse": float(train_rmse),
        "train_mae":  float(train_mae),
        "val_rmse":   float(val_rmse),
        "val_mae":    float(val_mae)
    }
    with open(metrics_path, "w") as f:
        json.dump(metrics, f)
    print(f"Metrics saved to {metrics_path}")

if __name__ == "__main__":
    train_and_evaluate()