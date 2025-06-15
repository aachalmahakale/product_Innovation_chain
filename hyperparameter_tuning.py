import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np

# 1. Load data
df = pd.read_parquet("features.parquet")
y  = df["avg_rating"]
X  = df.drop(columns=["product_id","product_name","about_product","avg_rating"])

# 2. Split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Define grid
param_dist = {
    "n_estimators": [50,100,200,400],
    "max_depth":    [None,10,20,30],
    "min_samples_split": [2,5,10],
    "min_samples_leaf":  [1,2,4],
    "max_features":      ["auto","sqrt","log2"]
}

# 4. Randomized Search
rf = RandomForestRegressor(random_state=42)
search = RandomizedSearchCV(
    rf, param_distributions=param_dist,
    n_iter=20, cv=3,
    scoring="neg_mean_squared_error",
    n_jobs=-1, random_state=42
)
search.fit(X_train, y_train)

# 5. Evaluate & save best
best = search.best_estimator_
preds = best.predict(X_val)
rmse = np.sqrt(mean_squared_error(y_val, preds))
print(f"Best params: {search.best_params_}")
print(f"Validation RMSE: {rmse:.3f}")

joblib.dump(best, "rf_model_tuned.joblib")
print("Saved tuned model to rf_model_tuned.joblib")