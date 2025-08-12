# linear_regression_model.py

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import json

# Load preprocessed data
df = pd.read_csv("../data/preprocessed_full.csv")

# Split features & target
X = df.drop("Price_RUR", axis=1)
y = df["Price_RUR"]

date_col = ' Date'  # adjust to your date column name

if date_col in X.columns:
    X[date_col] = X[date_col].str.strip()
    X[date_col] = pd.to_datetime(X[date_col], format="%d.%m.%Y %H:%M", dayfirst=True, errors='coerce')

    mask_valid_date = X[date_col].notna()
    X = X.loc[mask_valid_date].copy()
    y = y.loc[mask_valid_date].copy()

    X.loc[:, date_col] = X[date_col].view('int64') // 10**9
# Train-test split (same ratio as main model)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train simple linear regression
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# Predictions
y_pred_train = lr_model.predict(X_train)
y_pred_test = lr_model.predict(X_test)

# Metrics
metrics_train = {
    "MAE": mean_absolute_error(y_train, y_pred_train),
    "RMSE": mean_squared_error(y_train, y_pred_train, squared=False),
    "R2": r2_score(y_train, y_pred_train)
}

metrics_test = {
    "MAE": mean_absolute_error(y_test, y_pred_test),
    "RMSE": mean_squared_error(y_test, y_pred_test, squared=False),
    "R2": r2_score(y_test, y_pred_test)
}

# Save metrics
with open("linear_regression_metrics.json", "w") as f:
    json.dump({"train": metrics_train, "test": metrics_test}, f, indent=4)

# Save model
joblib.dump(lr_model, "linear_regression_model.pkl")

print("✅ Linear Regression complete.")
print("📊 Train metrics:", metrics_train)
print("📊 Test metrics:", metrics_test)
