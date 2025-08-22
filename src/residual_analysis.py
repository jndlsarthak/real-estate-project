import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

df = pd.read_csv("../data/merged_output.csv")
df.columns = df.columns.str.strip()
X = df.drop("Price, RUR", axis=1)
y = df["Price, RUR"]

def parse_dates(X):
    if isinstance(X, pd.DataFrame):
        date_series = X.iloc[:, 0]
    elif isinstance(X, pd.Series):
        date_series = X
    else:
        date_series = pd.Series(X)
    date_series = pd.to_datetime(date_series, errors='coerce')
    month = date_series.dt.month.fillna(0).astype(int)
    season = month.apply(lambda m: (m % 12 + 3) // 3)
    return pd.DataFrame({'Month': month, 'Season': season})


# Loading trained model
model = joblib.load("../training_artifacts/model.pkl")

if 'Additional description' in X.columns:
    X['Additional description'] = X['Additional description'].fillna("")

# Predicting on all available data
y_pred = model.predict(X)

# Calculating residuals
residuals = y - y_pred

# Saving residual metrics
metrics = {
    "MAE": mean_absolute_error(y, y_pred),
    "RMSE": mean_squared_error(y, y_pred, squared=False),
    "R2": r2_score(y, y_pred)
}
with open("../training_artifacts/metrics.json", "w") as f:
    import json
    json.dump(metrics, f, indent=4)

# Plotting residuals vs predicted
plt.figure(figsize=(10, 5))
sns.scatterplot(x=y_pred, y=residuals)
plt.axhline(0, color="red", linestyle="--")
plt.xlabel("Predicted Price")
plt.ylabel("Residuals")
plt.title("Residuals vs Predicted Price")
plt.tight_layout()
plt.savefig("../training_artifacts/residuals_vs_predicted.png")
plt.close()

# Plotting histogram of residuals
plt.figure(figsize=(8, 4))
sns.histplot(residuals, bins=50, kde=True)
plt.xlabel("Residuals")
plt.title("Distribution of Residuals")
plt.tight_layout()
plt.savefig("../training_artifacts/residual_distribution.png")
plt.close()

print(" Residual analysis complete. Plots saved.")

