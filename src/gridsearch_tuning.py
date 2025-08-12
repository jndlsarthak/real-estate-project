import os
import json
import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, RepeatedKFold, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

# Load data
df = pd.read_csv("../data/final01_processed_real_estate_data.csv")

df = df.loc[:, ~df.columns.duplicated()]  # drop duplicate columns
# ✅ Drop leakage columns before split
leakage_cols = ["Log_Price", "log_price", "price"]
for col in leakage_cols:
    if col in df.columns:
        print(f"⚠️ Dropping potential leakage column: {col}")
        df = df.drop(columns=[col])

# Define target and features
X = df.drop(columns=[" Price, RUR"])
y = df[" Price, RUR"]

# Drop high-cardinality or unnecessary features
drop_features = [' Address', ' Address on the website', ' Contacts']
for col in drop_features:
    if col in X.columns:
        X = X.drop(columns=col)

# ✅ Clean text features if present
if ' Additional description' in X.columns:
    print("✅ Cleaning text column: 'Additional description'")
    X[' Additional description'] = (
        X[' Additional description']
        .fillna("")
        .astype(str)
    )

# Define preprocessing functions
def parse_dates(df):
    df = df.copy()
    df[' Date'] = pd.to_datetime(df[' Date'], errors='coerce')
    df[' Year'] = df[' Date'].dt.year.fillna(0).astype(int)
    df[' Month'] = df[' Date'].dt.month.fillna(0).astype(int)
    return df[[' Year', ' Month']]

# Column categorization
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
text_features = [' Additional description'] if ' Additional description' in X.columns else []
date_features = [' Date'] if 'Date' in X.columns else []
categorical_features = X.select_dtypes(include='object').drop(columns=text_features + date_features, errors='ignore').columns.tolist()

# Preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
        ('text', TfidfVectorizer(max_features=100), ' Additional description') if text_features else ('drop_text', 'drop', []),
        ('date', FunctionTransformer(parse_dates, validate=False), ' Date') if date_features else ('drop_date', 'drop', [])
    ]
)

# Pipeline
pipeline = Pipeline(steps=[
    ('preprocess', preprocessor),
    ('model', XGBRegressor(random_state=42, n_jobs=-1))
])

# Parameter grid
param_grid = {
    "model__n_estimators": [100, 300],
    "model__max_depth": [3, 6],
    "model__learning_rate": [0.01, 0.1],
    "model__subsample": [0.7, 1.0],
    "model__colsample_bytree": [0.7, 1.0],
}

# ✅ Split into train/test before fitting
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# GridSearchCV on training set only
cv = RepeatedKFold(n_splits=5, n_repeats=2, random_state=42)
grid_search = GridSearchCV(
    pipeline,
    param_grid,
    scoring="neg_root_mean_squared_error",
    cv=cv,
    n_jobs=-1,
    verbose=1
)
grid_search.fit(X_train, y_train)

# Save best params and model
os.makedirs("../training_artifacts", exist_ok=True)
with open("../training_artifacts/best_params.json", "w") as f:
    json.dump(grid_search.best_params_, f, indent=4)
joblib.dump(grid_search.best_estimator_, "../training_artifacts/model.pkl")

# ✅ Evaluate on the test set
y_pred = grid_search.predict(X_test)
metrics = {
    "MAE": mean_absolute_error(y_test, y_pred),
    "RMSE": mean_squared_error(y_test, y_pred, squared=False),
    "R2": r2_score(y_test, y_pred)
}

with open("../training_artifacts/metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

print("✅ Best params and model saved.")
print("📊 Test set metrics:", metrics)


# Combine features and target into one DataFrame
df_full = X.copy()
df_full["Price_RUR"] = y  # Append target column

# Save combined dataset
df_full.to_csv("../data/preprocessed_full.csv", index=False)
