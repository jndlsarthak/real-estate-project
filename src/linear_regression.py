import os
import json
import pandas as pd
import joblib
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, RepeatedKFold, GridSearchCV
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

# Loading data 
df = pd.read_csv("../data/final01_processed_real_estate_data.csv")
df.columns = df.columns.str.strip()

leakage_cols = ["Log_Price", "log_price", "price"]
for col in leakage_cols:
    if col in df.columns:
        df = df.drop(columns=[col])

X = df.drop(columns=["Price, RUR"])
y = df["Price, RUR"]

drop_features = ['Address', 'Address on the website', 'Contacts']
for col in drop_features:
    if col in X.columns:
        X = X.drop(columns=[col])

if 'Additional description' in X.columns:
    X['Additional description'] = X['Additional description'].fillna("").astype(str)

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

numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
text_features = ['Additional description'] if 'Additional description' in X.columns else []
date_features = ['Date'] if 'Date' in X.columns else []
categorical_features = X.select_dtypes(include='object').drop(columns=text_features + date_features, errors='ignore').columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
        ('text', TfidfVectorizer(max_features=100), 'Additional description') if text_features else ('drop_text', 'drop', []),
        ('date', FunctionTransformer(parse_dates, validate=False), 'Date') if date_features else ('drop_date', 'drop', [])
    ]
)
nan_columns = X.columns[X.isnull().any()].tolist()
print("Columns with NaNs:", nan_columns)

# Elastic net pipeline
pipeline = Pipeline(steps=[
    ('preprocess', preprocessor),
    ('model', ElasticNet(max_iter=5000, random_state=42))
])

param_grid = {
    'model__alpha': [0.1, 1.0, 10.0],
    'model__l1_ratio': [0.1, 0.5, 0.9],
}

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

cv = RepeatedKFold(n_splits=3, n_repeats=2, random_state=42)

grid_search = GridSearchCV(
    pipeline,
    param_grid,
    scoring="neg_mean_absolute_error",
    cv=cv,
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)

# saving model & best parameters 
os.makedirs("../training_artifacts", exist_ok=True)
with open("../training_artifacts/elasticnet_best_params.json", "w") as f:
    json.dump(grid_search.best_params_, f, indent=4)
joblib.dump(grid_search.best_estimator_, "../training_artifacts/elasticnet_model.pkl")

y_pred = grid_search.predict(X_test)
metrics = {
    "MAE": mean_absolute_error(y_test, y_pred),
    "RMSE": mean_squared_error(y_test, y_pred, squared=False),
    "R2": r2_score(y_test, y_pred)
}
with open("../training_artifacts/elasticnet_metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

print("ElasticNet best params and model saved.")
print("Test set metrics (ElasticNet):", metrics)

# saving preprocessed data after Elastic Net
df_full = X.copy()
df_full["Price_RUR"] = y
df_full.to_csv("../data/preprocessed_full_elasticnet.csv", index=False)
