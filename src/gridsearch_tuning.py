import os
import json
import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, RepeatedKFold, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

# Loading data and strip column whitespace
df = pd.read_csv("../data/final01_processed_real_estate_data.csv")
df.columns = df.columns.str.strip()

# Dropping leakage columns
leakage_cols = ["Log_Price", "log_price", "price"]
for col in leakage_cols:
    if col in df.columns:
        df = df.drop(columns=[col])

# Defining features and target
X = df.drop(columns=["Price, RUR"])
y = df["Price, RUR"]

# Dropping high-cardinality or unnecessary columns
drop_features = ['Address', 'Address on the website', 'Contacts']
for col in drop_features:
    if col in X.columns:
        X = X.drop(columns=[col])

# Cleaning the text column
if 'Additional description' in X.columns:
    X['Additional description'] = X['Additional description'].fillna("").astype(str)

# Date parsing function:to extract Month and Season only
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

numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
text_features = ['Additional description'] if 'Additional description' in X.columns else []
date_features = ['Date'] if 'Date' in X.columns else []
categorical_features = X.select_dtypes(include='object').drop(columns=text_features + date_features, errors='ignore').columns.tolist()

# Building preprocess pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
        ('text', TfidfVectorizer(max_features=100), 'Additional description') if text_features else ('drop_text', 'drop', []),
        ('date', FunctionTransformer(parse_dates, validate=False), 'Date') if date_features else ('drop_date', 'drop', [])
    ]
)

pipeline = Pipeline(steps=[
    ('preprocess', preprocessor),
    ('model', XGBRegressor(
        objective='reg:absoluteerror',  
        n_estimators=500,    
        random_state=42,
        n_jobs=-1
    ))
])

param_grid = {
    "model__max_depth": [3, 5],
    "model__learning_rate": [0.01, 0.1],
    "model__subsample": [0.7, 1.0],
    "model__colsample_bytree": [0.7, 1.0],
}

# Train/test split
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Further split train into train/validation for hyperparameter tuning
X_train, X_valid, y_train, y_valid = train_test_split(
    X_train_full, y_train_full, test_size=0.2, random_state=42
)

cv = RepeatedKFold(n_splits=5, n_repeats=2, random_state=42)

grid_search = GridSearchCV(
    pipeline,
    param_grid,
    scoring='neg_mean_absolute_error',
    cv=cv,
    n_jobs=-1,
    verbose=1
)

# Fitting grid search WITHOUT early stopping
grid_search.fit(X_train, y_train)

# Saving best params and model
os.makedirs("../training_artifacts", exist_ok=True)
with open("../training_artifacts/best_params.json", 'w') as f:
    json.dump(grid_search.best_params_, f, indent=4)
joblib.dump(grid_search.best_estimator_, "../training_artifacts/model.pkl")

# Evaluating on test set
y_pred = grid_search.predict(X_test)
metrics = {
    "MAE": mean_absolute_error(y_test, y_pred),
    "RMSE": mean_squared_error(y_test, y_pred, squared=False),
    "R2": r2_score(y_test, y_pred)
}

with open("../training_artifacts/metrics.json", 'w') as f:
    json.dump(metrics, f, indent=4)

print("Best params and model saved.")
print("Test set metrics:", metrics)

df_full = X.copy()
df_full["Price_RUR"] = y
df_full.to_csv("../data/preprocessed_full.csv", index=False)
