# feature_importance.py

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from data_loader import load_data

TARGET_COLUMN = " Price, RUR"

# 1. Load data
X_train, X_test, y_train, y_test = load_data(target_column=TARGET_COLUMN)

# 2. Verify target is not in features
assert TARGET_COLUMN not in X_train.columns, f"❌ Target column '{TARGET_COLUMN}' found in features!"
print("✅ Target variable is not included in the feature set.")

# 3. Keep only numeric columns for feature importance
numeric_X_train = X_train.select_dtypes(include=[np.number])
non_numeric_cols = X_train.columns.difference(numeric_X_train.columns)
if len(non_numeric_cols) > 0:
    print(f"⚠️ Dropping non-numeric columns for feature importance: {list(non_numeric_cols)}")

# 4. Train RandomForest for feature importance
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(numeric_X_train, y_train)

importances = pd.Series(rf.feature_importances_, index=numeric_X_train.columns)
importances = importances.sort_values(ascending=False)
print("\nTop 10 feature importances:")
print(importances.head(10))

# 5. Check for duplicate columns
duplicate_cols = numeric_X_train.columns[numeric_X_train.columns.duplicated()].tolist()
if duplicate_cols:
    print(f"\n⚠️ Duplicate columns found: {duplicate_cols}")
else:
    print("\n✅ No duplicate columns found.")

# 6. Check for highly correlated features
corr_matrix = numeric_X_train.corr().abs()
high_corr_pairs = [
    (col1, col2, corr_matrix.loc[col1, col2])
    for col1 in corr_matrix.columns
    for col2 in corr_matrix.columns
    if col1 != col2 and corr_matrix.loc[col1, col2] > 0.95
]
if high_corr_pairs:
    print("\n⚠️ Highly correlated feature pairs (corr > 0.95):")
    for col1, col2, corr in high_corr_pairs:
        print(f"{col1} ↔ {col2} : {corr:.2f}")
else:
    print("\n✅ No highly correlated features above 0.95.")

print("\n🔍 Feature importance and duplicate/correlation check complete.")
