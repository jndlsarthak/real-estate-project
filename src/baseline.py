import numpy as np
import joblib
from sklearn.dummy import DummyRegressor
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from data_loader import load_data

# Loading the data
X_train, X_test, y_train, y_test = load_data()

# Dummy transformer 
class DummyTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return np.zeros((len(X), 1)) 

# Creating the pipeline
baseline_pipeline = Pipeline([
    ("dummy_transform", DummyTransformer()),
    ("dummy_model", DummyRegressor(strategy="mean"))
])

# Training the pipeline
baseline_pipeline.fit(X_train, y_train)

# Evaluation
y_pred = baseline_pipeline.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

# Printing results
print("Dummy Regressor (Mean Strategy) Results:")
print(f"MAE:  {mae:,.2f}")
print(f"MSE:  {mse:,.2f}")
print(f"RMSE: {rmse:,.2f}")
print(f"R²:   {r2:.4f}")

joblib.dump(baseline_pipeline, "../training_artifacts/baseline_dummy_model.joblib")
