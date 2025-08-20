#  ML System Design Document – Automated Valuation Model (AVM)

## Objective

Design a robust ML solution for real estate price prediction using structured listing data. This document outlines the proposed modeling family, validation strategy, evaluation metric, loss function, and feature engineering wishlist, along with business goals for model performance.

---

## 1. Modeling Family

### Baseline Model:
- **Dummy Regressor (Mean Predictor)**
  - Very simple and quick to implement
  - Helps validate the pipeline
  - Sets the absolute baseline for comparison

### Advanced Models:
- **XGBoost**
  - Widely used in regression problems

---

## 2. Validation Strategy

### Proposed Strategy:
- **5-Fold Cross-Validation**
  - Randomly splits the dataset into 5 folds
  - Each fold is used once as the validation set
  - Final performance = average of 5 runs

### Why:
- Reduces overfitting to a single validation set
- Suitable for non-temporal data
- Provides stable performance estimation

### Alternative (Future):
- **Time-Based Split** if dataset includes temporal trends (e.g., listing date)

---

## 3. Evaluation Metric

### Primary Metric:
- **MAE (Mean Absolute Error)**
  - Interpretable: average error in currency units
  - Less sensitive to outliers than RMSE
  - Suitable for pricing use cases

### Additional Metric (optional):
- **Relative MAE (%)** = (MAE / mean price) × 100

---

## 4. Loss Function

### Proposed:
- **MAE Loss** (if supported by model)
  - Matches evaluation metric
  - Directly minimizes absolute error

### Fallback:
- **MSE Loss** 
  - Use while monitoring MAE on validation set

### Optional for Advanced Use:
- **Quantile Loss**
  - Useful for predicting price ranges (e.g., 25th/50th/75th percentile)
  - Helps with risk-sensitive predictions

---

## 5. Feature Engineering Plan

### Baseline Features:
- Numeric: area, kitchen area, number of rooms, floor, total floors, year built
- Categorical: building type, region, district, material
- Log Transformation: Apply log(price) to handle skew and outliers more effectively. Predicted values can be exponentiated back to the original scale.

### Advanced Features:
- **Interaction Terms**: price per square meter, floor/total floors
- **Text Features** : TF-IDF embeddings of listing descriptions using Hugging Face transformers for better semantic understanding
- **Geo-Based Features** : distance from city center or proximity to metro stations, and neighborhood clusters
- **External Data**: average neighborhood price, amenities, transport access

---

## 6. Business Thresholds

- **Target MAE**: ≤ 10% of mean price
- Performance will be evaluated using:
  \[
  \text{Relative MAE (\%)} = \frac{\text{MAE}}{\text{mean(actual price)}} \times 100
  \]

- Will iterate on feature engineering or model complexity if threshold is not met.

---

## 7. Next Steps (Planned Work)

- Finalize preprocessing pipeline
- Implement baseline model with Linear Regression
- Run 5-Fold CV with MAE logging
- Start integrating tree-based models
- Track and compare performance on validation folds
- Begin feature importance analysis

---

## Summary

This document outlines the design of the ML system to be implemented over the coming time. It balances simplicity and interpretability with modern ML capabilities for structured data and geospatial insights.

