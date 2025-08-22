# Project Report: Real Estate Price Prediction Pipeline

## Introduction

The project aimed to predict real estate prices using structured data collected from the provided raw dataset . The workflow involved a sequence of data preprocessing, feature engineering, model training, tuning, evaluation, and analysis until the best predictive model was identified.

---

## Data Preprocessing and Feature Engineering 

Initially, raw data from an Excel sheet was converted to CSV and underwent exploratory data analysis (EDA) and cleaning to handle missing values and categorical inconsistencies. Basic cleaning of room numbers, area categorizations, and text columns was performed to prepare the data for modeling. Additionally, a distance feature was engineered to add geographical context.

---

## Pipeline Construction and Training

Starting from the cleaned dataset (`final01_processed_real_estate_data.csv`), multiple models were developed and evaluated:

1. **Baseline Model:**  
   A simple Dummy Regressor predicting the mean price was created as a baseline. It yielded expected low performance—MAE and RMSE were high, and R² was close to zero—serving as a reference point.

2. **Grid Search Model Tuning:**  
   The main predictive pipeline used an XGBoost regressor with extensive hyperparameter tuning via grid search and repeated cross-validation. The tuning process adjusted parameters like learning rate, max depth, column sampling, and subsample ratio to optimize model performance on validation folds.

3. **ElasticNet Regression:**  
   As a benchmarking alternative, an ElasticNet linear regression model was trained with grid-search for alpha and l1_ratio, providing a balance between L1 and L2 regularization. This model was easier to understand but did not perform quite as well as XGBoost.

---

## Challenges Noted

- **Geographical Feature Encoding:**  
  The address translations used initially for geocoding did not accurately map addresses in MapBox, leading to missing or incorrect distance calculations. To resolve this, the original untranslated address data was used for geocoding, and the resulting features were merged back.

- **Data Leakage:**  
  Incorporating a `Log_Price` feature during data exploration led to unexpectedly strong model results. This was identified as leakage because it correlates directly with the target. A function was added to automatically detect and drop such leakage columns during training to ensure model integrity.

- **Training Duration:**  
  Grid search tuning with multiple folds required significant computation time, delaying progress. This was mitigated by reducing the grid size once the best parameters were identified and focusing on the optimized parameter space for final model training.


---

## Model Metrics Comparison and Dataset Impact

Two versions of the processed dataset were used for training and evaluation:

- `final01_processed_real_estate_data.csv` — the original cleaned dataset.
- `merged_output.csv` — the dataset enriched with additional geographical features (distance and city flags) merged in.

The metrics for three models on both datasets are as follows:

### Metrics using `final01_processed_real_estate_data.csv`

| Model                 | MAE           | RMSE          | R²              |
|-----------------------|---------------|---------------|-----------------|
| Baseline Dummy Regressor | 1,352,127.61  | 1,803,556.03  | -0.0001         |
| XGBoost (GridSearchCV)  | 624,555.56    | 898,628.60    | 0.7517          |
| ElasticNet Regression   | 756,013.65    | 1,094,549.03  | 0.6317          |

### Metrics using `merged_output.csv` (with distance and city features)

| Model                 | MAE           | RMSE          | R²              |
|-----------------------|---------------|---------------|-----------------|
| Baseline Dummy Regressor | 1,352,127.61  | 1,803,556.03  | -0.0001         |
| XGBoost (GridSearchCV)  | 591,672.72    | 872,457.05    | 0.7660          |
| ElasticNet Regression   | 759,152.28    | 1,113,840.92  | 0.6186          |

---

**Observations:**

- The XGBoost model showed **improved metrics with the enriched dataset** (`merged_output.csv`), indicating the added geographical features contributed positively.
- The ElasticNet model performed slightly better on the original dataset.
- The baseline Dummy Regressor remained constant across datasets.
  
Given these results, the XGBoost model trained on the merged dataset was selected as the **final recommended model** due to its superior predictive performance and robustness.

Also, the Residual analysis showed that most prediction errors were small and randomly distributed around zero, indicating a good overall fit. However, a few outliers and slight skewness suggested occasional large prediction errors, which are common in real estate data.

The README and report have been updated to reflect this evaluation and decision clearly.

---

## Conclusion

The end-to-end modeling pipeline from raw data preprocessing through feature engineering and advanced model tuning showcases a strong predictive capability for real estate pricing. The inclusion of geospatial features added meaningful value, demonstrated by improved XGBoost performance.

Key challenges such as feature encoding, data leakage and mitigation were successfully addressed. Among the models evaluated, the XGBoost model tuned with grid search on the enriched dataset is recommended for deployment or further study.