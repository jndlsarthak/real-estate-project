# real-estate-project
## Overview

This project processes raw real estate Excel data, engineers features, and builds machine learning models to predict target metrics. The workflow consists of a sequence of scripts and notebooks, starting from the raw data sheet and producing final predictions, metrics, and supporting analysis.

---

## Dependencies

- **Python** (recommend ≥3.8)
- Libraries:  
  - `pandas`  
  - `numpy`  
  - `scikit-learn`  
  - `joblib`  
  - `matplotlib`  
  - `seaborn`  
  - `geopy` (with MapBox for geocoding)  
- MapBox API credentials for geocoding features are also needed.

---
## Running the Pipeline

Below is the sequence of scripts/notebooks, including expected inputs and outputs at each step:

### 1. Preprocessing
- **notebook/eda_preprocessing**  
  - **Input:** `data/datasheet.xlsx`  
  - **Output:** 
    - `data/raw_real_estate_data.csv`
    - `data/processed_real_estate_data.csv`
    - `data/final_processed_real_estate_data.csv`

### 2. Room Feature Cleaning
- **notebook/rooms_cleaning**  
  - **Input:** `data/final_processed_real_estate_data.csv`  
  - **Output:** `data/final01_processed_real_estate_data.csv`

### 3. Data Loading and Baseline Modeling
- **src/data_loader.py**  
  - **Input:** `data/final01_processed_real_estate_data.csv`  
  - **Output:** `load_data()`
- **src/baseline.py**  
  - **Input:** from data_loader output `load_data()`
  - **Output:** `training_artifacts/baseline_dummy_model.joblib`

### 4. Baseline Predictions
- **notebook/predict_baseline.ipynb**

### 5. Hyperparameter Tuning & Grid Search
- **src/gridsearch_tuning.py**  
  - **Input:** `data/final01_processed_real_estate_data.csv`  
  - **Output:**  
    - `training_artifacts/feature_target_correlation.csv`  
    - `training_artifacts/best_params.json`  
    - `training_artifacts/model.pkl`  
    - `training_artifacts/metrics.json`  
    - `data/preprocessed_full.csv`

### 6. Linear & ElasticNet Regression
- **src/linear_regression.py**  
  - **Input:** `data/final01_processed_real_estate_data.csv`  
  - **Output:**  
    - `training_artifacts/elasticnet_best_params.json`  
    - `training_artifacts/elasticnet_model.pkl`  
    - `training_artifacts/elasticnet_metrics.json`  
    - `data/preprocessed_full_elasticnet.csv`

### 7. Residual Analysis
- **src/residual_analysis.py**  
  - **Input:** `data/preprocessed_full.csv`  
  - **Output:**  
    - `training_artifacts/residual_distribution.png`  
    - `training_artifacts/residuals_vs_predicted.png`

### 8. Feature Engineering 
- **src/distance_feature.py**  
  - **Input:** `data/russianREdata.xlsx`  
  - **Output:** 
    - `data/distance_and_city_flag_only.csv`
    - `data/merged_output.csv`

---

## Usage

1. Ensure dependencies are installed and API keys are configured.
2. Run each script/notebook **in the sequence above**.
   - For Jupyter notebooks, open and execute all cells, or run via command line using:  
     ```
     jupyter nbconvert --to notebook --execute <notebook>.ipynb
     ```
   - For Python scripts, run:  
     ```
     python <script_name.py>
     ```
3. Each step will save its output files needed for subsequent stages.

---

## Notes

- Data must flow in the specified order as outputs from one step are required as inputs for the next.
- Review intermediate CSVs/artifacts if you need to validate processing at any stage.
- For troubleshooting, ensure file paths and names are consistent throughout the pipeline.
