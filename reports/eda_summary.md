### Summary of Week 1 : EDA & Preprocessing 

- Loaded and cleaned raw dataset (615 rows).

- Missing values in `Layout`, `Number of rooms`, and `Additional description` were handled:
  - Categorical values filled with `"Not specified"` or `"Unknown"`

- Target column `Price, RUR` had no invalid values, missing prices were dropped.
- Outliers were retained to preserve full dataset size.
- Generated:
  - Correlation heatmap
  - Histogram plots of numeric features
- Final cleaned dataset saved as `processed_real_estate_data.csv`