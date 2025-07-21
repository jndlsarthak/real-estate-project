### Summary of Week 1 : EDA & Preprocessing 

## 1. Dataset Overview

The dataset consists of real estate listings from Russia and the CIS region. It contains various features such as price, total area, living area, and kitchen area. The goal was to understand data quality, distribution, and relationships between features relevant for price prediction.

## 2. Data Cleaning

- Several numeric columns (such as `Price`, `Total area`, `Kitchen area`) had inconsistent formats, including the use of spaces as thousand separators and comma vs. period usage.
- These were systematically cleaned and converted into proper numeric types to ensure smooth downstream processing.
- Also, the Missing values in them` were handled:
  - Categorical values filled with `"Not specified"` or `"Unknown"`

## 3. Visual Analysis

Generated the following for initial insights:
- **Correlation heatmap**
- **Histogram plots** for all numeric features

## 4. Correlation Analysis

A correlation heatmap was generated to understand the relationship between numerical features:

### Key Insights:

- **Price** is moderately correlated with:
  - **Total Area (0.62)** – Strongest correlation with price, as expected.
  - **Kitchen Area (0.56)** – Also shows a moderate relationship, suggesting larger kitchens tend to be associated with higher prices.
  - **Living Area (0.45)** – Weaker correlation, possibly due to variability in layout or recording errors.

- **Total Area** is highly correlated with:
  - **Living Area (0.61)** – Logical, as living area is part of the total area.
- **Kitchen Area** and **Living Area** have a **very weak correlation (0.20)**, suggesting they vary independently.

## 5. Summary

- Significant preprocessing and cleaning were necessary to standardize numeric formats.
- Outliers were retained for now but flagged for future analysis or transformation.
- The strongest predictors of price appear to be **Total Area** and **Kitchen Area**.
- This EDA sets the foundation for informed feature selection and model design in upcoming steps.
- The final cleaned dataset was saved as `data/processed_real_estate_data.csv`.