import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(
    file_path: str = "../data/final01_processed_real_estate_data.csv",
    #file_path: str = "../data/merged_output.csv",   #the metrics recieved from this file were not as good as the original metrics
    target_column: str = " Price, RUR",
    test_size: float = 0.2,
    random_state: int = 42
):
    
    # Loads processed dataset and splits into train/test sets.
    # Returns:
    #     X_train, X_test, y_train, y_test
    df = pd.read_csv(file_path)

    # Dropping leakage columns if any
    leakage_cols = ["Log_Price", "log_price" , "price"]
    for col in leakage_cols:
        if col in df.columns:
            print(f" Dropping potential leakage column: {col}")
            df = df.drop(columns=[col])

    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    return X_train, X_test, y_train, y_test
if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_data()
    print("X_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)
    print("y_train mean:", y_train.mean())



df = pd.read_csv("../data/final01_processed_real_estate_data.csv")
#df = pd.read_csv ("../data/merged_output.csv")
def leakage_scan(df, target_col=" Price, RUR", threshold=0.95):
    
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataframe.")
    
    numeric_df = df.select_dtypes(include=['int64', 'float64'])
    corrs = numeric_df.corr()[target_col].abs().sort_values(ascending=False)
    corrs = corrs.drop(target_col, errors="ignore")
    suspicious = corrs[corrs > threshold]
    
    if suspicious.empty:
        print(" No suspiciously high correlations found.")
    else:
        print(" Potential leakage detected:")
        print(suspicious)
    
    return suspicious

# Running the check here 
suspicious_features = leakage_scan(df, target_col=" Price, RUR", threshold=0.95)

# dropping them automatically 
if not suspicious_features.empty:
    df = df.drop(columns=suspicious_features.index)
