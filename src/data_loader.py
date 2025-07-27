import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(
    file_path: str = "../data/final01_processed_real_estate_data.csv",
    target_column: str = " Price, RUR",
    test_size: float = 0.2,
    random_state: int = 42
):
    
    # Loads processed dataset and splits into train/test sets.
    # Returns:
    #     X_train, X_test, y_train, y_test
    
    df = pd.read_csv(file_path)
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
