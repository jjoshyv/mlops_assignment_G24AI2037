import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import joblib

def load_data():
    """
    Loads Boston housing dataset from the original URL (as required in assignment).
    Returns a DataFrame with feature columns and 'MEDV' target column.
    """
    data_url = "http://lib.stat.cmu.edu/datasets/boston"
    raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
    data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
    target = raw_df.values[1::2, 2]
    feature_names = [
        'CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE',
        'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT'
    ]
    df = pd.DataFrame(data, columns=feature_names)
    df['MEDV'] = target
    return df

def split_data(df, target_col='MEDV', test_size=0.2, random_state=42):
    X = df.drop(columns=[target_col])
    y = df[target_col].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    return X_train.reset_index(drop=True), X_test.reset_index(drop=True), y_train, y_test

def preprocess_fit_transform(X_train, X_test):
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler

def train_model(model, X_train, y_train):
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    return mse

def repeated_train_evaluate(model_factory, df, n_runs=5, test_size=0.2, random_seed=42, save_model_path=None):
    mses = []
    last_model = None
    last_scaler = None
    for i in range(n_runs):
        seed = random_seed + i
        X_train, X_test, y_train, y_test = split_data(df, test_size=test_size, random_state=seed)
        X_train_s, X_test_s, scaler = preprocess_fit_transform(X_train, X_test)
        model = model_factory()
        model = train_model(model, X_train_s, y_train)
        mse = evaluate_model(model, X_test_s, y_test)
        mses.append(mse)
        last_model = model
        last_scaler = scaler
    avg_mse = float(np.mean(mses))
    if save_model_path and last_model is not None:
        joblib.dump({'model': last_model, 'scaler': last_scaler}, save_model_path)
    return avg_mse, mses
