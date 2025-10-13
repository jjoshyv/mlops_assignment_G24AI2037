import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import joblib
import os

def load_data():
    """
    Load the Boston housing dataset.

    Strategy:
    1) If a file 'data/boston.csv' exists in the repository, read that (recommended).
    2) Otherwise attempt to download the original dataset URL (best-effort).
       If the HTTPS download fails due to SSL issues, raise a friendly error
       asking the user to place a local copy at data/boston.csv.
    """
    data_local = os.path.join(os.path.dirname(__file__), "data", "boston.csv")

    # 1) Try local file first
    if os.path.exists(data_local):
        print("✅ Using local dataset:", data_local)
        df = pd.read_csv(data_local)
        return df

    # 2) Fallback to online dataset (if local not found)
    print("🌐 Local dataset not found. Attempting to download from CMU server...")
    data_url = "http://lib.stat.cmu.edu/datasets/boston"

    try:
        raw_df = pd.read_csv(data_url, sep=r"\s+", skiprows=22, header=None)
        data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
        target = raw_df.values[1::2, 2]

        feature_names = [
            'CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE',
            'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT'
        ]

        df = pd.DataFrame(data, columns=feature_names)
        df['MEDV'] = target

        print("✅ Successfully loaded dataset from remote URL.")
        return df

    except Exception as e:
        raise RuntimeError(
            "❌ Could not load Boston dataset (SSL or network issue).\n"
            "Please download and place a local copy at 'data/boston.csv'.\n"
            "Dataset URL: http://lib.stat.cmu.edu/datasets/boston\n"
            f"Error: {e}"
        )

def split_data(df, target_col='MEDV', test_size=0.2, random_state=42):
    """
    Split the dataset into train/test sets.
    """
    X = df.drop(columns=[target_col])
    y = df[target_col].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    return X_train.reset_index(drop=True), X_test.reset_index(drop=True), y_train, y_test

def preprocess_fit_transform(X_train, X_test):
    """
    Fit StandardScaler on training data and transform both train/test sets.
    """
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled

def repeated_train_evaluate(model_factory,
                            df,
                            n_runs=5,
                            test_size=0.2,
                            random_seed=42,
                            save_model_path=None):
    """
    Repeatedly train a model returned by `model_factory()` and evaluate on a
    held-out test set. For each run we re-sample the train/test split using
    `random_seed + run_idx` to vary randomness.

    Parameters
    ----------
    model_factory : callable
        Zero-argument function that returns a fresh (unfitted) estimator.
    df : pandas.DataFrame
        Full dataset with features + target column 'MEDV' (as produced by load_data()).
    n_runs : int
        Number of repeated runs.
    test_size : float
        Fraction of data used as test set.
    random_seed : int
        Base random seed (each run uses random_seed + run_idx).
    save_model_path : str or None
        If provided, models will be saved with suffixes:
            "{save_model_path}_0.joblib", "{save_model_path}_1.joblib", ...
        The directory for save_model_path will be created if needed.

    Returns
    -------
    avg_mse : float
        Mean of MSEs across runs.
    mses : list[float]
        List of MSE values (one per run).
    """
    mses = []

    # prepare save dir if requested
    if save_model_path:
        save_dir = os.path.dirname(save_model_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

    for i in range(n_runs):
        seed = random_seed + i

        # 1) new train/test split for this run
        X_train, X_test, y_train, y_test = split_data(df, target_col='MEDV',
                                                      test_size=test_size,
                                                      random_state=seed)

        # 2) preprocess (fit scaler on train, apply to both)
        X_train_scaled, X_test_scaled = preprocess_fit_transform(X_train, X_test)

        # 3) create fresh model and fit
        model = model_factory()

        # If model accepts a random_state parameter and model_factory didn't set it,
        # try to set it here for reproducibility per-run.
        try:
            if hasattr(model, "set_params"):
                model.set_params(random_state=seed)
            else:
                setattr(model, "random_state", seed)
        except Exception:
            # ignore if model doesn't accept random_state
            pass

        model.fit(X_train_scaled, y_train)

        # 4) predict & evaluate
        preds = model.predict(X_test_scaled)
        mse = mean_squared_error(y_test, preds)
        mses.append(float(mse))

        # 5) optionally save model for this run
        if save_model_path:
            fname = f"{save_model_path}_{i}.joblib"
            joblib.dump(model, fname)

    avg_mse = float(sum(mses) / len(mses)) if mses else float("nan")
    return avg_mse, mses

