"""Helper: load and preprocess Rossmann dataset to match notebook pipeline.

Functions:
- load_raw(data_dir='dataset') -> (train, store, test)
- build_merged(train, store) -> store_merged
- get_feature_matrix(store_merged) -> (X, y, scaler, feature_columns)

This file mirrors the notebook's data contract: Date-as-index, sentinel for CompetitionDistance=200000,
PromoInterval month expansion, IterativeImputer for promo/competition dates, and get_dummies with prefix.
"""
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import MinMaxScaler


def load_raw(data_dir: str = "dataset"):
    data_dir = Path(data_dir)
    train = pd.read_csv(data_dir / "train.csv", parse_dates=["Date"], index_col="Date")
    store = pd.read_csv(data_dir / "store.csv")
    test = pd.read_csv(data_dir / "test.csv")
    return train, store, test


def build_merged(train: pd.DataFrame, store: pd.DataFrame) -> pd.DataFrame:
    """Merge train and store and perform notebook-style preprocessing."""
    df = pd.merge(train, store, on="Store", how="left")

    # Extract calendar features
    df["Year"] = df.index.year
    df["Month"] = df.index.month
    df["Day"] = df.index.day
    # pandas 1.1+ returns a DataFrame for isocalendar()
    try:
        df["WeekofYear"] = df.index.isocalendar().week
    except Exception:
        df["WeekofYear"] = df.index.week

    # Sales per customer (watch for division by zero)
    df["SalesPerCustomer"] = df["Sales"] / df["Customers"].replace({0: np.nan})

    # Notebook fills train NaNs with 0 early; follow similar conservative approach for train columns
    df["Sales"] = df["Sales"].fillna(0)
    df["Customers"] = df["Customers"].fillna(0)

    # Promo interval -> month flags
    df["PromoInterval"] = df["PromoInterval"].fillna("0").astype(str)
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sept", "Oct", "Nov", "Dec"]
    for month in months:
        df[f"Promo_{month}"] = df["PromoInterval"].apply(lambda x: 1 if month in x else 0)
    df.drop(columns=["PromoInterval"], inplace=True)

    # CompetitionDistance sentinel and logical zeros
    df["CompetitionDistance"] = df["CompetitionDistance"].fillna(200000)
    df.loc[df["CompetitionDistance"] == 200000, ["CompetitionOpenSinceMonth", "CompetitionOpenSinceYear"]] = 0
    df.loc[df["Promo2"] == 0, ["Promo2SinceWeek", "Promo2SinceYear"]] = 0

    # Iterative imputation for the remaining numeric date gaps
    cols_to_impute = ["CompetitionOpenSinceMonth", "CompetitionOpenSinceYear", "Promo2SinceWeek", "Promo2SinceYear"]
    imputer = IterativeImputer(random_state=42)
    try:
        df[cols_to_impute] = imputer.fit_transform(df[cols_to_impute])
    except Exception:
        # If imputation fails (small dataset or types), fallback to filling with zeros
        df[cols_to_impute] = df[cols_to_impute].fillna(0)

    # Ensure ints where expected
    for c in ["CompetitionOpenSinceMonth", "CompetitionOpenSinceYear", "Promo2SinceWeek", "Promo2SinceYear"]:
        df[c] = df[c].fillna(0).astype(int)

    # Clean StateHoliday
    df["StateHoliday"] = df["StateHoliday"].replace("0", 0)

    return df


def get_feature_matrix(store_merged: pd.DataFrame):
    """Return X, y, scaler, feature_columns mirroring notebook preprocessing.

    - One-hot encodes categorical_cols with prefix equal to column list
    - Scales numeric_cols with MinMaxScaler
    - Returns X (features DataFrame), y (Series), scaler (fitted scaler), feature_columns (list)
    """
    df = store_merged.copy()

    numeric_cols = [
        "Sales", "Customers", "Day", "Month",
        "CompetitionDistance", "CompetitionOpenSinceMonth", "CompetitionOpenSinceYear",
        "Promo2SinceWeek", "Promo2SinceYear", "Year", "WeekofYear", "DayOfWeek",
    ]
    categorical_cols = ["StoreType", "Assortment", "StateHoliday"]

    # One-hot encode with prefix as in the notebook
    df = pd.get_dummies(df, columns=categorical_cols, prefix=categorical_cols)

    # Scale numeric columns (MinMax)
    scaler = MinMaxScaler()
    # Some numeric columns may be missing in certain datasets; intersect
    to_scale = [c for c in numeric_cols if c in df.columns]
    if len(to_scale) > 0:
        df.loc[:, to_scale] = scaler.fit_transform(df.loc[:, to_scale])

    # Prepare X and y
    drop_cols = ["Store", "Sales", "Customers", "SalesPerCustomer"]
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])
    y = df["Sales"] if "Sales" in df.columns else None

    feature_columns = list(X.columns)
    return X, y, scaler, feature_columns


if __name__ == "__main__":
    # quick smoke-run
    print("data_loader module: import functions load_raw, build_merged, get_feature_matrix")
