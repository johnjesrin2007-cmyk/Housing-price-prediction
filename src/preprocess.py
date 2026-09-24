import pandas as pd
from typing import Optional, Tuple

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def preprocess_data(
    path: str,
    training: bool = True,
    target_col: Optional[str] = None
) -> Tuple[pd.DataFrame, Optional[pd.Series]]:

    # -------------- -----------
    # 📥 LOAD DATA
    # -------------------------
    df = pd.read_csv(path)

    if target_col:
       df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
       df = df.dropna(subset=[target_col])
    if df.empty:
        raise ValueError("Dataset is empty")

    df = pd.read_csv(path)

    if training:
        if target_col is None:
            target_col = "price_lakhs"

        df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
        df = df.dropna(subset=[target_col])

        if df.empty:
            raise ValueError("Dataset is empty")

        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found")

        X = df.drop(columns=[target_col])
        y = df[target_col]

        return X, y

    else:
        return df, None


def get_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    

    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object"]).columns.tolist()

    preprocess = ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler())
        ]), numeric_features),

        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ]), categorical_features)
    ])

    return preprocess


## dont use scalar for tree and boosting models