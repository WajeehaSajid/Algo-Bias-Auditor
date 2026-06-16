import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from typing import Tuple

def preprocess_adult(df: pd.DataFrame, drop_columns=None) -> Tuple[pd.DataFrame, pd.Series, Pipeline]:
    """Return feature matrix X, target y, and preprocessing pipeline.

    - Encodes categorical variables using one-hot encoding
    - Imputes if necessary (the dataset we'll use is already cleaned)
    """
    if drop_columns is None:
        drop_columns = []

    df = df.copy()
    y = df["income"].apply(lambda v: 1 if ">50K" in v else 0)
    X = df.drop(columns=["income"] + drop_columns)

    # Identify categorical and numeric cols
    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
    num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

    numeric_transformer = Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))])
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, num_cols),
            ("cat", categorical_transformer, cat_cols),
        ]
    )

    return X, y, preprocessor
