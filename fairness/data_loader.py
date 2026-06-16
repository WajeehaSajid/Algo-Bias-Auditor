import pandas as pd
from typing import Tuple

def load_adult_dataset(path: str = None) -> pd.DataFrame:
    """Load the Adult Income dataset.

    If `path` is None, load from UCI repository via URL.
    Returns a cleaned DataFrame with column names.
    """
    if path is None:
        url = (
            "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
        )
    else:
        url = path

    cols = [
        "age",
        "workclass",
        "fnlwgt",
        "education",
        "education-num",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "capital-gain",
        "capital-loss",
        "hours-per-week",
        "native-country",
        "income",
    ]

    df = pd.read_csv(url, header=None, names=cols, na_values=" ?",
                     skipinitialspace=True)

    # Basic clean: drop rows with missing values and strip whitespace
    df = df.dropna().reset_index(drop=True)
    df["income"] = df["income"].apply(lambda x: x.strip())

    # Create a binary target column `income_binary`: 1 if >50K, else 0
    df["income_binary"] = df["income"].apply(lambda v: 1 if ">50K" in v else 0)
    return df

def train_test_split(df: pd.DataFrame, target: str = "income", test_size: float = 0.2, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    from sklearn.model_selection import train_test_split

    # If a binary target column exists, prefer it
    if "income_binary" in df.columns and target not in ("income_binary",):
        y = df["income_binary"]
        X = df.drop(columns=["income", "income_binary"]) if "income" in df.columns else df.drop(columns=["income_binary"])
    else:
        X = df.drop(columns=[target])
        # fallback: map string labels to binary
        y = df[target].apply(lambda v: 1 if ">50K" in str(v) else 0)
    return train_test_split(X, y, test_size=test_size, random_state=random_state)
