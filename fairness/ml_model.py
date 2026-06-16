import pandas as pd
from typing import Tuple

def train_model(df: pd.DataFrame, target: str = "income_binary", test_size: float = 0.2, random_state: int = 42, C: float = 1.0):
    """Split data, scale numeric features, encode categoricals, train Logistic Regression.

    Returns: (trained_pipeline, test_accuracy, y_pred, y_test)
    """
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score

    # Prepare X and y
    if target in df.columns:
        y = df[target]
    elif "income_binary" in df.columns:
        y = df["income_binary"]
    else:
        # fallback to mapping string labels
        y = df["income"].apply(lambda v: 1 if ">50K" in str(v) else 0)

    X = df.drop(columns=[c for c in [target, "income", "income_binary"] if c in df.columns])

    # Identify numeric and categorical columns
    num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()

    # Preprocessing pipeline
    numeric_transformer = StandardScaler()
    # scikit-learn changed the `sparse` arg name to `sparse_output` in newer versions.
    try:
        categorical_transformer = OneHotEncoder(handle_unknown="ignore", sparse=False)
    except TypeError:
        categorical_transformer = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, num_cols),
            ("cat", categorical_transformer, cat_cols),
        ],
        remainder="drop",
    )

    clf = LogisticRegression(max_iter=1000, C=C, random_state=random_state)
    pipe = Pipeline(steps=[("preprocessor", preprocessor), ("clf", clf)])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    acc = float(accuracy_score(y_test, y_pred))

    return pipe, acc, y_pred, y_test
