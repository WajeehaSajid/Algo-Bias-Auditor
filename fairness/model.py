import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, roc_auc_score
from typing import Tuple

def train_logistic_regression(X_train, y_train, preprocessor, C: float = 1.0, random_state: int = 42) -> Pipeline:
    clf = LogisticRegression(max_iter=1000, C=C, random_state=random_state)
    pipe = Pipeline(steps=[("preprocessor", preprocessor), ("clf", clf)])
    pipe.fit(X_train, y_train)
    return pipe

def evaluate_model(pipe: Pipeline, X_test, y_test) -> dict:
    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1] if hasattr(pipe, "predict_proba") else None
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
    }
    if y_proba is not None:
        metrics["roc_auc"] = float(roc_auc_score(y_test, y_proba))
    return {"metrics": metrics, "y_pred": y_pred}
