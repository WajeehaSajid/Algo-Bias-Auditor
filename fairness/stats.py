import numpy as np
import pandas as pd
from typing import Tuple
import scipy.stats as st

def descriptive_stats(df: pd.DataFrame, group_by: str = None) -> pd.DataFrame:
    if group_by is None:
        return df.describe(include="all")
    else:
        return df.groupby(group_by).describe()

def mean_confidence_interval(data: np.ndarray, confidence: float = 0.95) -> Tuple[float, float]:
    a = 1.0 * np.array(data)
    n = len(a)
    m = np.mean(a)
    se = st.sem(a)
    h = se * st.t.ppf((1 + confidence) / 2., n-1)
    return m - h, m + h

def proportion_confidence_interval(count: int, nobs: int, alpha: float = 0.05) -> Tuple[float, float]:
    """Wilson score interval for a proportion (implemented without statsmodels).

    Returns (lower, upper).
    """
    if nobs == 0:
        return 0.0, 0.0
    p = count / nobs
    z = st.norm.ppf(1 - alpha / 2)
    denom = 1 + (z ** 2) / nobs
    centre = p + (z ** 2) / (2 * nobs)
    margin = z * ((p * (1 - p) / nobs) + (z ** 2) / (4 * nobs ** 2)) ** 0.5
    lower = (centre - margin) / denom
    upper = (centre + margin) / denom
    lower = max(0.0, lower)
    upper = min(1.0, upper)
    return lower, upper

def grouped_numeric_stats(df: pd.DataFrame, group_by: str, cols: list = None, confidence: float = 0.95) -> pd.DataFrame:
    """Compute mean, median, variance, n, and 95% CI for numeric columns grouped by `group_by`.

    Returns a tidy DataFrame with MultiIndex (group, column) and statistics.
    """
    if cols is None:
        cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    rows = []
    groups = df[group_by].unique()
    for g in groups:
        subset = df[df[group_by] == g]
        n = len(subset)
        for c in cols:
            series = subset[c].dropna().astype(float)
            if len(series) == 0:
                continue
            mean = float(series.mean())
            median = float(series.median())
            var = float(series.var(ddof=1))
            ci_low, ci_high = mean_confidence_interval(series.values, confidence=confidence)
            rows.append({"group": g, "column": c, "n": n, "mean": mean, "median": median, "variance": var, "ci_low": ci_low, "ci_high": ci_high})

    result = pd.DataFrame(rows).set_index(["group", "column"])
    return result
