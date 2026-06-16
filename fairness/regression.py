"""Correlation and regression analysis for the fairness audit app."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, Any, List, Tuple


# ── Correlation ───────────────────────────────────────────────────────────────

def correlation_matrix(df: pd.DataFrame, cols: List[str] = None) -> pd.DataFrame:
    """Return Pearson correlation matrix for numeric columns."""
    if cols is None:
        cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    return df[cols].corr(method="pearson")


def pairwise_correlation(
    df: pd.DataFrame, col_x: str, col_y: str
) -> Dict[str, Any]:
    """Pearson r, p-value, and interpretation for two columns."""
    x = df[col_x].dropna()
    y = df[col_y].dropna()
    common = x.index.intersection(y.index)
    x, y = x[common].values, y[common].values
    r, p_value = stats.pearsonr(x, y)
    return {
        "r": float(r),
        "r_squared": float(r ** 2),
        "p_value": float(p_value),
        "n": int(len(x)),
        "interpretation": _interpret_r(r),
        "significant": bool(p_value < 0.05),
    }


def covariance_matrix(df: pd.DataFrame, cols: List[str] = None) -> pd.DataFrame:
    """Return sample covariance matrix."""
    if cols is None:
        cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    return df[cols].cov()


# ── Simple Linear Regression (OLS) ────────────────────────────────────────────

def simple_ols(
    df: pd.DataFrame, x_col: str, y_col: str
) -> Dict[str, Any]:
    """
    Ordinary Least Squares simple linear regression y ~ x.
    Returns slope, intercept, R², SE, t-stat, p-value, residuals.
    """
    data = df[[x_col, y_col]].dropna()
    x = data[x_col].values
    y = data[y_col].values
    n = len(x)

    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    y_pred = slope * x + intercept
    residuals = y - y_pred
    ss_res = float(np.sum(residuals ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    # MSE and RMSE
    mse = ss_res / (n - 2) if n > 2 else 0.0
    rmse = float(np.sqrt(mse))

    # Confidence interval for slope (95%)
    t_crit = stats.t.ppf(0.975, df=n - 2)
    slope_ci = (float(slope - t_crit * std_err), float(slope + t_crit * std_err))

    return {
        "x_col": x_col,
        "y_col": y_col,
        "n": int(n),
        "slope": float(slope),
        "intercept": float(intercept),
        "r_value": float(r_value),
        "r_squared": float(r_squared),
        "adj_r_squared": float(1 - (1 - r_squared) * (n - 1) / (n - 2)) if n > 2 else 0.0,
        "std_err": float(std_err),
        "t_statistic": float(slope / std_err) if std_err > 0 else 0.0,
        "p_value": float(p_value),
        "mse": float(mse),
        "rmse": rmse,
        "slope_ci_95": slope_ci,
        "equation": f"{y_col} = {slope:.4f} × {x_col} + {intercept:.4f}",
        "x": x,
        "y": y,
        "y_pred": y_pred,
        "residuals": residuals,
        "significant": bool(p_value < 0.05),
        "conclusion": (
            f"The slope is statistically significant (p = {p_value:.4f} < 0.05). "
            f"{x_col} significantly predicts {y_col}. R² = {r_squared:.4f} means "
            f"{r_squared*100:.1f}% of variance in {y_col} is explained by {x_col}."
            if p_value < 0.05 else
            f"The slope is NOT statistically significant (p = {p_value:.4f} ≥ 0.05). "
            f"{x_col} does not significantly predict {y_col}."
        ),
    }


# ── Multiple Linear Regression (NumPy OLS via normal equations) ───────────────

def multiple_ols(
    df: pd.DataFrame, x_cols: List[str], y_col: str
) -> Dict[str, Any]:
    """Multiple OLS via normal equations. Returns coefficients and R²."""
    data = df[x_cols + [y_col]].dropna()
    X = data[x_cols].values
    y = data[y_col].values
    n, k = X.shape

    # Add intercept column
    X_b = np.column_stack([np.ones(n), X])
    try:
        beta = np.linalg.lstsq(X_b, y, rcond=None)[0]
    except np.linalg.LinAlgError:
        return {"error": "Could not solve normal equations."}

    y_pred = X_b @ beta
    residuals = y - y_pred
    ss_res = float(np.sum(residuals ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    adj_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - k - 1) if n > k + 1 else 0.0

    coef_dict = {"Intercept": float(beta[0])}
    for col, b in zip(x_cols, beta[1:]):
        coef_dict[col] = float(b)

    return {
        "x_cols": x_cols,
        "y_col": y_col,
        "n": int(n),
        "coefficients": coef_dict,
        "r_squared": float(r_squared),
        "adj_r_squared": float(adj_r_squared),
        "rmse": float(np.sqrt(ss_res / max(n - k - 1, 1))),
        "residuals": residuals,
        "y_pred": y_pred,
        "y": y,
    }


# ── Helpers ───────────────────────────────────────────────────────────────────

def _interpret_r(r: float) -> str:
    a = abs(r)
    direction = "positive" if r >= 0 else "negative"
    if a < 0.1:
        strength = "negligible"
    elif a < 0.3:
        strength = "weak"
    elif a < 0.5:
        strength = "moderate"
    elif a < 0.7:
        strength = "strong"
    else:
        strength = "very strong"
    return f"{strength.capitalize()} {direction} correlation"
