"""Hypothesis testing functions for the fairness audit app."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, Any, Tuple


# ── Two-sample t-test ──────────────────────────────────────────────────────────

def two_sample_ttest(
    df: pd.DataFrame,
    numeric_col: str,
    group_col: str,
    group_a: str,
    group_b: str,
    alpha: float = 0.05,
    equal_var: bool = False,
) -> Dict[str, Any]:
    """
    Welch's two-sample t-test (equal_var=False by default) comparing
    numeric_col between group_a and group_b in group_col.
    """
    a = df[df[group_col] == group_a][numeric_col].dropna().values
    b = df[df[group_col] == group_b][numeric_col].dropna().values

    t_stat, p_value = stats.ttest_ind(a, b, equal_var=equal_var)

    # Cohen's d effect size
    pooled_std = np.sqrt((np.std(a, ddof=1) ** 2 + np.std(b, ddof=1) ** 2) / 2)
    cohens_d = (np.mean(a) - np.mean(b)) / pooled_std if pooled_std > 0 else 0.0

    reject = bool(p_value < alpha)
    return {
        "test": "Welch's Two-Sample t-Test" if not equal_var else "Two-Sample t-Test",
        "group_a": group_a,
        "group_b": group_b,
        "n_a": int(len(a)),
        "n_b": int(len(b)),
        "mean_a": float(np.mean(a)),
        "mean_b": float(np.mean(b)),
        "std_a": float(np.std(a, ddof=1)),
        "std_b": float(np.std(b, ddof=1)),
        "t_statistic": float(t_stat),
        "p_value": float(p_value),
        "alpha": alpha,
        "reject_h0": reject,
        "cohens_d": float(cohens_d),
        "effect_size_label": _effect_size_label(abs(cohens_d)),
        "h0": f"H₀: Mean {numeric_col} of '{group_a}' = Mean {numeric_col} of '{group_b}'",
        "h1": f"H₁: Mean {numeric_col} of '{group_a}' ≠ Mean {numeric_col} of '{group_b}'",
        "conclusion": (
            f"Reject H₀ (p = {p_value:.4f} < α = {alpha}). "
            f"There IS a statistically significant difference in {numeric_col} between '{group_a}' and '{group_b}'."
            if reject else
            f"Fail to Reject H₀ (p = {p_value:.4f} ≥ α = {alpha}). "
            f"No statistically significant difference in {numeric_col} between '{group_a}' and '{group_b}'."
        ),
        "data_a": a,
        "data_b": b,
    }


# ── Chi-square test of independence ───────────────────────────────────────────

def chi_square_test(
    df: pd.DataFrame,
    col1: str,
    col2: str,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """Chi-square test of independence between two categorical columns."""
    contingency = pd.crosstab(df[col1], df[col2])
    chi2, p_value, dof, expected = stats.chi2_contingency(contingency)

    # Cramér's V effect size
    n = contingency.values.sum()
    min_dim = min(contingency.shape) - 1
    cramers_v = float(np.sqrt(chi2 / (n * min_dim))) if min_dim > 0 else 0.0

    reject = bool(p_value < alpha)
    return {
        "test": "Chi-Square Test of Independence",
        "col1": col1,
        "col2": col2,
        "chi2_statistic": float(chi2),
        "p_value": float(p_value),
        "degrees_of_freedom": int(dof),
        "alpha": alpha,
        "reject_h0": reject,
        "cramers_v": cramers_v,
        "effect_size_label": _cramers_label(cramers_v),
        "h0": f"H₀: '{col1}' and '{col2}' are independent.",
        "h1": f"H₁: '{col1}' and '{col2}' are NOT independent.",
        "conclusion": (
            f"Reject H₀ (χ² = {chi2:.3f}, p = {p_value:.4f} < α = {alpha}). "
            f"'{col1}' and '{col2}' are significantly associated."
            if reject else
            f"Fail to Reject H₀ (χ² = {chi2:.3f}, p = {p_value:.4f} ≥ α = {alpha}). "
            f"No significant association between '{col1}' and '{col2}'."
        ),
        "contingency_table": contingency,
        "expected_table": pd.DataFrame(expected, index=contingency.index, columns=contingency.columns),
    }


# ── One-sample z-test for proportion ──────────────────────────────────────────

def proportion_ztest(
    count: int,
    nobs: int,
    p0: float = 0.5,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """One-sample z-test: H₀: p = p0."""
    p_hat = count / nobs if nobs > 0 else 0.0
    se = np.sqrt(p0 * (1 - p0) / nobs) if nobs > 0 else 1.0
    z_stat = (p_hat - p0) / se if se > 0 else 0.0
    p_value = float(2 * (1 - stats.norm.cdf(abs(z_stat))))
    reject = bool(p_value < alpha)
    return {
        "test": "One-Sample Z-Test for Proportion",
        "p_hat": float(p_hat),
        "p0": p0,
        "z_statistic": float(z_stat),
        "p_value": p_value,
        "alpha": alpha,
        "reject_h0": reject,
        "h0": f"H₀: p = {p0}",
        "h1": f"H₁: p ≠ {p0}",
        "conclusion": (
            f"Reject H₀ (z = {z_stat:.3f}, p = {p_value:.4f} < α = {alpha})."
            if reject else
            f"Fail to Reject H₀ (z = {z_stat:.3f}, p = {p_value:.4f} ≥ α = {alpha})."
        ),
    }


# ── Helpers ───────────────────────────────────────────────────────────────────

def _effect_size_label(d: float) -> str:
    if d < 0.2:
        return "Negligible"
    elif d < 0.5:
        return "Small"
    elif d < 0.8:
        return "Medium"
    else:
        return "Large"


def _cramers_label(v: float) -> str:
    if v < 0.1:
        return "Negligible"
    elif v < 0.3:
        return "Small"
    elif v < 0.5:
        return "Medium"
    else:
        return "Large"
