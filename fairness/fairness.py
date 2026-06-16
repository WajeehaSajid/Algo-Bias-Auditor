"""Fairness metrics for the algorithmic bias audit app."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, Tuple, Any


def conditional_probability(
    df: pd.DataFrame,
    protected_attr: str,
    group_value,
    outcome_col: str = "income_binary",
) -> Tuple[float, int, int]:
    subset = df[df[protected_attr] == group_value]
    total = len(subset)
    if total == 0:
        return 0.0, 0, 0
    if outcome_col in subset.columns and pd.api.types.is_numeric_dtype(subset[outcome_col].dtype):
        successes = int(subset[outcome_col].sum())
    else:
        successes = int(subset[outcome_col].apply(lambda v: 1 if ">50K" in str(v) else 0).sum())
    return successes / total, successes, total


def demographic_parity(
    df: pd.DataFrame, protected_attr: str, outcome_col: str
) -> Dict[str, float]:
    groups = df[protected_attr].unique()
    results = {}
    for g in groups:
        subset = df[df[protected_attr] == g]
        if subset.empty:
            results[str(g)] = 0.0
            continue
        if pd.api.types.is_numeric_dtype(subset[outcome_col].dtype):
            results[str(g)] = float(subset[outcome_col].mean())
        else:
            count = subset[outcome_col].apply(lambda v: 1 if ">50K" in str(v) else 0).sum()
            results[str(g)] = count / len(subset)
    return results


def disparate_impact(
    df: pd.DataFrame,
    protected_attr: str,
    outcome_col: str,
    privileged_group: str = None,
) -> Dict[str, Any]:
    rates = demographic_parity(df, protected_attr, outcome_col)
    if not rates:
        return {}
    if privileged_group is None:
        privileged_group = max(rates, key=rates.get)
    priv_rate = rates.get(str(privileged_group), 0.0)
    results = {}
    for g, rate in rates.items():
        if g == str(privileged_group):
            di = 1.0
        else:
            di = rate / priv_rate if priv_rate > 0 else 0.0
        results[g] = {
            "positive_rate": rate,
            "disparate_impact": di,
            "passes_80_rule": di >= 0.8,
            "verdict": "PASSES" if di >= 0.8 else "FAILS",
        }
    return {
        "privileged_group": str(privileged_group),
        "privileged_rate": priv_rate,
        "groups": results,
        "overall_verdict": (
            "No Adverse Impact Detected"
            if all(v["passes_80_rule"] for v in results.values())
            else "Adverse Impact Detected"
        ),
    }


def equalized_odds(
    y_true: pd.Series,
    y_pred: pd.Series,
    protected: pd.Series,
) -> Dict[str, Dict[str, float]]:
    results = {}
    groups = protected.unique()
    for g in groups:
        mask = protected == g
        tp = int(((y_true == 1) & (y_pred == 1) & mask).sum())
        fn = int(((y_true == 1) & (y_pred == 0) & mask).sum())
        fp = int(((y_true == 0) & (y_pred == 1) & mask).sum())
        tn = int(((y_true == 0) & (y_pred == 0) & mask).sum())
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        results[str(g)] = {
            "TP": tp, "FN": fn, "FP": fp, "TN": tn,
            "TPR": float(tpr),
            "FPR": float(fpr),
            "Precision": float(precision),
            "F1": float(2 * precision * tpr / (precision + tpr)) if (precision + tpr) > 0 else 0.0,
        }
    return results


def equalized_odds_verdict(eo_results: Dict[str, Dict[str, float]], tol: float = 0.05) -> Dict[str, Any]:
    tprs = [v["TPR"] for v in eo_results.values()]
    fprs = [v["FPR"] for v in eo_results.values()]
    tpr_gap = float(max(tprs) - min(tprs))
    fpr_gap = float(max(fprs) - min(fprs))
    passes = tpr_gap <= tol and fpr_gap <= tol
    return {
        "tpr_gap": tpr_gap,
        "fpr_gap": fpr_gap,
        "tolerance": tol,
        "passes": passes,
        "verdict": (
            f"Equalized Odds satisfied (TPR gap={tpr_gap:.3f}, FPR gap={fpr_gap:.3f})"
            if passes else
            f"Equalized Odds VIOLATED (TPR gap={tpr_gap:.3f}, FPR gap={fpr_gap:.3f})"
        ),
    }


def wilson_ci(count: int, nobs: int, alpha: float = 0.05) -> Tuple[float, float]:
    if nobs == 0:
        return 0.0, 0.0
    p = count / nobs
    z = stats.norm.ppf(1 - alpha / 2)
    denom = 1 + (z ** 2) / nobs
    centre = p + (z ** 2) / (2 * nobs)
    margin = z * ((p * (1 - p) / nobs) + (z ** 2) / (4 * nobs ** 2)) ** 0.5
    return max(0.0, (centre - margin) / denom), min(1.0, (centre + margin) / denom)


def group_ci_table(
    df: pd.DataFrame,
    protected_attr: str,
    outcome_col: str = "income_binary",
    alpha: float = 0.05,
) -> pd.DataFrame:
    rows = []
    for g in df[protected_attr].unique():
        prob, successes, total = conditional_probability(df, protected_attr, g, outcome_col)
        lo, hi = wilson_ci(successes, total, alpha)
        rows.append({
            "Group": str(g),
            "N": total,
            "Positive Count": successes,
            "Positive Rate": round(prob, 4),
            "CI Lower (95%)": round(lo, 4),
            "CI Upper (95%)": round(hi, 4),
            "CI Width": round(hi - lo, 4),
        })
    return pd.DataFrame(rows).set_index("Group")
