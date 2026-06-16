"""Probability distribution fitting and analysis for the fairness audit app."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, Tuple, Any


def fit_normal(data: np.ndarray) -> Dict[str, float]:
    """Fit a Normal distribution and return parameters."""
    mu, sigma = stats.norm.fit(data)
    return {"mu": float(mu), "sigma": float(sigma)}


def fit_poisson(data: np.ndarray) -> Dict[str, float]:
    """Estimate Poisson lambda as the sample mean."""
    lam = float(np.mean(data))
    return {"lambda": lam}


def fit_binomial(data: np.ndarray, n: int = None) -> Dict[str, float]:
    """Estimate Binomial p given n (defaults to max of data)."""
    if n is None:
        n = int(np.max(data))
    p = float(np.mean(data) / n) if n > 0 else 0.0
    return {"n": n, "p": p}


def shapiro_wilk_test(data: np.ndarray) -> Dict[str, Any]:
    """Run Shapiro-Wilk normality test. Samples down to 5000 if needed."""
    sample = data if len(data) <= 5000 else np.random.choice(data, 5000, replace=False)
    stat, p_value = stats.shapiro(sample)
    return {
        "statistic": float(stat),
        "p_value": float(p_value),
        "normal": bool(p_value > 0.05),
        "conclusion": (
            f"p = {p_value:.4f} > 0.05 → Fail to reject H₀. Data appears normal."
            if p_value > 0.05
            else f"p = {p_value:.4f} ≤ 0.05 → Reject H₀. Data is NOT normally distributed."
        ),
    }


def ks_test_normal(data: np.ndarray) -> Dict[str, Any]:
    """Kolmogorov-Smirnov test against Normal distribution."""
    standardized = (data - np.mean(data)) / (np.std(data) + 1e-9)
    stat, p_value = stats.kstest(standardized, "norm")
    return {
        "statistic": float(stat),
        "p_value": float(p_value),
        "conclusion": (
            "KS test: Data fits Normal distribution well (p > 0.05)."
            if p_value > 0.05
            else "KS test: Significant deviation from Normal distribution (p ≤ 0.05)."
        ),
    }


def get_normal_pdf(mu: float, sigma: float, x_range: Tuple[float, float] = None, n: int = 300):
    """Return x, y arrays for Normal PDF curve."""
    if x_range is None:
        x_range = (mu - 4 * sigma, mu + 4 * sigma)
    x = np.linspace(x_range[0], x_range[1], n)
    y = stats.norm.pdf(x, mu, sigma)
    return x, y


def get_binomial_pmf(n: int, p: float):
    """Return k, prob arrays for Binomial PMF."""
    k = np.arange(0, n + 1)
    prob = stats.binom.pmf(k, n, p)
    return k, prob


def get_poisson_pmf(lam: float, k_max: int = None):
    """Return k, prob arrays for Poisson PMF."""
    if k_max is None:
        k_max = max(30, int(lam + 5 * np.sqrt(lam)))
    k = np.arange(0, k_max + 1)
    prob = stats.poisson.pmf(k, lam)
    return k, prob


def distribution_summary(dist_name: str, params: Dict) -> str:
    """Return a human-readable summary of distribution parameters."""
    if dist_name == "Normal":
        mu, sigma = params["mu"], params["sigma"]
        return (
            f"μ = {mu:.3f}, σ = {sigma:.3f}\n"
            f"Variance = σ² = {sigma**2:.3f}\n"
            f"68% of data lies within [{mu-sigma:.2f}, {mu+sigma:.2f}]\n"
            f"95% of data lies within [{mu-2*sigma:.2f}, {mu+2*sigma:.2f}]"
        )
    elif dist_name == "Binomial":
        n, p = int(params["n"]), params["p"]
        mean = n * p
        var = n * p * (1 - p)
        return (
            f"n = {n}, p = {p:.4f}\n"
            f"Mean = np = {mean:.3f}\n"
            f"Variance = np(1-p) = {var:.3f}"
        )
    elif dist_name == "Poisson":
        lam = params["lambda"]
        return (
            f"λ = {lam:.3f}\n"
            f"Mean = λ = {lam:.3f}\n"
            f"Variance = λ = {lam:.3f}"
        )
    return ""
