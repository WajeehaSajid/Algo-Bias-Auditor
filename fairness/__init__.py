"""Fairness audit package."""

from .data_loader import load_adult_dataset
from .fairness import (
    conditional_probability,
    demographic_parity,
    disparate_impact,
    equalized_odds,
    equalized_odds_verdict,
    group_ci_table,
    wilson_ci,
)
from .stats import grouped_numeric_stats, mean_confidence_interval
from .ml_model import train_model
from .distributions import (
    fit_normal, fit_binomial, fit_poisson,
    shapiro_wilk_test, ks_test_normal,
    get_normal_pdf, get_binomial_pmf, get_poisson_pmf,
    distribution_summary,
)
from .hypothesis import two_sample_ttest, chi_square_test, proportion_ztest
from .regression import correlation_matrix, pairwise_correlation, simple_ols, multiple_ols

__all__ = [
    "load_adult_dataset",
    "conditional_probability", "demographic_parity", "disparate_impact",
    "equalized_odds", "equalized_odds_verdict", "group_ci_table", "wilson_ci",
    "grouped_numeric_stats", "mean_confidence_interval",
    "train_model",
    "fit_normal", "fit_binomial", "fit_poisson",
    "shapiro_wilk_test", "ks_test_normal",
    "get_normal_pdf", "get_binomial_pmf", "get_poisson_pmf", "distribution_summary",
    "two_sample_ttest", "chi_square_test", "proportion_ztest",
    "correlation_matrix", "pairwise_correlation", "simple_ols", "multiple_ols",
]
