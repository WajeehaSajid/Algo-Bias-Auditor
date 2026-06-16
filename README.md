# Algorithmic Bias & Fairness Auditor

![Python](https://img.shields.io/badge/Python-3.13%2B-blue?style=for-the-badge&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-App-orange?style=for-the-badge&logo=streamlit)
![License](https://img.shields.io/badge/License-Academic-green?style=for-the-badge)

## Overview

Algorithmic Bias & Fairness Auditor is a modular Streamlit application for analyzing demographic bias in machine learning systems. It uses the UCI Adult Income dataset to explore how model outcomes vary across protected groups, and presents results through a Material Design 3 inspired interactive dashboard with seven analysis tabs.

---

## Features

### Data Overview
Displays dataset shape, a 200-row preview of the cleaned Adult Income data, missing value detection, and an income class distribution pie chart.

### Descriptive Statistics & Confidence Intervals
Computes grouped summary statistics (mean, median, variance, standard deviation) with 95% confidence intervals for key numeric variables, segmented by a user-selected protected attribute (sex or race). Includes histogram and box plot comparisons.

### Probability Distributions
Fits Normal, Binomial, and Poisson distributions to any numeric column with parameter estimation. Includes empirical histogram overlay, distribution parameter summary, Shapiro-Wilk and Kolmogorov-Smirnov normality tests, and Q-Q plots.

### Hypothesis Testing
- Two-Sample t-Test — compares means of a numeric variable across two demographic groups with Cohen's d effect size
- Chi-Square Test of Independence — tests association between two categorical variables with Cramer's V effect size and a contingency table heatmap

### Regression & Correlation
- Pearson Correlation Matrix across all numeric features with pairwise detail view
- Simple Linear Regression (OLS) — slope, intercept, R2, adjusted R2, RMSE, t-statistic, residual plot, and an interactive prediction tool
- Multiple Linear Regression (OLS) — multi-predictor model with coefficient table, bar chart, and an interactive prediction form for all selected predictors

### Fairness Audit
- Demographic Parity — conditional positive income probability per group
- Disparate Impact — 80% rule (EEOC standard) with pass/fail verdict per group
- Equalized Odds — trains a Logistic Regression model and computes TPR and FPR per group with gap metrics
- Wilson Score Confidence Intervals — 95% CIs for positive income rate per group

### AI Interpreter (Gemini-Powered)
Sends audit results to the Gemini API and generates a professional 4-paragraph plain-English interpretation covering demographic disparities, disparate impact analysis, statistical confidence, and bias remediation recommendations. Includes retry logic for API rate limits and a download button for the generated report.

---

## Tech Stack

| Library | Purpose |
|---|---|
| Streamlit | Interactive web application framework |
| Pandas | Data loading, cleaning, and transformation |
| Scikit-Learn | Logistic Regression model and evaluation |
| Plotly | Interactive charts and visualizations |
| SciPy | Statistical tests and distribution fitting |
| NumPy | Numerical computations |
| Google Gemini API | AI-powered audit interpretation |

---

## Installation & Usage

### 1. Clone the repository
```bash
git clone <your-repository-url>
cd algo-bias-auditor
```

### 2. Create and activate a virtual environment
```bash
python -m venv venv
```
On Windows:
```bash
venv\Scripts\activate
```
On macOS/Linux:
```bash
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit app
```bash
streamlit run app.py
```

### 5. Using the AI Interpreter (optional)
In the AI Interpreter tab, enter a valid [Google Gemini API key](https://aistudio.google.com) to generate an automated audit interpretation. The app uses `gemini-2.0-flash-lite` by default.

---

## Dataset

This project uses the UCI Adult Income dataset for fairness analysis.

Dataset link: https://archive.ics.uci.edu/ml/datasets/adult

The dataset is loaded automatically via `data_loader.py` — no manual download required.

---

## Project Structure

```
algo-bias-auditor/
├── app.py                    # Main Streamlit interface (7 tabs)
├── fairness/
│   ├── __init__.py
│   ├── data_loader.py        # Dataset loading and cleaning
│   ├── stats.py              # Grouped descriptive statistics
│   ├── distributions.py      # Distribution fitting and normality tests
│   ├── hypothesis.py         # t-Test and Chi-Square test logic
│   ├── regression.py         # OLS simple and multiple regression
│   ├── fairness.py           # Fairness metrics (DI, EO, CI, parity)
│   └── ml_model.py           # Logistic Regression training
├── logo_white.png
├── requirements.txt
└── README.md
```

---

## Notes

- The protected attribute (sex or race) and significance level (alpha) can be changed at any time from the sidebar and all tabs update accordingly.
- The Equalized Odds tab requires clicking Train Model before results appear, as model training is on-demand.
- The AI Interpreter tab requires a valid Gemini API key. Free-tier keys are subject to rate limits; the app automatically retries up to 3 times with exponential backoff.
- This repository is intended for academic fairness auditing. It can be extended with additional fairness metrics, alternative datasets, or more advanced model evaluation.

---

**Course:** MT2005 · Probability & Statistics · FAST-NUCES CFD · Spring 2026
