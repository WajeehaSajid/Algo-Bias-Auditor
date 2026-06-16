from __future__ import annotations

import json
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
import streamlit as st

from fairness.data_loader import load_adult_dataset
from fairness.fairness import (
    conditional_probability,
    disparate_impact,
    equalized_odds,
    equalized_odds_verdict,
    group_ci_table,
)
from fairness.ml_model import train_model
from fairness.stats import grouped_numeric_stats
from fairness.distributions import (
    fit_normal, fit_binomial, fit_poisson,
    shapiro_wilk_test, ks_test_normal,
    get_normal_pdf, get_binomial_pmf, get_poisson_pmf,
    distribution_summary,
)
from fairness.hypothesis import two_sample_ttest, chi_square_test
from fairness.regression import correlation_matrix, pairwise_correlation, simple_ols, multiple_ols

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Algorithmic Bias & Fairness Audit",
    page_icon="icon.png",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Google+Sans:wght@400;500;700&family=Roboto+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Google Sans', 'DM Sans', sans-serif;
}
h1, h2, h3 {
    font-family: 'Roboto Mono', monospace;
    letter-spacing: -0.3px;
}

/* ── Verdict badges ── */
.verdict-pass {
    background: linear-gradient(135deg, #003322 0%, #004d33 100%);
    color: #4ade80;
    border-radius: 10px;
    padding: 12px 18px;
    font-weight: 600;
    font-family: 'Roboto Mono', monospace;
    font-size: 0.9rem;
    margin: 8px 0;
    display: flex;
    align-items: center;
    gap: 8px;
    transition: background 0.2s;
}
.verdict-fail {
    background: linear-gradient(135deg, #2d0a1a 0%, #4a0e1e 100%);
    color: #fb7185;
    border-radius: 10px;
    padding: 12px 18px;
    font-weight: 600;
    font-family: 'Roboto Mono', monospace;
    font-size: 0.9rem;
    margin: 8px 0;
    display: flex;
    align-items: center;
    gap: 8px;
    transition: background 0.2s;
}

/* ── Stat card ── */
.stat-card {
    background: #1a1a3e;
    border: 1px solid #312e81;
    border-radius: 14px;
    padding: 16px 20px;
    margin: 6px 0;
    transition: border-color 0.2s;
}
.stat-card:hover {
    border-color: #6366f1;
}

/* ── Hypothesis box ── */
.hypothesis-box {
    background: linear-gradient(135deg, #1e0a3c 0%, #190833 100%);
    border-left: 3px solid #a855f7;
    border-radius: 0 10px 10px 0;
    padding: 12px 16px;
    font-family: 'Roboto Mono', monospace;
    font-size: 0.85rem;
    color: #d8b4fe;
    margin: 8px 0;
    line-height: 1.7;
}

/* ── AI response panel ── */
.ai-response {
    background: #0a1628;
    border: 1.5px solid #1d4ed8;
    border-radius: 14px;
    padding: 20px 24px;
    font-size: 0.95rem;
    line-height: 1.8;
    color: #bfdbfe;
    position: relative;
}
.ai-response::before {
    content: "AI ANALYSIS";
    display: block;
    font-family: 'Roboto Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 1.5px;
    color: #3b82f6;
    margin-bottom: 12px;
    font-weight: 500;
}

/* ── Prediction boxes ── */
.prediction-box {
    background: #0a1e2e;
    border: 1.5px solid #0284c7;
    border-radius: 12px;
    padding: 18px 22px;
    margin: 12px 0;
    font-family: 'Roboto Mono', monospace;
}
.prediction-result {
    background: #022c22;
    border: 1.5px solid #059669;
    border-radius: 10px;
    padding: 14px 20px;
    font-weight: 600;
    font-family: 'Roboto Mono', monospace;
    font-size: 1rem;
    margin: 10px 0;
    color: #34d399;
}

/* ── Streamlit metric overrides ── */
[data-testid="metric-container"] {
    background: #1a1a3e;
    border: 1px solid #312e81;
    border-radius: 14px;
    padding: 14px 16px;
}
[data-testid="metric-container"] label {
    color: #a5b4fc !important;
    font-size: 0.8rem;
}
[data-testid="stMetricValue"] {
    color: #e0e7ff !important;
    font-family: 'Roboto Mono', monospace;
}

/* ── Tab styling ── */
[data-baseweb="tab-list"] {
    gap: 6px;
    background: #202021 !important;
    padding-bottom: 4px;
}
[data-baseweb="tab"] {
    border-radius: 8px 8px 0 0 !important;
    font-family: 'Google Sans', sans-serif !important;
    font-size: 0.85rem !important;
    color: #9E9E9E !important;
    padding: 8px 16px !important;
    transition: background 0.2s;
}
[aria-selected="true"][data-baseweb="tab"] {
    background: #2F2F3A !important;
    color: #EDEDEF !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #1f1f1f !important;
    border-right: 1px solid #3B3B49;
}
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stMarkdown {
    color: #94a3b8 !important;
}

/* ── Dataframe ── */
[data-testid="stDataFrame"] {
    border: 1px solid #3B3B49;
    border-radius: 10px;
    overflow: hidden;
}

/* ── Global page background ── */
.stApp {
    background: #0f0f0f;
}
.block-container {
    background: #0f0f0f !important;
    padding-top: 2rem;
}

/* ── Divider ── */
hr {
    border-color: #3B3B49 !important;
    transition: background 0.2s;
}

/* ── Spinner / alerts ── */
.stAlert {
    border-radius: 10px !important;
    border-left-width: 3px !important;
    transition: background 0.2s;
}
</style>
""", unsafe_allow_html=True)

# ── Data loading ──────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    return load_adult_dataset()

if "df" not in st.session_state:
    with st.spinner("Just a moment. Loading Adult Income dataset..."):
        st.session_state.df = load_data()

df = st.session_state.df

NUMERIC_COLS = ["age", "education-num", "hours-per-week", "capital-gain", "capital-loss", "fnlwgt"]
NUMERIC_COLS = [c for c in NUMERIC_COLS if c in df.columns]
CAT_COLS = ["workclass", "education", "marital-status", "occupation", "relationship",
            "race", "sex", "native-country"]
CAT_COLS = [c for c in CAT_COLS if c in df.columns]

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.image("logo.png", width=256)
st.sidebar.title("Fairness Audit Project")
st.sidebar.markdown("**Adult Income Dataset**")
st.sidebar.markdown(f"`{df.shape[0]:,}` rows · `{df.shape[1]}` columns")
st.sidebar.divider()
protected_attr = st.sidebar.selectbox("🔒 Protected Attribute", ["sex", "race"], index=0)
alpha_level = st.sidebar.selectbox("Significance Level (α)", [0.05, 0.01, 0.10], index=0)
st.sidebar.divider()
st.sidebar.markdown(f"Made by:")
st.sidebar.markdown("24F0508 · Bilal Rauf")
st.sidebar.markdown("24F0806 · Wajeeha Sajid")
st.sidebar.markdown("24F0729 · Meerab Fatima")
st.sidebar.markdown("24F3123 · Sadia Sehar")
st.sidebar.markdown("24F3064 · Saba Ashraf")
st.sidebar.divider()
st.sidebar.caption("Made for the 2024 Spring Semester Probability & Statistics project.")
st.sidebar.caption("MT2005 · FAST-NUCES CFD")

# ── Title ─────────────────────────────────────────────────────────────────────
st.title("Algorithmic Bias & Fairness Audit")
st.caption("A statistical fairness audit of the UCI Adult Income dataset across demographic groups.")
st.divider()

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "Data Overview",
    "Descriptive Stats",
    "Distributions",
    "Hypothesis Testing",
    "Regression & Correlation",
    "Fairness Audit",
    "AI Interpreter",
])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — DATA OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Dataset Preview")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Records", f"{df.shape[0]:,}")
    col2.metric("Features", df.shape[1])
    col3.metric("Income > 50K", f"{df['income_binary'].sum():,}")
    col4.metric("Positive Rate", f"{df['income_binary'].mean()*100:.1f}%")
    st.dataframe(df.head(200), width='stretch')

    st.subheader("Missing Values")
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if missing.empty:
        st.success("Found no missing values in the cleaned dataset.")
    else:
        st.dataframe(missing.rename("Missing Count"), width='stretch')

    st.subheader("Class Distribution")
    class_fig = px.pie(
        df, names="income", color_discrete_sequence=["#6366f1","#f59e0b"],
        title="Income Class Distribution",
    )
    class_fig.update_layout(template="plotly_dark")
    st.plotly_chart(class_fig, width='stretch')

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — DESCRIPTIVE STATS
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Grouped Descriptive Statistics")
    st.caption(f"Statistics grouped by **{protected_attr}** with 95% Confidence Intervals")

    with st.spinner("Computing statistics..."):
        stats_df = grouped_numeric_stats(df, protected_attr,
                                         cols=["age","education-num","hours-per-week","capital-gain","capital-loss"])
    st.dataframe(stats_df.style.format("{:.3f}"), width='stretch')

    st.subheader("Distribution Visualization")
    hist_col = st.selectbox("Select variable", NUMERIC_COLS, key="hist_col")
    hist_fig = px.histogram(
        df, x=hist_col, color=protected_attr, barmode="overlay", opacity=0.75,
        nbins=40, title=f"{hist_col} Distribution by {protected_attr.capitalize()}",
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    hist_fig.update_layout(template="plotly_dark", title=dict(x=0.5))
    st.plotly_chart(hist_fig, width='stretch')

    st.subheader("Box Plot Comparison")
    box_col = st.selectbox("Variable for box plot", NUMERIC_COLS, key="box_col")
    box_fig = px.box(
        df, x=protected_attr, y=box_col, color=protected_attr,
        title=f"{box_col} by {protected_attr.capitalize()}",
        color_discrete_sequence=px.colors.qualitative.Pastel,
    )
    box_fig.update_layout(template="plotly_dark", showlegend=False, title=dict(x=0.5))
    st.plotly_chart(box_fig, width='stretch')

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — PROBABILITY DISTRIBUTIONS  (Member 1)
# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("Probability Distribution Explorer")
    st.caption("Fit theoretical distributions to real dataset columns and test for normality.")

    dist_col1, dist_col2 = st.columns([1, 2])
    with dist_col1:
        dist_type = st.radio("Distribution", ["Normal", "Binomial", "Poisson"], index=0)
        data_col = st.selectbox("Dataset Column", NUMERIC_COLS, key="dist_data_col")
        show_empirical = st.checkbox("Overlay empirical histogram", value=True)

    data_vals = df[data_col].dropna().values.astype(float)

    with dist_col2:
        if dist_type == "Normal":
            params = fit_normal(data_vals)
            x_curve, y_curve = get_normal_pdf(params["mu"], params["sigma"])

            fig = go.Figure()
            if show_empirical:
                fig.add_trace(go.Histogram(
                    x=data_vals, histnorm="probability density", name="Empirical",
                    opacity=0.5, marker_color="#6366f1",
                    nbinsx=50,
                ))
            fig.add_trace(go.Scatter(
                x=x_curve, y=y_curve, mode="lines", name="Normal PDF",
                line=dict(color="#f59e0b", width=3),
            ))
            fig.add_vline(x=params["mu"], line_dash="dash", line_color="#4ade80",
                          annotation_text=f"μ={params['mu']:.2f}")
            fig.add_vline(x=params["mu"]-params["sigma"], line_dash="dot", line_color="#94a3b8")
            fig.add_vline(x=params["mu"]+params["sigma"], line_dash="dot", line_color="#94a3b8",
                          annotation_text="±1σ")
            fig.update_layout(
                template="plotly_dark", title=f"Normal Fit: {data_col}",
                xaxis_title=data_col, yaxis_title="Density",
            )
            st.plotly_chart(fig, width='stretch')

        elif dist_type == "Poisson":
            params = fit_poisson(data_vals)
            k_vals, pmf_vals = get_poisson_pmf(params["lambda"])
            fig = go.Figure()
            fig.add_trace(go.Bar(x=k_vals, y=pmf_vals, name="Poisson PMF",
                                  marker_color="#f59e0b", opacity=0.85))
            if show_empirical:
                counts, bins = np.histogram(data_vals, bins=min(50, int(data_vals.max()-data_vals.min()+1)))
                bin_centers = (bins[:-1] + bins[1:]) / 2
                fig.add_trace(go.Bar(x=bin_centers, y=counts/counts.sum(),
                                      name="Empirical", marker_color="#6366f1", opacity=0.5))
            fig.update_layout(template="plotly_dark", barmode="overlay",
                               title=f"Poisson Fit: {data_col}", xaxis_title=data_col)
            st.plotly_chart(fig, width='stretch')

        elif dist_type == "Binomial":
            n_max = int(data_vals.max()) if len(data_vals) > 0 else 10
            n_max = min(n_max, 200)
            params = fit_binomial(data_vals, n=n_max)
            k_vals, pmf_vals = get_binomial_pmf(params["n"], params["p"])
            fig = go.Figure()
            fig.add_trace(go.Bar(x=k_vals, y=pmf_vals, name="Binomial PMF",
                                  marker_color="#f59e0b", opacity=0.85))
            fig.update_layout(template="plotly_dark", title=f"Binomial Fit: {data_col}")
            st.plotly_chart(fig, width='stretch')

    # Parameters summary
    st.subheader("Distribution Parameters")
    summary = distribution_summary(dist_type, params)
    st.code(summary, language=None)

    # Normality tests
    st.subheader("Normality Tests")
    st.caption("Shapiro-Wilk and Kolmogorov-Smirnov tests for normality.")

    nt_col1, nt_col2 = st.columns(2)
    with nt_col1:
        sw = shapiro_wilk_test(data_vals)
        st.markdown("**Shapiro-Wilk Test**")
        st.markdown(f'<div class="hypothesis-box">H₀: Data is normally distributed<br>W = {sw["statistic"]:.4f} &nbsp;|&nbsp; p = {sw["p_value"]:.4f}</div>', unsafe_allow_html=True)
        verdict_cls = "verdict-pass" if sw["normal"] else "verdict-fail"
        icon = "✅" if sw["normal"] else "❌"
        st.markdown(f'<div class="{verdict_cls}">{icon} {sw["conclusion"]}</div>', unsafe_allow_html=True)

    with nt_col2:
        ks = ks_test_normal(data_vals)
        st.markdown("**Kolmogorov-Smirnov Test**")
        st.markdown(f'<div class="hypothesis-box">H₀: Data follows Normal distribution<br>KS = {ks["statistic"]:.4f} &nbsp;|&nbsp; p = {ks["p_value"]:.4f}</div>', unsafe_allow_html=True)
        ks_pass = ks["p_value"] > 0.05
        verdict_cls = "verdict-pass" if ks_pass else "verdict-fail"
        icon = "✅" if ks_pass else "❌"
        st.markdown(f'<div class="{verdict_cls}">{icon} {ks["conclusion"]}</div>', unsafe_allow_html=True)

    # QQ Plot
    st.subheader("Q-Q Plot")
    sample_q = data_vals if len(data_vals) <= 3000 else np.random.choice(data_vals, 3000, replace=False)
    (osm, osr), (slope_qq, intercept_qq, r_qq) = stats.probplot(sample_q, dist="norm")
    qq_fig = go.Figure()
    qq_fig.add_trace(go.Scatter(x=osm, y=osr, mode="markers", name="Data Quantiles",
                                 marker=dict(color="#6366f1", size=3, opacity=0.6)))
    line_x = np.array([min(osm), max(osm)])
    qq_fig.add_trace(go.Scatter(x=line_x, y=slope_qq*line_x+intercept_qq,
                                  mode="lines", name="Normal Line",
                                  line=dict(color="#f59e0b", width=2)))
    qq_fig.update_layout(template="plotly_dark", title=f"Q-Q Plot: {data_col}",
                          xaxis_title="Theoretical Quantiles", yaxis_title="Sample Quantiles")
    st.plotly_chart(qq_fig, width='stretch')

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — HYPOTHESIS TESTING  (Member 2)
# ═══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.subheader("Hypothesis Testing Suite")

    ht_tabs = st.tabs(["Two-Sample t-Test", "Chi-Square Test of Independence"])

    # ── t-Test ────────────────────────────────────────────────────────────────
    with ht_tabs[0]:
        st.markdown("**Compare a numeric variable across two demographic groups**")
        t_col1, t_col2, t_col3 = st.columns(3)
        with t_col1:
            t_num_col = st.selectbox("Numeric Variable", NUMERIC_COLS, key="t_num")
        with t_col2:
            t_group_col = st.selectbox("Group By", ["sex", "race"], key="t_grp")
        with t_col3:
            t_alpha = st.selectbox("α level", [0.05, 0.01, 0.10], key="t_alpha")

        groups_available = sorted(df[t_group_col].dropna().unique().tolist())
        col_a, col_b = st.columns(2)
        group_a = col_a.selectbox("Group A", groups_available, index=0, key="ta")
        group_b = col_b.selectbox("Group B", groups_available,
                                    index=min(1, len(groups_available)-1), key="tb")

        if group_a != group_b:
            result = two_sample_ttest(df, t_num_col, t_group_col, group_a, group_b, alpha=t_alpha)

            st.markdown("**Hypotheses**")
            st.markdown(f'<div class="hypothesis-box">{result["h0"]}<br>{result["h1"]}</div>',
                        unsafe_allow_html=True)

            m1, m2, m3, m4 = st.columns(4)
            m1.metric(f"Mean ({group_a})", f"{result['mean_a']:.3f}")
            m2.metric(f"Mean ({group_b})", f"{result['mean_b']:.3f}")
            m3.metric("t-Statistic", f"{result['t_statistic']:.4f}")
            m4.metric("p-Value", f"{result['p_value']:.4f}")

            m5, m6 = st.columns(2)
            m5.metric("Cohen's d (Effect Size)", f"{result['cohens_d']:.4f}")
            m6.metric("Effect Size Label", result['effect_size_label'])

            verdict_cls = "verdict-fail" if result["reject_h0"] else "verdict-pass"
            icon = "❌" if result["reject_h0"] else "✅"
            st.markdown(f'<div class="{verdict_cls}">{icon} {result["conclusion"]}</div>',
                        unsafe_allow_html=True)

            # Visualization
            vio_fig = go.Figure()
            vio_fig.add_trace(go.Violin(y=result["data_a"], name=group_a, box_visible=True,
                                          meanline_visible=True, fillcolor="#6366f1",
                                          line_color="#818cf8", opacity=0.7))
            vio_fig.add_trace(go.Violin(y=result["data_b"], name=group_b, box_visible=True,
                                          meanline_visible=True, fillcolor="#f59e0b",
                                          line_color="#fbbf24", opacity=0.7))
            vio_fig.update_layout(template="plotly_dark",
                                    title=f"Distribution of {t_num_col}: {group_a} vs {group_b}",
                                    yaxis_title=t_num_col)
            st.plotly_chart(vio_fig, width='stretch')
        else:
            st.warning("Please select two different groups.")

    # ── Chi-Square ────────────────────────────────────────────────────────────
    with ht_tabs[1]:
        st.markdown("**Test whether two categorical variables are statistically independent**")
        chi_col1, chi_col2, chi_col3 = st.columns(3)
        cat_col1 = chi_col1.selectbox("Variable 1", CAT_COLS, index=CAT_COLS.index("sex") if "sex" in CAT_COLS else 0, key="chi1")
        cat_col2 = chi_col2.selectbox("Variable 2", CAT_COLS, index=CAT_COLS.index("occupation") if "occupation" in CAT_COLS else 1, key="chi2")
        chi_alpha = chi_col3.selectbox("α level", [0.05, 0.01, 0.10], key="chi_alpha")

        if cat_col1 != cat_col2:
            with st.spinner("Running Chi-Square test..."):
                chi_result = chi_square_test(df, cat_col1, cat_col2, alpha=chi_alpha)

            st.markdown("**Hypotheses**")
            st.markdown(f'<div class="hypothesis-box">{chi_result["h0"]}<br>{chi_result["h1"]}</div>',
                        unsafe_allow_html=True)

            cm1, cm2, cm3, cm4 = st.columns(4)
            cm1.metric("χ² Statistic", f"{chi_result['chi2_statistic']:.4f}")
            cm2.metric("p-Value", f"{chi_result['p_value']:.4f}")
            cm3.metric("Degrees of Freedom", chi_result['degrees_of_freedom'])
            cm4.metric("Cramér's V", f"{chi_result['cramers_v']:.4f}")

            st.metric("Effect Size", chi_result['effect_size_label'])

            verdict_cls = "verdict-fail" if chi_result["reject_h0"] else "verdict-pass"
            icon = "❌" if chi_result["reject_h0"] else "✅"
            st.markdown(f'<div class="{verdict_cls}">{icon} {chi_result["conclusion"]}</div>',
                        unsafe_allow_html=True)

            st.subheader("Contingency Table (Observed)")
            st.dataframe(chi_result["contingency_table"], width='stretch')

            # Heatmap of contingency table
            ct = chi_result["contingency_table"]
            hm_fig = px.imshow(
                ct.values, x=ct.columns.astype(str), y=ct.index.astype(str),
                color_continuous_scale="Viridis", text_auto=True,
                title=f"Contingency Table: {cat_col1} × {cat_col2}",
            )
            hm_fig.update_layout(template="plotly_dark")
            st.plotly_chart(hm_fig, width='stretch')
        else:
            st.warning("Please select two different categorical variables.")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5 — REGRESSION & CORRELATION  (Member 3)
# ═══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.subheader("Correlation & Regression Analysis")

    reg_tabs = st.tabs(["Correlation Matrix", "Simple Linear Regression", "Multiple Regression"])

    # ── Correlation ───────────────────────────────────────────────────────────
    with reg_tabs[0]:
        st.markdown("**Pearson Correlation Matrix — All Numeric Features**")
        corr = correlation_matrix(df, cols=NUMERIC_COLS + ["income_binary"])
        corr_fig = px.imshow(
            corr, text_auto=".2f", color_continuous_scale="RdBu_r",
            zmin=-1, zmax=1, title="Pearson Correlation Matrix",
            aspect="auto",
        )
        corr_fig.update_layout(template="plotly_dark", title=dict(x=0.5))
        st.plotly_chart(corr_fig, width='stretch')

        st.subheader("Pairwise Correlation Detail")
        pc1, pc2 = st.columns(2)
        pcol_x = pc1.selectbox("Variable X", NUMERIC_COLS, key="pcx")
        pcol_y = pc2.selectbox("Variable Y", NUMERIC_COLS,
                                 index=min(1, len(NUMERIC_COLS)-1), key="pcy")
        if pcol_x != pcol_y:
            pr = pairwise_correlation(df, pcol_x, pcol_y)
            pm1, pm2, pm3, pm4 = st.columns(4)
            pm1.metric("Pearson r", f"{pr['r']:.4f}")
            pm2.metric("R²", f"{pr['r_squared']:.4f}")
            pm3.metric("p-Value", f"{pr['p_value']:.4f}")
            pm4.metric("Interpretation", pr["interpretation"])
            sig_label = "✅ Significant (p < 0.05)" if pr["significant"] else "⚠️ Not significant (p ≥ 0.05)"
            st.info(sig_label)

    # ── Simple OLS ────────────────────────────────────────────────────────────
    with reg_tabs[1]:
        st.markdown("**Ordinary Least Squares — Simple Linear Regression**")
        r1, r2 = st.columns(2)
        ols_x = r1.selectbox("Predictor (X)", NUMERIC_COLS, key="olsx")
        ols_y = r2.selectbox("Response (Y)", NUMERIC_COLS,
                               index=min(1, len(NUMERIC_COLS)-1), key="olsy")

        if ols_x != ols_y:
            ols = simple_ols(df, ols_x, ols_y)
            st.markdown(f"**Regression Equation:** `{ols['equation']}`")

            om1, om2, om3, om4 = st.columns(4)
            om1.metric("Slope (β₁)", f"{ols['slope']:.4f}")
            om2.metric("Intercept (β₀)", f"{ols['intercept']:.4f}")
            om3.metric("R²", f"{ols['r_squared']:.4f}")
            om4.metric("Adj. R²", f"{ols['adj_r_squared']:.4f}")
            om5, om6, om7, om8 = st.columns(4)
            om5.metric("t-Statistic", f"{ols['t_statistic']:.4f}")
            om6.metric("p-Value", f"{ols['p_value']:.4f}")
            om7.metric("RMSE", f"{ols['rmse']:.4f}")
            om8.metric("n", ols['n'])

            verdict_cls = "verdict-pass" if ols["significant"] else "verdict-fail"
            st.markdown(f'<div class="{verdict_cls}">{"✅" if ols["significant"] else "⚠️"} {ols["conclusion"]}</div>',
                        unsafe_allow_html=True)

            # Scatter + regression line
            sample_idx = np.random.choice(len(ols["x"]), min(2000, len(ols["x"])), replace=False)
            scatter_fig = go.Figure()
            scatter_fig.add_trace(go.Scatter(
                x=ols["x"][sample_idx], y=ols["y"][sample_idx],
                mode="markers", name="Data", marker=dict(color="#6366f1", size=4, opacity=0.4),
            ))
            sort_idx = np.argsort(ols["x"])
            scatter_fig.add_trace(go.Scatter(
                x=ols["x"][sort_idx], y=ols["y_pred"][sort_idx],
                mode="lines", name="OLS Fit", line=dict(color="#f59e0b", width=2.5),
            ))
            scatter_fig.update_layout(
                template="plotly_dark",
                title=f"OLS Regression: {ols_y} ~ {ols_x}",
                xaxis_title=ols_x, yaxis_title=ols_y,
            )
            st.plotly_chart(scatter_fig, width='stretch')

            # Residual plot
            res_fig = go.Figure()
            res_fig.add_trace(go.Scatter(
                x=ols["y_pred"][sample_idx], y=ols["residuals"][sample_idx],
                mode="markers", marker=dict(color="#f87171", size=3, opacity=0.4), name="Residuals",
            ))
            res_fig.add_hline(y=0, line_dash="dash", line_color="#94a3b8")
            res_fig.update_layout(template="plotly_dark", title="Residual Plot",
                                    xaxis_title="Fitted Values", yaxis_title="Residuals")
            st.plotly_chart(res_fig, width='stretch')

            # ── Prediction Tool (Simple OLS) ──────────────────────────────────
            st.subheader("🔮 Make a Prediction")
            st.caption(
                f"Enter a value for **{ols_x}** to predict **{ols_y}** "
                f"using the fitted model: `{ols['equation']}`"
            )

            x_min = float(df[ols_x].min())
            x_max = float(df[ols_x].max())
            x_mean = float(df[ols_x].mean())

            pred_x_val = st.number_input(
                f"Input value for {ols_x}",
                min_value=x_min,
                max_value=x_max,
                value=round(x_mean, 2),
                step=(x_max - x_min) / 100,
                key="simple_pred_x",
            )

            predicted_y = ols["slope"] * pred_x_val + ols["intercept"]

            st.markdown(
                f'<div class="prediction-result">'
                f'Predicted {ols_y} = {ols["slope"]:.4f} × {pred_x_val:.4f} + {ols["intercept"]:.4f} '
                f'= <span style="font-size:1.2rem">{predicted_y:.4f}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

            # Show where the prediction falls on the regression line
            pred_fig = go.Figure()
            pred_fig.add_trace(go.Scatter(
                x=ols["x"][sample_idx], y=ols["y"][sample_idx],
                mode="markers", name="Data",
                marker=dict(color="#6366f1", size=4, opacity=0.3),
            ))
            pred_fig.add_trace(go.Scatter(
                x=ols["x"][sort_idx], y=ols["y_pred"][sort_idx],
                mode="lines", name="OLS Fit",
                line=dict(color="#f59e0b", width=2),
            ))
            pred_fig.add_trace(go.Scatter(
                x=[pred_x_val], y=[predicted_y],
                mode="markers", name="Your Prediction",
                marker=dict(color="#38bdf8", size=14, symbol="star"),
            ))
            pred_fig.add_annotation(
                x=pred_x_val, y=predicted_y,
                text=f"  ({pred_x_val:.2f}, {predicted_y:.2f})",
                showarrow=True, arrowhead=2,
                arrowcolor="#38bdf8", font=dict(color="#38bdf8", size=12),
            )
            pred_fig.update_layout(
                template="plotly_dark",
                title=f"Prediction: {ols_y} when {ols_x} = {pred_x_val:.2f}",
                xaxis_title=ols_x, yaxis_title=ols_y,
            )
            st.plotly_chart(pred_fig, width='stretch')

    # ── Multiple Regression ───────────────────────────────────────────────────
    with reg_tabs[2]:
        st.markdown("**Multiple Linear Regression via OLS (Normal Equations)**")
        default_x = NUMERIC_COLS[:3] if len(NUMERIC_COLS) >= 3 else NUMERIC_COLS[:1]
        mr_x = st.multiselect("Predictor Variables (X)", NUMERIC_COLS, default=default_x, key="mrx")
        mr_y = st.selectbox("Response Variable (Y)", NUMERIC_COLS + ["income_binary"], key="mry")

        if mr_x and mr_y not in mr_x:
            with st.spinner("Fitting multiple regression..."):
                mr = multiple_ols(df, mr_x, mr_y)
            if "error" not in mr:
                mm1, mm2, mm3 = st.columns(3)
                mm1.metric("R²", f"{mr['r_squared']:.4f}")
                mm2.metric("Adj. R²", f"{mr['adj_r_squared']:.4f}")
                mm3.metric("RMSE", f"{mr['rmse']:.4f}")

                st.subheader("Coefficients")
                coef_df = pd.DataFrame.from_dict(mr["coefficients"], orient="index", columns=["Coefficient"])
                st.dataframe(coef_df.style.format("{:.4f}"), width='stretch')

                coef_fig = px.bar(
                    coef_df.reset_index(), x="index", y="Coefficient",
                    title="OLS Coefficients", color="Coefficient",
                    color_continuous_scale="RdBu_r",
                )
                coef_fig.update_layout(template="plotly_dark", xaxis_title="Variable")
                st.plotly_chart(coef_fig, width='stretch')

                # ── Prediction Tool (Multiple OLS) ────────────────────────────
                st.subheader("🔮 Make a Prediction")
                st.caption(
                    f"Enter values for each predictor to get a predicted **{mr_y}** "
                    f"from the fitted multiple regression model."
                )

                pred_inputs = {}
                input_cols = st.columns(min(len(mr_x), 3))
                for i, col_name in enumerate(mr_x):
                    col_min = float(df[col_name].min())
                    col_max = float(df[col_name].max())
                    col_mean = float(df[col_name].mean())
                    with input_cols[i % len(input_cols)]:
                        pred_inputs[col_name] = st.number_input(
                            f"{col_name}",
                            min_value=col_min,
                            max_value=col_max,
                            value=round(col_mean, 2),
                            step=(col_max - col_min) / 100,
                            key=f"mr_pred_{col_name}",
                        )

                # Retrieve the intercept and coefficients from the fitted model.
                # mr["coefficients"] is expected to be a dict mapping variable name → coefficient,
                # with the intercept stored under the key "Intercept".
                intercept_val = mr["coefficients"].get("Intercept", 0.0)
                multi_pred = intercept_val + sum(
                    mr["coefficients"].get(col_name, 0.0) * val
                    for col_name, val in pred_inputs.items()
                )

                equation_parts = [f"{mr['coefficients'].get(c, 0.0):.4f}×{c}" for c in mr_x]
                equation_str = f"{intercept_val:.4f} + " + " + ".join(equation_parts)

                st.markdown(
                    f'<div class="prediction-result">'
                    f'Predicted {mr_y} = {equation_str} '
                    f'= <span style="font-size:1.2rem">{multi_pred:.4f}</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

                # Actual vs Predicted scatter to give context for the prediction
                if "y_pred" in mr and "y_actual" in mr:
                    avp_sample = min(2000, len(mr["y_actual"]))
                    avp_idx = np.random.choice(len(mr["y_actual"]), avp_sample, replace=False)
                    avp_fig = go.Figure()
                    avp_fig.add_trace(go.Scatter(
                        x=np.array(mr["y_actual"])[avp_idx],
                        y=np.array(mr["y_pred"])[avp_idx],
                        mode="markers",
                        name="Actual vs Predicted",
                        marker=dict(color="#a78bfa", size=4, opacity=0.4),
                    ))
                    avp_range = [
                        min(mr["y_actual"].min(), mr["y_pred"].min()),
                        max(mr["y_actual"].max(), mr["y_pred"].max()),
                    ]
                    avp_fig.add_trace(go.Scatter(
                        x=avp_range, y=avp_range,
                        mode="lines", name="Perfect Fit",
                        line=dict(color="#f59e0b", dash="dash", width=1.5),
                    ))
                    avp_fig.add_trace(go.Scatter(
                        x=[multi_pred], y=[multi_pred],
                        mode="markers", name="Your Prediction",
                        marker=dict(color="#38bdf8", size=14, symbol="star"),
                    ))
                    avp_fig.update_layout(
                        template="plotly_dark",
                        title=f"Actual vs Predicted: {mr_y}",
                        xaxis_title=f"Actual {mr_y}",
                        yaxis_title=f"Predicted {mr_y}",
                    )
                    st.plotly_chart(avp_fig, width='stretch')

            else:
                st.error(mr["error"])
        elif mr_y in mr_x:
            st.warning("Response variable cannot be a predictor.")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 6 — ADVANCED FAIRNESS AUDIT  (Member 4)
# ═══════════════════════════════════════════════════════════════════════════════
with tab6:
    st.subheader("Advanced Fairness Metrics")
    st.caption(f"Protected attribute: **{protected_attr}** | Outcome: income > $50K")

    fair_tabs = st.tabs(["Demographic Parity", "Disparate Impact", "Equalized Odds", "Confidence Intervals"])

    # ── Demographic Parity ────────────────────────────────────────────────────
    with fair_tabs[0]:
        st.markdown("**P(Income > 50K | group) — Conditional Probability by Group**")
        groups = list(df[protected_attr].dropna().unique())
        rows = []
        for g in groups:
            prob, successes, total = conditional_probability(df, protected_attr, g, "income_binary")
            rows.append({"Group": str(g), "P(Y=1|group)": prob, "Count>50K": successes, "Total": total})
        prob_df = pd.DataFrame(rows)
        st.dataframe(prob_df.set_index("Group").style.format("{:.4f}", subset=["P(Y=1|group)"]),
                      width='stretch')

        dp_fig = px.bar(
            prob_df, x="Group", y="P(Y=1|group)", color="Group",
            title=f"Positive Income Rate by {protected_attr.capitalize()}",
            color_discrete_sequence=px.colors.qualitative.Set2, text_auto=".3f",
        )
        dp_fig.update_layout(template="plotly_dark", showlegend=False, title=dict(x=0.5))
        st.plotly_chart(dp_fig, width='stretch')

    # ── Disparate Impact ──────────────────────────────────────────────────────
    with fair_tabs[1]:
        st.markdown("**Disparate Impact Ratio** — the 80% rule (EEOC standard)")
        st.caption("DI = P(Y=1 | unprivileged) / P(Y=1 | privileged). DI < 0.8 → adverse impact.")

        di = disparate_impact(df, protected_attr, "income_binary")
        if di:
            st.info(f"**Privileged group (highest positive rate):** {di['privileged_group']}  "
                    f"(rate = {di['privileged_rate']:.4f})")

            di_rows = []
            for g, vals in di["groups"].items():
                di_rows.append({
                    "Group": g,
                    "Positive Rate": vals["positive_rate"],
                    "Disparate Impact": vals["disparate_impact"],
                    "Passes 80% Rule": vals["verdict"],
                })
            di_df = pd.DataFrame(di_rows)
            st.dataframe(di_df.set_index("Group"), width='stretch')

            di_fig = go.Figure()
            di_fig.add_trace(go.Bar(
                x=di_df["Group"], y=di_df["Disparate Impact"],
                marker_color=["#4ade80" if v >= 0.8 else "#f87171" for v in di_df["Disparate Impact"]],
                text=[f"{v:.3f}" for v in di_df["Disparate Impact"]], textposition="outside",
                name="DI Ratio",
            ))
            di_fig.add_hline(y=0.8, line_dash="dash", line_color="#f59e0b",
                              annotation_text="80% Threshold", annotation_position="top right")
            di_fig.update_layout(template="plotly_dark", title="Disparate Impact Ratio by Group",
                                   yaxis_title="DI Ratio", title_x=0.5)
            st.plotly_chart(di_fig, width='stretch')

            overall_pass = di["overall_verdict"] == "No Adverse Impact Detected"
            verdict_cls = "verdict-pass" if overall_pass else "verdict-fail"
            icon = "✅" if overall_pass else "❌"
            st.markdown(f'<div class="{verdict_cls}">{icon} {di["overall_verdict"]}</div>',
                        unsafe_allow_html=True)

    # ── Equalized Odds ────────────────────────────────────────────────────────
    with fair_tabs[2]:
        st.markdown("**Equalized Odds** — TPR and FPR equality across groups")
        st.caption("Requires a trained model. Click below to train and evaluate.")

        if st.button("Train Model & Compute Equalized Odds", key="eo_btn"):
            with st.spinner("Training Logistic Regression..."):
                model, accuracy, y_pred, y_test = train_model(df)
                st.session_state.model_results = {
                    "model": model, "accuracy": accuracy,
                    "y_pred": y_pred, "y_test": y_test,
                }

        if "model_results" in st.session_state:
            res = st.session_state.model_results
            st.metric("Model Accuracy", f"{res['accuracy']:.4f}")

            y_test = res["y_test"]
            y_pred_s = pd.Series(res["y_pred"], index=y_test.index)
            protected_test = df.loc[y_test.index, protected_attr]

            eo = equalized_odds(y_test, y_pred_s, protected_test)
            eo_df = pd.DataFrame(eo).T.round(4)
            st.dataframe(eo_df, width='stretch')

            verdict = equalized_odds_verdict(eo)
            st.metric("TPR Gap", f"{verdict['tpr_gap']:.4f}")
            st.metric("FPR Gap", f"{verdict['fpr_gap']:.4f}")

            verdict_cls = "verdict-pass" if verdict["passes"] else "verdict-fail"
            icon = "✅" if verdict["passes"] else "❌"
            st.markdown(f'<div class="{verdict_cls}">{icon} {verdict["verdict"]}</div>',
                        unsafe_allow_html=True)

            # TPR/FPR grouped bar
            eo_plot = pd.DataFrame({
                "Group": list(eo.keys()) * 2,
                "Metric": ["TPR"] * len(eo) + ["FPR"] * len(eo),
                "Value": [v["TPR"] for v in eo.values()] + [v["FPR"] for v in eo.values()],
            })
            eo_fig = px.bar(eo_plot, x="Group", y="Value", color="Metric", barmode="group",
                             color_discrete_map={"TPR": "#4ade80", "FPR": "#f87171"},
                             title="TPR & FPR by Group (Equalized Odds)")
            eo_fig.update_layout(template="plotly_dark")
            st.plotly_chart(eo_fig, width='stretch')
        else:
            st.info("Train the model above to see Equalized Odds results.")

    # ── Confidence Intervals ──────────────────────────────────────────────────
    with fair_tabs[3]:
        st.markdown("**Wilson Score 95% Confidence Intervals** for positive rate per group")
        ci_table = group_ci_table(df, protected_attr, "income_binary")
        st.dataframe(ci_table.style.format("{:.4f}", subset=["Positive Rate","CI Lower (95%)","CI Upper (95%)","CI Width"]),
                      width='stretch')

        ci_fig = go.Figure()
        for g in ci_table.index:
            row = ci_table.loc[g]
            ci_fig.add_trace(go.Scatter(
                x=[g, g], y=[row["CI Lower (95%)"], row["CI Upper (95%)"]],
                mode="lines", line=dict(color="#94a3b8", width=4), showlegend=False,
            ))
            ci_fig.add_trace(go.Scatter(
                x=[g], y=[row["Positive Rate"]],
                mode="markers", marker=dict(color="#f59e0b", size=14, symbol="diamond"),
                name=g,
            ))
        ci_fig.update_layout(
            template="plotly_dark",
            title=f"95% Wilson CI for Positive Income Rate by {protected_attr.capitalize()}",
            yaxis_title="P(income > 50K)", xaxis_title="Group",
        )
        st.plotly_chart(ci_fig, width='stretch')

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 7 — AI INTERPRETER  (Member 5)
# ═══════════════════════════════════════════════════════════════════════════════
with tab7:
    st.subheader("AI-Powered Audit Interpreter")
    st.caption("Uses Gemini AI to generate a plain-English interpretation of your fairness audit results.")

    gemini_api_key = st.text_input(
        "Enter your Gemini API Key",
        type="password",
        placeholder="AIza...",
        key="gemini_key",
    )

    if st.button("🔍 Generate AI Interpretation", key="ai_btn", type="primary"):
        if not gemini_api_key:
            st.warning("Please enter your Gemini API key above.")
        else:
            di_data = disparate_impact(df, protected_attr, "income_binary")
            ci_tbl = group_ci_table(df, protected_attr, "income_binary")

            stats_df = grouped_numeric_stats(
                df, protected_attr, cols=["age", "hours-per-week", "education-num"]
            ).round(2)
            cols_to_keep = [c for c in stats_df.columns if any(k in c.lower() for k in ["mean", "std"])]
            stats_summary = stats_df[cols_to_keep].to_string() if cols_to_keep else stats_df.to_string()

            ci_trimmed = ci_tbl[["Positive Rate", "CI Lower (95%)", "CI Upper (95%)"]].round(4).to_string()

            di_summary = json.dumps(
                {g: {"positive_rate": round(v["positive_rate"], 4),
                      "disparate_impact": round(v["disparate_impact"], 4),
                      "passes_80_rule": v["passes_80_rule"]}
                 for g, v in di_data.get("groups", {}).items()},
                indent=2
            )

            model_acc = st.session_state.model_results["accuracy"] if "model_results" in st.session_state else "Not computed yet"

            prompt = f"""You are an expert in algorithmic fairness and statistics analyzing the UCI Adult Income dataset.
Protected attribute: '{protected_attr}'

DESCRIPTIVE STATS (mean & std by group):
{stats_summary}

DISPARATE IMPACT:
Privileged group: {di_data.get('privileged_group', 'N/A')}
{di_summary}
Verdict: {di_data.get('overall_verdict', 'N/A')}

95% CONFIDENCE INTERVALS:
{ci_trimmed}

Model accuracy: {model_acc}

Write a concise 4-paragraph professional audit interpretation covering:
1. Key demographic disparities in descriptive stats
2. What the disparate impact ratios mean, and whether the 80% rule passes
3. What the confidence intervals tell us about reliability
4. Recommendations to address bias found
Be specific with numbers. End with a one-sentence overall fairness verdict."""

            with st.spinner("Gemini is analyzing your audit results..."):
                try:
                    import urllib.request
                    import time

                    payload = json.dumps({
                        "contents": [{"parts": [{"text": prompt}]}]
                    }).encode()

                    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-lite:generateContent?key={gemini_api_key}"

                    max_retries = 3
                    for attempt in range(max_retries):
                        req = urllib.request.Request(
                            url,
                            data=payload,
                            headers={"Content-Type": "application/json"},
                            method="POST",
                        )
                        with urllib.request.urlopen(req) as resp:
                            data = json.loads(resp.read().decode())
                            ai_text = data["candidates"][0]["content"]["parts"][0]["text"]
                        break

                    st.session_state["ai_interpretation"] = ai_text
                except Exception as e:
                    st.error(f"API error: {e}")

    if "ai_interpretation" in st.session_state:
        st.markdown(
            f'<div class="ai-response">{st.session_state["ai_interpretation"].replace(chr(10), "<br>")}</div>',
            unsafe_allow_html=True,
        )
        st.download_button(
            "📥 Download Interpretation",
            data=st.session_state["ai_interpretation"],
            file_name="fairness_audit_interpretation.txt",
            mime="text/plain",
        )