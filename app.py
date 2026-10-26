"""Churn Signal — an interactive retention analytics app.

Run locally with:
    streamlit run app.py
"""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from churn.config import (
    ID_COLUMN,
    METRICS_PATH,
    PALETTE,
    RISK_BANDS,
    TEST_PREDICTIONS_PATH,
)
from churn.data import (
    DataValidationError,
    batch_scoring_template,
    clean_churn_data,
    load_clean_data,
    prepare_inference_data,
)
from churn.insights import local_what_if, retention_actions, risk_color, score_frame
from churn.modeling import load_model_bundle

ROOT = Path(__file__).resolve().parent

INK = PALETTE["ink"]
MUTED = PALETTE["muted"]
TEAL = PALETTE["teal"]
MINT = PALETTE["mint"]
CORAL = PALETTE["coral"]
AMBER = PALETTE["amber"]
CREAM = PALETTE["cream"]
RISK_ORDER = [band["name"] for band in RISK_BANDS]
RISK_COLORS = {band["name"]: band["color"] for band in RISK_BANDS}

st.set_page_config(
    page_title="Churn Signal · Customer retention intelligence",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

CSS = f"""
<style>
    :root {{
        --ink: {INK};
        --muted: {MUTED};
        --teal: {TEAL};
        --mint: {MINT};
        --coral: {CORAL};
        --amber: {AMBER};
        --cream: {CREAM};
        --line: #E4E8EC;
    }}
    html, body, [class*="css"] {{
        font-family: "Segoe UI", "Inter", -apple-system, BlinkMacSystemFont, sans-serif;
    }}
    .stApp {{
        background: #F7F5F0;
    }}
    .block-container {{
        max-width: 1320px;
        padding-top: 1.2rem;
        padding-bottom: 3rem;
    }}
    header[data-testid="stHeader"] {{
        background: rgba(247, 245, 240, 0.86);
        backdrop-filter: blur(8px);
    }}
    [data-testid="stToolbar"], [data-testid="stStatusWidget"], #MainMenu {{
        visibility: hidden;
        height: 0;
    }}
    [data-testid="stDecoration"] {{
        display: none;
    }}
    section[data-testid="stSidebar"] {{
        background: #0B1F33;
        border-right: 1px solid rgba(255,255,255,0.06);
    }}
    section[data-testid="stSidebar"] * {{
        color: #EAF1F5;
    }}
    section[data-testid="stSidebar"] .stRadio label p {{
        color: #D9E6EC;
        font-size: 0.95rem;
    }}
    section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {{
        color: #AFC3CD;
        font-size: 0.83rem;
    }}
    section[data-testid="stSidebar"] .stRadio [data-checked="true"] p {{
        color: #FFFFFF;
        font-weight: 600;
    }}
    section[data-testid="stSidebar"] hr {{
        border-color: rgba(255,255,255,0.12);
    }}
    .brand {{
        color: #FFFFFF;
        letter-spacing: 0.16em;
        font-size: 0.74rem;
        font-weight: 700;
        text-transform: uppercase;
        margin-bottom: 0.2rem;
    }}
    .brand-sub {{
        color: #8FB4C2;
        font-size: 0.78rem;
        line-height: 1.4;
        margin-bottom: 1.4rem;
    }}
    .hero {{
        background: linear-gradient(120deg, #0B1F33 0%, #103D4B 56%, #0E7C86 100%);
        border-radius: 24px;
        padding: 2.35rem 2.5rem 2rem 2.5rem;
        color: #FFFFFF;
        position: relative;
        overflow: hidden;
        box-shadow: 0 18px 48px rgba(11, 31, 51, 0.16);
        margin-bottom: 1.3rem;
    }}
    .hero:after {{
        content: "";
        position: absolute;
        width: 420px;
        height: 420px;
        border-radius: 50%;
        background: rgba(42, 157, 143, 0.16);
        right: -110px;
        top: -170px;
    }}
    .hero:before {{
        content: "";
        position: absolute;
        width: 260px;
        height: 260px;
        border-radius: 50%;
        border: 1px solid rgba(255,255,255,0.13);
        right: 90px;
        bottom: -150px;
    }}
    .eyebrow {{
        font-size: 0.72rem;
        letter-spacing: 0.18em;
        text-transform: uppercase;
        color: #8ED7D2;
        font-weight: 700;
        margin-bottom: 0.75rem;
    }}
    .hero h1 {{
        font-size: clamp(1.9rem, 3.2vw, 3.05rem);
        line-height: 1.06;
        letter-spacing: -0.035em;
        max-width: 850px;
        margin: 0 0 0.8rem 0;
        color: #FFFFFF;
        font-weight: 700;
    }}
    .hero p {{
        color: #C8DCE3;
        font-size: 1rem;
        line-height: 1.55;
        max-width: 760px;
        margin: 0;
    }}
    .chips {{
        display: flex;
        flex-wrap: wrap;
        gap: 0.55rem;
        margin-top: 1.35rem;
    }}
    .chip {{
        background: rgba(255,255,255,0.10);
        border: 1px solid rgba(255,255,255,0.15);
        border-radius: 999px;
        padding: 0.36rem 0.75rem;
        color: #EAF6F5;
        font-size: 0.77rem;
        letter-spacing: 0.01em;
    }}
    .page-head {{
        padding: 1.1rem 0 0.45rem 0;
        margin-bottom: 0.8rem;
    }}
    .page-head .eyebrow {{
        color: {TEAL};
    }}
    .page-head h1 {{
        color: {INK};
        font-size: clamp(1.7rem, 2.5vw, 2.35rem);
        line-height: 1.1;
        letter-spacing: -0.03em;
        margin: 0.2rem 0 0.45rem 0;
    }}
    .page-head p {{
        color: {MUTED};
        font-size: 0.98rem;
        max-width: 900px;
        line-height: 1.55;
        margin: 0;
    }}
    .kpi-card {{
        background: #FFFFFF;
        border: 1px solid var(--line);
        border-radius: 16px;
        padding: 1.05rem 1.1rem 0.95rem 1.1rem;
        box-shadow: 0 6px 18px rgba(11, 31, 51, 0.035);
        min-height: 116px;
    }}
    .kpi-label {{
        color: {MUTED};
        font-size: 0.74rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-weight: 600;
        margin-bottom: 0.35rem;
    }}
    .kpi-value {{
        color: {INK};
        font-size: 1.62rem;
        line-height: 1.1;
        letter-spacing: -0.035em;
        font-weight: 700;
    }}
    .kpi-detail {{
        color: {MUTED};
        font-size: 0.78rem;
        line-height: 1.35;
        margin-top: 0.45rem;
    }}
    .section-label {{
        color: {TEAL};
        font-size: 0.72rem;
        text-transform: uppercase;
        letter-spacing: 0.15em;
        font-weight: 700;
        margin: 1.8rem 0 0.25rem 0;
    }}
    .section-head {{
        color: {INK};
        font-size: 1.3rem;
        letter-spacing: -0.02em;
        font-weight: 700;
        margin: 0 0 0.25rem 0;
    }}
    .section-sub {{
        color: {MUTED};
        font-size: 0.9rem;
        margin: 0 0 0.85rem 0;
    }}
    .callout {{
        background: #EAF4F2;
        border-left: 4px solid {MINT};
        border-radius: 0 14px 14px 0;
        padding: 1rem 1.2rem;
        color: {INK};
        font-size: 0.93rem;
        line-height: 1.55;
        margin: 0.75rem 0 0.25rem 0;
    }}
    .callout strong {{
        color: {INK};
    }}
    .play-card {{
        background: #FFFFFF;
        border: 1px solid var(--line);
        border-radius: 16px;
        padding: 1.2rem 1.15rem;
        height: 100%;
        box-shadow: 0 6px 18px rgba(11, 31, 51, 0.03);
    }}
    .play-num {{
        color: {TEAL};
        font-size: 0.72rem;
        letter-spacing: 0.12em;
        font-weight: 700;
        text-transform: uppercase;
        margin-bottom: 0.55rem;
    }}
    .play-card h4 {{
        color: {INK};
        font-size: 1rem;
        line-height: 1.25;
        margin: 0 0 0.45rem 0;
    }}
    .play-card p {{
        color: {MUTED};
        font-size: 0.84rem;
        line-height: 1.5;
        margin: 0;
    }}
    .play-metric {{
        color: {CORAL};
        font-size: 0.8rem;
        font-weight: 700;
        margin-top: 0.75rem;
    }}
    .risk-chip {{
        display: inline-block;
        border-radius: 999px;
        padding: 0.25rem 0.65rem;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        margin-bottom: 0.5rem;
    }}
    .score-panel {{
        background: #FFFFFF;
        border: 1px solid var(--line);
        border-radius: 18px;
        padding: 1.15rem 1.15rem 1rem 1.15rem;
        box-shadow: 0 8px 24px rgba(11, 31, 51, 0.045);
    }}
    .score-value {{
        color: {INK};
        font-size: 2.7rem;
        line-height: 1;
        letter-spacing: -0.05em;
        font-weight: 700;
    }}
    .score-sub {{
        color: {MUTED};
        font-size: 0.82rem;
        line-height: 1.4;
        margin-top: 0.4rem;
    }}
    .mini-stat {{
        background: #F8FAF9;
        border: 1px solid #E8ECEF;
        border-radius: 12px;
        padding: 0.7rem 0.75rem;
        margin-top: 0.65rem;
    }}
    .mini-label {{
        color: {MUTED};
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 0.07em;
        font-weight: 600;
    }}
    .mini-value {{
        color: {INK};
        font-size: 1.05rem;
        font-weight: 700;
        margin-top: 0.2rem;
    }}
    .action-item {{
        display: flex;
        gap: 0.7rem;
        padding: 0.7rem 0;
        border-bottom: 1px solid #EDF0F2;
        color: {INK};
        font-size: 0.9rem;
        line-height: 1.45;
    }}
    .action-item:last-child {{
        border-bottom: none;
    }}
    .action-index {{
        color: {TEAL};
        font-weight: 700;
        font-size: 0.78rem;
        min-width: 1.4rem;
        padding-top: 0.08rem;
    }}
    .footer {{
        border-top: 1px solid #E0E5E9;
        margin-top: 2.5rem;
        padding-top: 1rem;
        color: #7B8B96;
        font-size: 0.76rem;
        line-height: 1.6;
    }}
    .footer a {{
        color: {TEAL};
        text-decoration: none;
    }}
    div[data-testid="stVerticalBlockBorderWrapper"] {{
        border-radius: 14px;
    }}
    .stButton > button, .stDownloadButton > button {{
        border-radius: 10px;
        border: 1px solid #CAD5DB;
        background: #FFFFFF;
        color: {INK};
        font-weight: 600;
        padding: 0.45rem 0.9rem;
    }}
    .stButton > button:hover, .stDownloadButton > button:hover {{
        border-color: {TEAL};
        color: {TEAL};
    }}
    .stFormSubmitButton > button {{
        background: {TEAL};
        color: #FFFFFF;
        border: 1px solid {TEAL};
        border-radius: 10px;
        font-weight: 700;
        padding: 0.55rem 1rem;
    }}
    .stFormSubmitButton > button:hover {{
        background: #0A6570;
        border-color: #0A6570;
        color: #FFFFFF;
    }}
    div[data-testid="stDataFrame"] {{
        border: 1px solid #E4E8EC;
        border-radius: 12px;
        overflow: hidden;
    }}
    div[data-testid="stPlotlyChart"] {{
        background: #FFFFFF;
        border: 1px solid #E8ECEF;
        border-radius: 16px;
        padding: 0.35rem 0.45rem 0.15rem 0.45rem;
        box-shadow: 0 4px 14px rgba(11, 31, 51, 0.025);
    }}
    .stTabs [data-baseweb="tab-list"] {{
        gap: 0.35rem;
        border-bottom: 1px solid #E2E7EA;
    }}
    .stTabs [data-baseweb="tab"] {{
        border-radius: 8px 8px 0 0;
        color: {MUTED};
        font-weight: 600;
        font-size: 0.85rem;
    }}
    .stTabs [aria-selected="true"] {{
        color: {TEAL};
        background: #EAF4F2;
    }}
    .stAlert {{
        border-radius: 12px;
    }}
    [data-testid="stCode"] pre,
    [data-testid="stCode"] code {{
        white-space: pre-wrap !important;
        overflow-wrap: anywhere !important;
        font-size: 0.76rem;
        line-height: 1.65;
    }}
    [data-testid="stCode"] code {{
        display: block !important;
        width: auto !important;
    }}
    @media (max-width: 760px) {{
        .hero {{ padding: 1.5rem 1.2rem; }}
        .block-container {{ padding-top: 0.7rem; }}
    }}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def get_bundle() -> dict[str, Any]:
    return load_model_bundle()


@st.cache_data(show_spinner=False)
def get_metrics() -> dict[str, Any]:
    return json.loads(METRICS_PATH.read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def get_training_frame() -> pd.DataFrame:
    return load_clean_data()


@st.cache_data(show_spinner=False)
def get_test_predictions() -> pd.DataFrame:
    return pd.read_csv(TEST_PREDICTIONS_PATH)


@st.cache_data(show_spinner=False)
def get_risk_deciles() -> pd.DataFrame:
    path = ROOT / "reports" / "evaluation" / "risk_deciles.csv"
    return pd.read_csv(path)


def fmt_money(value: float) -> str:
    if abs(value) >= 1_000_000:
        return f"${value / 1_000_000:.2f}M"
    if abs(value) >= 1_000:
        return f"${value / 1_000:.1f}K"
    return f"${value:,.0f}"


def fmt_pct(value: float, digits: int = 1) -> str:
    return f"{value * 100:.{digits}f}%"


def kpi_card(label: str, value: str, detail: str, accent: str = TEAL) -> str:
    return f"""
    <div class="kpi-card" style="border-top: 3px solid {accent};">
        <div class="kpi-label">{html.escape(label)}</div>
        <div class="kpi-value">{html.escape(value)}</div>
        <div class="kpi-detail">{html.escape(detail)}</div>
    </div>
    """


def kpi_row(items: list[tuple[str, str, str, str]]) -> None:
    columns = st.columns(len(items), gap="medium")
    for column, (label, value, detail, accent) in zip(columns, items):
        with column:
            st.markdown(kpi_card(label, value, detail, accent), unsafe_allow_html=True)


def hero(kicker: str, title: str, subtitle: str, chips: list[str]) -> None:
    chip_html = "".join(f'<span class="chip">{html.escape(chip)}</span>' for chip in chips)
    st.markdown(
        f"""
        <div class="hero">
            <div class="eyebrow">{html.escape(kicker)}</div>
            <h1>{html.escape(title)}</h1>
            <p>{html.escape(subtitle)}</p>
            <div class="chips">{chip_html}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def page_head(kicker: str, title: str, subtitle: str) -> None:
    st.markdown(
        f"""
        <div class="page-head">
            <div class="eyebrow">{html.escape(kicker)}</div>
            <h1>{html.escape(title)}</h1>
            <p>{html.escape(subtitle)}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section(title: str, subtitle: str = "", label: str = "") -> None:
    label_html = f'<div class="section-label">{html.escape(label)}</div>' if label else ""
    subtitle_html = f'<div class="section-sub">{html.escape(subtitle)}</div>' if subtitle else ""
    st.markdown(
        f'{label_html}<div class="section-head">{html.escape(title)}</div>{subtitle_html}',
        unsafe_allow_html=True,
    )


def chart_layout(fig: go.Figure, *, height: int = 360, title: str | None = None) -> go.Figure:
    layout: dict[str, Any] = {
        "template": "plotly_white",
        "height": height,
        "margin": dict(l=12, r=18, t=52 if title else 18, b=12),
        "font": dict(family="Segoe UI, Inter, sans-serif", color=INK, size=12),
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "rgba(0,0,0,0)",
        "legend": dict(orientation="h", yanchor="bottom", y=1.01, x=0, font=dict(size=10)),
        "hoverlabel": dict(bgcolor="white", font=dict(color=INK)),
    }
    if title:
        layout["title"] = dict(text=title, font=dict(size=15, color=INK), x=0.01, xanchor="left")
    fig.update_layout(**layout)
    fig.update_xaxes(gridcolor="#E8ECEF", zeroline=False, linecolor="#DDE3E7")
    fig.update_yaxes(gridcolor="#E8ECEF", zeroline=False, linecolor="#DDE3E7")
    return fig


def plot(fig: go.Figure, *, height: int = 360) -> None:
    st.plotly_chart(
        chart_layout(fig, height=height),
        width="stretch",
        config={"displayModeBar": False, "responsive": True},
    )


def footer() -> None:
    st.markdown(
        """
        <div class="footer">
            <strong>Churn Signal</strong> · A reproducible customer-retention portfolio project.<br>
            Data: IBM Telco Customer Churn (public Kaggle dataset) · Built with scikit-learn and Streamlit.<br>
            <a href="https://github.com/mlaharsh/Customer_Churn_Prediction" target="_blank">GitHub repository</a>
            &nbsp;·&nbsp;
            <a href="https://mlaharsh.github.io/Customer_Churn_Prediction/" target="_blank">Project showcase</a>
            &nbsp;·&nbsp;
            Model output is a decision-support signal, not a causal claim or a guarantee of customer behavior.
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Shared data
# ---------------------------------------------------------------------------
try:
    bundle = get_bundle()
    metrics = get_metrics()
    training_frame = get_training_frame()
    test_predictions = get_test_predictions()
    risk_deciles = get_risk_deciles()
except FileNotFoundError as exc:
    st.error(
        "The model artifacts are missing. Run `python -m churn.train` from the project root, then refresh this page."
    )
    st.code(str(exc))
    st.stop()

DATASET = metrics["dataset"]
THRESHOLD = float(metrics["decision_threshold"])
SELECTED = metrics["selected_model_display"]


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown('<div class="brand">Churn Signal</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="brand-sub">Retention intelligence for a telecom customer base.</div>',
        unsafe_allow_html=True,
    )
    page = st.radio(
        "Navigate",
        [
            "Overview",
            "Customer risk scorer",
            "Batch scoring",
            "Model performance",
            "Method & limits",
        ],
        label_visibility="collapsed",
    )
    st.divider()
    st.markdown(
        f"""
        <div style="font-size:0.72rem;letter-spacing:0.12em;text-transform:uppercase;color:#8FB4C2;font-weight:700;margin-bottom:0.5rem;">
            Live model
        </div>
        <div style="color:#FFFFFF;font-size:0.92rem;font-weight:600;">{html.escape(SELECTED)}</div>
        <div style="color:#AFC3CD;font-size:0.76rem;line-height:1.5;margin-top:0.3rem;">
            Threshold {THRESHOLD:.1%} · holdout ROC-AUC {metrics['test_metrics']['roc_auc']:.3f}
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.divider()
    st.markdown(
        """
        <div style="color:#8FB4C2;font-size:0.72rem;line-height:1.55;">
            The model is trained on 7,043 historical customer records.
            Scores are prioritization signals for retention teams, not causal effects.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div style="margin-top:1rem;font-size:0.72rem;color:#8FB4C2;">'
        '<a style="color:#8ED7D2;text-decoration:none;" href="https://github.com/mlaharsh/Customer_Churn_Prediction" target="_blank">'
        "View source on GitHub →</a></div>",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Overview
# ---------------------------------------------------------------------------
if page == "Overview":
    hero(
        "Customer intelligence · portfolio project",
        "Know who is about to leave — before they do.",
        "A reproducible churn model and a retention workbench built on the IBM Telco customer base. "
        "Explore what drives churn, score a customer scenario, and turn a holdout evaluation into a campaign decision.",
        [
            f"{DATASET['customers']:,} customers",
            f"{fmt_pct(DATASET['churn_rate'])} observed churn",
            f"{metrics['test_metrics']['roc_auc']:.3f} holdout ROC-AUC",
        ],
    )

    kpi_row(
        [
            (
                "Customers analyzed",
                f"{DATASET['customers']:,}",
                "One row per customer; no duplicate IDs.",
                TEAL,
            ),
            (
                "Observed churn rate",
                fmt_pct(DATASET["churn_rate"]),
                f"{DATASET['churners']:,} churners in the historical base.",
                CORAL,
            ),
            (
                "Annual revenue lost",
                fmt_money(DATASET["annual_churn_revenue"]),
                "Observed monthly charges from churned customers × 12.",
                AMBER,
            ),
            (
                "Holdout ROC-AUC",
                f"{metrics['test_metrics']['roc_auc']:.3f}",
                f"{SELECTED} · 20% stratified holdout.",
                MINT,
            ),
        ]
    )

    section(
        "The pattern is concentrated, not random.",
        "Churn is unevenly distributed across contract, tenure, service, and billing segments. "
        "Those overlapping segments are where a retention team gets the most leverage.",
        label="What the data says",
    )

    left, right = st.columns(2, gap="large")
    with left:
        contract = pd.DataFrame(metrics["segments"]["contract"]).sort_values("churn_rate")
        contract["label"] = contract["churn_rate"].map(lambda value: f"{value:.1%}")
        fig = px.bar(
            contract,
            x="churn_rate",
            y="segment",
            orientation="h",
            text="label",
            color="churn_rate",
            color_continuous_scale=[[0, MINT], [0.45, AMBER], [1, CORAL]],
            labels={"churn_rate": "Churn rate", "segment": ""},
        )
        fig.update_traces(textposition="outside", textfont_size=11)
        fig.update_layout(coloraxis_showscale=False, xaxis_tickformat=".0%")
        fig.update_xaxes(range=[0, 0.54])
        plot(fig, height=320)
        st.caption("Month-to-month contracts account for 88.5% of all observed churners despite being 55% of the customer base.")

    with right:
        tenure = pd.DataFrame(metrics["segments"]["tenure_band"])
        tenure["label"] = tenure["churn_rate"].map(lambda value: f"{value:.0%}")
        fig = px.line(
            tenure,
            x="segment",
            y="churn_rate",
            markers=True,
            text="label",
            labels={"segment": "Tenure band", "churn_rate": "Churn rate"},
            color_discrete_sequence=[TEAL],
        )
        fig.update_traces(line_width=3, marker_size=7, textposition="top center")
        fig.update_layout(yaxis_tickformat=".0%", yaxis_range=[0, 0.65])
        fig.update_yaxes(title="Churn rate")
        plot(fig, height=320)
        st.caption("The first six months carry a 52.9% churn rate; the 61–72 month cohort is down to 6.6%.")

    priority = DATASET["priority_segment"]
    st.markdown(
        f"""
        <div class="callout">
            <strong>The retention case:</strong> month-to-month customers represent
            <strong>{fmt_money(metrics['segments']['contract'][0]['annual_churn_revenue'])}</strong>
            of annualised churn revenue, and fiber-optic customers represent
            <strong>{fmt_money(metrics['segments']['internet_service'][0]['annual_churn_revenue'])}</strong>.
            The overlap is the actionable part: month-to-month fiber accounts are
            <strong>{fmt_pct(priority['share_of_customers'])}</strong> of the base, churn at
            <strong>{fmt_pct(priority['churn_rate'])}</strong>, and represent
            <strong>{fmt_pct(priority['share_of_churners'])}</strong> of observed churners and
            <strong>{fmt_pct(priority['share_of_churn_revenue'])}</strong> of annualised churn revenue.
        </div>
        """,
        unsafe_allow_html=True,
    )

    section(
        "Segment scorecard",
        "Observed churn and revenue loss by the dimensions a retention team can act on.",
        label="Where to look",
    )
    segment_tabs = st.tabs(["Contract", "Internet service", "Payment method", "Tenure"])
    segment_specs = [
        ("contract", "Contract"),
        ("internet_service", "Internet service"),
        ("payment_method", "Payment method"),
        ("tenure_band", "Tenure band"),
    ]
    for tab, (key, label) in zip(segment_tabs, segment_specs):
        with tab:
            frame = pd.DataFrame(metrics["segments"][key]).copy()
            frame = frame[
                ["segment", "customers", "churners", "churn_rate", "monthly_churn_revenue", "annual_churn_revenue"]
            ].rename(
                columns={
                    "segment": label,
                    "customers": "Customers",
                    "churners": "Churners",
                    "churn_rate": "Churn rate",
                    "monthly_churn_revenue": "Monthly revenue lost",
                    "annual_churn_revenue": "Annual revenue lost",
                }
            )
            st.dataframe(
                frame,
                width="stretch",
                hide_index=True,
                column_config={
                    "Churn rate": st.column_config.NumberColumn(format="%.1f%%"),
                    "Monthly revenue lost": st.column_config.NumberColumn(format="$%.0f"),
                    "Annual revenue lost": st.column_config.NumberColumn(format="$%.0f"),
                },
            )

    section(
        "Turn the signal into retention plays.",
        "These are operational starting points tied to observable account facts, not causal prescriptions.",
        label="What to do",
    )
    play_columns = st.columns(3, gap="medium")
    plays = [
        (
            "01",
            "Move the right customers to annual contracts",
            "Month-to-month churn is 42.7% versus 11.3% on one-year and 2.8% on two-year contracts. "
            "Target the highest-risk month-to-month accounts with a loyalty credit, not a blanket discount.",
            "42.7% → 11.3% churn rate",
        ),
        (
            "02",
            "Own the first 90 days",
            "Customers with 0–6 months of tenure churn at 52.9%. A structured onboarding check-in and service-activation review "
            "is the highest-leverage intervention in the lifecycle.",
            "52.9% churn in months 0–6",
        ),
        (
            "03",
            "Fix fiber value and support gaps",
            "Fiber-optic customers churn at 41.9%, and missing online security or tech support is associated with higher risk. "
            "Run a price-to-value and service-quality review before renewing the plan.",
            "41.9% fiber churn",
        ),
    ]
    for column, (number, title, body, metric) in zip(play_columns, plays):
        with column:
            st.markdown(
                f"""
                <div class="play-card">
                    <div class="play-num">{number}</div>
                    <h4>{title}</h4>
                    <p>{body}</p>
                    <div class="play-metric">{metric}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


# ---------------------------------------------------------------------------
# Risk scorer
# ---------------------------------------------------------------------------
elif page == "Customer risk scorer":
    page_head(
        "Scenario workbench",
        "Score a customer. See the drivers. Choose a play.",
        "Change the account facts below and score the scenario against the deployed model. "
        "The explanation is a one-at-a-time sensitivity check against training-set baselines — useful for prioritization, not proof of causation.",
    )

    defaults = {
        "gender": "Female",
        "SeniorCitizen": "No",
        "Partner": "No",
        "Dependents": "No",
        "tenure": 8,
        "PhoneService": "Yes",
        "MultipleLines": "Yes",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 95.5,
    }
    last = st.session_state.get("risk_inputs", defaults)

    def option_index(options: list[str], value: str) -> int:
        return options.index(value) if value in options else 0

    def scenario_result(inputs: dict[str, Any]) -> dict[str, Any]:
        raw = pd.DataFrame(
            [
                {
                    "customerID": "scenario-001",
                    **inputs,
                    "TotalCharges": round(float(inputs["MonthlyCharges"]) * int(inputs["tenure"]), 2),
                    "Churn": "No",
                }
            ]
        )
        cleaned = clean_churn_data(raw, require_target=False)
        scored = score_frame(bundle["pipeline"], cleaned, threshold=THRESHOLD, include_target=False)
        row = scored.iloc[0]
        what_if = local_what_if(bundle["pipeline"], cleaned.iloc[0], bundle["baselines"])
        actions = retention_actions(cleaned.iloc[0], float(row["churn_probability"]), THRESHOLD)
        return {"scored": row, "what_if": what_if, "actions": actions, "cleaned": cleaned.iloc[0]}

    if "risk_result" not in st.session_state:
        st.session_state["risk_result"] = scenario_result(defaults)

    form_column, score_column = st.columns([1.32, 0.68], gap="large")

    with form_column:
        with st.form("risk_form", border=True):
            st.markdown("#### Account profile")
            row1 = st.columns(2)
            with row1[0]:
                gender = st.selectbox("Gender", ["Female", "Male"], index=option_index(["Female", "Male"], last["gender"]))
            with row1[1]:
                senior = st.selectbox("Senior citizen", ["No", "Yes"], index=option_index(["No", "Yes"], last["SeniorCitizen"]))
            row2 = st.columns(2)
            with row2[0]:
                partner = st.selectbox("Partner", ["No", "Yes"], index=option_index(["No", "Yes"], last["Partner"]))
            with row2[1]:
                dependents = st.selectbox("Dependents", ["No", "Yes"], index=option_index(["No", "Yes"], last["Dependents"]))

            st.markdown("#### Account & billing")
            row3 = st.columns(2)
            with row3[0]:
                tenure = st.slider("Tenure (months)", 0, 72, int(last["tenure"]))
            with row3[1]:
                monthly = st.number_input(
                    "Monthly charges ($)",
                    min_value=18.0,
                    max_value=130.0,
                    value=float(last["MonthlyCharges"]),
                    step=0.5,
                )
            row4 = st.columns(2)
            with row4[0]:
                contract = st.selectbox(
                    "Contract",
                    ["Month-to-month", "One year", "Two year"],
                    index=option_index(["Month-to-month", "One year", "Two year"], last["Contract"]),
                )
            with row4[1]:
                paperless = st.selectbox(
                    "Paperless billing", ["No", "Yes"], index=option_index(["No", "Yes"], last["PaperlessBilling"])
                )
            payment = st.selectbox(
                "Payment method",
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)",
                ],
                index=option_index(
                    [
                        "Electronic check",
                        "Mailed check",
                        "Bank transfer (automatic)",
                        "Credit card (automatic)",
                    ],
                    last["PaymentMethod"],
                ),
            )

            st.markdown("#### Services")
            row5 = st.columns(3)
            with row5[0]:
                phone = st.selectbox("Phone service", ["No", "Yes"], index=option_index(["No", "Yes"], last["PhoneService"]))
            with row5[1]:
                multiple = st.selectbox(
                    "Multiple lines",
                    ["No phone service", "No", "Yes"],
                    index=option_index(["No phone service", "No", "Yes"], last["MultipleLines"]),
                )
            with row5[2]:
                internet = st.selectbox(
                    "Internet service",
                    ["DSL", "Fiber optic", "No"],
                    index=option_index(["DSL", "Fiber optic", "No"], last["InternetService"]),
                )
            service_labels = [
                ("OnlineSecurity", "Online security"),
                ("OnlineBackup", "Online backup"),
                ("DeviceProtection", "Device protection"),
                ("TechSupport", "Tech support"),
                ("StreamingTV", "Streaming TV"),
                ("StreamingMovies", "Streaming movies"),
            ]
            service_values: dict[str, str] = {}
            service_columns = st.columns(3)
            for index, (key, label) in enumerate(service_labels):
                with service_columns[index % 3]:
                    service_values[key] = st.selectbox(
                        label,
                        ["No", "Yes", "No internet service"],
                        index=option_index(["No", "Yes", "No internet service"], last[key]),
                    )

            submitted = st.form_submit_button("Score this customer", width="stretch")

        if submitted:
            inputs = {
                "gender": gender,
                "SeniorCitizen": senior,
                "Partner": partner,
                "Dependents": dependents,
                "tenure": tenure,
                "PhoneService": phone,
                "MultipleLines": multiple,
                "InternetService": internet,
                "OnlineSecurity": service_values["OnlineSecurity"],
                "OnlineBackup": service_values["OnlineBackup"],
                "DeviceProtection": service_values["DeviceProtection"],
                "TechSupport": service_values["TechSupport"],
                "StreamingTV": service_values["StreamingTV"],
                "StreamingMovies": service_values["StreamingMovies"],
                "Contract": contract,
                "PaperlessBilling": paperless,
                "PaymentMethod": payment,
                "MonthlyCharges": monthly,
            }
            st.session_state["risk_inputs"] = inputs
            st.session_state["risk_result"] = scenario_result(inputs)

    result = st.session_state["risk_result"]
    scored = result["scored"]
    probability = float(scored["churn_probability"])
    tier = str(scored["risk_tier"])
    color = risk_color(probability)
    flagged = probability >= THRESHOLD
    expected_annual = float(scored["expected_annual_revenue_at_risk"])

    with score_column:
        st.markdown(
            f"""
            <div class="score-panel" style="border-top: 4px solid {color};">
                <div class="risk-chip" style="background:{color}1F;color:{color};border:1px solid {color}66;">
                    {html.escape(tier)} risk
                </div>
                <div class="score-value">{probability:.1%}</div>
                <div class="score-sub">Predicted probability of churn for this scenario.</div>
                <div class="mini-stat">
                    <div class="mini-label">Expected annual revenue at risk</div>
                    <div class="mini-value">{fmt_money(expected_annual)}</div>
                </div>
                <div class="mini-stat">
                    <div class="mini-label">Outreach decision at {THRESHOLD:.1%} threshold</div>
                    <div class="mini-value" style="color:{color};">
                        {"Flag for retention" if flagged else "Monitor, do not prioritize"}
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        gauge = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=probability * 100,
                number={"suffix": "%", "font": {"size": 30, "color": INK}},
                gauge={
                    "axis": {
                        "range": [0, 100],
                        "tickwidth": 0,
                        "tickcolor": "#B9C6CD",
                        "showticklabels": False,
                    },
                    "bar": {"color": color, "thickness": 0.28},
                    "bgcolor": "#F3F6F7",
                    "borderwidth": 0,
                    "steps": [
                        {"range": [0, 30], "color": "#E4F3EF"},
                        {"range": [30, 55], "color": "#FBF3DC"},
                        {"range": [55, 75], "color": "#FCE8D8"},
                        {"range": [75, 100], "color": "#F8DFD8"},
                    ],
                    "threshold": {
                        "line": {"color": INK, "width": 3},
                        "thickness": 0.8,
                        "value": THRESHOLD * 100,
                    },
                },
            )
        )
        plot(gauge, height=230)
        st.caption(f"The dark marker is the deployed outreach threshold ({THRESHOLD:.1%}).")

    st.markdown("---")
    driver_tab, action_tab, detail_tab = st.tabs(
        ["Why this score", "Retention plays", "Scenario detail"]
    )
    with driver_tab:
        section(
            "What is moving the probability?",
            "Each bar resets one feature to its training-set baseline while holding the others fixed. "
            "Positive values increase risk; negative values reduce it. This is a sensitivity check, not a causal effect.",
            label="Explanation",
        )
        what_if = result["what_if"].sort_values("impact_pp")
        colors = [CORAL if value > 0 else MINT for value in what_if["impact_pp"]]
        labels = what_if["impact_pp"].map(lambda value: f"{value:+.1f} pp")
        fig = go.Figure(
            go.Bar(
                x=what_if["impact_pp"],
                y=what_if["feature_label"],
                orientation="h",
                marker_color=colors,
                text=labels,
                textposition="outside",
                hovertemplate="%{y}<br>%{x:+.1f} percentage points<extra></extra>",
            )
        )
        fig.add_vline(x=0, line_width=1, line_color="#B9C6CD")
        fig.update_layout(xaxis_title="Change in churn probability (percentage points)")
        plot(fig, height=360)
        with st.expander("See the current vs. baseline values"):
            table = result["what_if"][
                ["feature_label", "current_value", "baseline_value", "probability_if_baseline", "impact_pp"]
            ].copy()
            table["probability_if_baseline"] = table["probability_if_baseline"].map(lambda value: f"{value:.1%}")
            table["impact_pp"] = table["impact_pp"].map(lambda value: f"{value:+.1f} pp")
            st.dataframe(
                table.rename(
                    columns={
                        "feature_label": "Feature",
                        "current_value": "Current",
                        "baseline_value": "Training baseline",
                        "probability_if_baseline": "Probability at baseline",
                        "impact_pp": "Impact",
                    }
                ),
                width="stretch",
                hide_index=True,
            )

    with action_tab:
        section(
            "A retention specialist can start here.",
            "Rules are tied to observable account facts so an agent can act on them. "
            "They are prioritization suggestions, not validated causal interventions.",
            label="Next best action",
        )
        action_html = "".join(
            f'<div class="action-item"><div class="action-index">{index:02d}</div><div>{html.escape(action)}</div></div>'
            for index, action in enumerate(result["actions"], start=1)
        )
        st.markdown(f'<div class="score-panel">{action_html}</div>', unsafe_allow_html=True)

    with detail_tab:
        cleaned = result["cleaned"]
        detail = pd.DataFrame(
            [
                ("Tenure", f"{int(cleaned['tenure'])} months"),
                ("Monthly charges", f"${float(cleaned['MonthlyCharges']):,.2f}"),
                ("Total charges", f"${float(cleaned['TotalCharges']):,.2f}"),
                ("Average monthly spend", f"${float(cleaned['AvgMonthlySpend']):,.2f}"),
                ("Active services", f"{int(cleaned['ServiceCount'])}"),
                ("New customer", "Yes" if int(cleaned["IsNewCustomer"]) == 1 else "No"),
                ("Contract", str(cleaned["Contract"])),
                ("Internet service", str(cleaned["InternetService"])),
                ("Payment method", str(cleaned["PaymentMethod"])),
                ("Tech support", str(cleaned["TechSupport"])),
                ("Online security", str(cleaned["OnlineSecurity"])),
            ],
            columns=["Account fact", "Value"],
        )
        st.dataframe(detail, width="stretch", hide_index=True)


# ---------------------------------------------------------------------------
# Batch scoring
# ---------------------------------------------------------------------------
elif page == "Batch scoring":
    page_head(
        "Batch scoring",
        "Score a customer file in one pass.",
        "Upload the raw Telco schema or use the bundled template. Missing customer IDs are generated, "
        "blank TotalCharges values are handled explicitly, and every row gets a probability, risk tier, and revenue-at-risk estimate.",
    )

    top_left, top_right = st.columns([0.62, 0.38], gap="large")
    with top_left:
        uploaded = st.file_uploader(
            "Upload a CSV",
            type=["csv"],
            help="Use the template or a file with the same 20 feature columns. Churn is optional.",
        )
    with top_right:
        template = batch_scoring_template()
        st.download_button(
            "Download CSV template",
            data=template.to_csv(index=False).encode("utf-8"),
            file_name="telco_churn_batch_template.csv",
            mime="text/csv",
            width="stretch",
        )

    with st.expander("Expected columns and validation rules"):
        st.markdown(
            """
            The app accepts the raw Telco columns: `customerID`, `gender`, `SeniorCitizen`, `Partner`,
            `Dependents`, `tenure`, `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`,
            `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`,
            `Contract`, `PaperlessBilling`, `PaymentMethod`, `MonthlyCharges`, and `TotalCharges`.

            - `customerID` is optional; missing IDs are generated as `row-1`, `row-2`, ...
            - `Churn` is optional and ignored if present.
            - `Tenure` and `MonthlyCharges` must be numeric.
            - Blank `TotalCharges` is filled as `MonthlyCharges × tenure`; for new customers (`tenure=0`) that is zero.
            - Duplicate customer IDs are rejected so scores are not silently overwritten.
            """
        )

    if uploaded is not None:
        try:
            raw_batch = pd.read_csv(uploaded)
        except Exception as exc:  # pragma: no cover - user-facing guard
            st.error(f"Could not read that CSV: {exc}")
            st.stop()
        source_label = "Uploaded file"
    else:
        raw_batch = template.copy()
        source_label = "Bundled sample · first 25 rows"

    if len(raw_batch) > 50_000:
        st.error("Please score at most 50,000 rows at a time so the interactive app stays responsive.")
        st.stop()

    try:
        prepared = prepare_inference_data(raw_batch)
        scored = score_frame(bundle["pipeline"], prepared, threshold=THRESHOLD, include_target=False)
    except DataValidationError as exc:
        st.error(f"Data validation failed: {exc}")
        st.stop()
    except Exception as exc:  # pragma: no cover - user-facing guard
        st.error(f"Scoring failed: {exc}")
        st.stop()

    scored = scored.sort_values("churn_probability", ascending=False).reset_index(drop=True)
    flagged = scored[scored["churn_prediction"] == 1]
    critical = scored[scored["risk_tier"].isin(["High", "Critical"])]
    total_exposure = float(scored["expected_annual_revenue_at_risk"].sum())
    flagged_exposure = float(flagged["expected_annual_revenue_at_risk"].sum())

    st.caption(f"Source: {source_label} · {len(scored):,} customers scored · threshold {THRESHOLD:.1%}")
    kpi_row(
        [
            (
                "Customers scored",
                f"{len(scored):,}",
                "Rows with valid, unique customer IDs.",
                TEAL,
            ),
            (
                "Flagged for outreach",
                f"{len(flagged):,}",
                f"{len(flagged) / max(1, len(scored)):.1%} of the batch.",
                CORAL,
            ),
            (
                "High or critical risk",
                f"{len(critical):,}",
                "Risk tiers at or above 55% predicted churn.",
                AMBER,
            ),
            (
                "Expected annual revenue at risk",
                fmt_money(total_exposure),
                f"{fmt_money(flagged_exposure)} sits in flagged accounts.",
                MINT,
            ),
        ]
    )

    left, right = st.columns([1.2, 0.8], gap="large")
    with left:
        section("Scored accounts", "Sorted by predicted churn probability.", label="Results")
        display = scored[
            [
                "customerID",
                "churn_probability",
                "risk_tier",
                "churn_prediction",
                "expected_annual_revenue_at_risk",
                "tenure",
                "Contract",
                "InternetService",
                "PaymentMethod",
                "MonthlyCharges",
            ]
        ].copy()
        display["expected_annual_revenue_at_risk"] = display["expected_annual_revenue_at_risk"].round(2)
        display["MonthlyCharges"] = display["MonthlyCharges"].round(2)
        st.dataframe(
            display.head(250),
            width="stretch",
            hide_index=True,
            height=430,
            column_config={
                "customerID": "Customer ID",
                "churn_probability": st.column_config.NumberColumn("Churn probability", format="%.3f"),
                "risk_tier": "Risk tier",
                "churn_prediction": st.column_config.NumberColumn("Flag", format="%d"),
                "expected_annual_revenue_at_risk": st.column_config.NumberColumn(
                    "Expected annual revenue at risk", format="$%.0f"
                ),
                "tenure": "Tenure",
                "Contract": "Contract",
                "InternetService": "Internet",
                "PaymentMethod": "Payment method",
                "MonthlyCharges": st.column_config.NumberColumn("Monthly charges", format="$%.2f"),
            },
        )
        if len(scored) > 250:
            st.caption("Showing the first 250 rows; download the CSV for the full scored file.")
        st.download_button(
            "Download scored CSV",
            data=scored.to_csv(index=False).encode("utf-8"),
            file_name="churn_scores.csv",
            mime="text/csv",
        )

    with right:
        tier_counts = (
            scored["risk_tier"].value_counts().reindex(RISK_ORDER).fillna(0).astype(int).reset_index()
        )
        tier_counts.columns = ["Risk tier", "Customers"]
        fig = px.bar(
            tier_counts,
            x="Risk tier",
            y="Customers",
            color="Risk tier",
            color_discrete_map=RISK_COLORS,
            text="Customers",
            category_orders={"Risk tier": RISK_ORDER},
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(showlegend=False, title="Batch composition by risk tier")
        fig.update_yaxes(title="Customers")
        plot(fig, height=290)
        st.markdown(
            f"""
            <div class="callout" style="margin-top:0.5rem;">
                <strong>Campaign math:</strong> {len(flagged):,} flagged accounts carry
                <strong>{fmt_money(flagged_exposure)}</strong> of expected annual revenue exposure.
                Use the threshold slider on the model-performance page to test a tighter or looser campaign.
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.caption(
            "Expected revenue at risk = predicted churn probability × monthly charges × 12, summed across accounts. "
            "It is a prioritization estimate, not a forecast of booked loss."
        )


# ---------------------------------------------------------------------------
# Model performance
# ---------------------------------------------------------------------------
elif page == "Model performance":
    page_head(
        "Evaluation",
        "A model is only useful if the decision policy is explicit.",
        "The model ranks customers by churn risk; the threshold converts those ranks into outreach decisions. "
        "This page replays the saved, stratified 20% holdout evaluation and lets you test a different operating point.",
    )

    y_true = test_predictions["actual_churn"].astype(int).to_numpy()
    y_prob = test_predictions["churn_probability"].to_numpy()
    test_metrics = metrics["test_metrics"]

    kpi_row(
        [
            ("Holdout ROC-AUC", f"{test_metrics['roc_auc']:.3f}", "Ranking quality across all thresholds.", TEAL),
            ("Recall", fmt_pct(test_metrics["recall"]), f"At the {THRESHOLD:.1%} deployed threshold.", CORAL),
            ("Precision", fmt_pct(test_metrics["precision"]), "Share of contacted customers who churned.", AMBER),
            ("F1 score", f"{test_metrics['f1']:.3f}", "Balance of precision and recall.", MINT),
            ("Top-decile lift", f"{test_metrics['top_decile_lift']:.2f}×", "Churn capture versus a random contact.", "#8A6FBF"),
        ]
    )

    section(
        "Five candidates, one explicit selection rule.",
        "Models are compared with 5-fold stratified cross-validation ROC-AUC on the training split. "
        "The winner is evaluated once on the holdout set; the threshold is tuned on out-of-fold training predictions.",
        label="Model selection",
    )
    comparison = pd.DataFrame(metrics["model_comparison"])
    comparison_long = comparison.melt(
        id_vars=["display_name"],
        value_vars=["cv_roc_auc", "test_roc_auc"],
        var_name="Estimate",
        value_name="ROC-AUC",
    )
    comparison_long["Estimate"] = comparison_long["Estimate"].map(
        {"cv_roc_auc": "5-fold CV", "test_roc_auc": "Holdout"}
    )
    fig = px.bar(
        comparison_long,
        x="ROC-AUC",
        y="display_name",
        color="Estimate",
        barmode="group",
        orientation="h",
        color_discrete_map={"5-fold CV": TEAL, "Holdout": CORAL},
        category_orders={"display_name": comparison.sort_values("cv_roc_auc")["display_name"].tolist()},
        labels={"display_name": ""},
    )
    fig.update_xaxes(range=[0.78, 0.87], tickformat=".2f")
    fig.update_layout(legend_title_text="")
    plot(fig, height=360)
    with st.expander("See the comparison table"):
        table = comparison[
            ["display_name", "cv_roc_auc", "cv_roc_auc_std", "test_roc_auc", "test_pr_auc", "test_f1_at_0_5", "selected"]
        ].rename(
            columns={
                "display_name": "Model",
                "cv_roc_auc": "CV ROC-AUC",
                "cv_roc_auc_std": "CV std",
                "test_roc_auc": "Holdout ROC-AUC",
                "test_pr_auc": "Holdout PR-AUC",
                "test_f1_at_0_5": "F1 @ 0.5",
                "selected": "Selected",
            }
        )
        st.dataframe(table, width="stretch", hide_index=True)

    discrimination_tab, policy_tab, drivers_tab = st.tabs(
        ["Discrimination", "Decision policy", "Drivers"]
    )

    with discrimination_tab:
        left, right = st.columns(2, gap="large")
        with left:
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"Model (AUC {roc_auc_score(y_true, y_prob):.3f})", line=dict(color=TEAL, width=3)))
            fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="No-skill", line=dict(color="#B9C6CD", width=1, dash="dash")))
            fig.update_layout(xaxis_title="False positive rate", yaxis_title="True positive rate", title="ROC curve")
            fig.update_xaxes(range=[0, 1])
            fig.update_yaxes(range=[0, 1])
            plot(fig, height=380)
        with right:
            precision, recall, _ = precision_recall_curve(y_true, y_prob)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=recall, y=precision, mode="lines", name="Model", line=dict(color=CORAL, width=3)))
            fig.add_trace(go.Scatter(x=[0, 1], y=[float(np.mean(y_true)), float(np.mean(y_true))], mode="lines", name="Prevalence", line=dict(color="#B9C6CD", width=1, dash="dash")))
            fig.update_layout(xaxis_title="Recall", yaxis_title="Precision", title="Precision–recall curve")
            fig.update_xaxes(range=[0, 1])
            fig.update_yaxes(range=[0, 1])
            plot(fig, height=380)

        observed, predicted = calibration_curve(y_true, y_prob, n_bins=10, strategy="quantile")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=predicted, y=observed, mode="lines+markers", name="Model", line=dict(color=TEAL, width=2.5), marker=dict(size=7)))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Perfect calibration", line=dict(color="#B9C6CD", width=1, dash="dash")))
        fig.update_layout(xaxis_title="Mean predicted probability", yaxis_title="Observed churn rate", title="Calibration")
        fig.update_xaxes(range=[0, 1])
        fig.update_yaxes(range=[0, 1])
        plot(fig, height=340)
        st.caption(
            f"Brier score {test_metrics['brier']:.4f} · log loss {test_metrics['log_loss']:.4f}. "
            "Probabilities are useful for ranking and campaign sizing; they are not perfectly calibrated at every score."
        )

    with policy_tab:
        left, right = st.columns([0.4, 0.6], gap="large")
        with left:
            chosen_threshold = st.slider(
                "Outreach threshold",
                min_value=0.10,
                max_value=0.90,
                value=float(THRESHOLD),
                step=0.001,
                format="%.3f",
                help="Lower thresholds catch more churners but contact more retained customers.",
            )
            chosen_pred = (y_prob >= chosen_threshold).astype(int)
            chosen_precision = precision_score(y_true, chosen_pred, zero_division=0)
            chosen_recall = recall_score(y_true, chosen_pred, zero_division=0)
            chosen_f1 = f1_score(y_true, chosen_pred, zero_division=0)
            chosen_accuracy = accuracy_score(y_true, chosen_pred)
            st.markdown(
                f"""
                <div class="mini-stat">
                    <div class="mini-label">At {chosen_threshold:.3f}</div>
                    <div class="mini-value" style="color:{TEAL};">{chosen_pred.mean():.1%} of holdout customers flagged</div>
                    <div class="mini-value" style="font-size:0.9rem;color:{INK};margin-top:0.35rem;">
                        Precision {chosen_precision:.1%} · Recall {chosen_recall:.1%} · F1 {chosen_f1:.3f}
                    </div>
                    <div class="mini-value" style="font-size:0.8rem;color:{MUTED};margin-top:0.2rem;">
                        Accuracy {chosen_accuracy:.1%} · {int(chosen_pred.sum()):,} contacts
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.caption(
                "The deployed threshold was selected to maximize F1 on out-of-fold training predictions. "
                "A real campaign should set this from contact capacity and the relative cost of a missed churner versus a wasted contact."
            )
        with right:
            matrix = confusion_matrix(y_true, chosen_pred, labels=[0, 1])
            fig = px.imshow(
                matrix,
                text_auto=True,
                color_continuous_scale=["#F4F8F6", MINT, "#0E7C86"],
                labels=dict(x="Predicted", y="Actual", color="Customers"),
                x=["Retain", "Churn"],
                y=["Retained", "Churned"],
            )
            fig.update_layout(coloraxis_showscale=False, title=f"Confusion matrix · threshold {chosen_threshold:.3f}")
            fig.update_traces(textfont_size=16)
            plot(fig, height=330)

        thresholds = np.round(np.arange(0.05, 0.951, 0.01), 3)
        precision_curve = [precision_score(y_true, (y_prob >= value).astype(int), zero_division=0) for value in thresholds]
        recall_curve = [recall_score(y_true, (y_prob >= value).astype(int), zero_division=0) for value in thresholds]
        f1_curve = [f1_score(y_true, (y_prob >= value).astype(int), zero_division=0) for value in thresholds]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=thresholds, y=precision_curve, mode="lines", name="Precision", line=dict(color=TEAL, width=2.4)))
        fig.add_trace(go.Scatter(x=thresholds, y=recall_curve, mode="lines", name="Recall", line=dict(color=CORAL, width=2.4)))
        fig.add_trace(go.Scatter(x=thresholds, y=f1_curve, mode="lines", name="F1", line=dict(color=AMBER, width=2.4)))
        fig.add_vline(x=chosen_threshold, line_width=2, line_dash="dash", line_color=INK)
        fig.update_layout(xaxis_title="Probability threshold", yaxis_title="Score", title="Precision–recall trade-off")
        fig.update_xaxes(range=[0.05, 0.95])
        fig.update_yaxes(range=[0, 1.02])
        plot(fig, height=360)

        section(
            "Probability deciles are a campaign-sizing tool.",
            "Sorting the holdout set by predicted risk shows how much churn a limited-contact campaign can reach.",
            label="Ranking",
        )
        deciles = risk_deciles.sort_values("risk_decile", ascending=False).copy()
        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=deciles["risk_decile"].astype(str),
                y=deciles["churn_rate"],
                name="Observed churn rate",
                marker_color=[CORAL if value >= 0.5 else AMBER if value >= 0.25 else MINT for value in deciles["churn_rate"]],
                text=deciles["churn_rate"].map(lambda value: f"{value:.0%}"),
                textposition="outside",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=deciles["risk_decile"].astype(str),
                y=deciles["churn_capture"],
                name="Cumulative churn capture",
                mode="lines+markers",
                line=dict(color=INK, width=2),
                marker=dict(size=7),
                yaxis="y2",
            )
        )
        fig.update_layout(
            xaxis_title="Risk decile (10 = highest risk)",
            yaxis_title="Observed churn rate",
            yaxis2=dict(title="Cumulative capture", overlaying="y", side="right", tickformat=".0%", range=[0, 1.05]),
            legend=dict(orientation="h", y=1.12),
        )
        fig.update_yaxes(tickformat=".0%")
        plot(fig, height=380)
        st.caption(
            f"The top decile captures {deciles.iloc[0]['churn_capture']:.1%} of all holdout churners; "
            f"the top three deciles capture {deciles.iloc[2]['churn_capture']:.1%}. "
            "That is the practical case for prioritizing a limited outreach team."
        )

    with drivers_tab:
        importance = pd.DataFrame(metrics["feature_importance"]).head(14).sort_values("importance_mean")
        fig = px.bar(
            importance,
            x="importance_mean",
            y="feature",
            orientation="h",
            color="importance_mean",
            color_continuous_scale=[[0, "#CFE8E4"], [1, TEAL]],
            labels={"importance_mean": "Mean ROC-AUC drop when shuffled", "feature": ""},
            text=importance["importance_mean"].map(lambda value: f"{value:.3f}"),
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(coloraxis_showscale=False)
        fig.update_xaxes(range=[0, float(importance["importance_mean"].max()) * 1.18])
        plot(fig, height=420)
        st.caption(
            "Permutation importance measures how much holdout ROC-AUC drops when a feature is shuffled. "
            "Tenure, internet service, contract, monthly charges, and the new-customer flag carry most of the signal; "
            "correlated features can share importance."
        )

        section(
            "Segment view",
            "Observed churn rates for the dimensions used in the retention recommendations.",
            label="Context",
        )
        driver_tabs = st.tabs(["Contract", "Internet", "Payment", "Tenure"])
        for tab, key in zip(driver_tabs, ["contract", "internet_service", "payment_method", "tenure_band"]):
            with tab:
                frame = pd.DataFrame(metrics["segments"][key])
                st.dataframe(
                    frame[["segment", "customers", "churners", "churn_rate", "annual_churn_revenue"]].rename(
                        columns={
                            "segment": "Segment",
                            "customers": "Customers",
                            "churners": "Churners",
                            "churn_rate": "Churn rate",
                            "annual_churn_revenue": "Annual revenue lost",
                        }
                    ),
                    width="stretch",
                    hide_index=True,
                    column_config={
                        "Churn rate": st.column_config.NumberColumn(format="%.1f%%"),
                        "Annual revenue lost": st.column_config.NumberColumn(format="$%.0f"),
                    },
                )

    st.markdown(
        """
        <div class="callout">
            <strong>Read this honestly:</strong> this is a retrospective holdout evaluation on one historical dataset.
            It does not establish that a retention offer will change behavior, and the historical Telco base may not
            represent a current customer population. Use the model to prioritize experiments, then measure saved
            revenue against a holdout or matched control group.
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Method & limits
# ---------------------------------------------------------------------------
elif page == "Method & limits":
    page_head(
        "Method & limits",
        "Reproducible by design, explicit about what it cannot prove.",
        "Every number in this app is regenerated by a deterministic training pipeline. "
        "The model is intentionally simple to explain, deploy, and monitor.",
    )

    left, right = st.columns([0.55, 0.45], gap="large")
    with left:
        section(
            "From raw CSV to a retention decision.",
            "The pipeline is deterministic, stratified, and versioned.",
            label="How it works",
        )
        pipeline_steps = [
            ("01", "Validate the schema", "Check required columns, unique customer IDs, numeric tenure and charges, and the target labels."),
            ("02", "Engineer features", "Fill 11 blank TotalCharges records from tenure × monthly charges; add average monthly spend, service count, and a new-customer flag."),
            ("03", "Compare candidates", "Logistic regression, random forest, gradient boosting, histogram gradient boosting, and extra trees under 5-fold stratified CV."),
            ("04", "Tune the decision", "Select by CV ROC-AUC, tune the F1 threshold on out-of-fold predictions, then evaluate once on the 20% holdout."),
            ("05", "Export the bundle", "Refit the winning pipeline on all labeled rows and serialize the model, threshold, baselines, and metadata."),
        ]
        for number, title, body in pipeline_steps:
            st.markdown(
                f"""
                <div class="action-item">
                    <div class="action-index">{number}</div>
                    <div><strong>{title}.</strong> {body}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
    with right:
        section("Reproduce it locally.", "No hidden state and no notebook-only steps.", label="Reproducibility")
        st.code(
            """# after cloning the repository
cd Customer_Churn_Prediction
python -m venv .venv
.venv\\Scripts\\activate      # Windows
source .venv/bin/activate     # macOS / Linux
pip install -r requirements.txt
python -m churn.train
streamlit run app.py""",
            language="bash",
        )
        st.markdown(
            """
            **Artifacts written by training**
            - `models/churn_model.joblib`
            - `models/metrics.json`
            - `reports/evaluation/*.csv`
            - `reports/figures/*.png`
            """
        )
        st.caption(
            f"Selected model: {SELECTED} · threshold {THRESHOLD:.3f} · "
            f"holdout ROC-AUC {metrics['test_metrics']['roc_auc']:.4f}."
        )

    section(
        "Model card",
        "A concise record of intended use, evaluation, and limitations.",
        label="Governance",
    )
    card_left, card_right = st.columns(2, gap="large")
    with card_left:
        st.markdown(
            f"""
            **Intended use.** Prioritize retention outreach, size a campaign, and explain the observable
            account factors associated with churn risk in a telecom customer base.

            **Not intended for.** Causal claims, individual credit or pricing decisions, or replacing a
            retention agent's judgment. The output is a probability from a historical pattern, not a promise
            that a customer will leave.

            **Training data.** IBM Telco Customer Churn, {DATASET['rows']:,} customers and
            {DATASET['columns']} raw columns, with a {fmt_pct(DATASET['churn_rate'])} observed churn rate.
            The cleaned modeling frame adds three engineered features.

            **Selection.** {SELECTED} won a five-model comparison on 5-fold CV ROC-AUC
            ({metrics['model_comparison'][0]['cv_roc_auc']:.4f} ± {metrics['model_comparison'][0]['cv_roc_auc_std']:.4f}).
            """
        )
    with card_right:
        st.markdown(
            f"""
            **Evaluation.** One stratified 20% holdout ({len(test_predictions):,} customers):
            ROC-AUC {metrics['test_metrics']['roc_auc']:.4f}, PR-AUC {metrics['test_metrics']['pr_auc']:.4f},
            recall {fmt_pct(metrics['test_metrics']['recall'])}, precision {fmt_pct(metrics['test_metrics']['precision'])},
            F1 {metrics['test_metrics']['f1']:.4f}, Brier {metrics['test_metrics']['brier']:.4f}.

            **Decision policy.** A customer is flagged at a {THRESHOLD:.1%} churn probability. That threshold
            was chosen on out-of-fold training predictions to maximize F1 and should be re-tuned to campaign
            capacity and cost.

            **Known limitations.** Retrospective holdout only; no temporal validation; no causal uplift
            measurement; potential drift in pricing, services, and customer mix; correlated features share
            credit in permutation importance.
            """
        )

    section(
        "Data dictionary",
        "The raw schema is intentionally preserved so batch scoring can accept the public dataset format.",
        label="Reference",
    )
    dictionary = pd.DataFrame(
        [
            ("customerID", "Unique customer identifier", "String"),
            ("gender", "Customer gender", "Female / Male"),
            ("SeniorCitizen", "Senior-citizen indicator", "0/1 normalized to No/Yes"),
            ("Partner", "Has a partner", "No / Yes"),
            ("Dependents", "Has dependents", "No / Yes"),
            ("tenure", "Months with the company", "Integer, 0–72"),
            ("PhoneService", "Has phone service", "No / Yes"),
            ("MultipleLines", "Has multiple lines", "No phone service / No / Yes"),
            ("InternetService", "Internet product", "DSL / Fiber optic / No"),
            ("OnlineSecurity", "Has online security", "No / Yes / No internet service"),
            ("OnlineBackup", "Has online backup", "No / Yes / No internet service"),
            ("DeviceProtection", "Has device protection", "No / Yes / No internet service"),
            ("TechSupport", "Has tech support", "No / Yes / No internet service"),
            ("StreamingTV", "Streams TV", "No / Yes / No internet service"),
            ("StreamingMovies", "Streams movies", "No / Yes / No internet service"),
            ("Contract", "Contract term", "Month-to-month / One year / Two year"),
            ("PaperlessBilling", "Uses paperless billing", "No / Yes"),
            ("PaymentMethod", "Payment channel", "Electronic check / Mailed check / Bank transfer / Credit card"),
            ("MonthlyCharges", "Current monthly charge", "USD"),
            ("TotalCharges", "Lifetime charges", "USD; 11 blanks in source"),
            ("Churn", "Observed churn label", "No / Yes (training only)"),
        ],
        columns=["Field", "Definition", "Values"],
    )
    st.dataframe(dictionary, width="stretch", hide_index=True, height=520)

    section("Links", "Inspect the code, data, and generated evaluation.", label="Open source")
    st.markdown(
        f"""
        - [GitHub repository](https://github.com/mlaharsh/Customer_Churn_Prediction)
        - [Project showcase](https://mlaharsh.github.io/Customer_Churn_Prediction/)
        - [Evaluation summary](https://github.com/mlaharsh/Customer_Churn_Prediction/blob/master/reports/evaluation/summary.md)
        - [Metrics JSON](https://github.com/mlaharsh/Customer_Churn_Prediction/blob/master/models/metrics.json)
        - [Data source](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
        """
    )
    st.caption(
        "The app loads the serialized model bundle committed to the repository; it does not train on page load. "
        "Training is reproducible with the command above."
    )


footer()
