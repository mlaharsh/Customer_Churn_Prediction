"""Reproducible figures and tables for the README, docs, and model review."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from churn.config import FIGURES_DIR, PALETTE

INK = PALETTE["ink"]
MUTED = PALETTE["muted"]
TEAL = PALETTE["teal"]
MINT = PALETTE["mint"]
CORAL = PALETTE["coral"]
AMBER = PALETTE["amber"]
CREAM = PALETTE["cream"]

RISK_ORDER = ["Low", "Medium", "High", "Critical"]
RISK_COLORS = {
    "Low": MINT,
    "Medium": AMBER,
    "High": "#F4A261",
    "Critical": CORAL,
}


def _pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "axes.labelsize": 10,
            "axes.labelcolor": MUTED,
            "axes.edgecolor": "#D7DDE3",
            "axes.facecolor": "#FFFFFF",
            "figure.facecolor": "#FFFFFF",
            "text.color": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "grid.color": "#E7EBEF",
            "grid.linewidth": 0.8,
            "legend.frameon": False,
        }
    )
    return plt


def _finish(ax, *, grid_axis: str = "y") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#D7DDE3")
    ax.spines["bottom"].set_color("#D7DDE3")
    ax.grid(axis=grid_axis, alpha=0.9)
    ax.set_axisbelow(True)


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    fig.clear()
    return path


def _money(value: float) -> str:
    if value >= 1_000_000:
        return f"${value / 1_000_000:.1f}M"
    if value >= 1_000:
        return f"${value / 1_000:.0f}K"
    return f"${value:,.0f}"


def plot_churn_overview(frame: pd.DataFrame, path: Path = FIGURES_DIR / "churn_overview.png") -> Path:
    """Donut chart plus contract and tenure context for the executive summary."""

    plt = _pyplot()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), gridspec_kw={"width_ratios": [0.9, 1.2, 1.2]})

    counts = frame["Churn"].value_counts().reindex([0, 1]).fillna(0)
    axes[0].pie(
        counts.values,
        labels=["Retained", "Churned"],
        colors=[MINT, CORAL],
        autopct="%1.1f%%",
        startangle=90,
        counterclock=False,
        wedgeprops={"width": 0.38, "edgecolor": "white", "linewidth": 2},
        textprops={"color": INK, "fontsize": 10},
    )
    axes[0].text(0, 0, f"{int(counts.sum()):,}\ncustomers", ha="center", va="center", fontsize=12, color=INK)
    axes[0].set_title("Observed churn mix", loc="left")

    contract_order = ["Month-to-month", "One year", "Two year"]
    contract = (
        frame.groupby("Contract")["Churn"]
        .agg(["mean", "count"])
        .reindex(contract_order)
        .reset_index()
    )
    bars = axes[1].barh(
        contract["Contract"],
        contract["mean"] * 100,
        color=[CORAL, AMBER, MINT],
        height=0.58,
    )
    axes[1].invert_yaxis()
    for bar, rate, count in zip(bars, contract["mean"], contract["count"]):
        axes[1].text(
            rate * 100 + 1.1,
            bar.get_y() + bar.get_height() / 2,
            f"{rate:.1%}  ({int(count):,})",
            va="center",
            fontsize=9,
            color=INK,
        )
    axes[1].set_xlim(0, 55)
    axes[1].set_xlabel("Churn rate")
    axes[1].set_title("Contract length is the strongest segment signal", loc="left")
    _finish(axes[1], grid_axis="x")

    bins = [-0.1, 6, 12, 24, 36, 48, 60, 72]
    labels = ["0-6", "7-12", "13-24", "25-36", "37-48", "49-60", "61-72"]
    tenure = (
        frame.assign(TenureBand=pd.cut(frame["tenure"], bins=bins, labels=labels))
        .groupby("TenureBand", observed=True)["Churn"]
        .agg(["mean", "count"])
        .reset_index()
    )
    axes[2].plot(
        tenure["TenureBand"].astype(str),
        tenure["mean"] * 100,
        color=TEAL,
        linewidth=2.6,
        marker="o",
        markersize=7,
    )
    for x, rate in enumerate(tenure["mean"] * 100):
        axes[2].annotate(f"{rate:.0f}%", (x, rate), textcoords="offset points", xytext=(0, 10), ha="center", fontsize=8, color=TEAL)
    axes[2].set_ylim(0, 65)
    axes[2].set_ylabel("Churn rate")
    axes[2].set_xlabel("Tenure (months)")
    axes[2].set_title("Risk falls sharply after the first year", loc="left")
    _finish(axes[2])

    fig.suptitle("Customer churn at a glance", x=0.02, ha="left", fontsize=17, fontweight="bold", color=INK)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    return _save(fig, path)


def plot_segment_risk(frame: pd.DataFrame, path: Path = FIGURES_DIR / "segment_risk_profile.png") -> Path:
    """Three-panel comparison of the most actionable customer segments."""

    plt = _pyplot()
    segments = [
        ("Contract", "Contract", ["Month-to-month", "One year", "Two year"]),
        ("InternetService", "Internet service", ["Fiber optic", "DSL", "No"]),
        (
            "PaymentMethod",
            "Payment method",
            ["Electronic check", "Mailed check", "Credit card (automatic)", "Bank transfer (automatic)"],
        ),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, (column, title, order) in zip(axes, segments):
        summary = (
            frame.groupby(column)["Churn"]
            .agg(["mean", "count"])
            .reindex(order)
            .dropna()
            .reset_index()
        )
        colors = [CORAL if rate >= 0.35 else AMBER if rate >= 0.18 else MINT for rate in summary["mean"]]
        bars = ax.barh(summary[column], summary["mean"] * 100, color=colors, height=0.6)
        ax.invert_yaxis()
        for bar, rate, count in zip(bars, summary["mean"], summary["count"]):
            ax.text(
                rate * 100 + 1.0,
                bar.get_y() + bar.get_height() / 2,
                f"{rate:.1%} · {int(count):,}",
                va="center",
                fontsize=9,
                color=INK,
            )
        ax.set_xlim(0, max(50, float(summary["mean"].max()) * 100 + 12))
        ax.set_xlabel("Churn rate")
        ax.set_title(title, loc="left")
        _finish(ax, grid_axis="x")

    fig.suptitle("Where churn concentrates", x=0.02, ha="left", fontsize=17, fontweight="bold", color=INK)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    return _save(fig, path)


def plot_model_comparison(
    comparison: Iterable[dict[str, Any]],
    path: Path = FIGURES_DIR / "model_comparison.png",
) -> Path:
    """Cross-validated and holdout ROC-AUC for every candidate model."""

    plt = _pyplot()
    records = sorted(list(comparison), key=lambda item: item["cv_roc_auc"], reverse=True)
    names = [item["display_name"] for item in records]
    cv_scores = [item["cv_roc_auc"] for item in records]
    cv_std = [item.get("cv_roc_auc_std", 0.0) for item in records]
    test_scores = [item["test_roc_auc"] for item in records]
    y_positions = np.arange(len(names))
    height = 0.34

    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    ax.barh(
        y_positions - height / 2,
        cv_scores,
        xerr=cv_std,
        color=TEAL,
        height=height,
        alpha=0.92,
        label="5-fold CV ROC-AUC",
        error_kw={"ecolor": MUTED, "elinewidth": 0.8, "capsize": 2},
    )
    ax.barh(
        y_positions + height / 2,
        test_scores,
        color=CORAL,
        height=height,
        alpha=0.92,
        label="Holdout ROC-AUC",
    )
    ax.set_yticks(y_positions, names)
    ax.invert_yaxis()
    lower = max(0.72, min(cv_scores + test_scores) - 0.03)
    upper = max([cv + std for cv, std in zip(cv_scores, cv_std)] + test_scores) + 0.03
    ax.set_xlim(lower, upper)
    for position, cv, std, test in zip(y_positions, cv_scores, cv_std, test_scores):
        ax.text(cv + std + 0.002, position - height / 2, f"{cv:.4f}", va="center", fontsize=8.5, color=INK)
        ax.text(test + 0.002, position + height / 2, f"{test:.4f}", va="center", fontsize=8.5, color=CORAL)
    ax.set_xlabel("ROC-AUC (higher is better)")
    ax.set_title("Candidate model comparison", loc="left")
    ax.legend(loc="lower right", ncol=2, fontsize=9)
    _finish(ax, grid_axis="x")
    fig.tight_layout()
    return _save(fig, path)


def plot_roc_curve(
    y_true: pd.Series,
    probabilities: np.ndarray,
    path: Path = FIGURES_DIR / "roc_curve.png",
) -> Path:
    plt = _pyplot()
    fpr, tpr, _ = roc_curve(y_true, probabilities)
    auc = roc_auc_score(y_true, probabilities)
    fig, ax = plt.subplots(figsize=(5.6, 5.0))
    ax.plot(fpr, tpr, color=TEAL, linewidth=2.6, label=f"Selected model (AUC {auc:.4f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="#B8C2CC", linewidth=1.2, label="No-skill baseline")
    ax.fill_between(fpr, tpr, alpha=0.10, color=TEAL)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curve · holdout set", loc="left")
    ax.legend(loc="lower right", fontsize=9)
    _finish(ax)
    fig.tight_layout()
    return _save(fig, path)


def plot_precision_recall_curve(
    y_true: pd.Series,
    probabilities: np.ndarray,
    path: Path = FIGURES_DIR / "precision_recall_curve.png",
) -> Path:
    plt = _pyplot()
    precision, recall, _ = precision_recall_curve(y_true, probabilities)
    ap = average_precision_score(y_true, probabilities)
    prevalence = float(np.mean(y_true))
    fig, ax = plt.subplots(figsize=(5.6, 5.0))
    ax.plot(recall, precision, color=CORAL, linewidth=2.6, label=f"Selected model (AP {ap:.4f})")
    ax.axhline(prevalence, linestyle="--", color="#B8C2CC", linewidth=1.2, label=f"Prevalence {prevalence:.1%}")
    ax.fill_between(recall, precision, alpha=0.10, color=CORAL)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision–recall curve · holdout set", loc="left")
    ax.legend(loc="upper right", fontsize=9)
    _finish(ax)
    fig.tight_layout()
    return _save(fig, path)


def plot_confusion_matrix(
    y_true: pd.Series,
    predictions: np.ndarray,
    threshold: float,
    path: Path = FIGURES_DIR / "confusion_matrix.png",
) -> Path:
    plt = _pyplot()
    matrix = confusion_matrix(y_true, predictions, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    image = ax.imshow(matrix, cmap="BuGn", aspect="auto")
    labels = [["True negative", "False positive"], ["False negative", "True positive"]]
    for row in range(2):
        for column in range(2):
            value = matrix[row, column]
            color = "white" if value > matrix.max() * 0.55 else INK
            ax.text(column, row, f"{value:,}\n{labels[row][column]}", ha="center", va="center", fontsize=10, color=color)
    ax.set_xticks([0, 1], ["Predicted retain", "Predicted churn"])
    ax.set_yticks([0, 1], ["Actually retained", "Actually churned"])
    ax.set_title(f"Confusion matrix at threshold {threshold:.3f}", loc="left")
    ax.grid(False)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="Customers")
    fig.tight_layout()
    return _save(fig, path)


def plot_threshold_tradeoff(
    y_true: pd.Series,
    probabilities: np.ndarray,
    threshold: float,
    path: Path = FIGURES_DIR / "threshold_tradeoff.png",
) -> Path:
    plt = _pyplot()
    thresholds = np.round(np.arange(0.05, 0.96, 0.01), 2)
    precision = [precision_score(y_true, (probabilities >= value).astype(int), zero_division=0) for value in thresholds]
    recall = [recall_score(y_true, (probabilities >= value).astype(int), zero_division=0) for value in thresholds]
    f1 = [f1_score(y_true, (probabilities >= value).astype(int), zero_division=0) for value in thresholds]

    fig, ax = plt.subplots(figsize=(10, 4.6))
    ax.plot(thresholds, precision, color=TEAL, linewidth=2.2, label="Precision")
    ax.plot(thresholds, recall, color=CORAL, linewidth=2.2, label="Recall")
    ax.plot(thresholds, f1, color=AMBER, linewidth=2.2, label="F1")
    ax.axvline(threshold, color=INK, linestyle="--", linewidth=1.2)
    ax.text(threshold + 0.012, 0.04, f"Selected threshold\n{threshold:.3f}", fontsize=8, color=INK)
    ax.set_xlim(0.05, 0.95)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Probability threshold")
    ax.set_ylabel("Score")
    ax.set_title("Precision–recall trade-off by decision threshold", loc="left")
    ax.legend(loc="lower left", ncols=3, fontsize=9)
    _finish(ax)
    fig.tight_layout()
    return _save(fig, path)


def plot_feature_importance(
    importance: pd.DataFrame,
    path: Path = FIGURES_DIR / "feature_importance.png",
    top_n: int = 14,
) -> Path:
    plt = _pyplot()
    top = importance.head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9.2, 5.8))
    bars = ax.barh(
        top["feature"].str.replace("_", " "),
        top["importance_mean"],
        xerr=top["importance_std"],
        color=TEAL,
        alpha=0.9,
        height=0.62,
        error_kw={"ecolor": MUTED, "elinewidth": 0.8, "capsize": 2},
    )
    for bar, value, std in zip(bars, top["importance_mean"], top["importance_std"]):
        ax.text(
            value + std + 0.0015,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.3f}",
            va="center",
            fontsize=8,
            color=INK,
        )
    upper = float((top["importance_mean"] + top["importance_std"]).max())
    ax.set_xlim(0, upper * 1.18)
    ax.set_xlabel("Drop in holdout ROC-AUC when shuffled")
    ax.set_title("Permutation importance · selected model", loc="left")
    _finish(ax, grid_axis="x")
    fig.tight_layout()
    return _save(fig, path)


def plot_risk_distribution(
    scored: pd.DataFrame,
    path: Path = FIGURES_DIR / "risk_distribution.png",
) -> Path:
    plt = _pyplot()
    data = (
        scored.groupby("risk_tier")
        .agg(customers=("churn_probability", "size"), churn_rate=("actual_churn", "mean"))
        .reindex(RISK_ORDER)
        .dropna()
        .reset_index()
    )
    retained = []
    churned = []
    for tier in data["risk_tier"]:
        subset = scored[scored["risk_tier"] == tier]
        churned.append(int((subset["actual_churn"] == 1).sum()))
        retained.append(int((subset["actual_churn"] == 0).sum()))

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.4), gridspec_kw={"width_ratios": [1.2, 1]})
    colors = [RISK_COLORS.get(tier, TEAL) for tier in data["risk_tier"]]
    axes[0].bar(data["risk_tier"], retained, color="#DCE6EA", label="Retained")
    axes[0].bar(data["risk_tier"], churned, bottom=retained, color=colors, label="Churned")
    for index, (keep, churn) in enumerate(zip(retained, churned)):
        axes[0].text(index, keep + churn + 3, f"{keep + churn:,}", ha="center", fontsize=9, color=INK)
    axes[0].set_ylabel("Customers")
    axes[0].set_title("Risk tiers on the holdout set", loc="left")
    axes[0].legend(loc="upper right", ncols=2, fontsize=8)
    _finish(axes[0])

    bars = axes[1].bar(data["risk_tier"], data["churn_rate"] * 100, color=colors, width=0.62)
    for bar, rate in zip(bars, data["churn_rate"]):
        axes[1].text(bar.get_x() + bar.get_width() / 2, rate * 100 + 2, f"{rate:.1%}", ha="center", fontsize=9, color=INK)
    axes[1].set_ylim(0, 105)
    axes[1].set_ylabel("Observed churn rate")
    axes[1].set_title("Calibration by tier", loc="left")
    _finish(axes[1])

    fig.suptitle("Model risk bands separate churners from retained customers", x=0.02, ha="left", fontsize=17, fontweight="bold", color=INK)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    return _save(fig, path)


def plot_calibration(
    y_true: pd.Series,
    probabilities: np.ndarray,
    path: Path = FIGURES_DIR / "calibration_curve.png",
) -> Path:
    plt = _pyplot()
    observed, predicted = calibration_curve(y_true, probabilities, n_bins=10, strategy="quantile")
    fig, ax = plt.subplots(figsize=(5.6, 5.0))
    ax.plot(predicted, observed, marker="o", color=TEAL, linewidth=2.2, label="Selected model")
    ax.plot([0, 1], [0, 1], linestyle="--", color="#B8C2CC", linewidth=1.2, label="Perfect calibration")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Mean predicted churn probability")
    ax.set_ylabel("Observed churn rate")
    ax.set_title("Calibration · holdout set", loc="left")
    ax.legend(loc="upper left", fontsize=9)
    _finish(ax)
    fig.tight_layout()
    return _save(fig, path)


def plot_revenue_at_risk(
    scored: pd.DataFrame,
    path: Path = FIGURES_DIR / "revenue_at_risk.png",
) -> Path:
    plt = _pyplot()
    data = (
        scored.groupby("risk_tier")["expected_annual_revenue_at_risk"]
        .sum()
        .reindex(RISK_ORDER)
        .dropna()
        .reset_index()
    )
    fig, ax = plt.subplots(figsize=(8.6, 4.2))
    bars = ax.bar(
        data["risk_tier"],
        data["expected_annual_revenue_at_risk"],
        color=[RISK_COLORS.get(tier, TEAL) for tier in data["risk_tier"]],
        width=0.58,
    )
    for bar, value in zip(bars, data["expected_annual_revenue_at_risk"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + max(data["expected_annual_revenue_at_risk"]) * 0.015,
            _money(value),
            ha="center",
            fontsize=9,
            color=INK,
        )
    ax.set_ylabel("Expected annual revenue at risk")
    ax.set_title("Where the revenue exposure sits · holdout set", loc="left")
    ax.yaxis.set_major_formatter(lambda value, _: _money(value))
    _finish(ax)
    fig.tight_layout()
    return _save(fig, path)


def generate_report_assets(
    frame: pd.DataFrame,
    *,
    comparison: list[dict[str, Any]],
    importance: pd.DataFrame,
    scored_test: pd.DataFrame,
    y_test: pd.Series,
    probabilities: np.ndarray,
    threshold: float,
    figures_dir: Path = FIGURES_DIR,
) -> dict[str, Path]:
    """Generate every figure used by the README, docs, and Pages showcase."""

    figures_dir = Path(figures_dir)
    predictions = (probabilities >= threshold).astype(int)
    outputs = {
        "churn_overview": plot_churn_overview(frame, figures_dir / "churn_overview.png"),
        "segment_risk": plot_segment_risk(frame, figures_dir / "segment_risk_profile.png"),
        "model_comparison": plot_model_comparison(comparison, figures_dir / "model_comparison.png"),
        "roc_curve": plot_roc_curve(y_test, probabilities, figures_dir / "roc_curve.png"),
        "precision_recall_curve": plot_precision_recall_curve(
            y_test, probabilities, figures_dir / "precision_recall_curve.png"
        ),
        "confusion_matrix": plot_confusion_matrix(
            y_test, predictions, threshold, figures_dir / "confusion_matrix.png"
        ),
        "threshold_tradeoff": plot_threshold_tradeoff(
            y_test, probabilities, threshold, figures_dir / "threshold_tradeoff.png"
        ),
        "feature_importance": plot_feature_importance(
            importance, figures_dir / "feature_importance.png"
        ),
        "risk_distribution": plot_risk_distribution(
            scored_test, figures_dir / "risk_distribution.png"
        ),
        "calibration": plot_calibration(y_test, probabilities, figures_dir / "calibration_curve.png"),
        "revenue_at_risk": plot_revenue_at_risk(
            scored_test, figures_dir / "revenue_at_risk.png"
        ),
    }
    return outputs
