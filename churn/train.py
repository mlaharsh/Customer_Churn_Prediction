"""Command-line entry point that trains, evaluates, and exports the churn model."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import train_test_split

from churn.config import (
    DATA_PATH,
    FEATURE_IMPORTANCE_PATH,
    FIGURES_DIR,
    METRICS_PATH,
    MODEL_PATH,
    RANDOM_STATE,
    TEST_SIZE,
)
from churn.data import RAW_COLUMNS, load_clean_data, split_features_target
from churn.insights import score_frame, segment_summary
from churn.modeling import (
    _metric_summary,
    baselines_from_frame,
    save_model_artifact,
    train_and_select,
    write_metrics,
)
from churn.reporting import generate_report_assets


def build_dataset_summary(frame: pd.DataFrame) -> dict[str, Any]:
    """Summarize the canonical dataset for the model card and app."""

    churned = frame[frame["Churn"] == 1]
    monthly_revenue = float(frame["MonthlyCharges"].sum())
    churned_monthly_revenue = float(churned["MonthlyCharges"].sum())
    priority = frame[
        (frame["Contract"] == "Month-to-month")
        & (frame["InternetService"] == "Fiber optic")
    ]
    priority_churned = priority[priority["Churn"] == 1]
    priority_churned_revenue = float(priority_churned["MonthlyCharges"].sum())
    return {
        "source": "IBM Telco Customer Churn (public Kaggle dataset)",
        "rows": int(len(frame)),
        "columns": len(RAW_COLUMNS),
        "modeled_columns": int(frame.shape[1]),
        "customers": int(frame["customerID"].nunique()),
        "churners": int(frame["Churn"].sum()),
        "retained": int(len(frame) - frame["Churn"].sum()),
        "churn_rate": round(float(frame["Churn"].mean()), 4),
        "average_tenure_months": round(float(frame["tenure"].mean()), 2),
        "median_tenure_months": round(float(frame["tenure"].median()), 2),
        "average_monthly_charges": round(float(frame["MonthlyCharges"].mean()), 2),
        "monthly_revenue": round(monthly_revenue, 2),
        "annual_revenue": round(monthly_revenue * 12, 2),
        "monthly_churn_revenue": round(churned_monthly_revenue, 2),
        "annual_churn_revenue": round(churned_monthly_revenue * 12, 2),
        "priority_segment": {
            "name": "Month-to-month · Fiber optic",
            "customers": int(len(priority)),
            "churners": int(priority["Churn"].sum()),
            "churn_rate": round(float(priority["Churn"].mean()), 4),
            "share_of_customers": round(len(priority) / len(frame), 4),
            "share_of_churners": round(float(priority["Churn"].sum()) / float(frame["Churn"].sum()), 4),
            "monthly_churn_revenue": round(priority_churned_revenue, 2),
            "annual_churn_revenue": round(priority_churned_revenue * 12, 2),
            "share_of_churn_revenue": round(priority_churned_revenue / churned_monthly_revenue, 4),
        },
    }


def build_segments(frame: pd.DataFrame) -> dict[str, list[dict[str, Any]]]:
    """Build segment tables used by the app and evaluation report."""

    working = frame.copy()
    working["TenureBand"] = pd.cut(
        working["tenure"],
        bins=[-0.1, 6, 12, 24, 36, 48, 60, 72],
        labels=["0-6 months", "7-12 months", "13-24 months", "25-36 months", "37-48 months", "49-60 months", "61-72 months"],
    )
    dimensions = {
        "contract": "Contract",
        "internet_service": "InternetService",
        "payment_method": "PaymentMethod",
        "tenure_band": "TenureBand",
    }
    segments: dict[str, list[dict[str, Any]]] = {}
    for key, dimension in dimensions.items():
        summary = segment_summary(working, dimension).copy()
        summary["segment"] = summary["segment"].astype(str)
        for column in ["churn_rate"]:
            summary[column] = summary[column].astype(float).round(4)
        for column in ["monthly_revenue", "monthly_churn_revenue", "annual_churn_revenue"]:
            summary[column] = summary[column].astype(float).round(2)
        segments[key] = summary.to_dict(orient="records")
    return segments


def build_risk_deciles(scored_test: pd.DataFrame) -> pd.DataFrame:
    """Rank the holdout set into probability deciles for campaign sizing."""

    ranked = scored_test.copy()
    ranked["risk_decile"] = pd.qcut(
        ranked["churn_probability"].rank(method="first"),
        q=10,
        labels=list(range(1, 11)),
    ).astype(int)
    total_churners = max(1, int(ranked["actual_churn"].sum()))
    summary = (
        ranked.groupby("risk_decile")
        .agg(
            customers=("churn_probability", "size"),
            churners=("actual_churn", "sum"),
            churn_rate=("actual_churn", "mean"),
            average_predicted_probability=("churn_probability", "mean"),
            expected_annual_revenue_at_risk=("expected_annual_revenue_at_risk", "sum"),
        )
        .reset_index()
        .sort_values("risk_decile", ascending=False)
    )
    summary["churn_capture"] = summary["churners"].cumsum() / total_churners
    summary["average_predicted_probability"] = summary["average_predicted_probability"].round(4)
    summary["churn_rate"] = summary["churn_rate"].round(4)
    summary["churn_capture"] = summary["churn_capture"].round(4)
    summary["expected_annual_revenue_at_risk"] = summary["expected_annual_revenue_at_risk"].round(2)
    return summary.reset_index(drop=True)


def write_evaluation_tables(
    comparison: list[dict[str, Any]],
    scored_test: pd.DataFrame,
    risk_deciles: pd.DataFrame,
    segments: dict[str, list[dict[str, Any]]],
    importance: pd.DataFrame,
    evaluation_dir: Path,
) -> None:
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    comparison_frame = pd.DataFrame(comparison)
    if "selected" not in comparison_frame.columns:
        comparison_frame["selected"] = False
    comparison_frame["selected"] = comparison_frame["selected"].map(
        lambda value: bool(value) if pd.notna(value) else False
    )
    comparison_frame.to_csv(evaluation_dir / "model_comparison.csv", index=False)
    scored_test.to_csv(evaluation_dir / "test_predictions.csv", index=False)
    risk_deciles.to_csv(evaluation_dir / "risk_deciles.csv", index=False)
    for name, records in segments.items():
        pd.DataFrame(records).to_csv(evaluation_dir / f"segment_{name}.csv", index=False)
    importance.to_csv(evaluation_dir / "feature_importance.csv", index=False)


def write_evaluation_summary(
    *,
    dataset: dict[str, Any],
    comparison: list[dict[str, Any]],
    selected_name: str,
    threshold: float,
    test_metrics: dict[str, Any],
    importance: pd.DataFrame,
    path: Path,
) -> None:
    best = comparison[0]
    lines = [
        "# Evaluation summary",
        "",
        "Generated by `python -m churn.train`. This file is the source of truth for the headline numbers in the README.",
        "",
        "## Dataset",
        "",
        f"- Rows: **{dataset['rows']:,}** customers",
        f"- Observed churn rate: **{dataset['churn_rate']:.2%}** ({dataset['churners']:,} churners)",
        f"- Monthly recurring revenue represented: **${dataset['monthly_revenue']:,.0f}**",
        f"- Monthly revenue lost to observed churn: **${dataset['monthly_churn_revenue']:,.0f}**",
        "",
        "## Model selection",
        "",
        f"- Selected: **{best['display_name']}**",
        f"- 5-fold CV ROC-AUC: **{best['cv_roc_auc']:.4f}** (± {best['cv_roc_auc_std']:.4f})",
        f"- Holdout ROC-AUC: **{best['test_roc_auc']:.4f}**",
        f"- Decision threshold tuned on out-of-fold predictions: **{threshold:.3f}**",
        "",
        "| Candidate | CV ROC-AUC | Holdout ROC-AUC | Holdout F1 @ 0.5 |",
        "| --- | ---: | ---: | ---: |",
    ]
    for record in comparison:
        marker = " **(selected)**" if record["model"] == selected_name else ""
        lines.append(
            f"| {record['display_name']}{marker} | {record['cv_roc_auc']:.4f} | "
            f"{record['test_roc_auc']:.4f} | {record['test_f1_at_0_5']:.4f} |"
        )
    lines += [
        "",
        "## Holdout performance at the selected threshold",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Accuracy | {test_metrics['accuracy']:.4f} |",
        f"| Precision | {test_metrics['precision']:.4f} |",
        f"| Recall | {test_metrics['recall']:.4f} |",
        f"| F1 | {test_metrics['f1']:.4f} |",
        f"| ROC-AUC | {test_metrics['roc_auc']:.4f} |",
        f"| PR-AUC | {test_metrics['pr_auc']:.4f} |",
        f"| Brier score | {test_metrics['brier']:.4f} |",
        f"| Top-decile churn capture | {test_metrics['top_decile_capture']:.2%} |",
        f"| Top-decile lift | {test_metrics['top_decile_lift']:.2f}x |",
        "",
        "## Strongest permutation-importance features",
        "",
        "| Feature | Mean ROC-AUC drop | Share |",
        "| --- | ---: | ---: |",
    ]
    for _, row in importance.head(8).iterrows():
        lines.append(f"| {row['feature']} | {row['importance_mean']:.4f} | {row['importance_share']:.1%} |")
    lines += [
        "",
        "## Figures",
        "",
        "![Churn overview](../figures/churn_overview.png)",
        "",
        "![Model comparison](../figures/model_comparison.png)",
        "",
        "![Risk distribution](../figures/risk_distribution.png)",
        "",
        "See `reports/figures/` for ROC, precision–recall, calibration, threshold, and feature-importance charts.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def run_training(
    *,
    data_path: Path = DATA_PATH,
    figures: bool = True,
    cv_splits: int = 5,
) -> dict[str, Any]:
    """Train and export all model, metric, and figure artifacts."""

    started = time.time()
    frame = load_clean_data(data_path)
    x, y = split_features_target(frame)
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    print(f"Dataset: {len(frame):,} rows · {frame['Churn'].mean():.2%} churn", flush=True)
    print(f"Training candidates with {cv_splits}-fold CV ...", flush=True)
    result = train_and_select(
        x_train,
        y_train,
        x_test,
        y_test,
        n_splits=cv_splits,
    )
    selected_name = result["selected_name"]
    selected_pipeline = result["selected_pipeline"]
    threshold = result["threshold"]
    print(
        f"Selected {selected_name} · CV ROC-AUC {result['comparison'][0]['cv_roc_auc']:.4f} "
        f"· threshold {threshold:.3f}",
        flush=True,
    )

    baselines = baselines_from_frame(x_train)
    test_frame = x_test.copy()
    test_frame.insert(0, "customerID", frame.loc[x_test.index, "customerID"].to_numpy())
    test_frame["Churn"] = y_test.to_numpy()
    scored_test = score_frame(
        selected_pipeline,
        test_frame,
        threshold=threshold,
        include_target=True,
    )
    scored_test = scored_test[
        [
            "customerID",
            "churn_probability",
            "churn_prediction",
            "risk_tier",
            "actual_churn",
            "expected_monthly_revenue_at_risk",
            "expected_annual_revenue_at_risk",
            "tenure",
            "Contract",
            "InternetService",
            "PaymentMethod",
            "MonthlyCharges",
            "TotalCharges",
        ]
    ]
    default_metrics = _metric_summary(y_test, result["test_probabilities"], 0.5)
    risk_deciles = build_risk_deciles(scored_test)
    segments = build_segments(frame)
    dataset_summary = build_dataset_summary(frame)

    evaluation_dir = Path("reports") / "evaluation"
    write_evaluation_tables(
        comparison=result["comparison"],
        scored_test=scored_test,
        risk_deciles=risk_deciles,
        segments=segments,
        importance=result["permutation_importance"],
        evaluation_dir=evaluation_dir,
    )
    FEATURE_IMPORTANCE_PATH.parent.mkdir(parents=True, exist_ok=True)
    result["permutation_importance"].to_csv(FEATURE_IMPORTANCE_PATH, index=False)

    metrics_payload = write_metrics(
        selected_name=selected_name,
        threshold=threshold,
        comparison=result["comparison"],
        test_metrics=result["test_metrics"],
        importance=result["permutation_importance"],
        dataset_summary=dataset_summary,
        segment_summary=segments,
        path=METRICS_PATH,
    )
    metrics_payload["test_metrics_at_default_threshold"] = default_metrics
    metrics_payload["artifact"] = {
        "path": str(MODEL_PATH.relative_to(Path.cwd())) if MODEL_PATH.is_relative_to(Path.cwd()) else str(MODEL_PATH),
        "fit_scope": "all labeled rows (after holdout evaluation)",
        "random_state": RANDOM_STATE,
    }
    METRICS_PATH.write_text(
        json.dumps(metrics_payload, indent=2, default=str) + "\n",
        encoding="utf-8",
    )

    final_pipeline = clone(selected_pipeline)
    final_pipeline.fit(x, y)
    save_model_artifact(
        final_pipeline,
        threshold=threshold,
        selected_name=selected_name,
        baselines=baselines,
        metrics=metrics_payload,
        path=MODEL_PATH,
    )

    if figures:
        print("Generating figures ...", flush=True)
        generate_report_assets(
            frame,
            comparison=result["comparison"],
            importance=result["permutation_importance"],
            scored_test=scored_test,
            y_test=y_test,
            probabilities=result["test_probabilities"],
            threshold=threshold,
            figures_dir=FIGURES_DIR,
        )

    write_evaluation_summary(
        dataset=dataset_summary,
        comparison=result["comparison"],
        selected_name=selected_name,
        threshold=threshold,
        test_metrics=result["test_metrics"],
        importance=result["permutation_importance"],
        path=evaluation_dir / "summary.md",
    )

    elapsed = time.time() - started
    print(f"Artifacts written to models/ and reports/ in {elapsed:.1f}s", flush=True)
    print(
        f"Holdout ROC-AUC {result['test_metrics']['roc_auc']:.4f} · "
        f"F1 {result['test_metrics']['f1']:.4f} · recall {result['test_metrics']['recall']:.4f}",
        flush=True,
    )
    return {
        "selected_name": selected_name,
        "threshold": threshold,
        "test_metrics": result["test_metrics"],
        "comparison": result["comparison"],
        "dataset": dataset_summary,
        "elapsed_seconds": round(elapsed, 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA_PATH, help="Path to the raw Telco CSV.")
    parser.add_argument("--cv-splits", type=int, default=5, help="Number of stratified CV folds.")
    parser.add_argument("--no-figures", action="store_true", help="Skip matplotlib figure generation.")
    args = parser.parse_args()
    run_training(data_path=args.data, figures=not args.no_figures, cv_splits=args.cv_splits)


if __name__ == "__main__":
    main()
