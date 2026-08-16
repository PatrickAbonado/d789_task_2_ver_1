"""
Compare Performance Metrics & Outputs of All 4 Housing Models
=============================================================
Models compared
---------------
1. Classical Linear Regression          (regression.py)
2. MLPRegressor (Deep Learning)         (MLP.py)
3. 1-D CNN (Deep Learning)              (1D_CNN.py)
4. RandomForest Classifier              (classification.py)

How it works
------------
- Parses the text reports written by each pipeline (must be run first).
- Builds a unified comparison table for the three regression models
  (R², Adj-R², MAE, RMSE, MAPE) and a separate table for the
  classification model (Accuracy, Precision, Recall, F1, CV Accuracy).
- Writes a consolidated text report and a multi-panel visualization
  that highlights which feature-selection method won for each model
  and ranks the regression models by RMSE / R².

Expected report locations (relative to this script or project root)
------------------------------------------------------------------
  reports/regression_reports/regression_report.txt
  reports/mlp_reports/deep_learning_MLP_report.txt
  reports/1d_cnn_reports/model_report.txt
  reports/random_forest_reports/classification_report.txt

Usage
-----
  python compare_all_models.py
  # or, if reports live one level up:
  python compare_all_models.py --reports-root ../reports
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
# Script directory (works even if you rename / move the file)
SCRIPT_DIR = Path(__file__).resolve().parent

# Candidate locations for the "reports" folder, tried in order.
# Covers the common layouts for this project on Windows/Linux.
CANDIDATE_REPORT_ROOTS = [
    SCRIPT_DIR / "reports",                    # next to this script
    SCRIPT_DIR.parent / "reports",             # one level up (../reports)
    SCRIPT_DIR.parent.parent / "reports",      # two levels up
    Path.cwd() / "reports",                    # current working directory
    Path.cwd().parent / "reports",             # parent of cwd
]

DEFAULT_OUTPUT_DIR = None  # set later once reports_root is known
RANDOM_STATE = 42

# Report file relative paths (under the reports root)
REPORT_PATHS = {
    "LinearRegression": "regression_reports/regression_report.txt",
    "MLP": "mlp_reports/deep_learning_MLP_report.txt",
    "1D_CNN": "1d_cnn_reports/model_report.txt",
    "Classifier": "random_forest_reports/classification_report.txt",
}


def find_reports_root(explicit: Optional[Path] = None) -> Optional[Path]:
    """
    Locate the reports directory that contains at least one of the four
    model report files.  If `explicit` is given it is tried first.
    """
    candidates = []
    if explicit is not None:
        candidates.append(Path(explicit).resolve())
    candidates.extend(CANDIDATE_REPORT_ROOTS)

    seen = set()
    for root in candidates:
        root = root.resolve()
        if root in seen:
            continue
        seen.add(root)
        if not root.is_dir():
            continue
        # Does this folder contain any of the expected report files?
        for rel in REPORT_PATHS.values():
            if (root / rel).is_file():
                return root
    return None


# ---------------------------------------------------------------------------
# Helpers – parse numeric values from report text
# ---------------------------------------------------------------------------
def _find_float(pattern: str, text: str, default: Optional[float] = None) -> Optional[float]:
    m = re.search(pattern, text, flags=re.IGNORECASE | re.MULTILINE)
    if not m:
        return default
    raw = m.group(1).replace(",", "").replace("$", "").strip()
    try:
        return float(raw)
    except ValueError:
        return default


def _find_all_floats(pattern: str, text: str) -> List[float]:
    return [float(x.replace(",", "").replace("$", "")) for x in re.findall(pattern, text, flags=re.IGNORECASE)]


# ---------------------------------------------------------------------------
# Parsers for each model family
# ---------------------------------------------------------------------------
def parse_regression_report(text: str) -> Dict[str, Any]:
    """Parse classical LinearRegression report (SelectKBest vs RFE)."""
    result: Dict[str, Any] = {"model": "LinearRegression", "selectors": {}}

    # Split by feature-selection blocks
    blocks = re.split(r"-{10,}\s*FEATURE SELECTION METHOD:\s*", text)
    for block in blocks[1:]:
        name_m = re.match(r"([^\n]+)", block)
        if not name_m:
            continue
        name = name_m.group(1).strip()
        # Normalize name
        if "SelectKBest" in name:
            key = "SelectKBest"
        elif "RFE" in name:
            key = "RFE"
        else:
            key = name

        metrics = {
            "R2": _find_float(r"R2\s*[:=]\s*([\d.,]+)", block),
            "Adjusted_R2": _find_float(r"Adjusted R2\s*[:=]\s*([\d.,]+)", block),
            "MAE": _find_float(r"MAE\s*[:=]\s*([\d.,]+)", block),
            "RMSE": _find_float(r"RMSE\s*[:=]\s*([\d.,]+)", block),
            "MAPE": _find_float(r"MAPE\s*(?:%)?\s*[:=]\s*([\d.,]+)", block),
        }
        result["selectors"][key] = metrics

    # Also try the SUMMARY COMPARISON table if individual blocks failed
    if not result["selectors"]:
        # Fallback: look for the comparison table
        table_match = re.search(
            r"SUMMARY COMPARISON.*?Metric\s+SelectKBest\s+RFE\s*(.*?)(?:Best performing|$)",
            text, re.DOTALL | re.IGNORECASE
        )
        if table_match:
            rows = re.findall(
                r"(R2|Adjusted R2|MAE|RMSE|MAPE \(?%?\))\s+([\d.,]+)\s+([\d.,]+)",
                table_match.group(1), re.IGNORECASE
            )
            sk, rfe = {}, {}
            for metric, v1, v2 in rows:
                metric = metric.replace(" ", "_").replace("(%", "").replace(")", "").replace("%", "")
                sk[metric] = float(v1.replace(",", ""))
                rfe[metric] = float(v2.replace(",", ""))
            result["selectors"] = {"SelectKBest": sk, "RFE": rfe}

    return result


def parse_mlp_report(text: str) -> Dict[str, Any]:
    """Parse MLPRegressor report."""
    result: Dict[str, Any] = {"model": "MLP", "selectors": {}}

    blocks = re.split(r"-{10,}\s*Feature Selection Method:\s*", text)
    for block in blocks[1:]:
        name_m = re.match(r"([^\n]+)", block)
        if not name_m:
            continue
        name = name_m.group(1).strip()
        if "SelectKBest" in name:
            key = "SelectKBest"
        elif "RFE" in name:
            key = "RFE"
        else:
            key = name

        metrics = {
            "R2": _find_float(r"R-squared\s*[:=]\s*([\d.,]+)", block),
            "Adjusted_R2": _find_float(r"Adjusted R-squared\s*[:=]\s*([\d.,]+)", block),
            "MAE": _find_float(r"MAE\s*[:=]\s*([\d.,]+)", block),
            "RMSE": _find_float(r"RMSE\s*[:=]\s*([\d.,]+)", block),
            "MAPE": _find_float(r"MAPE\s*[:=]\s*([\d.,]+)", block),
        }
        result["selectors"][key] = metrics

    return result


def parse_cnn_report(text: str) -> Dict[str, Any]:
    """Parse 1-D CNN report."""
    result: Dict[str, Any] = {"model": "1D_CNN", "selectors": {}}

    # Look for the performance table
    # Format example:
    #   Method          RMSE ($)      MAE ($)        R2     RMSLE
    #   SelectKBest         ...         ...        ...      ...
    #   RFE                 ...         ...        ...      ...
    table = re.search(
        r"VALIDATION PERFORMANCE.*?Method\s+RMSE.*?RMSLE\s*(.*?)(?:Best feature selection|$)",
        text, re.DOTALL | re.IGNORECASE
    )
    if table:
        rows = re.findall(
            r"(SelectKBest|RFE)\s+([\d,]+)\s+([\d,]+)\s+([\d.]+)\s+([\d.]+)",
            table.group(1), re.IGNORECASE
        )
        for name, rmse, mae, r2, rmsle in rows:
            result["selectors"][name] = {
                "RMSE": float(rmse.replace(",", "")),
                "MAE": float(mae.replace(",", "")),
                "R2": float(r2),
                "RMSLE": float(rmsle),
                "MAPE": None,  # not reported
                "Adjusted_R2": None,
            }
    else:
        # Fallback line-by-line
        for name in ["SelectKBest", "RFE"]:
            block = re.search(rf"{name}.*?RMSE.*?([\d,]+).*?MAE.*?([\d,]+).*?R2.*?([\d.]+)", text, re.DOTALL | re.IGNORECASE)
            if block:
                result["selectors"][name] = {
                    "RMSE": float(block.group(1).replace(",", "")),
                    "MAE": float(block.group(2).replace(",", "")),
                    "R2": float(block.group(3)),
                    "RMSLE": None,
                    "MAPE": None,
                    "Adjusted_R2": None,
                }

    return result


def parse_classification_report(text: str) -> Dict[str, Any]:
    """Parse RandomForest classification report."""
    result: Dict[str, Any] = {"model": "Classifier", "selectors": {}}

    blocks = re.split(r"-{10,}\s*FEATURE SELECTION METHOD:\s*", text)
    for block in blocks[1:]:
        name_m = re.match(r"([^\n]+)", block)
        if not name_m:
            continue
        name = name_m.group(1).strip()
        if "SelectKBest" in name:
            key = "SelectKBest"
        elif "RFE" in name:
            key = "RFE"
        else:
            key = name

        metrics = {
            "Accuracy": _find_float(r"Test Accuracy\s*[:=]\s*([\d.]+)", block),
            "Precision_weighted": _find_float(r"Weighted Precision\s*[:=]\s*([\d.]+)", block),
            "Recall_weighted": _find_float(r"Weighted Recall\s*[:=]\s*([\d.]+)", block),
            "F1_weighted": _find_float(r"Weighted F1-score\s*[:=]\s*([\d.]+)", block),
            "CV_mean_accuracy": _find_float(r"5-fold CV Accuracy.*?([\d.]+)\s*\+/-", block),
            "CV_std_accuracy": _find_float(r"5-fold CV Accuracy.*?[\d.]+\s*\+/-\s*([\d.]+)", block),
        }
        result["selectors"][key] = metrics

    return result


PARSERS = {
    "LinearRegression": parse_regression_report,
    "MLP": parse_mlp_report,
    "1D_CNN": parse_cnn_report,
    "Classifier": parse_classification_report,
}


# ---------------------------------------------------------------------------
# Load & parse all available reports
# ---------------------------------------------------------------------------
def load_all_reports(reports_root: Path) -> Dict[str, Dict[str, Any]]:
    loaded = {}
    missing = []

    print(f"Using reports root: {reports_root}\n")

    for model_name, rel_path in REPORT_PATHS.items():
        full = reports_root / rel_path
        if not full.is_file():
            missing.append(str(full))
            continue
        text = full.read_text(encoding="utf-8", errors="replace")
        parser = PARSERS[model_name]
        loaded[model_name] = parser(text)
        print(f"[OK] Parsed {model_name:20s} from {full}")

    if missing:
        print("\n[WARNING] The following report files were not found:")
        for m in missing:
            print(f"          {m}")
        print("         Run the individual model scripts first to generate them.\n")

    return loaded


# ---------------------------------------------------------------------------
# Build comparison tables
# ---------------------------------------------------------------------------
def build_regression_table(reports: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    """Unified table for the three continuous-target models."""
    rows = []
    for model_name in ["LinearRegression", "MLP", "1D_CNN"]:
        if model_name not in reports:
            continue
        for sel_name, metrics in reports[model_name].get("selectors", {}).items():
            row = {
                "Model": model_name,
                "Feature_Selection": sel_name,
                "R2": metrics.get("R2"),
                "Adjusted_R2": metrics.get("Adjusted_R2"),
                "MAE": metrics.get("MAE"),
                "RMSE": metrics.get("RMSE"),
                "MAPE": metrics.get("MAPE"),
                "RMSLE": metrics.get("RMSLE"),
            }
            rows.append(row)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    # Rank by RMSE (lower better); fallback to R2 if RMSE missing
    if df["RMSE"].notna().any():
        df = df.sort_values("RMSE", ascending=True).reset_index(drop=True)
    else:
        df = df.sort_values("R2", ascending=False).reset_index(drop=True)
    return df


def build_classification_table(reports: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    if "Classifier" not in reports:
        return pd.DataFrame()
    rows = []
    for sel_name, metrics in reports["Classifier"].get("selectors", {}).items():
        rows.append({
            "Model": "RandomForestClassifier",
            "Feature_Selection": sel_name,
            "Accuracy": metrics.get("Accuracy"),
            "Precision_weighted": metrics.get("Precision_weighted"),
            "Recall_weighted": metrics.get("Recall_weighted"),
            "F1_weighted": metrics.get("F1_weighted"),
            "CV_mean_accuracy": metrics.get("CV_mean_accuracy"),
            "CV_std_accuracy": metrics.get("CV_std_accuracy"),
        })
    return pd.DataFrame(rows)


def best_per_model(reg_df: pd.DataFrame) -> pd.DataFrame:
    """Pick the better feature-selection method for each model (by RMSE)."""
    if reg_df.empty:
        return pd.DataFrame()
    best = (
        reg_df.dropna(subset=["RMSE"])
        .sort_values("RMSE")
        .groupby("Model", as_index=False)
        .first()
    )
    return best.sort_values("RMSE").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Text report
# ---------------------------------------------------------------------------
def write_text_report(
    reg_df: pd.DataFrame,
    clf_df: pd.DataFrame,
    best_df: pd.DataFrame,
    out_path: Path,
) -> str:
    lines: List[str] = []
    lines.append("=" * 78)
    lines.append("HOUSING PRICE MODELS — CROSS-MODEL COMPARISON REPORT")
    lines.append("=" * 78)
    lines.append("")
    lines.append("Models evaluated")
    lines.append("  1. LinearRegression          (classical, sklearn)")
    lines.append("  2. MLPRegressor               (deep learning, sklearn)")
    lines.append("  3. 1-D CNN                    (deep learning, PyTorch)")
    lines.append("  4. RandomForestClassifier     (classification of price quartiles)")
    lines.append("")
    lines.append("All three regression models share the same high-level pipeline:")
    lines.append("  ColumnTransformer (impute + scale / one-hot) → Feature selection")
    lines.append("  (SelectKBest or RFE) → Model.  Metrics are reported on a held-out")
    lines.append("  20 % validation / test split of Train.csv.")
    lines.append("")

    # ---- Regression comparison ----
    lines.append("-" * 78)
    lines.append("REGRESSION MODELS (continuous SalePrice target)")
    lines.append("-" * 78)
    if reg_df.empty:
        lines.append("  No regression reports could be parsed. Run the model scripts first.")
    else:
        # Pretty table
        display_cols = ["Model", "Feature_Selection", "R2", "Adjusted_R2", "MAE", "RMSE", "MAPE"]
        present = [c for c in display_cols if c in reg_df.columns]
        tbl = reg_df[present].copy()
        for col in ["R2", "Adjusted_R2"]:
            if col in tbl:
                tbl[col] = tbl[col].map(lambda x: f"{x:.4f}" if pd.notna(x) else "—")
        for col in ["MAE", "RMSE"]:
            if col in tbl:
                tbl[col] = tbl[col].map(lambda x: f"{x:,.0f}" if pd.notna(x) else "—")
        if "MAPE" in tbl:
            tbl["MAPE"] = tbl["MAPE"].map(lambda x: f"{x:.2f}%" if pd.notna(x) else "—")
        lines.append(tbl.to_string(index=False))
        lines.append("")

        if not best_df.empty:
            lines.append("Best feature-selection method per model (lowest RMSE):")
            for _, row in best_df.iterrows():
                r2_str = f"{row['R2']:.4f}" if pd.notna(row.get("R2")) else "—"
                rmse_str = f"${row['RMSE']:,.0f}" if pd.notna(row.get("RMSE")) else "—"
                lines.append(
                    f"  {row['Model']:<20s}  →  {row['Feature_Selection']:<12s}  "
                    f"RMSE={rmse_str}   R²={r2_str}"
                )
            lines.append("")
            winner = best_df.iloc[0]
            lines.append(
                f"Overall best regression configuration: "
                f"{winner['Model']} + {winner['Feature_Selection']} "
                f"(RMSE = ${winner['RMSE']:,.0f})"
            )
    lines.append("")

    # ---- Classification ----
    lines.append("-" * 78)
    lines.append("CLASSIFICATION MODEL (SalePrice → Low / Medium / High / Luxury quartiles)")
    lines.append("-" * 78)
    if clf_df.empty:
        lines.append("  No classification report could be parsed. Run classification.py first.")
    else:
        display_cols = [
            "Feature_Selection", "Accuracy", "Precision_weighted",
            "Recall_weighted", "F1_weighted", "CV_mean_accuracy"
        ]
        present = [c for c in display_cols if c in clf_df.columns]
        tbl = clf_df[present].copy()
        for col in present:
            if col != "Feature_Selection":
                tbl[col] = tbl[col].map(lambda x: f"{x:.4f}" if pd.notna(x) else "—")
        lines.append(tbl.to_string(index=False))
        lines.append("")
        if "Accuracy" in clf_df.columns and clf_df["Accuracy"].notna().any():
            best_clf = clf_df.loc[clf_df["Accuracy"].idxmax()]
            lines.append(
                f"Best classification configuration: {best_clf['Feature_Selection']} "
                f"(Test Accuracy = {best_clf['Accuracy']:.4f})"
            )
    lines.append("")

    # ---- Notes ----
    lines.append("-" * 78)
    lines.append("NOTES")
    lines.append("-" * 78)
    lines.append("• Classification and regression metrics are not directly comparable;")
    lines.append("  the classifier predicts a 4-class price band, while the others")
    lines.append("  predict continuous dollar amounts.")
    lines.append("• 1-D CNN reports RMSLE instead of MAPE; MAPE is shown as “—” for it.")
    lines.append("• All models used the same Ames Housing Train.csv and a stratified/")
    lines.append("  random 80/20 split (random_state=42) for fair comparison.")
    lines.append("• Feature selection (SelectKBest vs RFE) is the only controlled")
    lines.append("  difference inside each model family.")
    lines.append("")
    lines.append("=" * 78)

    report_text = "\n".join(lines)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report_text, encoding="utf-8")
    return report_text


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------
def plot_comparison(
    reg_df: pd.DataFrame,
    clf_df: pd.DataFrame,
    best_df: pd.DataFrame,
    out_path: Path,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(16, 12))
    fig.suptitle(
        "Housing Price Models — Cross-Model Comparison",
        fontsize=16, fontweight="bold", y=0.98
    )

    colors = {
        "LinearRegression": "#2E86AB",
        "MLP": "#E67E22",
        "1D_CNN": "#27AE60",
        "SelectKBest": "#3498DB",
        "RFE": "#E74C3C",
    }

    # ---- 1. RMSE by model + feature selection (grouped bar) ----
    ax1 = fig.add_subplot(2, 2, 1)
    if not reg_df.empty and reg_df["RMSE"].notna().any():
        pivot = reg_df.pivot(index="Model", columns="Feature_Selection", values="RMSE")
        pivot.plot(kind="bar", ax=ax1, color=[colors.get(c, "#888") for c in pivot.columns])
        ax1.set_ylabel("RMSE ($)")
        ax1.set_title("RMSE by Model & Feature Selection (lower better)")
        ax1.legend(title="Selector", fontsize=8)
        ax1.tick_params(axis="x", rotation=15)
        ax1.grid(axis="y", alpha=0.3)
        # annotate
        for container in ax1.containers:
            ax1.bar_label(container, fmt="%.0f", fontsize=7, padding=2)
    else:
        ax1.text(0.5, 0.5, "No RMSE data available", ha="center", va="center")
        ax1.set_title("RMSE by Model & Feature Selection")

    # ---- 2. R² by model + feature selection ----
    ax2 = fig.add_subplot(2, 2, 2)
    if not reg_df.empty and reg_df["R2"].notna().any():
        pivot = reg_df.pivot(index="Model", columns="Feature_Selection", values="R2")
        pivot.plot(kind="bar", ax=ax2, color=[colors.get(c, "#888") for c in pivot.columns])
        ax2.set_ylabel("R²")
        ax2.set_ylim(0, 1.05)
        ax2.set_title("R² by Model & Feature Selection (higher better)")
        ax2.legend(title="Selector", fontsize=8)
        ax2.tick_params(axis="x", rotation=15)
        ax2.grid(axis="y", alpha=0.3)
        for container in ax2.containers:
            ax2.bar_label(container, fmt="%.3f", fontsize=7, padding=2)
    else:
        ax2.text(0.5, 0.5, "No R² data available", ha="center", va="center")
        ax2.set_title("R² by Model & Feature Selection")

    # ---- 3. Best configuration ranking (horizontal bar of RMSE) ----
    ax3 = fig.add_subplot(2, 2, 3)
    if not best_df.empty:
        labels = [f"{r['Model']}\n({r['Feature_Selection']})" for _, r in best_df.iterrows()]
        vals = best_df["RMSE"].values
        bar_colors = [colors.get(r["Model"], "#888") for _, r in best_df.iterrows()]
        y_pos = np.arange(len(labels))
        bars = ax3.barh(y_pos, vals, color=bar_colors)
        ax3.set_yticks(y_pos)
        ax3.set_yticklabels(labels, fontsize=9)
        ax3.set_xlabel("RMSE ($)")
        ax3.set_title("Best Configuration per Model (ranked by RMSE)")
        ax3.invert_yaxis()
        ax3.grid(axis="x", alpha=0.3)
        for bar, v in zip(bars, vals):
            ax3.text(v, bar.get_y() + bar.get_height() / 2, f"  ${v:,.0f}",
                     va="center", fontsize=8)
    else:
        ax3.text(0.5, 0.5, "No best-model data", ha="center", va="center")
        ax3.set_title("Best Configuration per Model")

    # ---- 4. Classification metrics ----
    ax4 = fig.add_subplot(2, 2, 4)
    if not clf_df.empty and "Accuracy" in clf_df.columns:
        metrics = ["Accuracy", "Precision_weighted", "Recall_weighted", "F1_weighted"]
        x = np.arange(len(metrics))
        width = 0.35
        for i, (_, row) in enumerate(clf_df.iterrows()):
            vals = [row.get(m, 0) or 0 for m in metrics]
            ax4.bar(x + (i - 0.5) * width, vals, width,
                    label=row["Feature_Selection"],
                    color=colors.get(row["Feature_Selection"], "#888"))
        ax4.set_xticks(x)
        ax4.set_xticklabels(["Accuracy", "Precision", "Recall", "F1"], fontsize=9)
        ax4.set_ylim(0, 1.05)
        ax4.set_title("Classification Metrics (RandomForest)")
        ax4.legend(fontsize=8)
        ax4.grid(axis="y", alpha=0.3)
    else:
        ax4.text(0.5, 0.5, "No classification data available", ha="center", va="center")
        ax4.set_title("Classification Metrics")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Visualization saved → {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(reports_root: Optional[Path], output_dir: Optional[Path]) -> None:
    print("=" * 60)
    print("Loading model reports …")
    print("=" * 60)

    # Auto-discover reports folder if the user did not pass --reports-root
    # or if the passed path does not contain any reports.
    resolved_root = find_reports_root(reports_root)

    if resolved_root is None:
        print("\nNo model reports found.")
        print("Searched the following locations:")
        tried = []
        if reports_root is not None:
            tried.append(Path(reports_root).resolve())
        tried.extend(CANDIDATE_REPORT_ROOTS)
        for p in tried:
            print(f"  - {p.resolve()}")
        print()
        print("Please do one of the following:")
        print("  1. Run the four model scripts first so they write their reports:")
        print("       python regression.py")
        print("       python MLP.py")
        print("       python 1D_CNN.py")
        print("       python classification.py")
        print("  2. Or point this script at the folder that already contains the reports:")
        print("       python all_models_comparison.py --reports-root \"C:\\path\\to\\reports\"")
        print()
        print("Expected files inside the reports folder:")
        for rel in REPORT_PATHS.values():
            print(f"  - {rel}")
        return

    reports = load_all_reports(resolved_root)

    if not reports:
        print("\nReports folder was found, but none of the four report files could be parsed.")
        print(f"Folder: {resolved_root}")
        return

    # Default output next to the reports
    if output_dir is None:
        output_dir = resolved_root / "all_models_comparison"
    output_dir = Path(output_dir)

    reg_df = build_regression_table(reports)
    clf_df = build_classification_table(reports)
    best_df = best_per_model(reg_df)

    # Text report
    report_path = output_dir / "model_comparison_report.txt"
    report_text = write_text_report(reg_df, clf_df, best_df, report_path)
    print("\n" + report_text)
    print(f"\nText report saved → {report_path}")

    # Visualization
    viz_path = output_dir / "model_comparison_visual.png"
    plot_comparison(reg_df, clf_df, best_df, viz_path)

    # Also save the tables as CSV for further analysis
    if not reg_df.empty:
        csv_path = output_dir / "regression_comparison.csv"
        reg_df.to_csv(csv_path, index=False)
        print(f"Regression metrics CSV → {csv_path}")
    if not clf_df.empty:
        csv_path = output_dir / "classification_comparison.csv"
        clf_df.to_csv(csv_path, index=False)
        print(f"Classification metrics CSV → {csv_path}")

    print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare all four housing models (LinearRegression, MLP, 1D-CNN, Classifier)"
    )
    parser.add_argument(
        "--reports-root",
        type=Path,
        default=None,
        help="Folder that contains regression_reports/, mlp_reports/, "
             "1d_cnn_reports/, random_forest_reports/. "
             "If omitted, the script searches common relative locations.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for the comparison report and plots "
             "(default: <reports-root>/all_models_comparison)",
    )
    args = parser.parse_args()
    main(args.reports_root, args.output_dir)
