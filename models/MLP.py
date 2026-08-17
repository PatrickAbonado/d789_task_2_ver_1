"""
Deep Learning Housing Price Prediction Pipeline
=================================================
Predicts Ames Housing sale prices using a neural network (MLPRegressor)
built on top of a scikit-learn preprocessing pipeline, with two competing
feature selection strategies evaluated head-to-head.

Architecture
------------
1. Preprocessing (sklearn ColumnTransformer)
   - Numeric features:     median imputation -> StandardScaler
   - Categorical features: most-frequent imputation -> OneHotEncoder

2. Feature Selection (two algorithms, compared head-to-head)
   - SelectKBest   : univariate ANOVA F-test (f_regression), top 25 features
   - RFE           : Recursive Feature Elimination using a Ridge estimator
                      as the ranking model, top 25 features

3. Model (Deep Learning)
   - MLPRegressor: a multi-layer, fully-connected feed-forward neural
     network (the sklearn-native deep learning estimator), trained
     separately on each feature-selected representation.

4. Evaluation
   - Train.csv is the only labeled data available, so metrics are computed
     on a held-out validation split (80/20) taken from Train.csv.
   - If a Kaggle-style unlabeled Test.csv is present alongside Train.csv,
     final models (refit on 100% of Train.csv) are used to generate
     predictions for it.

Outputs
-------
   - housing_dl_model_report.txt : full metrics + architecture report
   - housing_dl_predictions.csv  : predictions for Test.csv (if present)
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_selection import SelectKBest, f_regression, RFE
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error,
)

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
TRAIN_PATH = "../Train.csv"
TEST_PATH = "../Test.csv"  # optional — only used if present
TARGET = "SalePrice"
ID_COL = "Id"
N_FEATURES = 25                 # features kept by each selector
RANDOM_STATE = 42
VALIDATION_SIZE = 0.20

HIDDEN_LAYERS = (128, 64, 32)   # deep MLP architecture (3 hidden layers)
MLP_PARAMS = dict(
    hidden_layer_sizes=HIDDEN_LAYERS,
    activation="relu",
    solver="adam",
    alpha=1e-3,                 # L2 regularization
    learning_rate_init=1e-3,
    max_iter=2000,
    early_stopping=True,
    n_iter_no_change=25,
    validation_fraction=0.1,
    random_state=RANDOM_STATE,
)

REPORT_PATH = "../reports/mlp_reports/MLP_report.txt"
PREDICTIONS_PATH = "../reports/mlp_reports/MLP_predictions.csv"
VISUALIZATION_PATH = "../reports/mlp_reports/MLP_visual_report.png"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()

    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        # drop="first" removes one reference level per categorical column so
        # the one-hot levels (plus the model's intercept/bias) can never sum
        # to a constant -- this is the "dummy variable trap" that produced
        # coefficients in the quintillions when RFE happened to retain most
        # levels of one column in the linear-regression pipeline.
        #
        # min_frequency=10 buckets any category level appearing in fewer than
        # 10 training rows into a single "infrequent" column instead of its
        # own dummy. RFE kept latching onto rare levels like RoofMatl_ClyTile,
        # RoofMatl_WdShake, Exterior2nd_ImStucc, Condition2_PosA, PoolQC_Gd --
        # levels with only a handful of examples, which is a classic overfit-
        # to-noise pattern (a rare level can look highly predictive on a tiny
        # sample by chance). Bucketing removes that opportunity for both
        # feature selectors while leaving common categories untouched.
        ("onehot", OneHotEncoder(handle_unknown="ignore", drop="first", min_frequency=10)),
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_pipeline, numeric_cols),
        ("cat", categorical_pipeline, categorical_cols),
    ])
    return preprocessor, numeric_cols, categorical_cols


def get_all_feature_names(prep, numeric_cols, categorical_cols):
    """Full post-encoding feature name list, before any selector is applied."""
    ohe = prep.named_transformers_["cat"].named_steps["onehot"]
    cat_feature_names = ohe.get_feature_names_out(categorical_cols).tolist()
    return numeric_cols + cat_feature_names


def to_dense(X):
    return X.toarray() if hasattr(X, "toarray") else X


def remove_collinear_features(X_train_proc, X_val_proc, feature_names, threshold=0.95):
    """Drop one feature from any pair correlated above `threshold` (computed on
    training data only). Near-duplicate numeric features -- e.g. GrLivArea is
    almost exactly 1stFlrSF + 2ndFlrSF + LowQualFinSF in this dataset -- can
    make linear estimators (SelectKBest's F-test denominator, RFE's Ridge
    ranking) unstable or redundant. Applying this before feature selection
    keeps both branches on the same de-duplicated, numerically stable footing.
    """
    feature_names = np.array(feature_names)
    corr_matrix = pd.DataFrame(X_train_proc, columns=feature_names).corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > threshold)]

    if to_drop:
        print(f"  Dropping {len(to_drop)} highly-collinear feature(s) (|r| > {threshold}):")
        for col in to_drop:
            partner = upper.index[upper[col] > threshold][0]
            r = corr_matrix.loc[partner, col]
            print(f"    {col:<30s} (r = {r:.3f} with {partner})")

    keep_mask = ~np.isin(feature_names, to_drop)
    return (
        X_train_proc[:, keep_mask],
        X_val_proc[:, keep_mask],
        feature_names[keep_mask].tolist(),
    )


def get_selected_feature_names(selector, feature_names):
    """Recover human-readable feature names for whichever columns a selector kept,
    given the (already collinearity-filtered) feature name list it was fit on."""
    mask = selector.get_support()
    return [name for name, keep in zip(feature_names, mask) if keep]


def evaluate(y_true, y_pred, n_features):
    n = len(y_true)
    r2 = r2_score(y_true, y_pred)
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - n_features - 1)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100
    return {
        "R2": r2,
        "Adjusted_R2": adj_r2,
        "MAE": mae,
        "RMSE": rmse,
        "MAPE_%": mape,
    }


def run_pipeline(selector_name, selector, X_train_raw, y_train, X_val_raw, y_val,
                  preprocessor, numeric_cols, categorical_cols):
    """Fit preprocessing + collinearity filter + feature selection + MLP on train,
    evaluate on validation."""
    # Fit preprocessor + selector on training data only (no leakage)
    prep = ColumnTransformer(transformers=preprocessor.transformers)
    X_train_proc = to_dense(prep.fit_transform(X_train_raw))
    X_val_proc = to_dense(prep.transform(X_val_raw))
    all_feature_names = get_all_feature_names(prep, numeric_cols, categorical_cols)

    print(f"  Feature matrix before collinearity filtering: {X_train_proc.shape[1]} columns")
    X_train_proc, X_val_proc, all_feature_names = remove_collinear_features(
        X_train_proc, X_val_proc, all_feature_names
    )
    print(f"  Feature matrix after collinearity filtering: {X_train_proc.shape[1]} columns")

    selector.fit(X_train_proc, y_train)
    X_train_sel = selector.transform(X_train_proc)
    X_val_sel = selector.transform(X_val_proc)

    feature_names = get_selected_feature_names(selector, all_feature_names)

    model = MLPRegressor(**MLP_PARAMS)
    model.fit(X_train_sel, y_train)

    val_pred = model.predict(X_val_sel)
    metrics = evaluate(y_val, val_pred, X_train_sel.shape[1])

    return {
        "name": selector_name,
        "preprocessor": prep,
        "selector": selector,
        "model": model,
        "feature_names": feature_names,
        "metrics": metrics,
        "n_iter": getattr(model, "n_iter_", None),
        "final_loss": getattr(model, "loss_", None),
        "loss_curve": getattr(model, "loss_curve_", None),
        "y_val": y_val,
        "val_pred": val_pred,
    }


def plot_results(results, save_path):
    """Build a single dashboard figure covering training curves, predicted-vs-actual,
    residuals, and a metrics comparison bar chart for both feature-selection pipelines."""
    fig, axes = plt.subplots(3, 2, figsize=(14, 16))
    colors = ["#2E86AB", "#E67E22"]

    # Row 1: MLP training loss curves
    for i, r in enumerate(results):
        ax = axes[0, i]
        if r["loss_curve"]:
            ax.plot(r["loss_curve"], color=colors[i])
        ax.set_title(f"Training Loss Curve\n{r['name']}")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Loss (MSE)")
        ax.grid(alpha=0.3)

    # Row 2: Predicted vs Actual scatter
    for i, r in enumerate(results):
        ax = axes[1, i]
        y_val, val_pred = r["y_val"], r["val_pred"]
        ax.scatter(y_val, val_pred, alpha=0.5, s=18, color=colors[i], edgecolor="none")
        lims = [min(y_val.min(), val_pred.min()), max(y_val.max(), val_pred.max())]
        ax.plot(lims, lims, "k--", linewidth=1, label="Perfect prediction")
        ax.set_title(f"Predicted vs Actual\n{r['name']}  (R\u00b2={r['metrics']['R2']:.3f})")
        ax.set_xlabel("Actual SalePrice")
        ax.set_ylabel("Predicted SalePrice")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    # Row 3, col 1: Residual distribution (both models overlaid)
    ax = axes[2, 0]
    for i, r in enumerate(results):
        residuals = r["y_val"] - r["val_pred"]
        ax.hist(residuals, bins=30, alpha=0.55, color=colors[i], label=r["name"])
    ax.axvline(0, color="k", linestyle="--", linewidth=1)
    ax.set_title("Residual Distribution (Actual - Predicted)")
    ax.set_xlabel("Residual ($)")
    ax.set_ylabel("Count")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # Row 3, col 2: Metrics comparison bar chart (normalized where needed)
    ax = axes[2, 1]
    metric_labels = ["R2", "Adjusted_R2", "MAPE_%"]
    display_labels = ["R\u00b2", "Adj. R\u00b2", "MAPE (%) /100"]
    x = np.arange(len(metric_labels))
    width = 0.35
    for i, r in enumerate(results):
        vals = []
        for m in metric_labels:
            v = r["metrics"][m]
            vals.append(v / 100 if m == "MAPE_%" else v)  # scale MAPE to fit 0-1 axis
        ax.bar(x + (i - 0.5) * width, vals, width, label=r["name"], color=colors[i])
    ax.set_xticks(x)
    ax.set_xticklabels(display_labels)
    ax.set_title("Performance Metrics Comparison")
    ax.set_ylabel("Score")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")

    fig.suptitle("Deep Learning Housing Price Model — Output & Performance Dashboard",
                  fontsize=15, fontweight="bold", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print(f"Loading {TRAIN_PATH} ...")
    df = pd.read_csv(TRAIN_PATH)
    df = df.drop(columns=[ID_COL]) if ID_COL in df.columns else df

    X = df.drop(columns=[TARGET])
    y = df[TARGET].values

    X_train_raw, X_val_raw, y_train, y_val = train_test_split(
        X, y, test_size=VALIDATION_SIZE, random_state=RANDOM_STATE
    )

    preprocessor, numeric_cols, categorical_cols = build_preprocessor(X)

    print("Numeric features:", len(numeric_cols))
    print("Categorical features:", len(categorical_cols))

    # -------------------- Feature Selector 1: SelectKBest --------------------
    print("\nTraining MLP with SelectKBest (ANOVA F-test) feature selection ...")
    kbest_selector = SelectKBest(score_func=f_regression, k=N_FEATURES)
    kbest_result = run_pipeline(
        "SelectKBest (f_regression)", kbest_selector,
        X_train_raw, y_train, X_val_raw, y_val,
        preprocessor, numeric_cols, categorical_cols,
    )

    # -------------------- Feature Selector 2: RFE ------------------------
    print("Training MLP with RFE (Ridge-ranked) feature selection ...")
    rfe_selector = RFE(estimator=Ridge(alpha=1.0), n_features_to_select=N_FEATURES)
    rfe_result = run_pipeline(
        "RFE (Ridge estimator)", rfe_selector,
        X_train_raw, y_train, X_val_raw, y_val,
        preprocessor, numeric_cols, categorical_cols,
    )

    results = [kbest_result, rfe_result]
    best = max(results, key=lambda r: r["metrics"]["R2"])

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------
    lines = []
    lines.append("=" * 72)
    lines.append("DEEP LEARNING HOUSING PRICE PREDICTION — MODEL REPORT")
    lines.append("=" * 72)
    lines.append(f"Training rows used     : {len(X_train_raw)}")
    lines.append(f"Validation rows used   : {len(X_val_raw)}  ({VALIDATION_SIZE:.0%} holdout)")
    lines.append(f"Target                 : {TARGET}")
    lines.append(f"Features kept per model: {N_FEATURES}")
    lines.append(f"Model architecture     : MLPRegressor, hidden layers {HIDDEN_LAYERS}, "
                  f"activation=relu, solver=adam, alpha={MLP_PARAMS['alpha']}")
    lines.append("")

    for r in results:
        m = r["metrics"]
        lines.append("-" * 72)
        lines.append(f"Feature Selection Method: {r['name']}")
        lines.append("-" * 72)
        lines.append(f"  Training iterations run : {r['n_iter']}")
        lines.append(f"  Final training loss     : {r['final_loss']:.4f}" if r["final_loss"] is not None else "")
        lines.append("  Validation performance:")
        lines.append(f"    R-squared            : {m['R2']:.4f}")
        lines.append(f"    Adjusted R-squared    : {m['Adjusted_R2']:.4f}")
        lines.append(f"    MAE                   : {m['MAE']:,.2f}")
        lines.append(f"    RMSE                  : {m['RMSE']:,.2f}")
        lines.append(f"    MAPE                  : {m['MAPE_%']:.2f}%")
        lines.append(f"  Selected features ({len(r['feature_names'])}):")
        for i in range(0, len(r["feature_names"]), 5):
            lines.append("    " + ", ".join(r["feature_names"][i:i + 5]))
        lines.append("")

    lines.append("=" * 72)
    lines.append("HEAD-TO-HEAD COMPARISON")
    lines.append("=" * 72)
    header = f"{'Metric':<20}{results[0]['name']:<28}{results[1]['name']:<28}"
    lines.append(header)
    for metric_key, label in [
        ("R2", "R-squared"), ("Adjusted_R2", "Adjusted R-squared"),
        ("MAE", "MAE"), ("RMSE", "RMSE"), ("MAPE_%", "MAPE (%)"),
    ]:
        row = f"{label:<20}"
        for r in results:
            val = r["metrics"][metric_key]
            row += f"{val:<28.4f}" if metric_key in ("R2", "Adjusted_R2") else f"{val:<28,.2f}"
        lines.append(row)
    lines.append("")
    lines.append(f"Best performing approach: {best['name']}  (R-squared = {best['metrics']['R2']:.4f})")
    lines.append("")

    report_text = "\n".join(lines)
    with open(REPORT_PATH, "w") as f:
        f.write(report_text)
    print(f"\nReport written to {REPORT_PATH}")
    print(report_text)

    # ------------------------------------------------------------------
    # Visualizations: training curves, predicted-vs-actual, residuals,
    # and a metrics comparison chart for both feature-selection pipelines
    # ------------------------------------------------------------------
    print(f"\nGenerating visualization dashboard ...")
    plot_results(results, VISUALIZATION_PATH)
    print(f"Visualizations written to {VISUALIZATION_PATH}")

    # ------------------------------------------------------------------
    # Refit best pipelines on 100% of Train.csv and predict on Test.csv
    # (only if an unlabeled Test.csv is available)
    # ------------------------------------------------------------------
    if os.path.exists(TEST_PATH):
        print(f"\n{TEST_PATH} found — generating holdout predictions ...")
        test_df = pd.read_csv(TEST_PATH)
        test_ids = test_df[ID_COL] if ID_COL in test_df.columns else pd.Series(range(len(test_df)))
        X_test_raw = test_df.drop(columns=[ID_COL]) if ID_COL in test_df.columns else test_df

        preds = {"Id": test_ids}
        for selector_name, selector_template in [
            ("SelectKBest", SelectKBest(score_func=f_regression, k=N_FEATURES)),
            ("RFE", RFE(estimator=Ridge(alpha=1.0), n_features_to_select=N_FEATURES)),
        ]:
            prep = ColumnTransformer(transformers=preprocessor.transformers)
            X_full_proc = to_dense(prep.fit_transform(X))
            X_test_proc = to_dense(prep.transform(X_test_raw))
            full_feature_names = get_all_feature_names(prep, numeric_cols, categorical_cols)

            X_full_proc, X_test_proc, full_feature_names = remove_collinear_features(
                X_full_proc, X_test_proc, full_feature_names
            )

            selector_template.fit(X_full_proc, y)
            X_full_sel = selector_template.transform(X_full_proc)
            X_test_sel = selector_template.transform(X_test_proc)

            final_model = MLPRegressor(**MLP_PARAMS)
            final_model.fit(X_full_sel, y)
            preds[f"SalePrice_{selector_name}"] = final_model.predict(X_test_sel)

        pred_df = pd.DataFrame(preds)
        pred_df.to_csv(PREDICTIONS_PATH, index=False)
        print(f"Predictions written to {PREDICTIONS_PATH}")
    else:
        print(f"\nNo {TEST_PATH} found next to {TRAIN_PATH} — skipping holdout predictions. "
              "(Validation metrics above are the primary performance report.)")


if __name__ == "__main__":
    main()
