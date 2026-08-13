"""
Housing Price Prediction - Classical Regression Pipeline
==========================================================
Dataset : Train.csv (Ames Housing / Kaggle House Prices - 1460 rows, 81 columns)
Target  : SalePrice

Pipeline components:
  1. Preprocessing   : sklearn ColumnTransformer
                         - numeric   -> SimpleImputer(median) + StandardScaler
                         - categorical -> SimpleImputer(most_frequent) + OneHotEncoder
  2. Feature selection (TWO algorithms, compared head-to-head):
                         - SelectKBest (f_regression / ANOVA F-test)
                         - RFE (Recursive Feature Elimination) with Linear Regression
  3. Model           : LinearRegression (classical linear regression)
  4. Reports         : model coefficients/summary + performance metrics
                        (R2, Adjusted R2, MAE, RMSE, MAPE) for both approaches,
                        written to a text report and printed to stdout.
"""

import warnings
warnings.filterwarnings("ignore")

import os
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
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error,
)

RANDOM_STATE = 42
N_FEATURES = 25  # number of features each selector will keep

# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------
df = pd.read_csv("../Train.csv")

# Id is just a row identifier, not predictive
df = df.drop(columns=["Id"])

target = "SalePrice"
X = df.drop(columns=[target])
y = df[target]

numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_features = X.select_dtypes(exclude=["int64", "float64"]).columns.tolist()

print(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")
print(f"Numeric features:     {len(numeric_features)}")
print(f"Categorical features: {len(categorical_features)}\n")

# ---------------------------------------------------------------------------
# 2. Train / test split
# ---------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE
)

# ---------------------------------------------------------------------------
# 3. Preprocessing (sklearn ColumnTransformer)
# ---------------------------------------------------------------------------
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore")),
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features),
])

# Fit preprocessor once so both feature-selection branches use identical
# transformed feature matrices (fair comparison).
X_train_proc = preprocessor.fit_transform(X_train)
X_test_proc = preprocessor.transform(X_test)

# Recover feature names after one-hot encoding
cat_feature_names = preprocessor.named_transformers_["cat"]["onehot"].get_feature_names_out(categorical_features)
all_feature_names = np.array(numeric_features + list(cat_feature_names))

if hasattr(X_train_proc, "toarray"):
    X_train_proc = X_train_proc.toarray()
    X_test_proc = X_test_proc.toarray()

print(f"Feature matrix after preprocessing: {X_train_proc.shape[1]} columns\n")

# ---------------------------------------------------------------------------
# 4. Feature Selection Algorithm #1: SelectKBest (f_regression / ANOVA F-test)
# ---------------------------------------------------------------------------
selector_kbest = SelectKBest(score_func=f_regression, k=N_FEATURES)
X_train_kbest = selector_kbest.fit_transform(X_train_proc, y_train)
X_test_kbest = selector_kbest.transform(X_test_proc)
kbest_features = all_feature_names[selector_kbest.get_support()]

model_kbest = LinearRegression()
model_kbest.fit(X_train_kbest, y_train)
pred_kbest = model_kbest.predict(X_test_kbest)

# ---------------------------------------------------------------------------
# 5. Feature Selection Algorithm #2: RFE (Recursive Feature Elimination)
# ---------------------------------------------------------------------------
rfe_estimator = LinearRegression()
selector_rfe = RFE(estimator=rfe_estimator, n_features_to_select=N_FEATURES, step=10)
X_train_rfe = selector_rfe.fit_transform(X_train_proc, y_train)
X_test_rfe = selector_rfe.transform(X_test_proc)
rfe_features = all_feature_names[selector_rfe.get_support()]

model_rfe = LinearRegression()
model_rfe.fit(X_train_rfe, y_train)
pred_rfe = model_rfe.predict(X_test_rfe)

# ---------------------------------------------------------------------------
# 6. Performance metrics helper
# ---------------------------------------------------------------------------
def compute_metrics(y_true, y_pred, n_features, n_samples):
    r2 = r2_score(y_true, y_pred)
    adj_r2 = 1 - (1 - r2) * (n_samples - 1) / (n_samples - n_features - 1)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100
    return {
        "R2": r2,
        "Adjusted R2": adj_r2,
        "MAE": mae,
        "RMSE": rmse,
        "MAPE (%)": mape,
    }

n_test = len(y_test)
metrics_kbest = compute_metrics(y_test, pred_kbest, N_FEATURES, n_test)
metrics_rfe = compute_metrics(y_test, pred_rfe, N_FEATURES, n_test)

# ---------------------------------------------------------------------------
# 7. Build report
# ---------------------------------------------------------------------------
lines = []
lines.append("=" * 70)
lines.append("HOUSING PRICE PREDICTION - MODEL REPORT")
lines.append("=" * 70)
lines.append(f"Dataset: Train.csv | Rows: {df.shape[0]} | Raw features: {df.shape[1]-1}")
lines.append(f"Train/Test split: {len(X_train)} / {len(X_test)} (80/20)")
lines.append(f"Preprocessing: SimpleImputer + StandardScaler (numeric), "
             f"SimpleImputer + OneHotEncoder (categorical) via ColumnTransformer")
lines.append(f"Features kept by each selector: {N_FEATURES}")
lines.append("")

for name, model, features, preds, metrics in [
    ("SelectKBest (f_regression)", model_kbest, kbest_features, pred_kbest, metrics_kbest),
    ("RFE (Recursive Feature Elimination)", model_rfe, rfe_features, pred_rfe, metrics_rfe),
]:
    lines.append("-" * 70)
    lines.append(f"FEATURE SELECTION METHOD: {name}")
    lines.append("-" * 70)
    lines.append(f"Model: Linear Regression")
    lines.append(f"Intercept: {model.intercept_:,.2f}")
    lines.append("")
    lines.append("Top 10 features selected (by |coefficient|):")
    coef_pairs = sorted(zip(features, model.coef_), key=lambda p: abs(p[1]), reverse=True)
    for feat, coef in coef_pairs[:10]:
        lines.append(f"    {feat:<30s} coef = {coef:>12,.2f}")
    lines.append("")
    lines.append("Performance metrics (on held-out test set):")
    for metric_name, value in metrics.items():
        lines.append(f"    {metric_name:<15s}: {value:,.4f}")
    lines.append("")

lines.append("=" * 70)
lines.append("SUMMARY COMPARISON")
lines.append("=" * 70)
lines.append(f"{'Metric':<15s}{'SelectKBest':>18s}{'RFE':>18s}")
for metric_name in metrics_kbest:
    lines.append(f"{metric_name:<15s}{metrics_kbest[metric_name]:>18,.4f}{metrics_rfe[metric_name]:>18,.4f}")

winner = "SelectKBest" if metrics_kbest["R2"] >= metrics_rfe["R2"] else "RFE"
lines.append("")
lines.append(f"Best performing feature selection method (by R2): {winner}")
lines.append("=" * 70)

report = "\n".join(lines)
print(report)

os.makedirs("../reports", exist_ok=True)
with open("../reports/regression_reports/regression_report.txt", "w") as f:
    f.write(report)

print("\nReport saved to reports/regression_reports/regression_report.txt")

# ---------------------------------------------------------------------------
# 8. Visualization: model outputs + performance metrics dashboard
# ---------------------------------------------------------------------------
plt.rcParams.update({"font.size": 9})
fig = plt.figure(figsize=(16, 14))
fig.suptitle("Housing Price Prediction \u2014 Model Report", fontsize=16, fontweight="bold", y=0.995)

results = [
    ("SelectKBest (f_regression)", "tab:blue", model_kbest, kbest_features, pred_kbest, metrics_kbest),
    ("RFE", "tab:orange", model_rfe, rfe_features, pred_rfe, metrics_rfe),
]

# --- Row 1: Actual vs Predicted scatter for each method ---
for i, (name, color, model, feats, preds, metrics) in enumerate(results):
    ax = fig.add_subplot(4, 2, i + 1)
    ax.scatter(y_test, preds, alpha=0.5, s=18, color=color, edgecolor="k", linewidth=0.2)
    lims = [min(y_test.min(), preds.min()), max(y_test.max(), preds.max())]
    ax.plot(lims, lims, "r--", linewidth=1, label="Perfect prediction")
    ax.set_xlabel("Actual SalePrice")
    ax.set_ylabel("Predicted SalePrice")
    ax.set_title(f"{name}\nActual vs. Predicted (R\u00b2 = {metrics['R2']:.3f})")
    ax.legend(fontsize=7)

# --- Row 2: Residual plots for each method ---
for i, (name, color, model, feats, preds, metrics) in enumerate(results):
    ax = fig.add_subplot(4, 2, i + 3)
    residuals = y_test.values - preds
    ax.scatter(preds, residuals, alpha=0.5, s=18, color=color, edgecolor="k", linewidth=0.2)
    ax.axhline(0, color="r", linestyle="--", linewidth=1)
    ax.set_xlabel("Predicted SalePrice")
    ax.set_ylabel("Residual (Actual - Predicted)")
    ax.set_title(f"{name}\nResiduals")

# --- Row 3: Top 10 feature coefficients for each method ---
for i, (name, color, model, feats, preds, metrics) in enumerate(results):
    ax = fig.add_subplot(4, 2, i + 5)
    coef_pairs = sorted(zip(feats, model.coef_), key=lambda p: abs(p[1]), reverse=True)[:10]
    labels = [p[0] for p in coef_pairs][::-1]
    values = [p[1] for p in coef_pairs][::-1]
    bar_colors = ["tab:green" if v >= 0 else "tab:red" for v in values]
    ax.barh(labels, values, color=bar_colors)
    ax.set_title(f"{name}\nTop 10 Coefficients")
    ax.set_xlabel("Coefficient value")
    ax.tick_params(axis="y", labelsize=7)

# --- Row 4, col 1: Grouped bar chart comparing metrics ---
ax_metrics = fig.add_subplot(4, 2, 7)
metric_names = list(metrics_kbest.keys())
x = np.arange(len(metric_names))
width = 0.35
# Normalize each metric's pair to [0,1] so different scales are readable together
kbest_vals = np.array([metrics_kbest[m] for m in metric_names], dtype=float)
rfe_vals = np.array([metrics_rfe[m] for m in metric_names], dtype=float)
ax_metrics.bar(x - width / 2, kbest_vals, width, label="SelectKBest", color="tab:blue")
ax_metrics.bar(x + width / 2, rfe_vals, width, label="RFE", color="tab:orange")
ax_metrics.set_xticks(x)
ax_metrics.set_xticklabels(metric_names, rotation=20, ha="right")
ax_metrics.set_title("Metric Comparison (raw values)")
ax_metrics.legend(fontsize=8)
for xi, (kv, rv) in enumerate(zip(kbest_vals, rfe_vals)):
    ax_metrics.text(xi - width / 2, kv, f"{kv:,.2f}", ha="center", va="bottom", fontsize=6, rotation=90)
    ax_metrics.text(xi + width / 2, rv, f"{rv:,.2f}", ha="center", va="bottom", fontsize=6, rotation=90)

# --- Row 4, col 2: R2 / Adjusted R2 head-to-head bar ---
ax_r2 = fig.add_subplot(4, 2, 8)
r2_metrics = ["R2", "Adjusted R2"]
x2 = np.arange(len(r2_metrics))
kbest_r2 = [metrics_kbest[m] for m in r2_metrics]
rfe_r2 = [metrics_rfe[m] for m in r2_metrics]
bars1 = ax_r2.bar(x2 - width / 2, kbest_r2, width, label="SelectKBest", color="tab:blue")
bars2 = ax_r2.bar(x2 + width / 2, rfe_r2, width, label="RFE", color="tab:orange")
ax_r2.set_xticks(x2)
ax_r2.set_xticklabels(r2_metrics)
ax_r2.set_ylim(0, 1)
ax_r2.set_title(f"R\u00b2 Comparison \u2014 Winner: {winner}")
ax_r2.legend(fontsize=8)
ax_r2.bar_label(bars1, fmt="%.3f", fontsize=7)
ax_r2.bar_label(bars2, fmt="%.3f", fontsize=7)

plt.tight_layout(rect=[0, 0, 1, 0.98])
viz_path = "../reports/regression_reports/regression_visual_report.png"
plt.savefig(viz_path, dpi=150, bbox_inches="tight")
plt.close(fig)

print(f"Visualization dashboard saved to {viz_path}")