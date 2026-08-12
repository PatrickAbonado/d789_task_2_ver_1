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

import numpy as np
import pandas as pd

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
df = pd.read_csv("Train.csv")

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

with open("reports/regression_report.txt", "w") as f:
    f.write(report)

print("\nReport saved to regression_report.txt")
