"""
Housing Price Prediction - Classical Regression Pipeline
==========================================================
Training data : Train.csv (Ames Housing / Kaggle House Prices - 1460 rows, 81 columns, has SalePrice)
Inference data: Test.csv  (1459 rows, 80 columns, NO SalePrice - unlabeled holdout set)

Since Test.csv has no ground-truth SalePrice, performance metrics (R2, MAE, etc.)
cannot be computed on it. Strategy:
  1. Split Train.csv into train/validation -> fit pipelines, evaluate + report
     performance metrics on the validation split (has ground truth).
  2. Refit the final pipelines on the FULL Train.csv.
  3. Use the refit pipelines to predict SalePrice for every row in Test.csv and
     write those predictions to a CSV (no metrics possible here - no labels).

Pipeline components:
  1. Preprocessing   : sklearn ColumnTransformer
                         - numeric   -> SimpleImputer(median) + StandardScaler
                         - categorical -> SimpleImputer(most_frequent) + OneHotEncoder
  2. Feature selection (TWO algorithms, compared head-to-head):
                         - SelectKBest (f_regression / ANOVA F-test)
                         - RFE (Recursive Feature Elimination) with Linear Regression
  3. Model           : LinearRegression (classical linear regression)
  4. Reports         : model coefficients/summary + performance metrics
                        (R2, Adjusted R2, MAE, RMSE, MAPE) on the validation split
                        for both approaches, written to a text report and printed
                        to stdout. Final Test.csv predictions saved to a CSV.
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
train_df = pd.read_csv("Train.csv")
test_df = pd.read_csv("Test.csv")  # unlabeled: no SalePrice column

target = "SalePrice"

# Keep Test.csv's Id column for the submission file, then drop Id from both
# (Id is just a row identifier, not predictive)
test_ids = test_df["Id"].copy()
train_df = train_df.drop(columns=["Id"])
test_df = test_df.drop(columns=["Id"])

X = train_df.drop(columns=[target])
y = train_df[target]
X_holdout = test_df  # Test.csv features, no labels available

numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_features = X.select_dtypes(exclude=["int64", "float64"]).columns.tolist()

print(f"Loaded training data: {train_df.shape[0]} rows, {train_df.shape[1]} columns (incl. SalePrice)")
print(f"Loaded Test.csv (unlabeled): {X_holdout.shape[0]} rows, {X_holdout.shape[1]} columns")
print(f"Numeric features:     {len(numeric_features)}")
print(f"Categorical features: {len(categorical_features)}\n")

# ---------------------------------------------------------------------------
# 2. Train / validation split (from Train.csv, since Test.csv has no labels
#    to score against -- validation metrics are computed here instead)
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

# Fit preprocessor once (on the train split) so both feature-selection
# branches use identical transformed feature matrices (fair comparison).
X_train_proc = preprocessor.fit_transform(X_train)
X_test_proc = preprocessor.transform(X_test)

# Recover feature names after one-hot encoding
cat_feature_names = preprocessor.named_transformers_["cat"]["onehot"].get_feature_names_out(categorical_features)
all_feature_names = np.array(numeric_features + list(cat_feature_names))

if hasattr(X_train_proc, "toarray"):
    X_train_proc = X_train_proc.toarray()
    X_test_proc = X_test_proc.toarray()

print(f"Feature matrix after preprocessing: {X_train_proc.shape[1]} columns\n")

# A second preprocessor fit on the FULL Train.csv (all 1460 rows) -- used
# later to transform Test.csv for the final predictions.
preprocessor_full = ColumnTransformer(transformers=[
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features),
])
X_full_proc = preprocessor_full.fit_transform(X)
X_holdout_proc = preprocessor_full.transform(X_holdout)
if hasattr(X_full_proc, "toarray"):
    X_full_proc = X_full_proc.toarray()
    X_holdout_proc = X_holdout_proc.toarray()

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
# 5b. Refit the winning approach on the FULL Train.csv, then predict on
#     Test.csv (the actual, unlabeled holdout set).
# ---------------------------------------------------------------------------
selector_kbest_full = SelectKBest(score_func=f_regression, k=N_FEATURES)
X_full_kbest = selector_kbest_full.fit_transform(X_full_proc, y)
X_holdout_kbest = selector_kbest_full.transform(X_holdout_proc)

selector_rfe_full = RFE(estimator=LinearRegression(), n_features_to_select=N_FEATURES, step=10)
X_full_rfe = selector_rfe_full.fit_transform(X_full_proc, y)
X_holdout_rfe = selector_rfe_full.transform(X_holdout_proc)

model_kbest_full = LinearRegression().fit(X_full_kbest, y)
model_rfe_full = LinearRegression().fit(X_full_rfe, y)

test_pred_kbest = model_kbest_full.predict(X_holdout_kbest)
test_pred_rfe = model_rfe_full.predict(X_holdout_rfe)

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
lines.append(f"Training source: Train.csv | Rows: {train_df.shape[0]} | Raw features: {train_df.shape[1]-1}")
lines.append(f"Inference source: Test.csv | Rows: {X_holdout.shape[0]} (unlabeled - no SalePrice)")
lines.append(f"Train/Validation split (from Train.csv): {len(X_train)} / {len(X_test)} (80/20)")
lines.append("NOTE: Test.csv has no ground-truth SalePrice, so performance metrics below")
lines.append("      are computed on the held-out VALIDATION split of Train.csv. The final")
lines.append("      models are then refit on all of Train.csv and used to predict Test.csv.")
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
    lines.append("Performance metrics (on held-out VALIDATION split of Train.csv):")
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
lines.append(f"Best performing feature selection method (by validation R2): {winner}")
lines.append("")
lines.append("=" * 70)
lines.append("TEST.CSV PREDICTIONS")
lines.append("=" * 70)
lines.append(f"Final models refit on the FULL Train.csv ({len(X)} rows), then used to")
lines.append(f"predict SalePrice for all {len(X_holdout)} rows of Test.csv.")
lines.append("No performance metrics are reportable here since Test.csv has no labels.")
lines.append(f"Predictions saved to: housing_predictions.csv "
             f"(columns: Id, SalePrice_SelectKBest, SalePrice_RFE)")
lines.append("=" * 70)

report = "\n".join(lines)
print(report)

with open("housing_model_report.txt", "w") as f:
    f.write(report)

print("\nReport saved to housing_model_report.txt")

# ---------------------------------------------------------------------------
# 8. Save Test.csv predictions
# ---------------------------------------------------------------------------
predictions_df = pd.DataFrame({
    "Id": test_ids,
    "SalePrice_SelectKBest": test_pred_kbest,
    "SalePrice_RFE": test_pred_rfe,
})
predictions_df.to_csv("housing_predictions.csv", index=False)
print("Test.csv predictions saved to housing_predictions.csv")
