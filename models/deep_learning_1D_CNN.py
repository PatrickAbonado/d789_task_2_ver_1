"""
Deep Learning House Price Prediction Pipeline
===============================================
Dataset : Ames Housing (Kaggle House Prices format) - Train.csv
Model   : 1-D Convolutional Neural Network (CNN) — NOT a plain MLP
Library : PyTorch (model) + scikit-learn (preprocessing / feature selection)

Pipeline stages
----------------
1. Load data, split target (SalePrice) and features, hold out a labeled
   validation split (Test.csv in Kaggle-style datasets is unlabeled, so all
   metric reporting is done on this validation split, per established
   practice on this project).
2. Preprocessing (scikit-learn ColumnTransformer):
     - numeric features  -> median imputation + StandardScaler
     - categorical features -> most-frequent imputation + OneHotEncoder
3. Feature selection (two algorithms, compared head-to-head):
     - SelectKBest (ANOVA F-test / f_regression)  -> filter method
     - RFE (Recursive Feature Elimination, Ridge estimator) -> wrapper method
4. Deep learning model: a 1-D CNN that treats the selected, scaled/encoded
   feature vector as a 1-D "signal" and learns local feature interactions
   via convolutional filters, followed by global pooling and a single
   linear read-out (this read-out is NOT a hidden MLP block - there is no
   stack of fully-connected hidden layers, which is what would make this
   an MLP).
5. Train one CNN per feature-selection method, evaluate both, and report
   which feature selection method produced the stronger downstream model.
6. Generate a text report and visualizations (training curves, predicted
   vs actual, residuals, feature-selection comparison, metric comparison).
7. If an unlabeled Test.csv is present alongside Train.csv, generate a
   predictions CSV using the best of the two trained CNNs, refit on the
   full labeled Train.csv.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_selection import SelectKBest, f_regression, RFE
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
RANDOM_STATE = 42
TRAIN_PATH = "../Train.csv"
TEST_PATH = "../Test.csv"          # optional, unlabeled
OUTPUT_DIR = "../reports/1d_cnn_reports"
K_FEATURES = 40                                         # features kept by each selector
VAL_SIZE = 0.2
BATCH_SIZE = 32
MAX_EPOCHS = 300
PATIENCE = 20                                            # early stopping
LEARNING_RATE = 1e-3
TARGET = "SalePrice"

np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ----------------------------------------------------------------------
# 1. Load data
# ----------------------------------------------------------------------
# The Ames Housing CSV uses the literal string "NA" for missing values, so
# it must be parsed explicitly rather than relying on pandas defaults.
df = pd.read_csv(TRAIN_PATH, na_values=["NA"], keep_default_na=True)

y = df[TARGET].astype(float)
X = df.drop(columns=[TARGET, "Id"])

numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

print(f"Loaded {X.shape[0]} rows, {X.shape[1]} raw features "
      f"({len(numeric_cols)} numeric, {len(categorical_cols)} categorical).")

# Train / validation split (Test.csv, if present, is unlabeled and is only
# used at the very end for generating predictions with the final model)
X_train_raw, X_val_raw, y_train, y_val = train_test_split(
    X, y, test_size=VAL_SIZE, random_state=RANDOM_STATE
)

# Log-transform the (right-skewed) target. The CNN is trained on this scale;
# predictions are exponentiated back before computing dollar-scale metrics.
y_train_log = np.log1p(y_train.values)
y_val_log = np.log1p(y_val.values)

# ----------------------------------------------------------------------
# 2. Preprocessing (scikit-learn)
# ----------------------------------------------------------------------
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore")),
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, numeric_cols),
    ("cat", categorical_transformer, categorical_cols),
])

X_train_proc = preprocessor.fit_transform(X_train_raw)
X_val_proc = preprocessor.transform(X_val_raw)

if hasattr(X_train_proc, "toarray"):
    X_train_proc = X_train_proc.toarray()
    X_val_proc = X_val_proc.toarray()

feature_names = preprocessor.get_feature_names_out()
print(f"Preprocessed feature matrix: {X_train_proc.shape[1]} columns "
      f"(after imputation, scaling, one-hot encoding).")

# ----------------------------------------------------------------------
# 3. Feature selection — two algorithms, compared head-to-head
# ----------------------------------------------------------------------
k = min(K_FEATURES, X_train_proc.shape[1])

# 3a. SelectKBest (ANOVA F-test / f_regression) — fast filter method
skb = SelectKBest(score_func=f_regression, k=k)
skb.fit(X_train_proc, y_train_log)
skb_mask = skb.get_support()
skb_scores = skb.scores_
skb_features = feature_names[skb_mask]

# 3b. RFE (Recursive Feature Elimination) — wrapper method, Ridge estimator
#     (Ridge is used instead of plain LinearRegression for stability, since
#     with hundreds of one-hot columns and only ~1200 training rows plain
#     OLS can be poorly conditioned.)
rfe_estimator = Ridge(alpha=10.0, random_state=RANDOM_STATE)
rfe = RFE(estimator=rfe_estimator, n_features_to_select=k, step=15)
rfe.fit(X_train_proc, y_train_log)
rfe_mask = rfe.get_support()
rfe_ranking = rfe.ranking_
rfe_features = feature_names[rfe_mask]

overlap = set(skb_features) & set(rfe_features)
print(f"SelectKBest chose {len(skb_features)} features; "
      f"RFE chose {len(rfe_features)} features; "
      f"{len(overlap)} features chosen by both.")

feature_sets = {
    "SelectKBest": skb_mask,
    "RFE": rfe_mask,
}

# ----------------------------------------------------------------------
# 4. 1-D CNN model (PyTorch) — not an MLP
# ----------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class HousePriceCNN(nn.Module):
    """
    1-D convolutional regressor.

    The selected feature vector (length = n_features) is treated as a
    single-channel 1-D signal. Stacked Conv1d blocks learn local patterns
    across neighbouring feature positions, batch norm stabilizes training,
    and adaptive average pooling collapses the learned feature maps before
    a single linear layer produces the regression output. There is no
    stack of fully-connected hidden layers (i.e. no MLP block) — the
    representation is learned entirely through convolution.
    """

    def __init__(self, n_features: int):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),   # global average pooling
        )
        self.dropout = nn.Dropout(0.2)
        self.readout = nn.Linear(128, 1)   # single linear read-out, not an MLP

    def forward(self, x):
        # x: (batch, n_features) -> (batch, 1, n_features)
        x = x.unsqueeze(1)
        x = self.conv_block(x)            # (batch, 128, 1)
        x = x.squeeze(-1)                 # (batch, 128)
        x = self.dropout(x)
        return self.readout(x).squeeze(-1)


def make_loader(X_arr, y_arr, batch_size, shuffle):
    X_t = torch.tensor(X_arr, dtype=torch.float32)
    y_t = torch.tensor(y_arr, dtype=torch.float32)
    ds = TensorDataset(X_t, y_t)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def train_cnn(X_tr, y_tr, X_va, y_va, tag):
    """Train one CNN and return the trained model + training history."""
    n_features = X_tr.shape[1]
    model = HousePriceCNN(n_features).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=8
    )
    loss_fn = nn.MSELoss()

    train_loader = make_loader(X_tr, y_tr, BATCH_SIZE, shuffle=True)
    X_va_t = torch.tensor(X_va, dtype=torch.float32).to(device)
    y_va_t = torch.tensor(y_va, dtype=torch.float32).to(device)

    history = {"train_loss": [], "val_loss": []}
    best_val = np.inf
    best_state = None
    epochs_no_improve = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        running_loss = 0.0
        n_samples = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            preds = model(xb)
            loss = loss_fn(preds, yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * xb.size(0)
            n_samples += xb.size(0)
        train_loss = running_loss / n_samples

        model.eval()
        with torch.no_grad():
            val_preds = model(X_va_t)
            val_loss = loss_fn(val_preds, y_va_t).item()

        scheduler.step(val_loss)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_state = {k_: v_.detach().cpu().clone() for k_, v_ in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epoch % 20 == 0 or epoch == 1:
            print(f"  [{tag}] epoch {epoch:3d}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")

        if epochs_no_improve >= PATIENCE:
            print(f"  [{tag}] early stopping at epoch {epoch} (best val_loss={best_val:.4f})")
            break

    model.load_state_dict(best_state)
    return model, history


def evaluate(model, X_arr, y_log_true, y_true_dollars):
    model.eval()
    with torch.no_grad():
        preds_log = model(torch.tensor(X_arr, dtype=torch.float32).to(device)).cpu().numpy()
    preds_dollars = np.expm1(preds_log)
    preds_dollars = np.clip(preds_dollars, 0, None)

    rmse = np.sqrt(mean_squared_error(y_true_dollars, preds_dollars))
    mae = mean_absolute_error(y_true_dollars, preds_dollars)
    r2 = r2_score(y_true_dollars, preds_dollars)
    rmsle = np.sqrt(mean_squared_error(y_log_true, preds_log))
    return {"rmse": rmse, "mae": mae, "r2": r2, "rmsle": rmsle}, preds_dollars


# ----------------------------------------------------------------------
# 5. Train + evaluate a CNN for each feature-selection method
# ----------------------------------------------------------------------
results = {}
histories = {}
predictions = {}

for name, mask in feature_sets.items():
    print(f"\n=== Training CNN on '{name}'-selected features ({mask.sum()} features) ===")
    X_tr_sel = X_train_proc[:, mask]
    X_va_sel = X_val_proc[:, mask]

    model, history = train_cnn(X_tr_sel, y_train_log, X_va_sel, y_val_log, tag=name)
    metrics, preds = evaluate(model, X_va_sel, y_val_log, y_val.values)

    results[name] = metrics
    histories[name] = history
    predictions[name] = preds

    print(f"  [{name}] VALIDATION  RMSE=${metrics['rmse']:,.0f}  "
          f"MAE=${metrics['mae']:,.0f}  R2={metrics['r2']:.4f}  RMSLE={metrics['rmsle']:.4f}")

best_method = min(results, key=lambda k_: results[k_]["rmse"])
print(f"\nBest feature selection method on validation RMSE: {best_method}")

# ----------------------------------------------------------------------
# 6. Visualizations
# ----------------------------------------------------------------------
colors = {"SelectKBest": "#2E86AB", "RFE": "#E4572E"}

# 6a. Training curves
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for ax, name in zip(axes, feature_sets.keys()):
    h = histories[name]
    ax.plot(h["train_loss"], label="Train loss", color=colors[name], alpha=0.8)
    ax.plot(h["val_loss"], label="Val loss", color=colors[name], linestyle="--")
    ax.set_title(f"Training curve — {name} features")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE loss (log-price scale)")
    ax.legend()
    ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/01_training_curves.png", dpi=150)
plt.close()

# 6b. Predicted vs actual
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
for ax, name in zip(axes, feature_sets.keys()):
    preds = predictions[name]
    ax.scatter(y_val.values, preds, alpha=0.5, color=colors[name], edgecolor="k", linewidth=0.3)
    lims = [min(y_val.values.min(), preds.min()), max(y_val.values.max(), preds.max())]
    ax.plot(lims, lims, "k--", linewidth=1, label="Perfect prediction")
    ax.set_xlabel("Actual SalePrice ($)")
    ax.set_ylabel("Predicted SalePrice ($)")
    ax.set_title(f"Predicted vs Actual — {name}\nR2={results[name]['r2']:.3f}")
    ax.legend()
    ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/02_predicted_vs_actual.png", dpi=150)
plt.close()

# 6c. Residuals
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for ax, name in zip(axes, feature_sets.keys()):
    preds = predictions[name]
    residuals = y_val.values - preds
    ax.scatter(preds, residuals, alpha=0.5, color=colors[name], edgecolor="k", linewidth=0.3)
    ax.axhline(0, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("Predicted SalePrice ($)")
    ax.set_ylabel("Residual (Actual - Predicted, $)")
    ax.set_title(f"Residual plot — {name}")
    ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/03_residual_plots.png", dpi=150)
plt.close()

# 6d. Metric comparison bar chart
metric_labels = ["rmse", "mae", "r2", "rmsle"]
fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))
for ax, m in zip(axes, metric_labels):
    vals = [results[n][m] for n in feature_sets.keys()]
    bars = ax.bar(list(feature_sets.keys()), vals,
                   color=[colors[n] for n in feature_sets.keys()])
    ax.set_title(m.upper())
    ax.grid(axis="y", alpha=0.3)
    for b, v in zip(bars, vals):
        label = f"${v:,.0f}" if m in ("rmse", "mae") else f"{v:.3f}"
        ax.text(b.get_x() + b.get_width() / 2, v, label, ha="center", va="bottom", fontsize=9)
plt.suptitle("Feature Selection Method Comparison (validation set)")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/04_metric_comparison.png", dpi=150)
plt.close()

# 6e. Top selected feature scores (SelectKBest) vs RFE ranking
fig, axes = plt.subplots(1, 2, figsize=(14, 7))

top_skb_idx = np.argsort(skb_scores)[::-1][:20]
axes[0].barh(feature_names[top_skb_idx][::-1], skb_scores[top_skb_idx][::-1], color=colors["SelectKBest"])
axes[0].set_title("Top 20 features — SelectKBest (F-score)")
axes[0].set_xlabel("F-statistic")

top_rfe_idx = np.argsort(rfe_ranking)[:20]
axes[1].barh(feature_names[top_rfe_idx][::-1], (21 - rfe_ranking[top_rfe_idx])[::-1], color=colors["RFE"])
axes[1].set_title("Top 20 features — RFE (rank 1 = most important)")
axes[1].set_xlabel("Inverted rank (higher = more important)")

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/05_feature_selection_comparison.png", dpi=150)
plt.close()

print("Saved 5 visualization files to", OUTPUT_DIR)

# ----------------------------------------------------------------------
# 7. Refit best pipeline on full labeled data + predict on Test.csv (if present)
# ----------------------------------------------------------------------
test_predictions_path = None
if os.path.exists(TEST_PATH):
    print(f"\nFound unlabeled {TEST_PATH} — generating holdout predictions with best model ({best_method}).")

    y_full_log = np.log1p(y.values)
    X_full_proc = preprocessor.fit_transform(X)
    if hasattr(X_full_proc, "toarray"):
        X_full_proc = X_full_proc.toarray()

    if best_method == "SelectKBest":
        selector_full = SelectKBest(score_func=f_regression, k=k)
    else:
        selector_full = RFE(estimator=Ridge(alpha=10.0, random_state=RANDOM_STATE),
                             n_features_to_select=k, step=15)
    selector_full.fit(X_full_proc, y_full_log)
    mask_full = selector_full.get_support()

    X_full_sel = X_full_proc[:, mask_full]
    # small internal validation split purely for early stopping on the refit
    X_tr_f, X_va_f, y_tr_f, y_va_f = train_test_split(
        X_full_sel, y_full_log, test_size=0.1, random_state=RANDOM_STATE
    )
    final_model, _ = train_cnn(X_tr_f, y_tr_f, X_va_f, y_va_f, tag=f"FINAL-{best_method}")

    test_df = pd.read_csv(TEST_PATH, na_values=["NA"], keep_default_na=True)
    test_ids = test_df["Id"]
    X_test_raw = test_df.drop(columns=["Id"])
    X_test_proc = preprocessor.transform(X_test_raw)
    if hasattr(X_test_proc, "toarray"):
        X_test_proc = X_test_proc.toarray()
    X_test_sel = X_test_proc[:, mask_full]

    final_model.eval()
    with torch.no_grad():
        test_preds_log = final_model(torch.tensor(X_test_sel, dtype=torch.float32).to(device)).cpu().numpy()
    test_preds = np.clip(np.expm1(test_preds_log), 0, None)

    out_df = pd.DataFrame({"Id": test_ids, "SalePrice": test_preds})
    test_predictions_path = f"{OUTPUT_DIR}/test_predictions.csv"
    out_df.to_csv(test_predictions_path, index=False)
    print(f"Saved test predictions to {test_predictions_path}")
else:
    print(f"\nNo {TEST_PATH} found — skipping unlabeled holdout predictions "
          f"(validation-set metrics/report/plots above are still complete).")

# Also save validation-set predictions for transparency
val_out = pd.DataFrame({
    "Actual_SalePrice": y_val.values,
    "Predicted_SalePrice_SelectKBest": predictions["SelectKBest"],
    "Predicted_SalePrice_RFE": predictions["RFE"],
})
val_pred_path = f"{OUTPUT_DIR}/validation_predictions.csv"
val_out.to_csv(val_pred_path, index=False)

# ----------------------------------------------------------------------
# 8. Text report
# ----------------------------------------------------------------------
report_lines = []
report_lines.append("=" * 70)
report_lines.append("DEEP LEARNING HOUSE PRICE PREDICTION — MODEL REPORT")
report_lines.append("=" * 70)
report_lines.append("")
report_lines.append("Dataset: Ames Housing (Kaggle House Prices format)")
report_lines.append(f"Rows: {X.shape[0]}   Raw features: {X.shape[1]}   "
                     f"Numeric: {len(numeric_cols)}   Categorical: {len(categorical_cols)}")
report_lines.append(f"Train/validation split: {int((1-VAL_SIZE)*100)}/{int(VAL_SIZE*100)} "
                     f"(random_state={RANDOM_STATE})")
report_lines.append("")
report_lines.append("-" * 70)
report_lines.append("1. PREPROCESSING (scikit-learn ColumnTransformer)")
report_lines.append("-" * 70)
report_lines.append("  Numeric features    : SimpleImputer(median) -> StandardScaler")
report_lines.append("  Categorical features: SimpleImputer(most_frequent) -> OneHotEncoder")
report_lines.append(f"  Resulting feature matrix width: {X_train_proc.shape[1]} columns")
report_lines.append("")
report_lines.append("-" * 70)
report_lines.append("2. FEATURE SELECTION (two algorithms compared)")
report_lines.append("-" * 70)
report_lines.append(f"  (a) SelectKBest (ANOVA F-test / f_regression), k={k}")
report_lines.append(f"      -> {len(skb_features)} features selected")
report_lines.append(f"  (b) RFE (Recursive Feature Elimination, Ridge estimator, step=15), k={k}")
report_lines.append(f"      -> {len(rfe_features)} features selected")
report_lines.append(f"  Overlap between the two selected feature sets: {len(overlap)} features")
report_lines.append("")
report_lines.append("  Top 10 SelectKBest features (by F-score):")
for i in np.argsort(skb_scores)[::-1][:10]:
    report_lines.append(f"    - {feature_names[i]:<35s} F={skb_scores[i]:.1f}")
report_lines.append("")
report_lines.append("  Top 10 RFE features (rank 1 = most important):")
for i in np.argsort(rfe_ranking)[:10]:
    report_lines.append(f"    - {feature_names[i]:<35s} rank={rfe_ranking[i]}")
report_lines.append("")
report_lines.append("-" * 70)
report_lines.append("3. MODEL ARCHITECTURE — 1-D Convolutional Neural Network (not an MLP)")
report_lines.append("-" * 70)
report_lines.append("  Input  : selected feature vector treated as a 1-channel 1-D signal")
report_lines.append("  Layers : Conv1d(1->32,k=3) -> BN -> ReLU")
report_lines.append("           Conv1d(32->64,k=3) -> BN -> ReLU -> MaxPool1d(2)")
report_lines.append("           Conv1d(64->128,k=3) -> BN -> ReLU -> AdaptiveAvgPool1d(1)")
report_lines.append("           Dropout(0.2) -> Linear(128->1)   [single linear read-out]")
report_lines.append("  Loss   : MSE on log1p(SalePrice)")
report_lines.append("  Optim  : Adam(lr=1e-3, weight_decay=1e-5) with ReduceLROnPlateau")
report_lines.append(f"  Training: up to {MAX_EPOCHS} epochs, early stopping patience={PATIENCE}, "
                     f"batch size={BATCH_SIZE}")
report_lines.append("")
report_lines.append("-" * 70)
report_lines.append("4. VALIDATION PERFORMANCE (SalePrice held in dollars unless noted)")
report_lines.append("-" * 70)
report_lines.append(f"  {'Method':<14s}{'RMSE ($)':>14s}{'MAE ($)':>14s}{'R2':>10s}{'RMSLE':>10s}")
for name in feature_sets.keys():
    m = results[name]
    report_lines.append(f"  {name:<14s}{m['rmse']:>14,.0f}{m['mae']:>14,.0f}{m['r2']:>10.4f}{m['rmsle']:>10.4f}")
report_lines.append("")
report_lines.append(f"  Best feature selection method (lowest validation RMSE): {best_method}")
report_lines.append("")
report_lines.append("-" * 70)
report_lines.append("5. INTERPRETATION")
report_lines.append("-" * 70)
rmse_gap_pct = abs(results["SelectKBest"]["rmse"] - results["RFE"]["rmse"]) / \
    max(results["SelectKBest"]["rmse"], results["RFE"]["rmse"]) * 100
worse_method = "RFE" if best_method == "SelectKBest" else "SelectKBest"
report_lines.append(
    f"  Both feature-selection strategies feed the SAME 1-D CNN architecture, so\n"
    f"  the performance gap between them isolates the effect of feature selection\n"
    f"  method rather than any difference in model capacity. SelectKBest is a\n"
    f"  univariate filter (ranks each feature independently of the others using an\n"
    f"  F-test against the target), while RFE is a wrapper method that iteratively\n"
    f"  removes the least useful feature(s) according to a fitted Ridge model, so it\n"
    f"  can capture feature interactions that a univariate filter misses. In this run,\n"
    f"  {best_method} outperformed {worse_method} by roughly {rmse_gap_pct:.0f}% lower validation RMSE.\n"
    f"  Note this differs from the classical linear-model comparison on this project,\n"
    f"  where SelectKBest previously outperformed RFE with a plain LinearRegression\n"
    f"  estimator; here RFE uses a regularized Ridge estimator (which is more stable\n"
    f"  with many rare one-hot categories) and feeds a non-linear CNN, both of which\n"
    f"  change which features end up being most useful downstream."
)
report_lines.append("")
if test_predictions_path:
    report_lines.append("-" * 70)
    report_lines.append("6. HOLDOUT (Test.csv) PREDICTIONS")
    report_lines.append("-" * 70)
    report_lines.append(f"  The best model ({best_method} feature set) was refit on the FULL")
    report_lines.append(f"  labeled Train.csv and used to score the unlabeled Test.csv.")
    report_lines.append(f"  Predictions saved to: {test_predictions_path}")
else:
    report_lines.append("-" * 70)
    report_lines.append("6. HOLDOUT (Test.csv) PREDICTIONS")
    report_lines.append("-" * 70)
    report_lines.append("  No Test.csv was found alongside Train.csv in this run, so no")
    report_lines.append("  unlabeled holdout predictions were generated. All metrics above")
    report_lines.append("  come from the labeled validation split.")
report_lines.append("")
report_lines.append("-" * 70)
report_lines.append("7. OUTPUT FILES")
report_lines.append("-" * 70)
report_lines.append("  01_training_curves.png            - train/val loss curves per feature-selection method")
report_lines.append("  02_predicted_vs_actual.png        - predicted vs actual SalePrice scatter")
report_lines.append("  03_residual_plots.png             - residuals vs predicted value")
report_lines.append("  04_metric_comparison.png          - RMSE/MAE/R2/RMSLE bar comparison")
report_lines.append("  05_feature_selection_comparison.png - top features per selection method")
report_lines.append("  validation_predictions.csv        - actual vs predicted on validation split")
if test_predictions_path:
    report_lines.append("  test_predictions.csv              - final predictions on unlabeled Test.csv")
report_lines.append("  model_report.txt                  - this report")
report_lines.append("=" * 70)

report_text = "\n".join(report_lines)
report_path = f"{OUTPUT_DIR}/model_report.txt"
with open(report_path, "w") as f:
    f.write(report_text)

print("\n" + report_text)
print(f"\nReport written to {report_path}")
