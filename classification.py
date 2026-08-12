"""
Housing Price CLASSIFICATION Model (Ames Housing Dataset)
============================================================
Use case : Predict a housing PRICE CATEGORY (Low / Medium / High / Luxury)

Pipeline components (all required elements of the assignment):
  1. Target engineering  : SalePrice (continuous) -> 4 price-range classes
  2. Preprocessing        : sklearn ColumnTransformer
                              - numeric   : SimpleImputer(median) + StandardScaler
                              - categorical: SimpleImputer(constant) + OneHotEncoder
  3. Feature selection #1 : SelectKBest (ANOVA F-test, f_classif)   -- filter method
  4. Feature selection #2 : RFE (Recursive Feature Elimination)     -- wrapper method
  5. Classifier            : RandomForestClassifier
  6. Reporting             : classification_report, confusion matrix, accuracy,
                              precision/recall/F1, 5-fold cross validation,
                              feature-selection comparison, saved plots + text report
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_recall_fscore_support, ConfusionMatrixDisplay
)

RANDOM_STATE = 42

# ------------------------------------------------------------------
# 1. LOAD DATA
# ------------------------------------------------------------------
df = pd.read_csv("Train.csv")
df = df.drop(columns=["Id"])

# ------------------------------------------------------------------
# 2. TARGET ENGINEERING: turn the continuous SalePrice into 4 classes
#    using quartile cut points -> a genuine classification target.
# ------------------------------------------------------------------
labels = ["Low", "Medium", "High", "Luxury"]
df["PriceCategory"] = pd.qcut(df["SalePrice"], q=4, labels=labels)

print("Class distribution (target = PriceCategory):")
print(df["PriceCategory"].value_counts().reindex(labels))
print()

X = df.drop(columns=["SalePrice", "PriceCategory"])
y = df["PriceCategory"]

numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_features = X.select_dtypes(include=["object"]).columns.tolist()
print(f"Numeric features   : {len(numeric_features)}")
print(f"Categorical features: {len(categorical_features)}")
print()

# ------------------------------------------------------------------
# 3. TRAIN / TEST SPLIT (stratified so all 4 classes stay balanced)
# ------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

# ------------------------------------------------------------------
# 4. PREPROCESSING (scikit-learn ColumnTransformer)
# ------------------------------------------------------------------
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
])

# ------------------------------------------------------------------
# 5. TWO FEATURE SELECTION ALGORITHMS
#    (a) SelectKBest w/ ANOVA F-test  -> filter method
#    (b) RFE w/ Logistic Regression   -> wrapper method
# ------------------------------------------------------------------
K_BEST = 40
N_RFE_FEATURES = 40

feature_selectors = {
    "SelectKBest (f_classif)": SelectKBest(score_func=f_classif, k=K_BEST),
    "RFE (LogisticRegression)": RFE(
        estimator=LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
        n_features_to_select=N_RFE_FEATURES,
        step=25  # remove 25 features per iteration -> keeps runtime reasonable
    ),
}

classifier = RandomForestClassifier(
    n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1
)

results = {}
fitted_pipelines = {}

for fs_name, selector in feature_selectors.items():
    pipe = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("feature_selection", selector),
        ("classifier", classifier)
    ])

    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="weighted"
    )
    cv_scores = cross_val_score(
        pipe, X_train, y_train, cv=StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE),
        scoring="accuracy", n_jobs=-1
    )

    results[fs_name] = {
        "accuracy": acc,
        "precision_weighted": precision,
        "recall_weighted": recall,
        "f1_weighted": f1,
        "cv_mean_accuracy": cv_scores.mean(),
        "cv_std_accuracy": cv_scores.std(),
        "y_pred": y_pred,
        "classification_report": classification_report(y_test, y_pred, target_names=labels)
    }
    fitted_pipelines[fs_name] = pipe

# ------------------------------------------------------------------
# 6. REPORTING
# ------------------------------------------------------------------
report_lines = []
report_lines.append("=" * 70)
report_lines.append("HOUSING PRICE CATEGORY CLASSIFICATION - MODEL REPORT")
report_lines.append("=" * 70)
report_lines.append(f"Dataset: Train.csv (Ames Housing) | rows={df.shape[0]}, raw cols={df.shape[1]-1}")
report_lines.append(f"Target : PriceCategory derived from SalePrice quartiles -> {labels}")
report_lines.append(f"Classifier: RandomForestClassifier(n_estimators=300)")
report_lines.append(f"Train/Test split: 80/20 (stratified), random_state={RANDOM_STATE}")
report_lines.append("")

for fs_name, res in results.items():
    report_lines.append("-" * 70)
    report_lines.append(f"FEATURE SELECTION METHOD: {fs_name}")
    report_lines.append("-" * 70)
    report_lines.append(f"Test Accuracy            : {res['accuracy']:.4f}")
    report_lines.append(f"Weighted Precision        : {res['precision_weighted']:.4f}")
    report_lines.append(f"Weighted Recall           : {res['recall_weighted']:.4f}")
    report_lines.append(f"Weighted F1-score         : {res['f1_weighted']:.4f}")
    report_lines.append(f"5-fold CV Accuracy (train): {res['cv_mean_accuracy']:.4f} +/- {res['cv_std_accuracy']:.4f}")
    report_lines.append("")
    report_lines.append("Classification Report:")
    report_lines.append(res["classification_report"])
    report_lines.append("")

# Head-to-head comparison table
report_lines.append("=" * 70)
report_lines.append("FEATURE SELECTION METHOD COMPARISON")
report_lines.append("=" * 70)
comp_df = pd.DataFrame({
    fs_name: {
        "Accuracy": res["accuracy"],
        "Precision (weighted)": res["precision_weighted"],
        "Recall (weighted)": res["recall_weighted"],
        "F1 (weighted)": res["f1_weighted"],
        "CV Mean Accuracy": res["cv_mean_accuracy"],
    }
    for fs_name, res in results.items()
}).T
report_lines.append(comp_df.round(4).to_string())
report_lines.append("")

best_method = comp_df["Accuracy"].idxmax()
report_lines.append(f"Best performing feature selection method (by test accuracy): {best_method}")
report_lines.append("")

report_text = "\n".join(report_lines)
print(report_text)

with open("reports/classification_report.txt", "w") as f:
    f.write(report_text)

# ------------------------------------------------------------------
# 7. VISUAL REPORTS: confusion matrices for both feature selectors
# ------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
for ax, (fs_name, res) in zip(axes, results.items()):
    cm = confusion_matrix(y_test, res["y_pred"], labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"Confusion Matrix\n{fs_name}\nAcc={res['accuracy']:.3f}")
plt.tight_layout()
plt.savefig("reports/classification_confusion_matrices.png", dpi=150)
plt.close()

# Bar chart comparing metrics across the two feature-selection methods
metrics_to_plot = ["Accuracy", "Precision (weighted)", "Recall (weighted)", "F1 (weighted)"]
fig, ax = plt.subplots(figsize=(9, 5))
comp_df[metrics_to_plot].plot(kind="bar", ax=ax)
ax.set_ylim(0, 1)
ax.set_ylabel("Score")
ax.set_title("Performance by Feature Selection Method")
ax.legend(loc="lower right")
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig("reports/classification_metric_comparison.png", dpi=150)
plt.close()

print("\nSaved: classification_report.txt, confusion_matrices.png, classification_metric_comparison.png")
