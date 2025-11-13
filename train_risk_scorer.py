# train_risk_scorer.py
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, precision_recall_curve, auc, f1_score
)
import joblib

# Optional: SHAP
try:
    import shap
    HAS_SHAP = True
except Exception:
    HAS_SHAP = False

# -----------------------
# CONFIG
# -----------------------
DATA_PATH = "data/realistic_fraud_detection_dataset.csv"  # update if different
RND = 42
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# -----------------------
# LOAD
# -----------------------
df = pd.read_csv(DATA_PATH)
print("Loaded:", df.shape)
print(df.head())

# -----------------------
# QUICK EDA (print)
# -----------------------
print("\nClass distribution:\n", df['is_fraud'].value_counts(normalize=True) * 100)
print("\nAmount stats:\n", df['amount'].describe())

# -----------------------
# FEATURE ENGINEERING
# -----------------------
# basic features: transaction amount, merchant_category, country, payment_method, age, tenure
# create derived features: amount_log, high_amount_flag
df['amount_log'] = np.log1p(df['amount'])
df['high_amount_flag'] = (df['amount'] > df['amount'].quantile(0.99)).astype(int)

# If you have timestamps, extract hour/day features — optional
if 'transaction_time' in df.columns:
    df['transaction_time'] = pd.to_datetime(df['transaction_time'])
    df['hour'] = df['transaction_time'].dt.hour
    df['dayofweek'] = df['transaction_time'].dt.dayofweek

# choose features
cat_features = [c for c in ['merchant_category', 'country', 'payment_method'] if c in df.columns]
num_features = ['amount_log', 'customer_age', 'customer_tenure', 'high_amount_flag']
if 'hour' in df.columns:
    num_features += ['hour', 'dayofweek']

X = df[cat_features + num_features]
y = df['is_fraud']

# -----------------------
# SPLIT
# -----------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RND, stratify=y
)
print("Train size:", X_train.shape, "Test size:", X_test.shape)

# -----------------------
# PREPROCESS PIPELINE
# -----------------------
categorical_transformer = Pipeline(steps=[
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

numeric_transformer = Pipeline(steps=[
    ("scaler", StandardScaler())
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, num_features),
        ("cat", categorical_transformer, cat_features)
    ],
    remainder='drop'
)

# -----------------------
# MODEL PIPELINE (RandomForest baseline)
# -----------------------
rf_pipeline = Pipeline(steps=[
    ("pre", preprocessor),
    ("rf", RandomForestClassifier(n_estimators=200, random_state=RND, n_jobs=-1, class_weight='balanced'))
])

# Fit RF
print("\nTraining RandomForest...")
rf_pipeline.fit(X_train, y_train)
print("Done.")

# Save RF
joblib.dump(rf_pipeline, os.path.join(MODEL_DIR, "rf_pipeline.pkl"))

# -----------------------
# MODEL PIPELINE (XGBoost)
# -----------------------
xgb_pipeline = Pipeline(steps=[
    ("pre", preprocessor),
    ("xgb", XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=RND,
        n_jobs=-1
    ))
])

print("\nTraining XGBoost...")
xgb_pipeline.fit(X_train, y_train)
joblib.dump(xgb_pipeline, os.path.join(MODEL_DIR, "xgb_pipeline.pkl"))
print("XGBoost saved.")

# -----------------------
# EVALUATION FUNCTION
# -----------------------
def evaluate_model(pipeline, X_test, y_test, name="model"):
    proba = pipeline.predict_proba(X_test)[:, 1]
    preds = (proba >= 0.5).astype(int)  # default threshold
    print(f"\n=== Evaluation: {name} (threshold=0.5) ===")
    print(classification_report(y_test, preds, digits=4))
    print("ROC AUC:", roc_auc_score(y_test, proba))
    # confusion
    cm = confusion_matrix(y_test, preds)
    print("Confusion matrix:\n", cm)

    # Precision-Recall curve area
    precision, recall, thresholds = precision_recall_curve(y_test, proba)
    pr_auc = auc(recall, precision)
    print("PR AUC:", pr_auc)

    # Return values for plotting or threshold tuning
    return proba, precision, recall, thresholds

# Evaluate both models
rf_proba, rf_precision, rf_recall, rf_thresholds = evaluate_model(rf_pipeline, X_test, y_test, "RandomForest")
xgb_proba, xgb_precision, xgb_recall, xgb_thresholds = evaluate_model(xgb_pipeline, X_test, y_test, "XGBoost")

# -----------------------
# PLOT Precision-Recall for both
# -----------------------
plt.figure(figsize=(8,6))
plt.plot(rf_recall, rf_precision, label=f"RF PR AUC={auc(rf_recall, rf_precision):.4f}")
plt.plot(xgb_recall, xgb_precision, label=f"XGB PR AUC={auc(xgb_recall, xgb_precision):.4f}")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(MODEL_DIR, "pr_curve.png"))
plt.show()

# -----------------------
# TUNE DECISION THRESHOLD (tradeoff precision vs recall)
# -----------------------
def tune_threshold(proba, precision, recall, thresholds, target_recall=0.8):
    # find threshold that gives recall >= target_recall and highest precision
    candidates = [(t, p, r) for t,p,r in zip(thresholds, precision[:-1], recall[:-1]) if r >= target_recall]
    if len(candidates)==0:
        # fallback: choose threshold that maximizes F1
        f1s = (2 * precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-9)
        idx = np.argmax(f1s)
        return thresholds[idx], precision[:-1][idx], recall[:-1][idx]
    # choose highest precision among those
    best = max(candidates, key=lambda x: x[1])
    return best[0], best[1], best[2]

thr, prec_at_thr, rec_at_thr = tune_threshold(xgb_proba, xgb_precision, xgb_recall, xgb_thresholds, target_recall=0.8)
print(f"\nChosen threshold for target recall 0.8 -> thr={thr:.4f}, precision={prec_at_thr:.4f}, recall={rec_at_thr:.4f}")

# Final predictions with chosen threshold
final_preds = (xgb_proba >= thr).astype(int)
print("\nFinal classification report with tuned threshold:")
print(classification_report(y_test, final_preds, digits=4))

# Save final XGBoost pipeline as "best" with threshold metadata (store threshold as separate file)
joblib.dump({"pipeline": xgb_pipeline, "threshold": thr}, os.path.join(MODEL_DIR, "best_model_with_threshold.pkl"))

# -----------------------
# SHAP explainability (optional)
# -----------------------
if HAS_SHAP:
    print("\nRunning SHAP explainability (this will take time)...")
    # get preprocessor and X_test transformed for shap
    pre = xgb_pipeline.named_steps['pre']
    X_test_trans = pre.transform(X_test)
    model_xgb = xgb_pipeline.named_steps['xgb']
    explainer = shap.TreeExplainer(model_xgb)
    shap_values = explainer.shap_values(X_test_trans)
    shap.summary_plot(shap_values, features=X_test_trans, show=True)
    # save a small shap summary plot to file
    plt.figure()
    shap.summary_plot(shap_values, features=X_test_trans, show=False)
    plt.savefig(os.path.join(MODEL_DIR, "shap_summary.png"))
else:
    print("\nSHAP not installed or failed to import. To enable SHAP, pip install shap and rerun.")

print("\nALL DONE. Models and artifacts are in the 'models/' folder.")
