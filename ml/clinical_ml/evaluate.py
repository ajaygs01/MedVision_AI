"""
Independent Evaluation script for Clinical ML Model.
Evaluates on the held-out Test Split and generates clinical metrics & feature importances.
"""

import os
import sys
from pathlib import Path
import json
import joblib
import numpy as np

_project_root = Path(__file__).resolve().parents[2]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
)

from ml.clinical_ml.config import (
    MODEL_SAVE_PATH,
    REPORTS_DIR,
    METRICS_SAVE_PATH,
)
from ml.clinical_ml.dataset import get_train_val_test_data


def evaluate_clinical_model():
    print("=" * 60)
    print("MediVision AI - Clinical ML Model Evaluation (Test Split)")
    print("=" * 60)

    if not MODEL_SAVE_PATH.exists():
        raise FileNotFoundError(f"Model checkpoint not found at: {MODEL_SAVE_PATH}. Run train.py first.")

    # 1. Load model bundle
    bundle = joblib.load(MODEL_SAVE_PATH)
    pipeline = bundle["pipeline"]
    feature_cols = bundle["feature_cols"]

    # 2. Load held-out test split
    data = get_train_val_test_data()
    X_test, y_test = data["X_test"], data["y_test"]

    print(f"Evaluating on held-out test split: {len(X_test)} samples")

    # 3. Model forward pass
    test_preds = pipeline.predict(X_test)
    test_probs = pipeline.predict_proba(X_test)[:, 1]

    # 4. Compute metrics
    acc = accuracy_score(y_test, test_preds)
    bal_acc = balanced_accuracy_score(y_test, test_preds)
    auc = roc_auc_score(y_test, test_probs)
    f1 = f1_score(y_test, test_preds)
    prec = precision_score(y_test, test_preds)
    rec = recall_score(y_test, test_preds)

    cm = confusion_matrix(y_test, test_preds)
    tn, fp, fn, tp = cm.ravel()
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    print("\n--- TEST SPLIT CLINICAL METRICS ---")
    print(f"Accuracy:          {acc:.4f}")
    print(f"Balanced Accuracy: {bal_acc:.4f}")
    print(f"ROC-AUC:           {auc:.4f}")
    print(f"F1-Score:          {f1:.4f}")
    print(f"Sensitivity/Recall:{rec:.4f}")
    print(f"Specificity:       {spec:.4f}")
    print(f"Precision:         {prec:.4f}")
    print(f"\nConfusion Matrix:\nTN={tn}, FP={fp}\nFN={fn}, TP={tp}")

    # 5. Extract Feature Importances
    classifier = pipeline.named_steps["classifier"]
    encoded_features = bundle.get("encoded_feature_names", [])

    importances = classifier.feature_importances_
    ranked_features = []
    if len(encoded_features) == len(importances):
        indices = np.argsort(importances)[::-1]
        for idx in indices[:10]:
            ranked_features.append({
                "feature": encoded_features[idx],
                "importance": round(float(importances[idx]), 4),
            })

    print("\n--- TOP 5 PREDICTIVE CLINICAL MARKERS ---")
    for item in ranked_features[:5]:
        print(f"  - {item['feature']}: {item['importance']:.4f}")

    # 6. Save metrics JSON
    os.makedirs(REPORTS_DIR, exist_ok=True)
    report = {
        "model_name": "Clinical Tabular Risk Predictor (Random Forest)",
        "dataset": "UCI Cleveland Clinic Heart Disease Cohort",
        "test_samples": int(len(X_test)),
        "metrics": {
            "accuracy": round(float(acc), 4),
            "balanced_accuracy": round(float(bal_acc), 4),
            "roc_auc": round(float(auc), 4),
            "f1": round(float(f1), 4),
            "sensitivity_recall": round(float(rec), 4),
            "specificity": round(float(spec), 4),
            "precision": round(float(prec), 4),
        },
        "confusion_matrix": {
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp),
        },
        "top_features": ranked_features,
        "disclaimer": "Research prototype. Must not be used as standalone clinical medical advice.",
    }

    with open(METRICS_SAVE_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\n[SAVED] Comprehensive metrics report written to: {METRICS_SAVE_PATH}")
    return report


if __name__ == "__main__":
    evaluate_clinical_model()
