"""
Training pipeline for Clinical Tabular Machine Learning.
Cardiovascular Patient Risk Predictor (Random Forest Ensemble).
"""

import os
import sys
from pathlib import Path
import joblib
import numpy as np

_project_root = Path(__file__).resolve().parents[2]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
)

from ml.clinical_ml.config import (
    MODEL_ARTIFACTS_DIR,
    MODEL_SAVE_PATH,
    MODEL_PARAMS,
    RANDOM_SEED,
)
from ml.clinical_ml.dataset import (
    get_train_val_test_data,
    build_preprocessor,
)


def train_clinical_model():
    print("=" * 60)
    print("MediVision AI - Clinical ML Training Pipeline")
    print("=" * 60)

    # 1. Load data
    data = get_train_val_test_data()
    X_train, y_train = data["X_train"], data["y_train"]
    X_val, y_val = data["X_val"], data["y_val"]

    print(f"Loaded {len(X_train)} training records, {len(X_val)} validation records.")

    # 2. Build preprocessor & classifier pipeline
    preprocessor = build_preprocessor()
    classifier = RandomForestClassifier(**MODEL_PARAMS)

    full_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", classifier),
    ])

    # 3. 5-Fold Stratified Cross-Validation on Train split
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    cv_auc_scores = cross_val_score(full_pipeline, X_train, y_train, cv=cv, scoring="roc_auc")
    cv_acc_scores = cross_val_score(full_pipeline, X_train, y_train, cv=cv, scoring="accuracy")

    print(f"5-Fold Cross-Validation ROC-AUC: {cv_auc_scores.mean():.4f} (+/- {cv_auc_scores.std():.4f})")
    print(f"5-Fold Cross-Validation Accuracy: {cv_acc_scores.mean():.4f} (+/- {cv_acc_scores.std():.4f})")

    # 4. Fit pipeline on full training split
    print("\nFitting full pipeline on training split...")
    full_pipeline.fit(X_train, y_train)

    # 5. Evaluate on Validation split
    val_preds = full_pipeline.predict(X_val)
    val_probs = full_pipeline.predict_proba(X_val)[:, 1]

    val_acc = accuracy_score(y_val, val_preds)
    val_auc = roc_auc_score(y_val, val_probs)
    val_f1 = f1_score(y_val, val_preds)
    val_prec = precision_score(y_val, val_preds)
    val_rec = recall_score(y_val, val_preds)

    print("\n--- Validation Split Performance ---")
    print(f"Accuracy:  {val_acc:.4f}")
    print(f"ROC-AUC:   {val_auc:.4f}")
    print(f"F1-Score:  {val_f1:.4f}")
    print(f"Precision: {val_prec:.4f}")
    print(f"Recall:    {val_rec:.4f}")

    # 6. Extract One-Hot Encoded Feature Names for Attribution
    fitted_preprocessor = full_pipeline.named_steps["preprocessor"]
    fitted_classifier = full_pipeline.named_steps["classifier"]

    cat_feature_names = fitted_preprocessor.named_transformers_["cat"]["encoder"].get_feature_names_out(
        data["feature_cols"][5:] # categorical cols
    ) if hasattr(fitted_preprocessor.named_transformers_["cat"]["encoder"], "get_feature_names_out") else []

    all_encoded_features = list(data["feature_cols"][:5]) + list(cat_feature_names)

    # 7. Save model artifact bundle
    os.makedirs(MODEL_ARTIFACTS_DIR, exist_ok=True)
    bundle = {
        "pipeline": full_pipeline,
        "feature_cols": data["feature_cols"],
        "encoded_feature_names": all_encoded_features,
        "val_metrics": {
            "accuracy": round(float(val_acc), 4),
            "roc_auc": round(float(val_auc), 4),
            "f1": round(float(val_f1), 4),
            "precision": round(float(val_prec), 4),
            "recall": round(float(val_rec), 4),
        },
        "cv_metrics": {
            "mean_roc_auc": round(float(cv_auc_scores.mean()), 4),
            "std_roc_auc": round(float(cv_auc_scores.std()), 4),
            "mean_accuracy": round(float(cv_acc_scores.mean()), 4),
            "std_accuracy": round(float(cv_acc_scores.std()), 4),
        },
        "model_type": "RandomForestClassifier",
        "dataset_name": "UCI Cleveland Heart Disease Cohort",
    }

    joblib.dump(bundle, MODEL_SAVE_PATH)
    print(f"\n[SUCCESS] Model artifact saved to: {MODEL_SAVE_PATH}")
    print(f"File size: {os.path.getsize(MODEL_SAVE_PATH) / 1024:.1f} KB")
    return bundle


if __name__ == "__main__":
    train_clinical_model()
