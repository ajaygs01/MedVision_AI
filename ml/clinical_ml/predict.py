"""
Inference prediction engine for Clinical Tabular Risk Predictor.
Loads trained Random Forest pipeline and evaluates patient clinical metrics.
"""

import os
import sys
from pathlib import Path
import joblib
import pandas as pd
from typing import Dict, Any, Optional

_project_root = Path(__file__).resolve().parents[2]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from ml.clinical_ml.config import (
    MODEL_SAVE_PATH,
    RISK_THRESHOLDS,
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
)


class ClinicalRiskPredictor:
    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path or str(MODEL_SAVE_PATH)
        self.bundle: Optional[Dict[str, Any]] = None
        self.pipeline = None
        self.feature_cols = []
        self.encoded_feature_names = []
        self._load_model()

    def _load_model(self):
        if os.path.exists(self.model_path):
            try:
                self.bundle = joblib.load(self.model_path)
                self.pipeline = self.bundle["pipeline"]
                self.feature_cols = self.bundle["feature_cols"]
                self.encoded_feature_names = self.bundle.get("encoded_feature_names", [])
            except Exception as e:
                print(f"Warning: Failed to load clinical model from {self.model_path}: {e}")
                self.pipeline = None
        else:
            self.pipeline = None

    def is_ready(self) -> bool:
        return self.pipeline is not None

    def predict(self, patient_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executes forward inference on a patient clinical profile.
        
        Expected patient_dict keys:
        - age (int/float): Age in years
        - sex (int): 1=male, 0=female
        - cp (int): Chest pain type (1-4)
        - trestbps (float): Resting blood pressure in mm Hg
        - chol (float): Serum cholesterol in mg/dl
        - fbs (int): Fasting blood sugar > 120 mg/dl (1 or 0)
        - restecg (int): Resting ECG results (0, 1, 2)
        - thalach (float): Maximum heart rate achieved
        - exang (int): Exercise induced angina (1 or 0)
        - oldpeak (float): ST depression induced by exercise
        - slope (int): Slope of peak exercise ST segment (1, 2, 3)
        - ca (float): Number of major vessels (0-3)
        - thal (float): Thallium stress test (3, 6, 7)
        """
        if not self.is_ready():
            return {
                "status": "not_trained",
                "risk_probability": None,
                "risk_tier": "Unconfigured",
                "prediction": "Model not trained",
                "top_contributing_factors": [],
                "message": "Clinical ML model weights have not been trained or loaded.",
                "disclaimer": "Academic research prototype. Results are not clinically validated.",
            }

        # Convert to single-row DataFrame
        df = pd.DataFrame([patient_dict])

        # Validate required columns
        for col in self.feature_cols:
            if col not in df.columns:
                df[col] = None

        # Ensure correct column ordering
        df = df[self.feature_cols]

        try:
            # Predict probability of cardiovascular disease (Class 1)
            prob_positive = float(self.pipeline.predict_proba(df)[0, 1])
            pred_class = int(prob_positive >= 0.5)

            # Categorize clinical risk tier
            if prob_positive < RISK_THRESHOLDS["low"]:
                tier = "Low Risk"
                label = "Low Cardiovascular Disease Risk"
            elif prob_positive < RISK_THRESHOLDS["moderate"]:
                tier = "Moderate Risk"
                label = "Moderate Cardiovascular Risk"
            else:
                tier = "Elevated Risk"
                label = "Elevated Cardiovascular Risk / Possible Pathology"

            # Compute patient-specific risk factor impact highlights
            risk_factors = []
            if float(patient_dict.get("chol", 0)) >= 240:
                risk_factors.append("Hypercholesterolemia (Cholesterol >= 240 mg/dl)")
            if float(patient_dict.get("trestbps", 0)) >= 140:
                risk_factors.append("Stage 2 Hypertension (Resting BP >= 140 mm Hg)")
            if int(patient_dict.get("exang", 0)) == 1:
                risk_factors.append("Exercise-Induced Angina (Exang = Yes)")
            if float(patient_dict.get("oldpeak", 0)) >= 2.0:
                risk_factors.append("Significant ST-Segment Depression (Oldpeak >= 2.0)")
            if float(patient_dict.get("ca", 0)) >= 1:
                risk_factors.append(f"Coronary Calcification Detected (Vessels: {int(patient_dict.get('ca', 0))})")
            if float(patient_dict.get("thalach", 0)) < 120 and float(patient_dict.get("age", 0)) < 60:
                risk_factors.append("Sub-maximal Chronotropic Response (Low Max Heart Rate)")

            return {
                "status": "success",
                "risk_probability": round(prob_positive, 4),
                "predicted_class_id": pred_class,
                "risk_tier": tier,
                "prediction": label,
                "model_name": "Random Forest Cardiovascular Risk Predictor",
                "dataset_source": "UCI Cleveland Clinic Cohort",
                "top_contributing_factors": risk_factors,
                "message": "Inference completed successfully via Scikit-Learn ensemble pipeline.",
                "disclaimer": "Academic research prototype. Results are not clinically validated and must not be used as standalone medical diagnosis.",
            }

        except Exception as exc:
            return {
                "status": "error",
                "risk_probability": None,
                "risk_tier": "Error",
                "prediction": "Evaluation Error",
                "top_contributing_factors": [],
                "message": f"Inference pipeline execution error: {str(exc)}",
                "disclaimer": "Academic research prototype.",
            }


clinical_risk_predictor = ClinicalRiskPredictor()
