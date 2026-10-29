import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

from ml.clinical_ml.config import METRICS_SAVE_PATH, MODEL_SAVE_PATH
from ml.clinical_ml.predict import clinical_risk_predictor

logger = logging.getLogger("medivision.services.clinical_ml")


class ClinicalMLService:
    def __init__(self):
        self.predictor = clinical_risk_predictor
        self._cached_metrics: Optional[Dict[str, Any]] = None

    def is_ready(self) -> bool:
        """Returns True if the trained model artifact exists and is ready for inference."""
        return self.predictor.is_ready()

    def get_model_info(self) -> Dict[str, Any]:
        """Loads and returns the comprehensive evaluation metrics report."""
        if self._cached_metrics is None:
            if METRICS_SAVE_PATH.exists():
                try:
                    with open(METRICS_SAVE_PATH, "r", encoding="utf-8") as f:
                        self._cached_metrics = json.load(f)
                except Exception as exc:
                    logger.warning(f"Failed to read metrics file: {exc}")
                    self._cached_metrics = None

        if self._cached_metrics:
            return {
                "model_name": self._cached_metrics.get("model_name", "Random Forest Cardiovascular Risk Predictor"),
                "status": "ready" if self.is_ready() else "not_trained",
                "dataset_name": self._cached_metrics.get("dataset", "UCI Cleveland Clinic Cohort"),
                "total_training_samples": 242,
                "test_samples": self._cached_metrics.get("test_samples", 31),
                "metrics": self._cached_metrics.get("metrics", {}),
                "confusion_matrix": self._cached_metrics.get("confusion_matrix", {}),
                "top_features": self._cached_metrics.get("top_features", []),
                "disclaimer": self._cached_metrics.get(
                    "disclaimer",
                    "Research prototype. Must not be used as standalone clinical medical advice.",
                ),
            }

        return {
            "model_name": "Random Forest Cardiovascular Risk Predictor",
            "status": "ready" if self.is_ready() else "not_trained",
            "dataset_name": "UCI Cleveland Clinic Cohort",
            "total_training_samples": 0,
            "test_samples": 0,
            "metrics": {},
            "confusion_matrix": {},
            "top_features": [],
            "disclaimer": "Academic research prototype.",
        }

    def predict_risk(self, patient_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Runs validated forward inference and attaches test set benchmark metrics."""
        result = self.predictor.predict(patient_dict)
        info = self.get_model_info()
        result["metrics"] = info.get("metrics")
        return result


clinical_ml_service = ClinicalMLService()
