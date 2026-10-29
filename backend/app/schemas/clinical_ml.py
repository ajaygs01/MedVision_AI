from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class ClinicalPatientInput(BaseModel):
    """
    Standard 13 clinical physiological attributes from the UCI Cleveland Clinic Cohort.
    """
    age: float = Field(..., ge=1, le=120, description="Age in years (e.g., 58)")
    sex: int = Field(..., ge=0, le=1, description="Biological sex (1 = male, 0 = female)")
    cp: int = Field(..., ge=1, le=4, description="Chest pain type (1 = typical angina, 2 = atypical angina, 3 = non-anginal, 4 = asymptomatic)")
    trestbps: float = Field(..., ge=50, le=300, description="Resting blood pressure in mm Hg (e.g., 130)")
    chol: float = Field(..., ge=50, le=800, description="Serum cholesterol in mg/dl (e.g., 240)")
    fbs: int = Field(..., ge=0, le=1, description="Fasting blood sugar > 120 mg/dl (1 = true, 0 = false)")
    restecg: int = Field(..., ge=0, le=2, description="Resting ECG results (0 = normal, 1 = ST-T wave abnormality, 2 = LV hypertrophy)")
    thalach: float = Field(..., ge=40, le=250, description="Maximum heart rate achieved in bpm (e.g., 150)")
    exang: int = Field(..., ge=0, le=1, description="Exercise-induced angina (1 = yes, 0 = no)")
    oldpeak: float = Field(..., ge=0.0, le=10.0, description="ST depression induced by exercise relative to rest (e.g., 1.5)")
    slope: int = Field(..., ge=1, le=3, description="Slope of peak exercise ST segment (1 = upsloping, 2 = flat, 3 = downsloping)")
    ca: float = Field(..., ge=0.0, le=3.0, description="Number of major vessels (0-3) colored by fluoroscopy")
    thal: float = Field(..., description="Thallium stress test (3 = normal, 6 = fixed defect, 7 = reversible defect)")
    patient_id: Optional[str] = Field(default=None, description="Optional anonymous patient identifier")


class ClinicalRiskResponse(BaseModel):
    id: Optional[str] = Field(default=None, description="Persisted analysis record ID in MongoDB Atlas")
    status: str = Field(description="'success', 'not_trained', or 'error'")
    risk_probability: Optional[float] = Field(default=None, description="Predicted probability of cardiovascular pathology [0.0 - 1.0]")
    predicted_class_id: Optional[int] = Field(default=None, description="0 for Low Risk (<0.5), 1 for Elevated Risk (>=0.5)")
    risk_tier: str = Field(description="Clinical risk tier: 'Low Risk', 'Moderate Risk', or 'Elevated Risk'")
    prediction: str = Field(description="Human-readable prediction summary")
    model_name: str = Field(default="Random Forest Cardiovascular Risk Predictor")
    dataset_source: str = Field(default="UCI Cleveland Clinic Cohort")
    top_contributing_factors: List[str] = Field(default_factory=list, description="Patient-specific high-risk physiological factors")
    metrics: Optional[Dict[str, Any]] = Field(default=None, description="Validation metrics achieved on held-out test cohort")
    message: str = Field(description="Execution summary message")
    disclaimer: str = Field(
        default="Academic research prototype. Results are not clinically validated and must not be used as standalone medical diagnosis."
    )


class ClinicalModelInfoResponse(BaseModel):
    model_name: str
    status: str
    dataset_name: str
    total_training_samples: int
    test_samples: int
    metrics: Dict[str, Any]
    confusion_matrix: Dict[str, int]
    top_features: List[Dict[str, Any]]
    disclaimer: str
