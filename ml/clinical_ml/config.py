"""
Configuration parameters for Clinical Tabular Machine Learning Pipeline.
Cardiovascular Disease Risk Prediction (UCI Cleveland Cohort).
"""

from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = PROJECT_ROOT / "datasets" / "clinical_heart"
DATASET_PATH = DATASET_DIR / "processed.cleveland.data"

MODEL_ARTIFACTS_DIR = PROJECT_ROOT / "backend" / "model_artifacts" / "clinical_ml"
MODEL_SAVE_PATH = MODEL_ARTIFACTS_DIR / "random_forest_risk_model.joblib"

REPORTS_DIR = PROJECT_ROOT / "model_artifacts" / "clinical_ml"
METRICS_SAVE_PATH = REPORTS_DIR / "metrics.json"

# Column definitions (UCI Cleveland 14-attribute standard)
COLUMN_NAMES = [
    "age",       # Age in years [29 - 77]
    "sex",       # Sex (1 = male, 0 = female)
    "cp",        # Chest pain type (1 = typical angina, 2 = atypical angina, 3 = non-anginal, 4 = asymptomatic)
    "trestbps",  # Resting blood pressure in mm Hg [94 - 200]
    "chol",      # Serum cholesterol in mg/dl [126 - 564]
    "fbs",       # Fasting blood sugar > 120 mg/dl (1 = true, 0 = false)
    "restecg",   # Resting electrocardiographic results (0 = normal, 1 = ST-T wave abnormality, 2 = LV hypertrophy)
    "thalach",   # Maximum heart rate achieved [71 - 202]
    "exang",     # Exercise-induced angina (1 = yes, 0 = no)
    "oldpeak",   # ST depression induced by exercise relative to rest [0.0 - 6.2]
    "slope",     # Slope of peak exercise ST segment (1 = upsloping, 2 = flat, 3 = downsloping)
    "ca",        # Number of major vessels (0-3) colored by fluoroscopy
    "thal",      # Thallium stress test (3 = normal, 6 = fixed defect, 7 = reversible defect)
    "target",    # Diagnosis: 0 = <50% diameter narrowing (absence), 1-4 = >50% narrowing (presence)
]

NUMERICAL_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CATEGORICAL_FEATURES = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
TARGET_COLUMN = "target"

# Reproducibility & Splitting
RANDOM_SEED = 42
TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
TEST_RATIO = 0.10

# Model Hyperparameters (Optimized for medical tabular reliability)
MODEL_PARAMS = {
    "n_estimators": 150,
    "max_depth": 5,
    "min_samples_split": 4,
    "min_samples_leaf": 2,
    "class_weight": "balanced",
    "random_state": RANDOM_SEED,
    "n_jobs": -1,
}

# Clinical Risk Thresholds
RISK_THRESHOLDS = {
    "low": 0.35,
    "moderate": 0.65,
}
