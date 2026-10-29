import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services.clinical_ml import clinical_ml_service


@pytest.fixture
def client():
    return TestClient(app)


def test_clinical_ml_service_is_ready():
    assert clinical_ml_service.is_ready() is True
    info = clinical_ml_service.get_model_info()
    assert info["status"] == "ready"
    assert "UCI Cleveland" in info["dataset_name"]
    assert "accuracy" in info["metrics"]
    assert info["metrics"]["accuracy"] > 0.80


def test_clinical_ml_model_info_endpoint(client):
    response = client.get("/api/clinical-ml/model-info")
    assert response.status_code == 200
    data = response.json()
    assert data["model_name"] == "Clinical Tabular Risk Predictor (Random Forest)" or "Random Forest" in data["model_name"]
    assert data["status"] == "ready"
    assert "metrics" in data
    assert "roc_auc" in data["metrics"]
    assert len(data["top_features"]) > 0


def test_clinical_ml_predict_low_risk(client):
    # Typical asymptomatic / healthy profile
    patient_payload = {
        "age": 35.0,
        "sex": 0,
        "cp": 1,
        "trestbps": 115.0,
        "chol": 180.0,
        "fbs": 0,
        "restecg": 0,
        "thalach": 165.0,
        "exang": 0,
        "oldpeak": 0.0,
        "slope": 1,
        "ca": 0.0,
        "thal": 3.0,
        "patient_id": "TEST_HEALTHY_01",
    }
    response = client.post("/api/clinical-ml/predict", json=patient_payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["risk_probability"] is not None
    assert data["risk_probability"] < 0.50
    assert data["predicted_class_id"] == 0
    assert "Low" in data["risk_tier"]


def test_clinical_ml_predict_high_risk(client):
    # Patient with multiple cardiovascular risk indicators
    patient_payload = {
        "age": 67.0,
        "sex": 1,
        "cp": 4, # asymptomatic chest pain (highest risk presentation in Cleveland)
        "trestbps": 160.0, # Stage 2 hypertension
        "chol": 286.0, # Hypercholesterolemia
        "fbs": 1,
        "restecg": 2, # LV hypertrophy
        "thalach": 108.0, # Chronotropic incompetence
        "exang": 1, # Angina on exertion
        "oldpeak": 2.6, # Marked ST depression
        "slope": 2, # Flat slope
        "ca": 3.0, # 3 vessels colored
        "thal": 7.0, # Reversible defect
        "patient_id": "TEST_HIGH_RISK_02",
    }
    response = client.post("/api/clinical-ml/predict", json=patient_payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["risk_probability"] is not None
    assert data["risk_probability"] >= 0.50
    assert data["predicted_class_id"] == 1
    assert data["risk_tier"] in ["Moderate Risk", "Elevated Risk"]
    assert len(data["top_contributing_factors"]) > 0


def test_clinical_ml_predict_invalid_data(client):
    # Missing required fields
    response = client.post("/api/clinical-ml/predict", json={"age": 55})
    assert response.status_code == 422


def test_clinical_ml_dataset_stratified_split():
    from ml.clinical_ml.dataset import load_raw_dataset, split_dataset
    df = load_raw_dataset()
    assert len(df) == 303
    assert "binary_target" in df.columns

    train_df, val_df, test_df = split_dataset(df)
    assert len(train_df) == 242
    assert len(val_df) == 30
    assert len(test_df) == 31

    # Verify stratified balance is preserved across splits
    overall_pos_rate = df["binary_target"].mean()
    train_pos_rate = train_df["binary_target"].mean()
    val_pos_rate = val_df["binary_target"].mean()
    test_pos_rate = test_df["binary_target"].mean()

    assert abs(overall_pos_rate - train_pos_rate) < 0.05
    assert abs(overall_pos_rate - val_pos_rate) < 0.10
    assert abs(overall_pos_rate - test_pos_rate) < 0.10


def test_clinical_ml_preprocessor_pipeline():
    from ml.clinical_ml.dataset import build_preprocessor, get_train_val_test_data
    data = get_train_val_test_data()
    X_train = data["X_train"]
    preprocessor = build_preprocessor()

    # Fit only on train to prevent leakage
    preprocessor.fit(X_train)
    transformed = preprocessor.transform(X_train)
    assert transformed.shape[0] == len(X_train)
    assert transformed.shape[1] > len(data["feature_cols"])  # One-hot expansion


def test_clinical_ml_history_persistence_and_report(client):
    # Execute valid inference
    payload = {
        "age": 58.0,
        "sex": 1,
        "cp": 3,
        "trestbps": 140.0,
        "chol": 211.0,
        "fbs": 1,
        "restecg": 0,
        "thalach": 165.0,
        "exang": 0,
        "oldpeak": 1.8,
        "slope": 2,
        "ca": 0.0,
        "thal": 3.0,
        "patient_id": "TEST_PERSIST_99",
    }
    response = client.post("/api/clinical-ml/predict", json=payload)
    assert response.status_code == 200
    res_data = response.json()
    assert res_data["status"] == "success"

    # Test direct PDF report synthesis for clinical ML
    report_payload = {
        "modality": "clinical_ml",
        "model_name": res_data["model_name"],
        "filename": "patient_TEST_PERSIST_99.json",
        "file_type": "tabular_ehr",
        "prediction": res_data["prediction"],
        "predicted_class_id": res_data["predicted_class_id"],
        "confidence": res_data["risk_probability"],
        "risk_probability": res_data["risk_probability"],
        "risk_tier": res_data["risk_tier"],
        "top_contributing_factors": res_data["top_contributing_factors"],
        "patient_data": payload,
        "status": "success",
    }
    rep_res = client.post("/api/reports/generate", json=report_payload)
    assert rep_res.status_code == 200
    assert rep_res.headers["content-type"] == "application/pdf"
    assert rep_res.content.startswith(b"%PDF-")
