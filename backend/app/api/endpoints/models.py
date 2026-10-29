import copy
import sys
from pathlib import Path
from fastapi import APIRouter
from app.schemas.status import ModelItemStatus, ModelsStatusResponse
from app.services.brain_mri import brain_mri_service
from app.services.chest_xray import chest_xray_service
from app.services.clinical_ml import clinical_ml_service
from app.services.skin_lesion import skin_lesion_service
from app.services.xai import xai_service
from app.services.yolo import yolo_service

router = APIRouter(tags=["Model Status"])

BASE_MODELS = {
    "cnn_chest_xray": ModelItemStatus(
        name="Chest X-ray ResNet50 Classifier",
        modality="Chest Radiograph (CXR - DICOM / Image)",
        architecture="ResNet-50 ImageNet Transfer Learning",
        framework="PyTorch 2.x / Torchvision",
        status="unconfigured",
        trained=False,
        weights_path=None,
        next_stage="Stage 2 Training with RSNA Pneumonia Detection Dataset",
        description="Convolutional neural network for binary detection of pneumonia and lung opacity on chest radiographs.",
    ),
    "unet_brain_mri": ModelItemStatus(
        name="Brain MRI Lesion Segmenter",
        modality="Brain MRI (FLAIR / T1ce / T2)",
        architecture="Attention U-Net / 3D U-Net",
        framework="PyTorch / MONAI",
        status="unconfigured",
        trained=False,
        weights_path=None,
        next_stage="Stage 2 Slice Extraction & Normalization (BraTS Dataset)",
        description="Encoder-decoder segmentation architecture with skip connections for volumetric brain tumor and lesion delineation.",
    ),
    "cnn_skin_lesion": ModelItemStatus(
        name="Skin Lesion Multi-Class Classifier",
        modality="Dermoscopy / Epiluminescence",
        architecture="EfficientNet-B4 / Vision Transformer (ViT)",
        framework="PyTorch / Timm",
        status="unconfigured",
        trained=False,
        weights_path=None,
        next_stage="Stage 2 Lesion Color Augmentation & Preprocessing (ISIC Archive)",
        description="High-resolution dermatological classifier for melanoma, basal cell carcinoma, and benign keratosis.",
    ),
    "yolo_detection": ModelItemStatus(
        name="Anatomical Lesion & Foreign Object Detector",
        modality="Multi-Modal Medical Imaging",
        architecture="YOLOv8 / YOLOv11 Medical Variant",
        framework="Ultralytics PyTorch",
        status="unconfigured",
        trained=False,
        weights_path=None,
        next_stage="Stage 3 Bounding Box Annotation & Anchor Optimization",
        description="Real-time single-stage object detector providing spatial bounding boxes and anatomical region of interest localization.",
    ),
    "clinical_ml": ModelItemStatus(
        name="Clinical Tabular Risk Predictor",
        modality="EHR & Tabular Patient Metrics",
        architecture="XGBoost / LightGBM / Random Forest Ensemble",
        framework="Scikit-Learn / XGBoost",
        status="unconfigured",
        trained=False,
        weights_path=None,
        next_stage="Stage 3 Tabular Imputation & Feature Engineering (MIMIC / UCI)",
        description="Gradient-boosted decision trees for cardiovascular risk stratification, sepsis onset, and patient readmission.",
    ),
    "xai_engine": ModelItemStatus(
        name="Explainable AI & Attribution Engine",
        modality="Interpretable AI Attribution",
        architecture="Grad-CAM, Score-CAM, SHAP & Integrated Gradients",
        framework="Captum / SHAP / PyTorch",
        status="unconfigured",
        trained=False,
        weights_path=None,
        next_stage="Stage 4 Feature Attribution Layer Hooks & Heatmap Generation",
        description="Visual and mathematical interpretability framework providing saliency heatmaps and feature contribution weights.",
    ),
    "rag_assistant": ModelItemStatus(
        name="Medical Research & Literature Assistant",
        modality="Biomedical Literature & Clinical Guidelines",
        architecture="Retrieval-Augmented Generation (Vector Store + BioBERT + LLM)",
        framework="LangChain / LlamaIndex / FAISS",
        status="unconfigured",
        trained=False,
        weights_path=None,
        next_stage="Stage 4 PubMed & Clinical Trial Knowledge Base Indexing",
        description="Specialized conversational research copilot grounded in peer-reviewed biomedical literature and evidence-based protocols.",
    ),
}


@router.get("/models/status", response_model=ModelsStatusResponse)
async def get_models_status():
    models = copy.deepcopy(BASE_MODELS)
    active_trained = 0

    # Dynamically inspect Chest X-ray CNN checkpoint status
    if chest_xray_service.is_ready() or chest_xray_service.is_model_trained():
        models["cnn_chest_xray"].status = "ready"
        models["cnn_chest_xray"].trained = True
        models["cnn_chest_xray"].weights_path = str(chest_xray_service.checkpoint_path)
        active_trained += 1
    else:
        models["cnn_chest_xray"].status = "unconfigured"
        models["cnn_chest_xray"].trained = False
        models["cnn_chest_xray"].weights_path = None

    # Dynamically inspect Brain MRI 3D U-Net status
    if brain_mri_service.is_ready():
        models["unet_brain_mri"].status = "ready"
        models["unet_brain_mri"].trained = True
        models["unet_brain_mri"].weights_path = str(brain_mri_service.checkpoint_path)
        active_trained += 1
    else:
        mri_stat = brain_mri_service.get_status().get("status", "unconfigured")
        models["unet_brain_mri"].status = mri_stat
        models["unet_brain_mri"].trained = False
        models["unet_brain_mri"].weights_path = None

    # Dynamically inspect Clinical ML Random Forest model status
    if clinical_ml_service.is_ready():
        models["clinical_ml"].status = "ready"
        models["clinical_ml"].trained = True
        models["clinical_ml"].weights_path = "backend/model_artifacts/clinical_ml/random_forest_risk_model.joblib"
        active_trained += 1
    else:
        models["clinical_ml"].status = "unconfigured"
        models["clinical_ml"].trained = False
        models["clinical_ml"].weights_path = None

    # Dynamically inspect Explainable AI (XAI) Grad-CAM engine status
    if xai_service.is_ready():
        models["xai_engine"].status = "ready"
        models["xai_engine"].trained = True
        models["xai_engine"].weights_path = "ResNet-50 Layer4 Bottleneck Hooks"
        active_trained += 1
    else:
        models["xai_engine"].status = "unconfigured"
        models["xai_engine"].trained = False
        models["xai_engine"].weights_path = None

    # Dynamically inspect YOLO Lesion Detection status
    if yolo_service.is_ready():
        models["yolo_detection"].status = "ready"
        models["yolo_detection"].trained = True
        models["yolo_detection"].weights_path = str(yolo_service.checkpoint_path)
        active_trained += 1
    else:
        models["yolo_detection"].status = "unconfigured"
        models["yolo_detection"].trained = False
        models["yolo_detection"].weights_path = None

    # Dynamically inspect Skin Lesion Classifier status
    if skin_lesion_service.is_ready():
        models["cnn_skin_lesion"].status = "ready"
        models["cnn_skin_lesion"].trained = True
        models["cnn_skin_lesion"].weights_path = str(skin_lesion_service.checkpoint_path)
        active_trained += 1
    else:
        models["cnn_skin_lesion"].status = "unconfigured"
        models["cnn_skin_lesion"].trained = False
        models["cnn_skin_lesion"].weights_path = None

    return ModelsStatusResponse(
        stage=2,
        system_status="stage2_pipeline_active",
        total_models=len(models),
        active_trained_count=active_trained,
        models=models,
        disclaimer=(
            "Stage 2 Chest X-ray and Clinical ML pipelines active with XAI Grad-CAM attribution. "
            "Results are academic research prototypes and must not be used as standalone clinical diagnoses."
        ),
    )
