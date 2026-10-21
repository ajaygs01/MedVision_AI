import copy
import sys
from pathlib import Path
from fastapi import APIRouter
from app.schemas.status import ModelItemStatus, ModelsStatusResponse
from app.services.chest_xray import chest_xray_service

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

    return ModelsStatusResponse(
        stage=2,
        system_status="stage2_pipeline_active",
        total_models=len(models),
        active_trained_count=active_trained,
        models=models,
        disclaimer=(
            "Stage 2 Chest X-ray pipeline implemented. Results are academic research prototypes "
            "and must not be used as standalone clinical diagnoses."
        ),
    )
