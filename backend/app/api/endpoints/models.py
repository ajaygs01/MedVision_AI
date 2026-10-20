from fastapi import APIRouter
from app.schemas.status import ModelItemStatus, ModelsStatusResponse

router = APIRouter(tags=["Model Status"])

STAGE_1_MODELS = {
    "cnn_chest_xray": ModelItemStatus(
        name="Chest X-ray Multi-Abnormality Classifier",
        modality="Chest Radiograph (CXR)",
        architecture="DenseNet-121 / ResNet-50 Pretrained Backbone",
        framework="PyTorch 2.x / Torchvision",
        status="unconfigured",
        trained=False,
        weights_path=None,
        next_stage="Stage 2 Data Ingestion & Preprocessing (NIH ChestX-ray14 / CheXpert)",
        description="Deep convolutional neural network for automated screening of cardiomegaly, pneumonia, effusion, and atelectasis.",
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
    return ModelsStatusResponse(
        stage=1,
        system_status="foundation_ready",
        total_models=len(STAGE_1_MODELS),
        active_trained_count=0,
        models=STAGE_1_MODELS,
        disclaimer="Stage 1 foundation only: No clinical prediction models are trained or executing inferences.",
    )
