import base64
import io
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    HRFlowable,
    Image as RLImage,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)
from PIL import Image as PILImage

logger = logging.getLogger("medivision.services.reports")

PRIMARY_COLOR = colors.HexColor("#014BAA")
DARK_NEUTRAL = colors.HexColor("#1A1512")
MUTED_TEXT = colors.HexColor("#61554E")
LIGHT_BG = colors.HexColor("#F8F3F0")
BORDER_COLOR = colors.HexColor("#DFD7D0")
AMBER_COLOR = colors.HexColor("#D97706")
EMERALD_COLOR = colors.HexColor("#059669")
ROSE_COLOR = colors.HexColor("#E11D48")


def _decode_base64_image(b64_str: Optional[str], max_width: float = 2.7 * inch, max_height: float = 2.7 * inch) -> Optional[RLImage]:
    """
    Decodes a base64 data URL or raw base64 string into a ReportLab Flowable Image,
    maintaining aspect ratio within bounding limits.
    """
    if not b64_str or not isinstance(b64_str, str):
        return None
    try:
        clean_b64 = b64_str.split(",", 1)[1] if "," in b64_str else b64_str
        image_bytes = base64.b64decode(clean_b64)
        pil_img = PILImage.open(io.BytesIO(image_bytes))
        w, h = pil_img.size
        if w <= 0 or h <= 0:
            return None

        aspect = float(h) / float(w)
        target_w = max_width
        target_h = target_w * aspect
        if target_h > max_height:
            target_h = max_height
            target_w = target_h / aspect

        return RLImage(io.BytesIO(image_bytes), width=target_w, height=target_h)
    except Exception as exc:
        logger.warning(f"Failed to parse base64 image for PDF report: {exc}")
        return None


def generate_analysis_pdf(record_id: str, record_data: Dict[str, Any]) -> bytes:
    """
    Generates a valid, publication-grade clinical research audit PDF report for
    either a Chest X-ray (CNN / Grad-CAM) or Clinical ML (Tabular EHR) case.
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        leftMargin=36,
        rightMargin=36,
        topMargin=36,
        bottomMargin=36,
    )

    styles = getSampleStyleSheet()

    # Custom typography styles
    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=18,
        leading=22,
        textColor=PRIMARY_COLOR,
    )
    subtitle_style = ParagraphStyle(
        "ReportSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=12,
        textColor=MUTED_TEXT,
    )
    section_heading = ParagraphStyle(
        "SectionHeading",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=11,
        leading=14,
        textColor=PRIMARY_COLOR,
        spaceAfter=4,
    )
    meta_key_style = ParagraphStyle(
        "MetaKey",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8,
        leading=10,
        textColor=DARK_NEUTRAL,
    )
    meta_val_style = ParagraphStyle(
        "MetaVal",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8,
        leading=10,
        textColor=DARK_NEUTRAL,
    )
    meta_val_mono = ParagraphStyle(
        "MetaValMono",
        parent=styles["Normal"],
        fontName="Courier",
        fontSize=7.5,
        leading=9,
        textColor=PRIMARY_COLOR,
    )
    table_cell_style = ParagraphStyle(
        "TableCell",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8,
        leading=10,
        textColor=DARK_NEUTRAL,
    )
    disclaimer_style = ParagraphStyle(
        "Disclaimer",
        parent=styles["Normal"],
        fontName="Helvetica-Oblique",
        fontSize=7.5,
        leading=10,
        textColor=DARK_NEUTRAL,
    )

    story = []

    # 1. Header Banner with Official MediVision Logo
    logo_path = Path(__file__).resolve().parents[1] / "static" / "logo.png"
    logo_flowable = None
    if logo_path.exists():
        try:
            logo_flowable = RLImage(str(logo_path), width=0.55 * inch, height=0.55 * inch)
        except Exception:
            logo_flowable = None

    if logo_flowable:
        header_table_data = [
            [
                logo_flowable,
                Paragraph("<b>MediVision AI Clinical Research Platform</b><br/><font size='8' color='#61554E'>Automated Diagnostic Evaluation & Saliency Attribution Audit Report</font>", title_style),
                Paragraph(f"<b>REPORT ID:</b> {record_id[:16]}<br/><b>DATE:</b> {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}<br/><font color='#E11D48'><b>RESEARCH USE ONLY</b></font>", meta_key_style),
            ],
        ]
        header_table = Table(header_table_data, colWidths=[0.7 * inch, 4.3 * inch, 2.5 * inch])
        header_table.setStyle(TableStyle([
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("ALIGN", (2, 0), (2, -1), "RIGHT"),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
        ]))
    else:
        header_table_data = [
            [
                Paragraph("<b>MediVision AI Clinical Research Platform</b>", title_style),
                Paragraph(f"<b>REPORT ID:</b> {record_id[:16]}<br/><b>DATE:</b> {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}", meta_key_style),
            ],
            [
                Paragraph("Automated Diagnostic Evaluation & Saliency Attribution Audit Report", subtitle_style),
                Paragraph("CONFIDENTIAL • RESEARCH USE ONLY", ParagraphStyle("RedAlert", parent=meta_key_style, textColor=ROSE_COLOR)),
            ]
        ]
        header_table = Table(header_table_data, colWidths=[5.0 * inch, 2.5 * inch])
        header_table.setStyle(TableStyle([
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("ALIGN", (1, 0), (1, -1), "RIGHT"),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 1),
        ]))
    story.append(header_table)
    story.append(Spacer(1, 6))
    story.append(HRFlowable(width="100%", thickness=1.5, color=PRIMARY_COLOR, spaceAfter=8))

    # 2. Case Telemetry & Parameters Table
    modality = record_data.get("modality", "chest_xray")
    model_name = record_data.get("model_name", "MediVision AI Engine")
    filename = record_data.get("filename", "unknown_study")
    timestamp = record_data.get("timestamp", datetime.now(timezone.utc).isoformat())
    view_mode = record_data.get("view_mode")

    modality_labels = {
        "chest_xray": "Chest Radiography (CXR)",
        "clinical_ml": "Clinical Tabular EHR",
        "brain_mri": "Brain MRI Volumetric Segmentation",
        "skin_lesion": "Dermoscopy Epiluminescence Classifier",
        "yolo_detection": "Anatomical & Foreign Object YOLO Localization",
        "research_assistant": "Medical Literature & Research Assistant",
    }
    modality_label = modality_labels.get(modality, modality.replace("_", " ").title())

    meta_grid = [
        [
            Paragraph("<b>Analysis Record ID:</b>", meta_key_style),
            Paragraph(record_id, meta_val_mono),
            Paragraph("<b>Modality:</b>", meta_key_style),
            Paragraph(modality_label, meta_val_style),
        ],
        [
            Paragraph("<b>Analyzed File / Entity:</b>", meta_key_style),
            Paragraph(filename, meta_val_mono),
            Paragraph("<b>Inference Model:</b>", meta_key_style),
            Paragraph(model_name, meta_val_style),
        ],
        [
            Paragraph("<b>Study Timestamp:</b>", meta_key_style),
            Paragraph(timestamp, meta_val_style),
            Paragraph("<b>Pipeline Status:</b>", meta_key_style),
            Paragraph(str(record_data.get("status", "success")).upper(), meta_val_style),
        ],
    ]
    if view_mode:
        meta_grid.append([
            Paragraph("<b>Attribution View Mode:</b>", meta_key_style),
            Paragraph(str(view_mode).upper(), meta_val_style),
            Paragraph("<b>Target Convolution:</b>", meta_key_style),
            Paragraph("ResNet-50 layer4 (Bottleneck Conv)", meta_val_style),
        ])

    meta_table = Table(meta_grid, colWidths=[1.5 * inch, 2.3 * inch, 1.4 * inch, 2.3 * inch])
    meta_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), LIGHT_BG),
        ("BOX", (0, 0), (-1, -1), 0.5, BORDER_COLOR),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, BORDER_COLOR),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 10))

    # 3. Model Findings & Diagnostic Summary
    story.append(Paragraph("Diagnostic Inference Findings", section_heading))

    if modality == "chest_xray":
        prediction = record_data.get("prediction", "Classified")
        confidence = record_data.get("confidence")
        class_id = record_data.get("predicted_class_id")
        threshold = record_data.get("threshold", 0.5)

        conf_str = f"{confidence * 100:.2f}%" if confidence is not None else "N/A"
        class_color = ROSE_COLOR if class_id == 1 else EMERALD_COLOR

        findings_data = [
            [
                Paragraph("<b>Predicted Classification:</b>", meta_key_style),
                Paragraph(f"<b><font color='{class_color.hexval()}'>{prediction}</font></b>", meta_val_style),
                Paragraph("<b>Model Confidence:</b>", meta_key_style),
                Paragraph(f"<b>{conf_str}</b> (Sigmoid Probability)", meta_val_style),
            ],
            [
                Paragraph("<b>Target Pathology Class:</b>", meta_key_style),
                Paragraph(f"Class {class_id} ({'Abnormal / Opacity' if class_id == 1 else 'Normal'})", meta_val_style),
                Paragraph("<b>Operating Threshold:</b>", meta_key_style),
                Paragraph(f"{threshold} (Binary Cutoff)", meta_val_style),
            ]
        ]
        if record_data.get("explanation"):
            findings_data.append([
                Paragraph("<b>Gradient Attribution:</b>", meta_key_style),
                Paragraph(str(record_data.get("explanation")), meta_val_style),
                Paragraph("<b>Attribution Hook:</b>", meta_key_style),
                Paragraph("layer4 ReLU feature flow", meta_val_style),
            ])
    elif modality == "clinical_ml":
        prediction = record_data.get("prediction", "Risk Evaluated")
        risk_tier = record_data.get("risk_tier", "Unstratified")
        risk_prob = record_data.get("risk_probability") or record_data.get("confidence")
        prob_str = f"{risk_prob * 100:.2f}%" if risk_prob is not None else "N/A"

        tier_color = ROSE_COLOR if "Elevated" in str(risk_tier) else (AMBER_COLOR if "Moderate" in str(risk_tier) else EMERALD_COLOR)

        findings_data = [
            [
                Paragraph("<b>Cardiovascular Risk Tier:</b>", meta_key_style),
                Paragraph(f"<b><font color='{tier_color.hexval()}'>{risk_tier}</font></b>", meta_val_style),
                Paragraph("<b>Calculated Risk Score:</b>", meta_key_style),
                Paragraph(f"<b>{prob_str}</b> (Random Forest Ensemble)", meta_val_style),
            ],
            [
                Paragraph("<b>Diagnostic Summary:</b>", meta_key_style),
                Paragraph(prediction, meta_val_style),
                Paragraph("<b>Ensemble Specification:</b>", meta_key_style),
                Paragraph("150 Decision Trees, Balanced Weights", meta_val_style),
            ]
        ]
    elif modality == "brain_mri":
        prediction = record_data.get("prediction", "Volumetric Brain Lesion Analysis")
        vol_metrics = record_data.get("volumetric_metrics") or {}
        wt_cm3 = vol_metrics.get("whole_tumor_cm3", "N/A")
        et_cm3 = vol_metrics.get("enhancing_tumor_cm3", "N/A")

        findings_data = [
            [
                Paragraph("<b>Dominant Finding:</b>", meta_key_style),
                Paragraph(prediction, meta_val_style),
                Paragraph("<b>Whole Tumor Volume (WT):</b>", meta_key_style),
                Paragraph(f"<b>{wt_cm3} cm³</b> (NCR + ED + ET)", meta_val_style),
            ],
            [
                Paragraph("<b>Active Enhancing Core (ET):</b>", meta_key_style),
                Paragraph(f"<b>{et_cm3} cm³</b> (Active Gadolinium)", meta_val_style),
                Paragraph("<b>Segmentation Architecture:</b>", meta_key_style),
                Paragraph("3D U-Net (BraTS 4-Channel mpMRI)", meta_val_style),
            ]
        ]
        if record_data.get("is_experimental") or "Experimental" in str(record_data.get("model_name", "")):
            findings_data.append([
                Paragraph("<b>Model Prototype Class:</b>", meta_key_style),
                Paragraph("<font color='#B45309'><b>Experimental Prototype (10p / 3e)</b></font>", meta_val_style),
                Paragraph("<b>Clinical Validation Status:</b>", meta_key_style),
                Paragraph("<font color='#B45309'><b>NOT Clinically Validated</b></font>", meta_val_style),
            ])
    elif modality == "research_assistant":
        query = record_data.get("query") or record_data.get("filename", "Clinical Research Query")
        prediction = record_data.get("prediction") or "Evidence Synthesized"
        findings_data = [
            [
                Paragraph("<b>Literature Query:</b>", meta_key_style),
                Paragraph(query, meta_val_style),
                Paragraph("<b>Retrieval Method:</b>", meta_key_style),
                Paragraph("Dense Vector RAG + BioLinkBERT", meta_val_style),
            ],
            [
                Paragraph("<b>Synthesis Status:</b>", meta_key_style),
                Paragraph(prediction, meta_val_style),
                Paragraph("<b>Grounding Safeguard:</b>", meta_key_style),
                Paragraph("Constrained Citation Enforcement", meta_val_style),
            ]
        ]
    else:
        # Modality fallback for brain_mri, skin_lesion, yolo_detection
        prediction = record_data.get("prediction", "Analysis Evaluated")
        confidence = record_data.get("confidence")
        conf_str = f"{confidence * 100:.2f}%" if confidence is not None else "Evaluated"
        findings_data = [
            [
                Paragraph("<b>Modality Evaluation:</b>", meta_key_style),
                Paragraph(prediction, meta_val_style),
                Paragraph("<b>Evaluation Score:</b>", meta_key_style),
                Paragraph(conf_str, meta_val_style),
            ],
            [
                Paragraph("<b>Model Architecture:</b>", meta_key_style),
                Paragraph(model_name, meta_val_style),
                Paragraph("<b>Dataset Source:</b>", meta_key_style),
                Paragraph(record_data.get("dataset", "Certified Ground Truth Cohort"), meta_val_style),
            ]
        ]

    findings_table = Table(findings_data, colWidths=[1.8 * inch, 2.0 * inch, 1.8 * inch, 1.9 * inch])
    findings_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.white),
        ("BOX", (0, 0), (-1, -1), 0.5, BORDER_COLOR),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, BORDER_COLOR),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(findings_table)
    story.append(Spacer(1, 10))

    # 4. Modality Specific Visual / Clinical Features Section
    if modality == "chest_xray":
        story.append(Paragraph("Visual Evidence & Explainable AI (Grad-CAM Saliency)", section_heading))

        orig_img_flowable = _decode_base64_image(record_data.get("image_preview"), max_width=3.4 * inch, max_height=2.8 * inch)
        gradcam_flowable = _decode_base64_image(
            record_data.get("gradcam_preview") or record_data.get("heatmap_preview"),
            max_width=3.4 * inch,
            max_height=2.8 * inch
        )

        if orig_img_flowable and gradcam_flowable:
            img_table_data = [
                [
                    Paragraph("<b>Original Chest Radiograph (CXR)</b>", meta_key_style),
                    Paragraph("<b>Grad-CAM Convolutional Attribution (layer4)</b>", meta_key_style),
                ],
                [
                    orig_img_flowable,
                    gradcam_flowable,
                ],
                [
                    Paragraph("<font size='7' color='#61554E'>Input radiograph normalized to 224x224</font>", table_cell_style),
                    Paragraph("<font size='7' color='#61554E'>Positive ReLU gradient flow overlaying anatomical focus</font>", table_cell_style),
                ]
            ]
            img_table = Table(img_table_data, colWidths=[3.7 * inch, 3.8 * inch])
            img_table.setStyle(TableStyle([
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("BOX", (0, 0), (-1, -1), 0.5, BORDER_COLOR),
                ("INNERGRID", (0, 0), (-1, -1), 0.5, BORDER_COLOR),
                ("BACKGROUND", (0, 0), (-1, 0), LIGHT_BG),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]))
            story.append(img_table)
        elif orig_img_flowable:
            img_table = Table([
                [Paragraph("<b>Original Chest Radiograph (CXR)</b>", meta_key_style)],
                [orig_img_flowable],
            ], colWidths=[7.5 * inch])
            img_table.setStyle(TableStyle([
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("BOX", (0, 0), (-1, -1), 0.5, BORDER_COLOR),
                ("BACKGROUND", (0, 0), (-1, 0), LIGHT_BG),
                ("PADDING", (0, 0), (-1, -1), 6),
            ]))
            story.append(img_table)
        else:
            story.append(Paragraph(
                "<i>Visual radiograph payload was not archived in telemetry storage for this record.</i>",
                subtitle_style
            ))
    elif modality == "clinical_ml":
        # Clinical Tabular EHR
        story.append(Paragraph("Patient Clinical Attributes & Physiological Profile", section_heading))
        patient_data = record_data.get("patient_data") or {}

        cp_map = {1: "Typical Angina", 2: "Atypical Angina", 3: "Non-Anginal", 4: "Asymptomatic"}
        ecg_map = {0: "Normal", 1: "ST-T Wave Abnormality", 2: "LV Hypertrophy"}
        thal_map = {3: "Normal", 6: "Fixed Defect", 7: "Reversible Defect"}

        # Grid of clinical parameters
        clinical_grid = [
            [
                Paragraph("<b>Parameter</b>", meta_key_style),
                Paragraph("<b>Recorded Value</b>", meta_key_style),
                Paragraph("<b>Parameter</b>", meta_key_style),
                Paragraph("<b>Recorded Value</b>", meta_key_style),
            ],
            [
                Paragraph("Age", meta_val_style),
                Paragraph(f"{patient_data.get('age', 'N/A')} yrs", meta_val_style),
                Paragraph("Sex", meta_val_style),
                Paragraph("Male" if patient_data.get("sex") == 1 else "Female", meta_val_style),
            ],
            [
                Paragraph("Chest Pain Type (cp)", meta_val_style),
                Paragraph(cp_map.get(patient_data.get("cp"), str(patient_data.get("cp", "N/A"))), meta_val_style),
                Paragraph("Resting Blood Pressure", meta_val_style),
                Paragraph(f"{patient_data.get('trestbps', 'N/A')} mm Hg", meta_val_style),
            ],
            [
                Paragraph("Serum Cholesterol (chol)", meta_val_style),
                Paragraph(f"{patient_data.get('chol', 'N/A')} mg/dl", meta_val_style),
                Paragraph("Fasting Blood Sugar > 120", meta_val_style),
                Paragraph("Yes (>120)" if patient_data.get("fbs") == 1 else "No (<=120)", meta_val_style),
            ],
            [
                Paragraph("Resting ECG (restecg)", meta_val_style),
                Paragraph(ecg_map.get(patient_data.get("restecg"), str(patient_data.get("restecg", "N/A"))), meta_val_style),
                Paragraph("Max Heart Rate (thalach)", meta_val_style),
                Paragraph(f"{patient_data.get('thalach', 'N/A')} bpm", meta_val_style),
            ],
            [
                Paragraph("Exercise Induced Angina", meta_val_style),
                Paragraph("Yes" if patient_data.get("exang") == 1 else "No", meta_val_style),
                Paragraph("ST Depression (oldpeak)", meta_val_style),
                Paragraph(f"{patient_data.get('oldpeak', 'N/A')} mm", meta_val_style),
            ],
            [
                Paragraph("ST Slope", meta_val_style),
                Paragraph(f"Type {patient_data.get('slope', 'N/A')}", meta_val_style),
                Paragraph("Fluoroscopy Vessels (ca)", meta_val_style),
                Paragraph(f"{patient_data.get('ca', 'N/A')} colored", meta_val_style),
            ],
            [
                Paragraph("Thallium Stress Scintigraphy", meta_val_style),
                Paragraph(thal_map.get(patient_data.get("thal"), str(patient_data.get("thal", "N/A"))), meta_val_style),
                Paragraph("Patient Entity ID", meta_val_style),
                Paragraph(str(patient_data.get("patient_id", "Anonymous")), meta_val_style),
            ],
        ]
        clin_table = Table(clinical_grid, colWidths=[2.0 * inch, 1.75 * inch, 2.0 * inch, 1.75 * inch])
        clin_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), LIGHT_BG),
            ("BOX", (0, 0), (-1, -1), 0.5, BORDER_COLOR),
            ("INNERGRID", (0, 0), (-1, -1), 0.5, BORDER_COLOR),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ("LEFTPADDING", (0, 0), (-1, -1), 5),
            ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ]))
        story.append(clin_table)

        # Contributing physiological risk factors
        top_factors = record_data.get("top_contributing_factors") or []
        if top_factors:
            story.append(Spacer(1, 6))
            factors_p = [Paragraph("<b>Top Patient-Specific Contributing Risk Factors:</b>", meta_key_style)]
            for tf in top_factors:
                factors_p.append(Paragraph(f"• <b>{tf}</b>", meta_val_style))
            factors_table = Table([[Paragraph("<br/>".join([f"• {f}" for f in top_factors]), meta_val_style)]], colWidths=[7.5 * inch])
            factors_table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#FEF3C7")),
                ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#F59E0B")),
                ("PADDING", (0, 0), (-1, -1), 6),
            ]))
            story.append(Paragraph("<b>Key Physiological Risk Factors Detected:</b>", section_heading))
            story.append(factors_table)

    elif modality == "brain_mri":
        story.append(Paragraph("Volumetric Neuroimaging & Delineated Lesion Slices", section_heading))

        mri_flowable = _decode_base64_image(record_data.get("image_preview"), max_width=3.4 * inch, max_height=2.8 * inch)
        overlay_flowable = _decode_base64_image(
            record_data.get("gradcam_preview") or record_data.get("heatmap_preview"),
            max_width=3.4 * inch,
            max_height=2.8 * inch
        )

        slices_info = record_data.get("slices") or {}
        plane_str = str(slices_info.get("plane", "Axial")).capitalize()
        slice_num = slices_info.get("slice_idx", "Optimal")
        total_s = slices_info.get("total_slices", "")
        slice_caption = f"{plane_str} Slice {slice_num}" + (f" of {total_s}" if total_s else "")

        if mri_flowable and overlay_flowable:
            img_table_data = [
                [
                    Paragraph(f"<b>Input Brain MRI ({slice_caption})</b>", meta_key_style),
                    Paragraph("<b>3D U-Net Color-Coded Lesion Segmentation Overlay</b>", meta_key_style),
                ],
                [
                    mri_flowable,
                    overlay_flowable,
                ],
                [
                    Paragraph("<font size='7' color='#61554E'>Multi-parametric MRI co-registered volume</font>", table_cell_style),
                    Paragraph("<font size='7' color='#61554E'>Red: Necrotic Core (NCR) | Green: Edema (ED) | Gold: Enhancing Tumor (ET)</font>", table_cell_style),
                ]
            ]
            img_table = Table(img_table_data, colWidths=[3.7 * inch, 3.8 * inch])
            img_table.setStyle(TableStyle([
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("BOX", (0, 0), (-1, -1), 0.5, BORDER_COLOR),
                ("INNERGRID", (0, 0), (-1, -1), 0.5, BORDER_COLOR),
                ("BACKGROUND", (0, 0), (-1, 0), LIGHT_BG),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]))
            story.append(img_table)
        elif mri_flowable:
            img_table = Table([
                [Paragraph(f"<b>Brain MRI ({slice_caption})</b>", meta_key_style)],
                [mri_flowable],
            ], colWidths=[7.5 * inch])
            img_table.setStyle(TableStyle([
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("BOX", (0, 0), (-1, -1), 0.5, BORDER_COLOR),
                ("BACKGROUND", (0, 0), (-1, 0), LIGHT_BG),
                ("PADDING", (0, 0), (-1, -1), 6),
            ]))
            story.append(img_table)

        # Volumetric Subregion Breakdown Table
        vol_data = record_data.get("volumetric_metrics") or {}
        vol_table_data = [
            [
                Paragraph("<b>Brain Lesion Subregion</b>", meta_key_style),
                Paragraph("<b>BraTS Tissue Subtype</b>", meta_key_style),
                Paragraph("<b>Volume (cm³)</b>", meta_key_style),
                Paragraph("<b>Clinical Description</b>", meta_key_style),
            ],
            [
                Paragraph("Whole Tumor (WT)", meta_val_style),
                Paragraph("Labels 1 + 2 + 4", meta_val_style),
                Paragraph(f"{vol_data.get('whole_tumor_cm3', 'N/A')} cm³", meta_val_style),
                Paragraph("Complete abnormal tissue boundary including edema", meta_val_style),
            ],
            [
                Paragraph("Tumor Core (TC)", meta_val_style),
                Paragraph("Labels 1 + 4", meta_val_style),
                Paragraph(f"{vol_data.get('tumor_core_cm3', 'N/A')} cm³", meta_val_style),
                Paragraph("Resectable core excluding peritumoral edema", meta_val_style),
            ],
            [
                Paragraph("Enhancing Tumor (ET)", meta_val_style),
                Paragraph("Label 4 (Active)", meta_val_style),
                Paragraph(f"{vol_data.get('enhancing_tumor_cm3', 'N/A')} cm³", meta_val_style),
                Paragraph("Active breakdown of blood-brain barrier", meta_val_style),
            ],
            [
                Paragraph("Peritumoral Edema (ED)", meta_val_style),
                Paragraph("Label 2 (Infiltrative)", meta_val_style),
                Paragraph(f"{vol_data.get('edema_cm3', 'N/A')} cm³", meta_val_style),
                Paragraph("Fluid accumulation and vasogenic edema infiltration", meta_val_style),
            ],
        ]
        vol_table = Table(vol_table_data, colWidths=[2.0 * inch, 1.8 * inch, 1.4 * inch, 2.3 * inch])
        vol_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), LIGHT_BG),
            ("BOX", (0, 0), (-1, -1), 0.5, BORDER_COLOR),
            ("INNERGRID", (0, 0), (-1, -1), 0.5, BORDER_COLOR),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("LEFTPADDING", (0, 0), (-1, -1), 5),
            ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ]))
        story.append(Spacer(1, 8))
        story.append(Paragraph("Quantitative Volumetric Tumor Delineation (BraTS Standard)", section_heading))
        story.append(vol_table)

    elif modality == "research_assistant":
        story.append(Paragraph("Evidence-Based Literature Review & Retrieved Sources", section_heading))
        query_text = record_data.get("query") or record_data.get("filename", "Clinical Research Topic")
        resp_text = record_data.get("response") or record_data.get("prediction", "Retrieved and synthesized literature analysis.")
        sources = record_data.get("sources") or []

        ra_table_data = [
            [Paragraph("<b>Research Query:</b>", meta_key_style)],
            [Paragraph(query_text, meta_val_style)],
            [Paragraph("<b>Evidence Synthesis & Clinical Discussion:</b>", meta_key_style)],
            [Paragraph(resp_text, table_cell_style)],
        ]
        if sources:
            src_lines = "<br/>".join([f"• <b>[{idx+1}]</b> {src}" for idx, src in enumerate(sources)])
            ra_table_data.extend([
                [Paragraph("<b>Grounded Citations & DOI References:</b>", meta_key_style)],
                [Paragraph(src_lines, meta_val_style)],
            ])
        ra_table = Table(ra_table_data, colWidths=[7.5 * inch])
        ra_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.white),
            ("BOX", (0, 0), (-1, -1), 0.5, BORDER_COLOR),
            ("INNERGRID", (0, 0), (-1, -1), 0.5, BORDER_COLOR),
            ("PADDING", (0, 0), (-1, -1), 6),
        ]))
        story.append(ra_table)

    story.append(Spacer(1, 12))

    # 5. Regulatory Safeguard & Academic Disclaimer Box
    disclaimer_text = (
        "<b>ACADEMIC RESEARCH & CLINICAL VALIDATION PROTOCOL NOTICE:</b><br/>"
        "This diagnostic report was synthesized automatically by the MediVision AI Research Platform. "
        "All visual heatmaps, classification confidences, and clinical risk stratifications are strictly investigational outputs "
        "designed for algorithm validation and retrospective benchmarking against certified ground truth cohorts. "
        "<b>This report does NOT constitute a certified medical diagnosis, radiological interpretation, or actionable clinical directive.</b> "
        "All findings must be independently reviewed and verified by a licensed healthcare professional before any clinical decisions."
    )
    disc_table = Table([[Paragraph(disclaimer_text, disclaimer_style)]], colWidths=[7.5 * inch])
    disc_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#FFFBEB")),
        ("BOX", (0, 0), (-1, -1), 1.0, colors.HexColor("#F59E0B")),
        ("PADDING", (0, 0), (-1, -1), 8),
    ]))
    story.append(disc_table)

    # Build the document
    doc.build(story)
    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes
