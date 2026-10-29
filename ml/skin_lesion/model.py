import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights


def build_skin_lesion_model(num_classes: int = 7, pretrained: bool = False) -> nn.Module:
    """
    Builds an EfficientNet-B0 backbone adapted for multi-class dermatoscopy classification.
    Runs efficiently on CPU with a 7-class linear head.
    """
    if pretrained:
        weights = EfficientNet_B0_Weights.DEFAULT
        model = efficientnet_b0(weights=weights)
    else:
        model = efficientnet_b0(weights=None)

    # Replace classifier head for 7 diagnostic classes
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model
