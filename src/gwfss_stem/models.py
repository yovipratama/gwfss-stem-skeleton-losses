"""Backbone / decoder factory (segmentation_models_pytorch)."""
import segmentation_models_pytorch as smp

BACKBONES = {
    # name: (architecture class, encoder name)  -- all ImageNet-pretrained encoders
    "deeplabv3plus_r101": (smp.DeepLabV3Plus, "resnet101"),
    "segformer_b1": (smp.Segformer, "mit_b1"),
    "segformer_b2": (smp.Segformer, "mit_b2"),
    "upernet_convnext_t": (smp.UPerNet, "tu-convnext_tiny"),
}


def build_model(name, num_classes=4, pretrained=True):
    arch, encoder = BACKBONES[name]
    return arch(encoder_name=encoder, encoder_weights="imagenet" if pretrained else None, classes=num_classes)
