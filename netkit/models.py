"""Pool members.

Architecture is one of the axes the pool randomises over.  A committee whose
members share a backbone can still disagree, but disagreement bought from
initialisation and augmentation alone tends to be shallow; different families
fail on different images.
"""
from __future__ import annotations

import timm
import torch
import torch.nn as nn


ARCHITECTURES = [
    "mobilenetv3_small_100",
    "efficientnet_b0",
    "resnet18",
    "convnext_atto",
    "regnety_004",
]


class PoolMember(nn.Module):
    """A timm backbone with a fresh linear head.

    The head is separate from the backbone's own classifier so that the number
    of classes is set here rather than by the checkpoint, and so that a member
    can later be probed for features without touching the head.
    """

    def __init__(self, arch, num_classes, pretrained=True, drop_rate=0.0,
                 image_size=224):
        super().__init__()
        if num_classes < 2:
            raise ValueError("num_classes must be at least 2")
        self.arch = arch
        self.num_classes = num_classes
        self.image_size = image_size
        self.backbone = timm.create_model(
            arch, pretrained=pretrained, num_classes=0, drop_rate=drop_rate)
        # num_features is not the width the pooled backbone actually emits for
        # every family (MobileNetV3 reports 960 but returns 1280 after its
        # conv_head), so measure it instead of trusting the attribute
        self.head = nn.Linear(self._feature_dim(image_size), num_classes)

    def _feature_dim(self, image_size):
        was_training = self.backbone.training
        self.backbone.eval()
        with torch.no_grad():
            out = self.backbone(torch.zeros(1, 3, image_size, image_size))
        if was_training:
            self.backbone.train()
        return int(out.shape[1])

    def forward(self, x):
        return self.head(self.backbone(x))

    @torch.no_grad()
    def probabilities(self, x):
        """Softmax outputs."""
        return torch.softmax(self.forward(x), dim=1)


def build(arch, num_classes, pretrained=True, drop_rate=0.0, device=None,
          image_size=224):
    m = PoolMember(arch, num_classes, pretrained, drop_rate, image_size)
    return m.to(device) if device is not None else m


def load_member(path, device=None, map_location="cpu"):
    """Rebuild a stored member from its checkpoint, architecture included."""
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    m = PoolMember(ckpt["arch"], ckpt["num_classes"], pretrained=False,
                   image_size=ckpt.get("image_size", 224))
    m.load_state_dict(ckpt["state_dict"])
    m.eval()
    return m.to(device) if device is not None else m


def save_member(model, path, extra=None):
    payload = {
        "arch": model.arch,
        "num_classes": model.num_classes,
        "image_size": model.image_size,
        "state_dict": model.state_dict(),
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)
