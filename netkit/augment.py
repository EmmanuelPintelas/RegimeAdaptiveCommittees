"""Augmentation setups.

A pool is useful only if its members differ: if every member sees the same view of
the data they end up making near-identical predictions.  The setups below are
therefore not interchangeable variations on one recipe, they deliberately disagree
about what a training image should look like.

The photometric operators follow the pool-generation design of GeNeX; the
geometric and occlusion setups are added because the benchmarks here are
domain-shift benchmarks rather than a single medical corpus.
"""
from __future__ import annotations

import random

import cv2
import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image, ImageEnhance

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class RandomRotation:
    """Rotate by one of a fixed set of angles, chosen uniformly."""

    def __init__(self, angles):
        self.angles = list(angles)

    def __call__(self, img):
        return TF.rotate(img, random.choice(self.angles))


class CLAHE:
    """Contrast-limited adaptive histogram equalisation on the L channel."""

    def __init__(self, clip_limit=2.0, tile_grid_size=(8, 8)):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size

    def __call__(self, img):
        if isinstance(img, torch.Tensor):
            img = TF.to_pil_image(img)
        lab = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=self.clip_limit,
                                tileGridSize=self.tile_grid_size)
        merged = cv2.merge((clahe.apply(l), a, b))
        return Image.fromarray(cv2.cvtColor(merged, cv2.COLOR_LAB2RGB))


class Sharpen:
    def __init__(self, factor=2.0):
        self.factor = factor

    def __call__(self, img):
        if isinstance(img, torch.Tensor):
            img = TF.to_pil_image(img)
        return ImageEnhance.Sharpness(img).enhance(self.factor)


class Maybe:
    """Apply an operator with probability p."""

    def __init__(self, op, p=0.5):
        self.op = op
        self.p = p

    def __call__(self, img):
        return self.op(img) if random.random() < self.p else img


def _finish(size):
    return [T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)]


def build_setups(size=224):
    """Seven training pipelines, returned as an ordered dict of name -> transform.

    They differ along axes that are known to change what a network latches onto:
    geometry, blur, sharpness, local contrast, colour, and occlusion.
    """
    resize = [T.Resize((size, size), antialias=True)]
    setups = {
        "light_geometric": T.Compose(
            resize + [RandomRotation([0, 90, 270]),
                      T.RandomHorizontalFlip()] + _finish(size)),
        "heavy_geometric": T.Compose(
            resize + [RandomRotation([0, 90, 180, 270]),
                      T.RandomHorizontalFlip(),
                      T.RandomResizedCrop(size, scale=(0.6, 1.0), antialias=True)]
            + _finish(size)),
        "blur": T.Compose(
            resize + [RandomRotation([0, 90, 270]),
                      Maybe(T.GaussianBlur((3, 3), (0.1, 2.0)), 0.5)] + _finish(size)),
        "sharpen": T.Compose(
            resize + [RandomRotation([0, 90, 270]), Maybe(Sharpen(), 0.5)]
            + _finish(size)),
        "clahe": T.Compose(
            resize + [RandomRotation([0, 90, 270]), Maybe(CLAHE(), 0.5)]
            + _finish(size)),
        "colour": T.Compose(
            resize + [T.RandomHorizontalFlip(),
                      T.ColorJitter(0.4, 0.4, 0.4, 0.1),
                      Maybe(T.RandomGrayscale(p=1.0), 0.2)] + _finish(size)),
        "occlusion": T.Compose(
            resize + [T.RandomHorizontalFlip()] + _finish(size)
            + [T.RandomErasing(p=0.5, scale=(0.02, 0.2))]),
    }
    return setups


def eval_transform(size=224):
    """Deterministic pipeline used for validation and scoring."""
    return T.Compose([T.Resize((size, size), antialias=True),
                      T.ToTensor(),
                      T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
