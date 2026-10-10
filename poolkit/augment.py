"""Named augmentation pipelines for pool members.

A member's recipe stores the name of its augmentation, and that name has to mean
the same pipeline for as long as the pool exists: it is what makes a stored
member reproducible and what lets a later analysis group members by how they
were trained.  Everything a pipeline depends on is therefore fixed in this file
or passed in explicitly from the frozen pool configuration.  There is no
module-level state that another run could have set differently.

The operators and setups reproduce those the earlier pools were built with:

    plain            resize only
    light_geometric  rotation by 0/90/270 and a horizontal flip
    heavy_geometric  rotation by 0/90/180/270, flip, random resized crop
    blur, sharpen, clahe, colour, occlusion
    heavy_all3       rotation, flip, sharpen, CLAHE, channel gain/gamma and dropout
    heavy_all4       one of: heavy_all3, a random pixel family, heavy_geometric
    channel_jitter   dihedral symmetry with channel gain/gamma and dropout
    heavy_micro      dihedral, crop, CLAHE, sharpen, channel jitter, erasing
    random_family    flip, then one pixel family at one radius drawn per image
    family:NAME@EPS  flip, then one pixel family at a fixed radius

The pixel families act on [0, 1] images between to-tensor and normalisation.
Their radius zone and mixture for heavy_all4 and random_family come from the
pool configuration ("augment": {"random_family": ...}).
"""
from __future__ import annotations

import random

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image, ImageEnhance

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# --------------------------------------------------------------------------- #
# image operators
# --------------------------------------------------------------------------- #
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
        clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)
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


class Dihedral:
    """One of the eight symmetries of the square, drawn uniformly."""

    def __call__(self, img):
        k = random.randrange(4)
        if k:
            img = TF.rotate(img, 90 * k)
        return TF.hflip(img) if random.random() < 0.5 else img


class ChannelGainGamma:
    """Per-channel gain and gamma on a [0, 1] tensor, with black kept black."""

    def __init__(self, gain=0.3, gamma=0.4):
        self.gain = float(gain)
        self.gamma = float(gamma)

    def __call__(self, x):
        shape = (x.shape[-3], 1, 1)
        g = 1.0 + self.gain * (2.0 * torch.rand(shape, dtype=x.dtype) - 1.0)
        e = torch.exp(self.gamma * (2.0 * torch.rand(shape, dtype=x.dtype) - 1.0))
        return (x.clamp_min(0.0) ** e * g).clamp_(0.0, 1.0)


class ChannelDropout:
    """Zero one channel with probability p, on a [0, 1] tensor."""

    def __init__(self, p=0.2):
        self.p = float(p)

    def __call__(self, x):
        if random.random() >= self.p:
            return x
        out = x.clone()
        out[..., random.randrange(x.shape[-3]), :, :] = 0.0
        return out


class OneOf:
    """Apply exactly one of several branches, drawn uniformly per call."""

    def __init__(self, *branches):
        if len(branches) < 2:
            raise ValueError("OneOf needs at least two branches")
        self.branches = [T.Compose(list(b)) if isinstance(b, (list, tuple)) else b
                         for b in branches]

    def __call__(self, img):
        return random.choice(self.branches)(img)


# --------------------------------------------------------------------------- #
# pixel families on [0, 1] tensors
# --------------------------------------------------------------------------- #
class _Family:
    name = ""

    def __init__(self, eps):
        self.eps = float(eps)

    def draw(self, x):
        raise NotImplementedError


class GaussianPixel(_Family):
    name = "gauss"

    def draw(self, x):
        return (x + self.eps * torch.randn_like(x)).clamp_(0.0, 1.0)


class UniformLinf(_Family):
    name = "linf"

    def draw(self, x):
        return (x + self.eps * (2.0 * torch.rand_like(x) - 1.0)).clamp_(0.0, 1.0)


class Photometric(_Family):
    """Brightness, contrast, saturation and a random amount of blur."""
    name = "photometric"

    def draw(self, x):
        b = x.shape[0]

        def factors():
            return (1.0 + self.eps * (2.0 * torch.rand(b, dtype=x.dtype) - 1.0)).view(b, 1, 1, 1)

        out = (x * factors()).clamp(0.0, 1.0)
        mean = out.mean(dim=(1, 2, 3), keepdim=True)
        out = ((out - mean) * factors() + mean).clamp(0.0, 1.0)
        grey = out.mean(dim=1, keepdim=True)
        out = ((out - grey) * factors() + grey).clamp(0.0, 1.0)
        if x.shape[-1] >= 16:
            k = torch.tensor([[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]], dtype=x.dtype)
            k = (k / k.sum()).view(1, 1, 3, 3).expand(x.shape[1], 1, 3, 3)
            blurred = F.conv2d(F.pad(out, (1, 1, 1, 1), mode="reflect"), k, groups=x.shape[1])
            mix = torch.rand(b, 1, 1, 1, dtype=x.dtype)
            out = (1.0 - mix) * out + mix * blurred
        return out.clamp(0.0, 1.0)


class StainJitter(_Family):
    """Per-channel gain and bias."""
    name = "stain"

    def draw(self, x):
        shape = (x.shape[0], x.shape[1], 1, 1)
        gain = 1.0 + self.eps * (2.0 * torch.rand(shape, dtype=x.dtype) - 1.0)
        bias = self.eps * (2.0 * torch.rand(shape, dtype=x.dtype) - 1.0)
        return (x * gain + bias).clamp(0.0, 1.0)


FAMILIES = {c.name: c for c in (GaussianPixel, UniformLinf, Photometric, StainJitter)}

DEFAULT_RANDOM_FAMILY = {
    "families": ["gauss", "linf", "photometric", "stain"],
    "ranges": {"gauss": [0.05, 0.25], "linf": [0.05, 0.25],
               "photometric": [0.40, 0.80], "stain": [0.15, 0.35]},
    "identity_prob": 0.25,
}


class PixelFamily:
    """One family at a fixed radius, applied to a single CHW image."""

    def __init__(self, name, eps):
        if name not in FAMILIES:
            raise ValueError(f"unknown pixel family {name!r}; expected one of {sorted(FAMILIES)}")
        self.family = FAMILIES[name](eps)

    def __call__(self, x):
        return self.family.draw(x.unsqueeze(0)).squeeze(0)


class RandomFamily:
    """One family at one radius drawn per call, or nothing with identity_prob."""

    def __init__(self, families, ranges, identity_prob):
        self.families = [str(f) for f in families]
        self.ranges = {f: (float(ranges[f][0]), float(ranges[f][1])) for f in self.families}
        self.identity_prob = float(identity_prob)
        for f in self.families:
            lo, hi = self.ranges[f]
            if f not in FAMILIES or not 0.0 < lo <= hi:
                raise ValueError(f"random_family: bad family or range for {f!r}")
        if not 0.0 <= self.identity_prob < 1.0:
            raise ValueError("random_family: identity_prob must be in [0, 1)")

    def __call__(self, x):
        if random.random() < self.identity_prob:
            return x
        name = random.choice(self.families)
        eps = random.uniform(*self.ranges[name])
        return FAMILIES[name](eps).draw(x.unsqueeze(0)).squeeze(0)


def _finish(pixel_op=None):
    ops = [T.ToTensor()]
    if pixel_op is not None:
        ops.append(pixel_op)
    ops.append(T.Normalize(IMAGENET_MEAN, IMAGENET_STD))
    return ops


def _random_family(params):
    cfg = dict(DEFAULT_RANDOM_FAMILY)
    cfg.update((params or {}).get("random_family") or {})
    return RandomFamily(cfg["families"], cfg["ranges"], cfg["identity_prob"])


SETUPS = ("plain", "light_geometric", "heavy_geometric", "blur", "sharpen", "clahe",
          "colour", "occlusion", "heavy_all3", "heavy_all4", "channel_jitter",
          "heavy_micro", "random_family")


def build_transform(name, size, params=None):
    """The training pipeline called ``name`` at ``size`` pixels."""
    params = params or {}
    resize = [T.Resize((size, size), antialias=True)]
    flip = ([T.RandomHorizontalFlip()]
            if params.get("family_geometry", "flip") == "flip" else [])
    rot3 = RandomRotation([0, 90, 270])

    def channel_micro():
        return T.Compose([Maybe(ChannelGainGamma(0.3, 0.4), 0.2),
                          Maybe(ChannelDropout(0.15), 0.2)])

    if name == "plain":
        ops = resize + _finish()
    elif name == "light_geometric":
        ops = resize + [rot3, T.RandomHorizontalFlip()] + _finish()
    elif name == "heavy_geometric":
        ops = resize + [RandomRotation([0, 90, 180, 270]), T.RandomHorizontalFlip(),
                        T.RandomResizedCrop(size, scale=(0.6, 1.0), antialias=True)] + _finish()
    elif name == "blur":
        ops = resize + [rot3, Maybe(T.GaussianBlur((3, 3), (0.1, 2.0)), 0.5)] + _finish()
    elif name == "sharpen":
        ops = resize + [rot3, Maybe(Sharpen(), 0.5)] + _finish()
    elif name == "clahe":
        ops = resize + [rot3, Maybe(CLAHE(), 0.5)] + _finish()
    elif name == "colour":
        ops = resize + [T.RandomHorizontalFlip(), T.ColorJitter(0.4, 0.4, 0.4, 0.1),
                        Maybe(T.RandomGrayscale(p=1.0), 0.2)] + _finish()
    elif name == "occlusion":
        ops = (resize + [T.RandomHorizontalFlip()] + _finish()
               + [T.RandomErasing(p=0.5, scale=(0.02, 0.2))])
    elif name == "heavy_all3":
        ops = (resize + [rot3, T.RandomHorizontalFlip(), Maybe(Sharpen(), 0.3), Maybe(CLAHE(), 0.3)]
               + _finish(channel_micro()))
    elif name == "heavy_all4":
        a1 = ([RandomRotation([0, 90, 270]), T.RandomHorizontalFlip(), Maybe(Sharpen(), 0.3),
               Maybe(CLAHE(), 0.3)] + _finish(channel_micro()))
        a2 = _finish(_random_family(params))
        b = ([RandomRotation([0, 90, 180, 270]), T.RandomHorizontalFlip(),
              T.RandomResizedCrop(size, scale=(0.6, 1.0), antialias=True)] + _finish())
        ops = resize + [OneOf(OneOf(a1, a2), b)]
    elif name == "channel_jitter":
        ops = (resize + [Dihedral()]
               + _finish(T.Compose([ChannelGainGamma(0.3, 0.4), ChannelDropout(0.2)])))
    elif name == "heavy_micro":
        ops = (resize + [Dihedral(), T.RandomResizedCrop(size, scale=(0.5, 1.0), antialias=True),
                         Maybe(CLAHE(), 0.5), Maybe(Sharpen(), 0.3)]
               + _finish(T.Compose([ChannelGainGamma(0.3, 0.4), ChannelDropout(0.15)]))
               + [T.RandomErasing(p=0.3, scale=(0.02, 0.2))])
    elif name == "random_family":
        ops = resize + flip + _finish(_random_family(params))
    elif isinstance(name, str) and name.startswith("family:"):
        try:
            fam, eps = name[len("family:"):].split("@")
            eps = float(eps)
        except ValueError:
            raise ValueError(f"expected family:NAME@EPS, got {name!r}") from None
        ops = resize + flip + _finish(PixelFamily(fam, eps))
    else:
        raise ValueError(f"unknown augmentation {name!r}; expected one of {SETUPS} "
                         f"or family:NAME@EPS")
    return T.Compose(ops)


def check_setup(name, params=None):
    """True if ``name`` builds and runs on a dummy image under ``params``."""
    tf = build_transform(name, 32, params)
    tf(Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8)))
    return True


def eval_transform(size):
    """The deterministic pipeline for validation and every stored prediction."""
    return T.Compose([T.Resize((size, size), antialias=True), T.ToTensor(),
                      T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
