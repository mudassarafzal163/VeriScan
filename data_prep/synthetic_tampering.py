"""Synthetic image-tampering generator (for training/evaluating the forensic
manipulation detector).

Why synthetic: public tampering benchmarks (CASIA v2, Columbia, Coverage,
IMD2020) could not be downloaded in the build environment. Following the
standard practice of those datasets, tampered images are created from
authentic photos with three operations:
  * splice     - region from a *different* photo with a different
                 compression/noise history pasted in (feathered mask)
  * copy-move  - region duplicated inside the same photo (optionally rescaled)
  * retouch    - local blur / denoise / noise injection (object removal style)
Authentic counterparts receive the same global processing (random resize and
final JPEG re-save at the same quality range), so that 'was re-compressed'
cannot be used as a shortcut; only *local inconsistency* separates classes.
Limitations: synthetic edits are cruder than skilled manual forgeries and the
detector will not catch every real-world manipulation.
"""
from __future__ import annotations

import io
import random

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from ml.image.forensics import limit_size, to_rgb


def _jpeg(img: Image.Image, q: int) -> Image.Image:
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=q)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def _mask(size, box, rng, feather):
    m = Image.new("L", size, 0)
    d = ImageDraw.Draw(m)
    if rng.random() < 0.5:
        d.rectangle(box, fill=255)
    else:
        d.ellipse(box, fill=255)
    return m.filter(ImageFilter.GaussianBlur(feather)) if feather else m


def _box(w, h, rng, lo=0.15, hi=0.4):
    bw, bh = int(w * rng.uniform(lo, hi)), int(h * rng.uniform(lo, hi))
    x, y = rng.randint(0, max(w - bw, 1)), rng.randint(0, max(h - bh, 1))
    return x, y, x + bw, y + bh


def _global(img, rng):
    img = limit_size(to_rgb(img), rng.choice([640, 768, 900, 1024]))
    return img


def authentic(img: Image.Image, rng: random.Random):
    img = _global(img, rng)
    if rng.random() < 0.5:  # simulate an earlier camera/app compression
        img = _jpeg(img, rng.randint(80, 98))
    return _jpeg(img, rng.randint(85, 95)), Image.new("L", img.size, 0)


def splice(img: Image.Image, donor: Image.Image, rng: random.Random) -> Image.Image:
    img = _global(img, rng)
    if rng.random() < 0.5:
        img = _jpeg(img, rng.randint(80, 98))
    w, h = img.size
    x0, y0, x1, y1 = _box(w, h, rng)
    donor = to_rgb(donor).resize((x1 - x0, y1 - y0), Image.BICUBIC)
    mode = rng.random()
    if mode < 0.4:
        donor = _jpeg(donor, rng.randint(40, 75))
    elif mode < 0.7:
        arr = np.asarray(donor, np.float32) + np.random.default_rng(rng.randint(0, 1 << 30)).normal(0, rng.uniform(3, 8), (donor.size[1], donor.size[0], 3))
        donor = Image.fromarray(arr.clip(0, 255).astype(np.uint8))
    else:
        donor = donor.filter(ImageFilter.GaussianBlur(rng.uniform(0.8, 1.8)))
    m = _mask((x1 - x0, y1 - y0), (0, 0, x1 - x0 - 1, y1 - y0 - 1), rng, rng.choice([0, 1, 2]))
    img.paste(donor, (x0, y0), m)
    full = Image.new("L", img.size, 0)
    full.paste(m, (x0, y0))
    return _jpeg(img, rng.randint(85, 95)), full


def copy_move(img: Image.Image, rng: random.Random) -> Image.Image:
    img = _global(img, rng)
    if rng.random() < 0.5:
        img = _jpeg(img, rng.randint(80, 98))
    w, h = img.size
    x0, y0, x1, y1 = _box(w, h, rng, 0.12, 0.3)
    patch = img.crop((x0, y0, x1, y1))
    s = rng.uniform(0.8, 1.25)
    patch = patch.resize((max(8, int(patch.width * s)), max(8, int(patch.height * s))), Image.BICUBIC)
    if rng.random() < 0.5:
        patch = _jpeg(patch, rng.randint(50, 80))
    tx, ty = rng.randint(0, max(w - patch.width, 1)), rng.randint(0, max(h - patch.height, 1))
    m = _mask(patch.size, (0, 0, patch.width - 1, patch.height - 1), rng, rng.choice([0, 1]))
    img.paste(patch, (tx, ty), m)
    full = Image.new("L", img.size, 0)
    full.paste(m, (tx, ty))
    return _jpeg(img, rng.randint(85, 95)), full


def retouch(img: Image.Image, rng: random.Random) -> Image.Image:
    img = _global(img, rng)
    if rng.random() < 0.5:
        img = _jpeg(img, rng.randint(80, 98))
    w, h = img.size
    box = _box(w, h, rng, 0.12, 0.35)
    region = img.crop(box)
    op = rng.random()
    if op < 0.4:
        region = region.filter(ImageFilter.GaussianBlur(rng.uniform(1.5, 4)))
    elif op < 0.7:
        region = region.filter(ImageFilter.MedianFilter(rng.choice([5, 7])))
    else:
        arr = np.asarray(region, np.float32) + np.random.default_rng(rng.randint(0, 1 << 30)).normal(0, rng.uniform(4, 10), (region.size[1], region.size[0], 3))
        region = Image.fromarray(arr.clip(0, 255).astype(np.uint8))
    m = _mask(region.size, (0, 0, region.width - 1, region.height - 1), rng, rng.choice([0, 2, 4]))
    img.paste(region, box[:2], m)
    full = Image.new("L", img.size, 0)
    full.paste(m, box[:2])
    return _jpeg(img, rng.randint(85, 95)), full


def make_variants(img, donors, rng, n_auth=2, n_tamp=2):
    """Yield (image, mask, label, op) tuples for one source photo."""
    for _ in range(n_auth):
        yield (*authentic(img, rng), 0, "authentic")
    ops = ["splice", "copy_move", "retouch"]
    for k in range(n_tamp):
        op = ops[(k + rng.randint(0, 2)) % 3]
        if op == "splice":
            yield (*splice(img, rng.choice(donors), rng), 1, op)
        elif op == "copy_move":
            yield (*copy_move(img, rng), 1, op)
        else:
            yield (*retouch(img, rng), 1, op)
