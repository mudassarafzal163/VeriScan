"""Block-level forensic manipulation detector.

Stage 1 (block classifier): the image (native resolution) is divided into
32x32 blocks. Each block is described by
  * JPEG-ghost curve (Farid 2009): squared error after re-saving at quality
    55/65/75/85/95, raw (log) and relative to the image median. A region
    previously compressed at a different quality has its error minimum at a
    different quality than the rest of the image.
  * noise residual level (image - median filter), raw and relative
  * texture energy (gradient magnitude) so the model can discount busy areas
  * neighbourhood contrast of the ghost/noise values (edits are contiguous)
A gradient-boosting model predicts P(block is tampered). It is trained with
the pixel masks of the synthetic tampering set.
Stage 2 (image score): logistic regression on the distribution of block
probabilities (max, top-k mean, mean, std, size of the largest suspicious
connected region).
"""
from __future__ import annotations

import io

import cv2
import numpy as np
from PIL import Image

from ml.image.forensics import limit_size, to_rgb

BLOCK = 32
# Analyse at native resolution: resampling destroys the 8x8 JPEG grid that
# JPEG-ghost relies on. Only very large images are downscaled (documented
# limitation: their forensic score is less reliable).
MAX_SIDE = 1600
GHOST_Q = (55, 65, 75, 85, 95)


def _prep(img: Image.Image, mode: str = "RGB") -> Image.Image:
    """Limit size, then crop (never stretch) to a multiple of BLOCK from the
    top-left corner so blocks stay aligned with the JPEG 8x8 grid."""
    img = limit_size(to_rgb(img) if mode == "RGB" else img.convert(mode), MAX_SIDE)
    w, h = img.size
    if w < BLOCK or h < BLOCK:
        return img.resize((max(w, BLOCK), max(h, BLOCK)))
    return img.crop((0, 0, w - w % BLOCK, h - h % BLOCK))


def _block_mean(m: np.ndarray, nby: int, nbx: int) -> np.ndarray:
    return m[:nby * BLOCK, :nbx * BLOCK].reshape(nby, BLOCK, nbx, BLOCK).mean(axis=(1, 3))


def _neigh_contrast(v: np.ndarray) -> np.ndarray:
    k = np.ones((3, 3), np.float32) / 8
    k[1, 1] = 0
    nb = cv2.filter2D(v.astype(np.float32), -1, k, borderType=cv2.BORDER_REFLECT)
    return v - nb


def block_features(img: Image.Image):
    """Return (features [nby*nbx, d], (nby, nbx), prepared image)."""
    img = _prep(img)
    arr = np.asarray(img, np.float32)
    gray = arr.mean(axis=2)
    h, w = gray.shape
    nby, nbx = h // BLOCK, w // BLOCK
    feats = []
    ghosts = []
    for q in GHOST_Q:
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=q)
        buf.seek(0)
        re = np.asarray(Image.open(buf).convert("RGB"), np.float32)
        ghosts.append(_block_mean(((arr - re) ** 2).mean(axis=2), nby, nbx))
    ghosts = np.stack(ghosts, -1)  # nby,nbx,5
    lg = np.log1p(ghosts)
    rel = lg - np.median(lg.reshape(-1, len(GHOST_Q)), axis=0)
    # normalised ghost curve shape (where the minimum lies)
    shape = ghosts / (ghosts.sum(-1, keepdims=True) + 1e-6)
    shape_rel = shape - np.median(shape.reshape(-1, len(GHOST_Q)), axis=0)
    med = cv2.medianBlur(np.clip(gray, 0, 255).astype(np.uint8), 3).astype(np.float32)
    res = np.abs(gray - med)
    noise = np.log1p(_block_mean(res, nby, nbx))
    noise_sq = np.log1p(_block_mean(res ** 2, nby, nbx))
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1)
    tex = np.log1p(_block_mean(np.sqrt(gx ** 2 + gy ** 2), nby, nbx))
    noise_rel = noise - np.median(noise)
    tex_rel = tex - np.median(tex)
    # noise unexplained by texture (robust linear fit across blocks)
    A = np.stack([tex.ravel(), np.ones(tex.size)], 1)
    coef, *_ = np.linalg.lstsq(A, noise.ravel(), rcond=None)
    noise_resid = noise - (A @ coef).reshape(noise.shape)
    coef_g, *_ = np.linalg.lstsq(A, lg[..., 3].ravel(), rcond=None)
    ghost_resid = lg[..., 3] - (A @ coef_g).reshape(noise.shape)
    feats = [lg, rel, shape_rel,
             np.stack([noise, noise_sq, noise_rel, tex, tex_rel, noise_resid, ghost_resid,
                       _neigh_contrast(noise_resid), _neigh_contrast(ghost_resid),
                       _neigh_contrast(rel[..., 1]), _neigh_contrast(rel[..., 2]),
                       np.full_like(noise, np.median(lg[..., 3]))], -1)]
    F = np.concatenate(feats, -1).reshape(nby * nbx, -1)
    return np.nan_to_num(F.astype(np.float32)), (nby, nbx), img


def block_labels(mask: Image.Image, grid, size):
    """Fraction of tampered pixels per block (mask resized like the image)."""
    m = np.asarray(_prep(mask, "L"), np.float32) / 255.0
    assert m.shape == (size[1], size[0]), (m.shape, size)
    nby, nbx = grid
    return _block_mean(m, nby, nbx).ravel()


def image_features(block_probs: np.ndarray, grid) -> np.ndarray:
    p = np.sort(block_probs)[::-1]
    n = len(p)
    pm = block_probs.reshape(grid)
    bin_ = (pm >= 0.5).astype(np.uint8)
    if bin_.any():
        _, _, stats, _ = cv2.connectedComponentsWithStats(bin_, connectivity=4)
        largest = stats[1:, cv2.CC_STAT_AREA].max() / n
    else:
        largest = 0.0
    return np.array([p[0], p[:3].mean(), p[:max(1, n // 20)].mean(), p.mean(), p.std(),
                     (p >= 0.5).mean(), largest], np.float32)


def score(bundle, img: Image.Image):
    """Returns (image_score, block_prob_map [nby,nbx], prepared image)."""
    F, grid, prep = block_features(img)
    bp = bundle["block_model"].predict_proba(F)[:, 1]
    s = float(bundle["image_model"].predict_proba(image_features(bp, grid)[None])[0, 1])
    return s, bp.reshape(grid), prep
