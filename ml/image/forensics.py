"""Classical image-forensics utilities: loading, ELA, noise residual, metadata
inspection and heatmap rendering. (The learned block-level detector lives in
ml/image/manipulation.py.)

Signals
  * Error Level Analysis (ELA): re-save as JPEG and measure per-pixel error.
    Regions pasted from a differently-compressed source often show a
    different error level than the rest of the image.
  * Noise residual: image minus its median-filtered version. Spliced regions
    often have a different sensor-noise level.
  * Metadata: EXIF camera fields, editing-software tags and generator
    parameters written by AI tools (e.g. Stable Diffusion 'parameters' chunk).
These are *indicators*, not proof: heavy recompression (social media) erases
many traces, and clean images can show false positives.
"""
from __future__ import annotations

import io

import cv2
import numpy as np
from PIL import Image, ExifTags

AI_METADATA_KEYS = ("parameters", "prompt", "workflow", "sd-metadata", "dream", "invokeai")
AI_SOFTWARE_HINTS = ("stable diffusion", "midjourney", "dall", "firefly", "novelai", "comfyui",
                     "automatic1111", "invoke", "imagen", "leonardo")
EDIT_SOFTWARE_HINTS = ("photoshop", "gimp", "lightroom", "affinity", "pixelmator", "snapseed",
                       "picsart", "canva", "paint.net", "faceapp")


def load_rgb(path_or_file) -> Image.Image:
    img = Image.open(path_or_file)
    img.load()
    if getattr(img, "is_animated", False):
        img.seek(0)
    return img


def to_rgb(img: Image.Image) -> Image.Image:
    if img.mode in ("RGBA", "LA", "P"):
        img = img.convert("RGBA")
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[-1])
        return bg
    return img.convert("RGB")


def limit_size(img: Image.Image, max_side: int = 1024) -> Image.Image:
    w, h = img.size
    s = max(w, h)
    if s <= max_side:
        return img
    r = max_side / s
    return img.resize((max(1, int(w * r)), max(1, int(h * r))), Image.LANCZOS)


def ela_map(img: Image.Image, quality: int = 90) -> np.ndarray:
    """Return ELA magnitude map (float32, HxW, 0..255)."""
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=quality)
    buf.seek(0)
    re = Image.open(buf).convert("RGB")
    diff = np.abs(np.asarray(img, np.float32) - np.asarray(re, np.float32))
    return diff.max(axis=2)


def noise_residual(img: Image.Image) -> np.ndarray:
    g = np.asarray(img.convert("L"), np.float32)
    med = cv2.medianBlur(g.astype(np.uint8), 3).astype(np.float32)
    return np.abs(g - med)


def metadata_report(img: Image.Image) -> dict:
    rep = {"format": img.format, "mode": img.mode, "size": list(img.size), "exif": {},
           "ai_generator_tags": [], "editing_software": [], "has_camera_exif": False,
           "jpeg_quality_estimate": None}
    try:
        exif = img.getexif()
        for k, v in exif.items():
            name = ExifTags.TAGS.get(k, str(k))
            if name in ("Make", "Model", "Software", "DateTime", "Artist", "LensModel"):
                rep["exif"][name] = str(v)[:120]
        rep["has_camera_exif"] = "Make" in rep["exif"] or "Model" in rep["exif"]
        sw = rep["exif"].get("Software", "").lower()
        rep["editing_software"] = [s for s in EDIT_SOFTWARE_HINTS if s in sw]
        rep["ai_generator_tags"] += [s for s in AI_SOFTWARE_HINTS if s in sw]
    except Exception:
        pass
    info = getattr(img, "info", {}) or {}
    for k, v in info.items():
        kl = str(k).lower()
        if any(a in kl for a in AI_METADATA_KEYS):
            rep["ai_generator_tags"].append(f"metadata key '{k}'")
        if isinstance(v, (str, bytes)):
            vs = v.decode("latin1", "ignore") if isinstance(v, bytes) else v
            vl = vs.lower()[:4000]
            rep["ai_generator_tags"] += [f"'{s}' in {k}" for s in AI_SOFTWARE_HINTS if s in vl]
            rep["editing_software"] += [s for s in EDIT_SOFTWARE_HINTS if s in vl and s not in rep["editing_software"]]
    rep["ai_generator_tags"] = sorted(set(rep["ai_generator_tags"]))
    if img.format == "JPEG":
        rep["jpeg_quality_estimate"] = _ijg_quality(img)
    return rep


def _ijg_quality(img: Image.Image):
    """Estimate IJG quality by matching the luminance quantisation table."""
    q = getattr(img, "quantization", None)
    if not q:
        return None
    table = np.array(q.get(0, list(q.values())[0]), np.float32)
    std = np.array([16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55, 14, 13, 16, 24, 40, 57, 69, 56,
                    14, 17, 22, 29, 51, 87, 80, 62, 18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81, 104, 113, 92,
                    49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99], np.float32)
    best, best_err = None, 1e18
    for qual in range(1, 101):
        s = 5000 / qual if qual < 50 else 200 - 2 * qual
        t = np.clip(np.floor((std * s + 50) / 100), 1, 255)
        err = np.abs(np.sort(t) - np.sort(table)).sum()
        if err < best_err:
            best, best_err = qual, err
    return int(best)


def heatmap_overlay(img: Image.Image, values: np.ndarray, alpha: float = 0.45) -> Image.Image:
    """Overlay a relative map (min-max normalised), e.g. a class activation map."""
    img = to_rgb(img)
    v = values.astype(np.float32)
    v = (v - v.min()) / (v.max() - v.min() + 1e-6)
    v = cv2.resize(v, img.size, interpolation=cv2.INTER_CUBIC)
    col = cv2.applyColorMap((v * 255).astype(np.uint8), cv2.COLORMAP_JET)[:, :, ::-1]
    out = (np.asarray(img, np.float32) * (1 - alpha) + col.astype(np.float32) * alpha).clip(0, 255)
    return Image.fromarray(out.astype(np.uint8))


def heatmap_overlay_abs(img: Image.Image, probs: np.ndarray, alpha: float = 0.5) -> Image.Image:
    """Overlay a probability map in [0,1] WITHOUT re-normalising, so a clean
    image stays blue and only genuinely high-probability blocks turn red."""
    img = to_rgb(img)
    v = cv2.resize(np.clip(probs.astype(np.float32), 0, 1), img.size, interpolation=cv2.INTER_NEAREST)
    col = cv2.applyColorMap((v * 255).astype(np.uint8), cv2.COLORMAP_JET)[:, :, ::-1]
    a = (alpha * (0.35 + 0.65 * v))[..., None]
    out = np.asarray(img, np.float32) * (1 - a) + col.astype(np.float32) * a
    return Image.fromarray(out.clip(0, 255).astype(np.uint8))


def ela_visual(ela: np.ndarray) -> Image.Image:
    scale = 255.0 / max(np.percentile(ela, 99.5), 1.0)
    v = np.clip(ela * scale, 0, 255).astype(np.uint8)
    return Image.fromarray(cv2.applyColorMap(v, cv2.COLORMAP_INFERNO)[:, :, ::-1])
