"""Image analysis: combines three independent image models + metadata.

  1. manipulation_score  - block-level forensic detector (JPEG-ghost and
     noise-residual features, gradient boosting) trained with pixel masks of
     a synthetic tampering set (splicing, copy-move, local retouch). Visual
     evidence: per-block tamper-probability heatmap + ELA map.
  2. ai_generated_score  - EfficientNet-B0 embedding + logistic regression,
     trained on a real-vs-AI dataset such as CIFAKE. OPTIONAL: only active if
     models/ai_generated_effnet_lr.joblib exists (see training/train_ai_generated.py).
  3. context_score       - EfficientNet-B0 embedding + logistic regression
     trained on MediaEval VMU images labelled by whether they were shared in
     misleading posts. This is NOT a pixel-tampering detector: most MediaEval
     'fake' images are real photos re-used out of context. Visual evidence: CAM.

Image authenticity score = 1 - max(available technical scores), where the
technical scores are manipulation and AI-generation (+ hard metadata
evidence). The context score is reported separately and used in fusion.
"""
from __future__ import annotations

import io
import base64

import joblib
import numpy as np
from PIL import Image

import config
from ml.image import forensics as fx
from ml.image import manipulation as mn


def _b64(img: Image.Image, max_side: int = 512) -> str:
    img = fx.limit_size(fx.to_rgb(img), max_side)
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def _load(path):
    return joblib.load(path) if path.exists() else None


class ImageAnalyzer:
    def __init__(self):
        self.manip = _load(config.MANIPULATION_MODEL)
        self.context = _load(config.IMAGE_CONTEXT_MODEL)
        self.ai = _load(config.AI_GENERATED_MODEL)
        self._cnn = None

    @property
    def cnn(self):
        if self._cnn is None and (self.context is not None or self.ai is not None):
            from ml.image.cnn_features import get_extractor
            self._cnn = get_extractor()
        return self._cnn

    def status(self) -> dict:
        return {"manipulation": self.manip is not None, "context": self.context is not None,
                "ai_generated": self.ai is not None}

    # ---------------------------------------------------------------- parts
    def manipulation(self, img: Image.Image, with_visuals: bool = True) -> dict:
        res = {"available": self.manip is not None}
        if self.manip is None:
            return res
        score, bmap, prep = mn.score(self.manip, img)
        res["score"] = round(score, 4)
        res["suspicious_block_fraction"] = round(float((bmap >= 0.5).mean()), 4)
        res["max_block_prob"] = round(float(bmap.max()), 4)
        res["test_metrics"] = self.manip.get("test_metrics", {})
        if with_visuals:
            res["heatmap"] = _b64(fx.heatmap_overlay_abs(prep, bmap))
            res["ela"] = _b64(fx.ela_visual(fx.ela_map(prep, 90)))
        return res

    def cnn_heads(self, img: Image.Image, with_visuals: bool = True) -> dict:
        out = {"context": {"available": self.context is not None},
               "ai_generated": {"available": self.ai is not None}, "concepts": []}
        if self.cnn is None:
            return out
        pooled, logits, fmap = self.cnn.forward([img])
        out["concepts"] = self.cnn.top_concepts(logits[0])
        out["embedding"] = pooled[0]
        for key, bundle in (("context", self.context), ("ai_generated", self.ai)):
            if bundle is None:
                continue
            pipe = bundle["pipeline"]
            p = float(pipe.predict_proba(pooled)[0, 1])
            out[key]["score"] = round(p, 4)
            out[key]["test_metrics"] = bundle.get("test_metrics", {})
            if with_visuals:
                sc, clf = pipe.named_steps["scaler"], pipe.named_steps["clf"]
                cam = self.cnn.cam(fmap[0], clf.coef_.ravel(), sc.scale_)
                out[key]["cam"] = _b64(fx.heatmap_overlay(fx.limit_size(img, 512), np.maximum(cam, 0)))
        return out

    # ---------------------------------------------------------------- main
    def analyze(self, img_or_path, with_visuals: bool = True) -> dict:
        img = fx.load_rgb(img_or_path) if not isinstance(img_or_path, Image.Image) else img_or_path
        meta = fx.metadata_report(img)
        rgb = fx.to_rgb(img)
        if min(rgb.size) < 32:
            raise ValueError("Image too small for analysis (min 32x32 px)")
        manip = self.manipulation(rgb, with_visuals)
        heads = self.cnn_heads(rgb, with_visuals)
        warnings, reasons = [], []

        technical = {}
        if manip.get("available"):
            technical["manipulation"] = manip["score"]
        if heads["ai_generated"].get("available"):
            technical["ai_generated"] = heads["ai_generated"]["score"]
        else:
            warnings.append("AI-generated image detector is not trained in this installation "
                            "(needs CIFAKE or a similar dataset). AI-generation is only checked via metadata.")
        if meta["ai_generator_tags"]:
            technical["metadata_ai_tag"] = 0.99
            reasons.append("Metadata contains AI-generator traces: " + ", ".join(meta["ai_generator_tags"]))
        if meta["editing_software"]:
            reasons.append("Metadata shows editing software: " + ", ".join(meta["editing_software"])
                           + " (editing is not necessarily deceptive).")
        if not meta["has_camera_exif"]:
            warnings.append("No camera EXIF data (common for screenshots and social-media re-uploads; "
                            "not evidence of fakery by itself).")
        q = meta.get("jpeg_quality_estimate")
        if q is not None and q < 75:
            warnings.append(f"Low JPEG quality (~{q}): recompression weakens forensic traces.")
        if max(rgb.size) < 300:
            warnings.append("Low resolution image: forensic analysis is unreliable.")

        authenticity = None
        if technical:
            authenticity = round(1.0 - max(technical.values()), 4)
            if manip.get("available"):
                s = manip["score"]
                if s >= 0.5:
                    reasons.append(f"Forensic detector found localised compression/noise inconsistencies (score {s:.2f}); see heatmap.")
                else:
                    reasons.append(f"No strong localised compression/noise inconsistency found (score {s:.2f}).")
        ctx = heads["context"]
        if ctx.get("available"):
            reasons.append(f"Content-based misleading-use score {ctx['score']:.2f} (dataset-specific, weak cue; "
                           "it does not test whether pixels were altered).")
        return {
            "authenticity_score": authenticity,
            "technical_scores": technical,
            "manipulation": manip,
            "ai_generated": heads["ai_generated"],
            "context": ctx,
            "concepts": heads["concepts"],
            "metadata": meta,
            "reasons": reasons,
            "warnings": warnings,
            "preview": _b64(rgb) if with_visuals else None,
            "_embedding": heads.get("embedding"),
        }
