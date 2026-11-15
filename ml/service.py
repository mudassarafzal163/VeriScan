"""DetectionService: the single entry point used by the web app (and tests).

Loads every model once, routes inputs to the right modality analysers,
fuses the results and builds a human-readable explanation.
"""
from __future__ import annotations

import threading
import time

import config
from ml.fusion.consistency import ConsistencyChecker
from ml.fusion.fusion import FusionEngine
from ml.image import forensics as fx
from ml.image.image_analyzer import ImageAnalyzer
from ml.text.text_model import TextModel

DISCLAIMER = ("Scores are outputs of statistical models trained on specific public datasets. They are NOT the "
              "probability that a story is true or false, and they cannot verify facts. Use them to prioritise "
              "human fact-checking, not to replace it.")


class DetectionService:
    def __init__(self):
        self.text_models = {"news": TextModel(config.TEXT_NEWS_MODEL, "news-headline (FakeNewsNet)"),
                            "social": TextModel(config.TEXT_SOCIAL_MODEL, "social-post (MediaEval)")}
        self.images = ImageAnalyzer()
        self.fusion = FusionEngine()
        self.consistency = ConsistencyChecker()
        self._video = None
        self._lock = threading.Lock()

    @property
    def video(self):
        if self._video is None:
            from ml.video.video_analyzer import VideoAnalyzer
            self._video = VideoAnalyzer(self.images)
        return self._video

    def status(self) -> dict:
        return {"text_news": self.text_models["news"].available,
                "text_social": self.text_models["social"].available,
                **{f"image_{k}": v for k, v in self.images.status().items()},
                "fusion_validated": self.fusion.validated}

    # ------------------------------------------------------------------ text
    def analyze_text(self, text: str, kind: str = "news") -> dict:
        t0 = time.time()
        if kind not in self.text_models:
            raise ValueError("kind must be 'news' or 'social'")
        if not text or len(text.strip()) < 3:
            raise ValueError("Please enter at least a few words of text.")
        if len(text) > 20000:
            text = text[:20000]
        tm = self.text_models[kind]
        res = tm.explain(text)
        fused = self.fusion.fuse({"text": res["fake_score"]})
        reasons = []
        if res["terms_toward_fake"]:
            reasons.append("Words most associated with fake items in training data: "
                           + ", ".join(t["term"] for t in res["terms_toward_fake"][:5]))
        if res["terms_toward_real"]:
            reasons.append("Words most associated with real items: "
                           + ", ".join(t["term"] for t in res["terms_toward_real"][:5]))
        st = res["style_stats"]
        if st["exclamations"] >= 2 or st["all_caps_words"] >= 2:
            reasons.append("Sensational style (exclamation marks / ALL-CAPS words) detected.")
        return self._finish("text", fused, {"text": res}, reasons, res["warnings"], t0,
                            summary=text[:200], text_input=text)

    # ----------------------------------------------------------------- image
    def analyze_image(self, path: str) -> dict:
        t0 = time.time()
        res = self.images.analyze(path)
        res.pop("_embedding", None)
        flags = []
        if res["metadata"]["ai_generator_tags"]:
            flags.append("AI-generator metadata found in the image file.")
        ai = res["ai_generated"]
        if ai.get("available") and ai["score"] >= 0.8:
            flags.append(f"AI-generated image detector score {ai['score']:.2f}.")
        man = res["manipulation"]
        fused = self.fusion.fuse({"manipulation": man.get("score"),
                                  "image_context": res["context"].get("score")}, flags)
        return self._finish("image", fused, {"image": res}, res["reasons"], res["warnings"], t0,
                            summary=str(path).split("/")[-1].split("\\")[-1])

    # ----------------------------------------------------------------- video
    def analyze_video(self, path: str) -> dict:
        t0 = time.time()
        res = self.video.analyze(path)
        flags = []
        if res["ai_generated_mean"] is not None and res["ai_generated_mean"] >= 0.8:
            flags.append("Frames look AI-generated according to the image detector.")
        fused = self.fusion.fuse({"video": res["manipulation_mean"]}, flags)
        return self._finish("video", fused, {"video": res}, res["reasons"], res["warnings"], t0,
                            summary=str(path).split("/")[-1].split("\\")[-1])

    # ------------------------------------------------------------ multimodal
    def analyze_multimodal(self, text: str | None, image_path: str | None, video_path: str | None = None,
                           kind: str = "social") -> dict:
        t0 = time.time()
        if not (text and text.strip()) and not image_path and not video_path:
            raise ValueError("Provide at least one of: text, image, video.")
        parts, reasons, warnings, flags, scores = {}, [], [], [], {}
        if text and text.strip():
            tr = self.text_models[kind].explain(text)
            parts["text"] = tr
            scores["text"] = tr["fake_score"]
            warnings += tr["warnings"]
            if tr["terms_toward_fake"]:
                reasons.append("Text: fake-associated words - " + ", ".join(t["term"] for t in tr["terms_toward_fake"][:5]))
            if kind == "news":
                warnings.append("Fusion weights were validated with the social-post text model; with the "
                                "news-headline model the weighting is an approximation.")
        img_obj = None
        if image_path:
            ir = self.images.analyze(image_path)
            emb = ir.pop("_embedding", None)
            parts["image"] = ir
            scores["manipulation"] = ir["manipulation"].get("score")
            scores["image_context"] = ir["context"].get("score")
            reasons += ["Image: " + r for r in ir["reasons"]]
            warnings += ir["warnings"]
            if ir["metadata"]["ai_generator_tags"]:
                flags.append("AI-generator metadata found in the image file.")
            if ir["ai_generated"].get("available") and ir["ai_generated"]["score"] >= 0.8:
                flags.append(f"AI-generated image detector score {ir['ai_generated']['score']:.2f}.")
            img_obj = fx.to_rgb(fx.load_rgb(image_path))
            if text and text.strip():
                c = self.consistency.score(text, img_obj, emb, ir["concepts"])
                parts["consistency"] = c
                if c["available"] and c["score"] < 0.3:
                    flags.append(f"Low text-image consistency ({c['backend']} score {c['score']:.2f}): the image "
                                 "may not depict what the text describes (possible out-of-context use).")
                elif c["available"]:
                    reasons.append(f"Text-image consistency ({c['backend']}) score {c['score']:.2f}.")
                if c["concept_overlap"]:
                    reasons.append("Objects recognised in the image that are also mentioned in the text: "
                                   + ", ".join(c["concept_overlap"]))
        if video_path:
            vr = self.video.analyze(video_path)
            parts["video"] = vr
            scores["video"] = vr["manipulation_mean"]
            reasons += ["Video: " + r for r in vr["reasons"]]
            warnings += vr["warnings"]
        fused = self.fusion.fuse(scores, flags)
        summary = (text or "")[:160] or (image_path or video_path or "")
        return self._finish("multimodal", fused, parts, reasons, warnings, t0, summary=str(summary),
                            text_input=text)

    # ---------------------------------------------------------------- common
    def _finish(self, modality, fused, parts, reasons, warnings, t0, summary="", text_input=None):
        warnings = list(dict.fromkeys(list(warnings) + fused["warnings"]))
        return {
            "modality": modality,
            "verdict": fused["verdict"],
            "score": fused["score"],
            "confidence": None if fused["score"] is None else round(abs(fused["score"] - 0.5) * 2, 4),
            "fusion": fused,
            "parts": parts,
            "reasons": reasons,
            "warnings": warnings,
            "flags": fused["flags"],
            "disclaimer": DISCLAIMER,
            "elapsed_s": round(time.time() - t0, 2),
            "summary": summary,
            "text_input": text_input,
        }


_SERVICE = None
_SLOCK = threading.Lock()


def get_service() -> DetectionService:
    global _SERVICE
    with _SLOCK:
        if _SERVICE is None:
            _SERVICE = DetectionService()
        return _SERVICE
