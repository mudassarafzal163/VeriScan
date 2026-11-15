"""Reliability-weighted late fusion with explicit conflict/insufficiency checks.

fused = sum_m w_m * s_m / sum_m w_m   over AVAILABLE modalities m
where s_m is the modality's fake-score and w_m = max(AUC_val(m) - 0.5, 0) was
measured on a held-out validation set (MediaEval 2015 test, events unseen by
the base models). Modalities that could not be validated (AI-generated
detector, CLIP consistency, video) do not move the number; they raise
evidence flags that are shown with the verdict.

Verdict bands (on the model score, NOT a probability that the story is true):
  score >= 0.65 -> LIKELY FAKE,  score <= 0.35 -> LIKELY REAL,  else UNCERTAIN
"""
from __future__ import annotations

import joblib

import config

FAKE_T, REAL_T = 0.65, 0.35
DEFAULT_WEIGHTS = {"text": 1.0, "image_context": 0.0, "manipulation": 0.1, "video": 0.1}
NAMES = {"text": "Text", "image_context": "Image content (misleading-use)",
         "manipulation": "Image forensics (tampering)", "video": "Video frame forensics"}


class FusionEngine:
    def __init__(self):
        b = joblib.load(config.FUSION_MODEL) if config.FUSION_MODEL.exists() else None
        self.weights = dict(DEFAULT_WEIGHTS)
        self.validated = b is not None
        if b:
            self.weights.update(b["weights"])
            # video technical score is frame-level manipulation evidence -> same reliability
            self.weights["video"] = b["weights"].get("manipulation", DEFAULT_WEIGHTS["video"])
            self.validation_auc = b.get("validation_auc", {})
            self.test_metrics = b.get("test_metrics", {})
        else:
            self.validation_auc, self.test_metrics = {}, {}

    @staticmethod
    def verdict(score):
        if score is None:
            return "INSUFFICIENT EVIDENCE"
        if score >= FAKE_T:
            return "LIKELY FAKE"
        if score <= REAL_T:
            return "LIKELY REAL"
        return "UNCERTAIN"

    def fuse(self, scores: dict, flags: list | None = None) -> dict:
        flags = list(flags or [])
        avail = {k: v for k, v in scores.items() if v is not None and k in self.weights}
        warnings = []
        used = {k: self.weights[k] for k in avail if self.weights[k] > 0}
        if not used and avail:
            used = {k: 1.0 for k in avail}
            warnings.append("Available modalities had no validated reliability; equal weights were used "
                            "(uncalibrated).")
        if not used:
            return {"score": None, "verdict": "INSUFFICIENT EVIDENCE", "contributions": [], "flags": flags,
                    "warnings": ["No modality produced a usable score."], "weights": {}}
        tot = sum(used.values())
        score = sum(used[k] * avail[k] for k in used) / tot
        contributions = [{"modality": k, "name": NAMES.get(k, k), "score": round(avail[k], 4),
                          "weight": round(used.get(k, 0) / tot, 4),
                          "push": round(used.get(k, 0) / tot * (avail[k] - 0.5), 4)}
                         for k in avail]
        contributions.sort(key=lambda c: -abs(c["push"]))
        # conflicts between strongly opinionated modalities
        strong = {k: v for k, v in avail.items() if abs(v - 0.5) >= 0.3}
        if any(v >= 0.8 for v in strong.values()) and any(v <= 0.2 for v in strong.values()):
            hi = [NAMES.get(k, k) for k, v in strong.items() if v >= 0.8]
            lo = [NAMES.get(k, k) for k, v in strong.items() if v <= 0.2]
            warnings.append(f"Conflicting evidence: {', '.join(hi)} suggest fake while {', '.join(lo)} suggest real.")
        if len(used) == 1:
            warnings.append("Only one modality contributed to the final score.")
        v = self.verdict(score)
        if flags and v == "LIKELY REAL":
            v = "UNCERTAIN"
            warnings.append("Verdict lowered to UNCERTAIN because of the evidence flags below.")
        return {"score": round(score, 4), "verdict": v, "contributions": contributions,
                "weights": {k: round(w / tot, 4) for k, w in used.items()}, "flags": flags, "warnings": warnings}
