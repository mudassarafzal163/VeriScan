"""Text-image consistency (does the image plausibly belong to the text?).

Two back-ends:
  * 'learned' (EXPERIMENTAL, trained here, NOT used at inference because its
    held-out AUC did not beat chance - reported as a negative result): tweet text -> TF-IDF -> SVD -> ridge
    regression onto the (PCA-reduced) EfficientNet image embedding. The
    consistency score is the cosine similarity between the image embedding
    predicted from the text and the actual one, mapped to [0,1]. Trained on
    matched MediaEval post/image pairs. It only knows the visual vocabulary
    of the training events, so on unrelated topics it is weak; its measured
    matched-vs-mismatched AUC is reported in reports/.
  * 'clip' (used at inference when available): OpenAI CLIP ViT-B/32 via
    HuggingFace transformers, a general-purpose zero-shot image-text model.
    Used automatically if `transformers` is installed and the weights can be
    downloaded (pip install -r requirements-optional.txt). Without it, the
    consistency check is reported as unavailable.
A low score means 'the image does not look like what the text describes'.
That can indicate out-of-context image reuse, but also just a loosely
illustrated story - it is reported as a warning, not as proof.
"""
from __future__ import annotations

import os

import numpy as np

import config
from ml.preprocessing.text_preprocess import clean_text

CONSISTENCY_MODEL = config.MODELS_DIR / "consistency_ridge.joblib"


def build_learned(texts, embeddings, seed=config.RANDOM_SEED):
    from sklearn.decomposition import PCA, TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    text_pipe = make_pipeline(
        TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3, max_features=50000, sublinear_tf=True),
        TruncatedSVD(n_components=200, random_state=seed))
    T = text_pipe.fit_transform([clean_text(t) for t in texts])
    img_pipe = make_pipeline(StandardScaler(), PCA(n_components=64, random_state=seed))
    Y = img_pipe.fit_transform(embeddings)
    ridge = Ridge(alpha=10.0).fit(T, Y)
    return {"text_pipe": text_pipe, "img_pipe": img_pipe, "ridge": ridge}


def learned_scores(bundle, texts, embeddings) -> np.ndarray:
    T = bundle["text_pipe"].transform([clean_text(t) for t in texts])
    pred = bundle["ridge"].predict(T)
    act = bundle["img_pipe"].transform(embeddings)
    num = (pred * act).sum(1)
    den = np.linalg.norm(pred, axis=1) * np.linalg.norm(act, axis=1) + 1e-9
    return (num / den + 1) / 2  # cosine -> [0,1]


class ConsistencyChecker:
    def __init__(self):
        self._clip = None
        self._clip_failed = os.environ.get("FND_DISABLE_CLIP", "0") == "1"

    def _try_clip(self):
        if self._clip is not None or self._clip_failed:
            return self._clip
        try:
            import torch
            from transformers import CLIPModel, CLIPProcessor
            name = "openai/clip-vit-base-patch32"
            self._clip = (CLIPModel.from_pretrained(name).eval(), CLIPProcessor.from_pretrained(name), torch)
        except Exception:
            self._clip_failed = True
        return self._clip

    def clip_score(self, text, img):
        clip = self._try_clip()
        if clip is None:
            return None
        model, proc, torch = clip
        with torch.no_grad():
            inp = proc(text=[text[:300]], images=[img], return_tensors="pt", padding=True, truncation=True)
            out = model(**inp)
            t = out.text_embeds / out.text_embeds.norm(dim=-1, keepdim=True)
            i = out.image_embeds / out.image_embeds.norm(dim=-1, keepdim=True)
            cos = float((t * i).sum())
        # CLIP cosines for matching pairs are typically 0.25-0.35, unrelated ~0.1-0.18
        return float(np.clip((cos - 0.12) / (0.32 - 0.12), 0, 1))

    def score(self, text: str, img, embedding, concepts) -> dict:
        res = {"available": False, "backend": None, "score": None, "concept_overlap": []}
        words = set(clean_text(text).split())
        for c in concepts or []:
            toks = {w for w in c["label"].lower().replace("-", " ").split() if len(w) > 2}
            if toks & words:
                res["concept_overlap"].append(c["label"])
        s = self.clip_score(text, img) if img is not None else None
        if s is not None:
            res.update(available=True, backend="clip", score=round(s, 4))
        else:
            res["note"] = ("CLIP is not installed/downloadable, so semantic text-image consistency was not "
                           "checked. The learned fallback model scored below chance on held-out events and is "
                           "therefore not used for decisions (see Evaluation report).")
        return res
