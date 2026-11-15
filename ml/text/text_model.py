"""Text fake-news classifier: TF-IDF features + linear classifier.

The model is a scikit-learn Pipeline(TfidfVectorizer -> LogisticRegression).
Because it is linear, each word's contribution to the decision is exactly
tfidf(word) * coefficient(word), which gives faithful (not post-hoc
approximated) word-level explanations.
"""
from __future__ import annotations

import re
from pathlib import Path

import joblib

from ml.preprocessing.text_preprocess import clean_text, text_stats


class TextModel:
    def __init__(self, path: Path, name: str):
        self.path = Path(path)
        self.name = name
        self.bundle = None
        if self.path.exists():
            self.bundle = joblib.load(self.path)

    @property
    def available(self) -> bool:
        return self.bundle is not None

    @property
    def pipeline(self):
        return self.bundle["pipeline"]

    def predict_proba(self, texts):
        cleaned = [clean_text(t) for t in texts]
        return self.pipeline.predict_proba(cleaned)[:, 1]

    def _word_contributions(self, cleaned: str, top_k: int):
        vec = self.pipeline.named_steps["tfidf"]
        clf = self.pipeline.named_steps["clf"]
        coef = clf.coef_.ravel()
        x = vec.transform([cleaned])
        feats = vec.get_feature_names_out()
        contrib = []
        if vec.analyzer == "char_wb":
            # char_wb n-grams never cross word boundaries, so each n-gram's
            # contribution can be split exactly over the words containing it
            # (equally per occurrence) -> word-level contributions.
            ngram_c = {feats[j]: float(v * coef[j]) for j, v in zip(x.indices, x.data)}
            lo, hi = vec.ngram_range
            occ, per_word = {}, []
            for w in cleaned.split():
                padded = f" {w} "
                grams = [padded[i:i + n] for n in range(lo, hi + 1) for i in range(len(padded) - n + 1)]
                grams = [g for g in grams if g in ngram_c]
                per_word.append((w, grams))
                for g in grams:
                    occ[g] = occ.get(g, 0) + 1
            words = {}
            for w, grams in per_word:
                words[w] = words.get(w, 0.0) + sum(ngram_c[g] / occ[g] for g in grams)
            contrib = [(w, c) for w, c in words.items() if abs(c) > 1e-9]
            contrib.sort(key=lambda t: abs(t[1]), reverse=True)
            fake = [{"term": t, "weight": round(w, 4)} for t, w in contrib if w > 0][:top_k]
            real = [{"term": t, "weight": round(-w, 4)} for t, w in contrib if w < 0][:top_k]
            return fake, real, list(ngram_c.items())
        for j, v in zip(x.indices, x.data):
            contrib.append((feats[j], float(v * coef[j])))
        contrib.sort(key=lambda t: abs(t[1]), reverse=True)
        fake = [{"term": t, "weight": round(w, 4)} for t, w in contrib if w > 0][:top_k]
        real = [{"term": t, "weight": round(-w, 4)} for t, w in contrib if w < 0][:top_k]
        return fake, real, contrib

    def explain(self, text: str, top_k: int = 8) -> dict:
        if not self.available:
            raise RuntimeError(f"{self.name} model not trained: {self.path} missing")
        if not text or not text.strip():
            raise ValueError("Empty text")
        cleaned = clean_text(text)
        prob = float(self.pipeline.predict_proba([cleaned])[0, 1])
        fake_terms, real_terms, contrib = self._word_contributions(cleaned, top_k)
        n_known = len(contrib)
        warnings = []
        n_words = len(cleaned.split())
        if n_words < 5:
            warnings.append("Very short input: the text model has little evidence to work with.")
        if n_known < 3:
            warnings.append("Most words are not in the model vocabulary; the score mostly reflects the prior.")
        if self.bundle.get("domain_note"):
            warnings.append(self.bundle["domain_note"])
        return {
            "model": self.name,
            "fake_score": round(prob, 4),
            "known_terms": n_known,
            "terms_toward_fake": fake_terms,
            "terms_toward_real": real_terms,
            "highlighted_html": highlight(text, {t["term"]: t["weight"] for t in fake_terms},
                                          {t["term"]: t["weight"] for t in real_terms}),
            "style_stats": text_stats(text),
            "warnings": warnings,
            "test_metrics": self.bundle.get("test_metrics", {}),
        }


def highlight(text: str, fake_terms: dict, real_terms: dict) -> str:
    """Wrap unigram matches in <mark> tags (HTML-escaped)."""
    import html
    out = []
    for tok in re.split(r"(\s+)", text):
        key = re.sub(r"[^\w#']", "", tok.lower())
        esc = html.escape(tok)
        if key and key in fake_terms:
            out.append(f'<mark class="hl-fake" title="pushes toward FAKE ({fake_terms[key]:+.3f})">{esc}</mark>')
        elif key and key in real_terms:
            out.append(f'<mark class="hl-real" title="pushes toward REAL ({real_terms[key]:+.3f})">{esc}</mark>')
        else:
            out.append(esc)
    return "".join(out)
