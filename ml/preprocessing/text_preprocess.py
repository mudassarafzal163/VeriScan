"""Text cleaning shared by training and inference (identical pipeline avoids
train/serve skew)."""
import re
import unicodedata

URL_RE = re.compile(r"(https?://\S+|www\.\S+|\b\S+\.(com|org|net|co)/\S*)", re.I)
MENTION_RE = re.compile(r"@\w+")
HTML_RE = re.compile(r"<[^>]+>")
RT_RE = re.compile(r"^\s*RT\b[:\s]*", re.I)
MULTISPACE_RE = re.compile(r"\s+")
NON_TEXT_RE = re.compile(r"[^\w\s#!?'%$.,-]", re.UNICODE)


def clean_text(text: str, keep_hashtags: bool = True) -> str:
    """Normalise a headline / article / tweet.

    - unicode NFKC normalisation and lower-casing
    - replaces URLs and @mentions with placeholder tokens (their presence is
      a signal, their exact value is noise)
    - strips HTML tags and the retweet prefix
    """
    if text is None:
        return ""
    text = unicodedata.normalize("NFKC", str(text))
    text = text.replace("\\n", " ")
    text = HTML_RE.sub(" ", text)
    text = RT_RE.sub("", text)
    text = URL_RE.sub(" urltoken ", text)
    text = MENTION_RE.sub(" usertoken ", text)
    if not keep_hashtags:
        text = text.replace("#", " ")
    text = NON_TEXT_RE.sub(" ", text)
    text = MULTISPACE_RE.sub(" ", text).strip().lower()
    return text


def text_stats(text: str) -> dict:
    """Simple stylistic indicators shown to the user (not used as truth)."""
    raw = text or ""
    words = raw.split()
    n = max(len(words), 1)
    letters = [c for c in raw if c.isalpha()]
    return {
        "n_words": len(words),
        "n_chars": len(raw),
        "exclamations": raw.count("!"),
        "questions": raw.count("?"),
        "uppercase_ratio": round(sum(c.isupper() for c in letters) / max(len(letters), 1), 3),
        "all_caps_words": sum(1 for w in words if len(w) > 2 and w.isupper()),
        "has_url": bool(URL_RE.search(raw)),
        "avg_word_len": round(sum(len(w) for w in words) / n, 2),
    }
