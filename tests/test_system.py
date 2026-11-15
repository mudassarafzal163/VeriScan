"""End-to-end and unit tests.  Run:  python -m pytest -q"""
import io
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SAMPLES = ROOT / "sample_inputs"


@pytest.fixture(scope="session")
def client(tmp_path_factory):
    os.environ["FND_DB_PATH"] = str(tmp_path_factory.mktemp("db") / "test.db")
    from app import create_app
    app = create_app(testing=True)
    return app.test_client()


# ------------------------------------------------------------------ units
def test_clean_text():
    from ml.preprocessing.text_preprocess import clean_text
    out = clean_text("RT @bob: BREAKING!! See http://t.co/abc <b>now</b>")
    assert "urltoken" in out and "usertoken" in out and "<b>" not in out
    assert out == out.lower()
    assert clean_text(None) == ""


def test_text_model_explanation_is_exact():
    import config
    from ml.preprocessing.text_preprocess import clean_text
    from ml.text.text_model import TextModel
    m = TextModel(config.TEXT_NEWS_MODEL, "news")
    text = "Shocking: celebrity secretly arrested, insiders claim"
    r = m.explain(text)
    clf = m.pipeline.named_steps["clf"]
    x = m.pipeline.named_steps["tfidf"].transform([clean_text(text)])
    logit = float((x @ clf.coef_.ravel())[0] + clf.intercept_[0])
    assert abs(1 / (1 + np.exp(-logit)) - r["fake_score"]) < 1e-3
    assert 0 <= r["fake_score"] <= 1
    assert r["terms_toward_fake"] or r["terms_toward_real"]


def test_social_model_word_level_terms():
    import config
    from ml.text.text_model import TextModel
    r = TextModel(config.TEXT_SOCIAL_MODEL, "social").explain("Shark swimming on the flooded highway #sandy")
    terms = [t["term"] for t in r["terms_toward_fake"] + r["terms_toward_real"]]
    assert terms and all(len(t) > 1 or t.isalnum() for t in terms)
    assert "<mark" in r["highlighted_html"]


def test_text_rejects_empty():
    from ml.service import get_service
    with pytest.raises(ValueError):
        get_service().analyze_text("  ")


def test_fusion_rules():
    from ml.fusion.fusion import FusionEngine
    f = FusionEngine()
    assert f.fuse({})["verdict"] == "INSUFFICIENT EVIDENCE"
    r = f.fuse({"text": 0.95, "manipulation": 0.05})
    assert 0 <= r["score"] <= 1 and abs(sum(r["weights"].values()) - 1) < 1e-6
    assert any("Conflicting" in w for w in r["warnings"])
    r = f.fuse({"text": 0.05}, flags=["AI metadata"])
    assert r["verdict"] == "UNCERTAIN"  # flags prevent a confident REAL verdict


def test_manipulation_localises_splice():
    import joblib
    import config
    from ml.image import manipulation as mn
    bundle = joblib.load(config.MANIPULATION_MODEL)
    img = Image.open(SAMPLES / "images" / "synthetic_tampered_splice_5.jpg")
    mask = Image.open(SAMPLES / "images" / "synthetic_tampered_splice_5_groundtruth_mask.png")
    s, bmap, prep = mn.score(bundle, img)
    lab = mn.block_labels(mask, bmap.shape, prep.size) >= 0.5
    assert s > 0.5
    assert bmap.ravel()[lab].mean() > bmap.ravel()[~lab].mean() + 0.2


def test_metadata_detects_ai_tags(tmp_path):
    from PIL import PngImagePlugin
    from ml.image import forensics as fx
    info = PngImagePlugin.PngInfo()
    info.add_text("parameters", "a photo of a flood, Steps: 30, Sampler: Euler a, Model: stable diffusion 1.5")
    p = tmp_path / "gen.png"
    Image.new("RGB", (64, 64), (120, 130, 140)).save(p, pnginfo=info)
    rep = fx.metadata_report(fx.load_rgb(p))
    assert rep["ai_generator_tags"]


def test_video_analyzer_detects_cut():
    from ml.service import get_service
    r = get_service().analyze_video(str(SAMPLES / "videos" / "sample_two_scene_clip.mp4"))
    v = r["parts"]["video"]
    assert v["metadata"]["duration_s"] == pytest.approx(7.0, abs=0.2)
    assert v["temporal"]["scene_cuts"] >= 1 and len(v["frames"]) == 12


def test_ai_generated_training_script_smoke(tmp_path):
    """Exercises the full CIFAKE-style training code path on a tiny dummy set
    (labels are synthetic; the resulting metric is meaningless and not reported)."""
    from training import train_ai_generated as t
    rng = np.random.default_rng(0)
    for split in ("train", "test"):
        for cls, smooth in (("REAL", False), ("FAKE", True)):
            d = tmp_path / "ds" / split / cls
            d.mkdir(parents=True)
            for i in range(6):
                a = rng.integers(0, 255, (48, 48, 3), dtype=np.uint8)
                if smooth:
                    a = np.full_like(a, int(a.mean()))
                Image.fromarray(a).save(d / f"{i}.png")
    out = tmp_path / "ai.joblib"
    m = t.main(["--root", str(tmp_path / "ds"), "--out", str(out), "--report-name", "_smoke_ai.json"])
    assert out.exists() and "roc_auc" in m
    import config
    (config.REPORTS_DIR / "_smoke_ai.json").unlink(missing_ok=True)


# -------------------------------------------------------------------- API
def test_pages(client):
    for p in ("/", "/text", "/image", "/video", "/multimodal", "/history", "/about"):
        assert client.get(p).status_code == 200, p
    assert client.get("/does-not-exist").status_code == 404


def test_api_text_and_history(client):
    r = client.post("/api/analyze/text", json={"text": "You won't believe what this senator did!!!", "kind": "news"})
    assert r.status_code == 200
    d = r.get_json()
    assert d["verdict"] in ("LIKELY FAKE", "LIKELY REAL", "UNCERTAIN") and d["disclaimer"]
    h = client.get(f"/api/history/{d['id']}").get_json()
    assert h["result"]["score"] == d["score"]
    assert client.get(f"/history/{d['id']}").status_code == 200
    assert client.delete(f"/api/history/{d['id']}").status_code == 200
    assert client.get(f"/api/history/{d['id']}").status_code == 404


def test_api_errors(client):
    assert client.post("/api/analyze/text", json={"text": ""}).status_code == 400
    assert client.post("/api/analyze/text", json={"text": "hello world", "kind": "x"}).status_code == 400
    bad = {"image": (io.BytesIO(b"not an image"), "x.jpg")}
    assert client.post("/api/analyze/image", data=bad, content_type="multipart/form-data").status_code == 400
    bad = {"image": (io.BytesIO(b"x"), "x.exe")}
    assert client.post("/api/analyze/image", data=bad, content_type="multipart/form-data").status_code == 400
    bad = {"video": (io.BytesIO(b"garbage"), "x.mp4")}
    assert client.post("/api/analyze/video", data=bad, content_type="multipart/form-data").status_code == 422
    assert client.post("/api/analyze/multimodal", data={}, content_type="multipart/form-data").status_code == 400


def test_api_multimodal(client):
    s = json.loads((SAMPLES / "multimodal" / "samples.json").read_text(encoding="utf-8"))[0]
    with open(SAMPLES / "multimodal" / s["image"], "rb") as f:
        r = client.post("/api/analyze/multimodal", data={"text": s["text"], "kind": "social", "image": (f, s["image"])},
                        content_type="multipart/form-data")
    assert r.status_code == 200
    d = r.get_json()
    mods = {c["modality"] for c in d["fusion"]["contributions"]}
    assert {"text", "manipulation", "image_context"} <= mods
    assert d["parts"]["image"]["manipulation"]["heatmap"].startswith("data:image/jpeg;base64,")
