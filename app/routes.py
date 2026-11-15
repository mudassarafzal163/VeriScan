"""Page routes and JSON API."""
from __future__ import annotations

import logging
import uuid
from pathlib import Path

from flask import Blueprint, abort, jsonify, render_template, request
from PIL import Image, UnidentifiedImageError
from werkzeug.utils import secure_filename

import config
from app import database
from evaluation.metrics import load_json

log = logging.getLogger("routes")
pages = Blueprint("pages", __name__)
api = Blueprint("api", __name__)


class InputError(ValueError):
    pass


def service():
    from ml.service import get_service
    return get_service()


# ------------------------------------------------------------------ helpers
def _save_upload(field: str, allowed: set, kind: str):
    f = request.files.get(field)
    if f is None or not f.filename:
        return None, None
    name = secure_filename(f.filename) or f"upload.{kind}"
    ext = name.rsplit(".", 1)[-1].lower() if "." in name else ""
    if ext not in allowed:
        raise InputError(f"Unsupported {kind} type '.{ext}'. Allowed: {', '.join(sorted(allowed))}")
    stored = config.UPLOAD_DIR / f"{uuid.uuid4().hex}.{ext}"
    f.save(stored)
    if stored.stat().st_size == 0:
        stored.unlink(missing_ok=True)
        raise InputError("Uploaded file is empty.")
    if kind == "image":
        try:
            with Image.open(stored) as im:
                im.verify()
        except (UnidentifiedImageError, OSError, SyntaxError):
            stored.unlink(missing_ok=True)
            raise InputError("The uploaded file is not a valid image.")
    return stored, f.filename


def _rel(p: Path | None):
    return str(p.relative_to(config.ROOT / "app" / "static")).replace("\\", "/") if p else None


def _respond(result, kind=None, file_name=None, stored=None):
    det_id = database.save_detection(result, text_kind=kind, file_name=file_name, stored_file=_rel(stored))
    result["id"] = det_id
    return jsonify(result)


def _run(fn):
    try:
        return fn()
    except InputError as e:
        return jsonify(error=str(e)), 400
    except ValueError as e:
        return jsonify(error=str(e)), 422
    except Exception as e:  # noqa: BLE001
        log.exception("analysis failed")
        return jsonify(error=f"Analysis failed: {type(e).__name__}: {e}"), 500


def _metrics_summary():
    out = {}
    tn = load_json("text_news_results.json")
    if tn:
        out["Text - news headlines (FakeNewsNet test)"] = tn["test_deployed"]
    ts = load_json("text_social_results.json")
    if ts:
        out["Text - social posts (MediaEval 2016 test)"] = ts["test"]["test2016"]
    ic = load_json("image_context_results.json")
    if ic:
        out["Image content (MediaEval 2016 test images)"] = ic["test"]["test2016"]
    mr = load_json("manipulation_results.json")
    if mr:
        out["Image forensics (synthetic tampering test)"] = mr["test_image_level"]
    fr = load_json("fusion_results.json")
    if fr:
        out["Multimodal fusion (MediaEval 2016 test)"] = fr["fusion"]["reliability_weighted_fusion"]
    return out


# -------------------------------------------------------------------- pages
@pages.route("/")
def index():
    return render_template("index.html", stats=database.stats(), status=service().status(),
                           metrics=_metrics_summary(), active="home")


def _text_samples():
    import json
    f = config.ROOT / "sample_inputs" / "texts.json"
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else []


@pages.route("/text")
def text_page():
    return render_template("text.html", samples=_text_samples(), active="text")


@pages.route("/image")
def image_page():
    return render_template("image.html", active="image")


@pages.route("/video")
def video_page():
    return render_template("video.html", active="video")


@pages.route("/multimodal")
def multimodal_page():
    return render_template("multimodal.html", active="multimodal")


@pages.route("/history")
def history_page():
    modality = request.args.get("modality") or None
    verdict = request.args.get("verdict") or None
    rows = database.list_detections(200, 0, modality, verdict)
    return render_template("history.html", rows=rows, modality=modality, verdict=verdict, active="history")


@pages.route("/history/<int:det_id>")
def history_detail(det_id):
    d = database.get_detection(det_id)
    if not d:
        abort(404)
    return render_template("detail.html", d=d, active="history")


@pages.route("/about")
def about_page():
    reports = {n: load_json(f"{n}_results.json") for n in
               ("text_news", "text_social", "image_context", "manipulation", "fusion")}
    return render_template("about.html", reports=reports, summary=load_json("dataset_summary.json"),
                           status=service().status(), active="about")


# ---------------------------------------------------------------------- API
@api.get("/status")
def api_status():
    return jsonify(models=service().status(), stats=database.stats())


@api.post("/analyze/text")
def api_text():
    data = request.get_json(silent=True) or request.form
    text = (data.get("text") or "").strip()
    kind = data.get("kind", "news")

    def go():
        if not text:
            raise InputError("Text is required.")
        if kind not in ("news", "social"):
            raise InputError("kind must be 'news' or 'social'.")
        return _respond(service().analyze_text(text, kind), kind=kind)
    return _run(go)


@api.post("/analyze/image")
def api_image():
    def go():
        stored, name = _save_upload("image", config.ALLOWED_IMAGE_EXT, "image")
        if not stored:
            raise InputError("Please choose an image file.")
        r = service().analyze_image(str(stored))
        r["summary"] = name
        return _respond(r, file_name=name, stored=stored)
    return _run(go)


@api.post("/analyze/video")
def api_video():
    def go():
        stored, name = _save_upload("video", config.ALLOWED_VIDEO_EXT, "video")
        if not stored:
            raise InputError("Please choose a video file.")
        r = service().analyze_video(str(stored))
        r["summary"] = name
        return _respond(r, file_name=name, stored=stored)
    return _run(go)


@api.post("/analyze/multimodal")
def api_multimodal():
    def go():
        text = (request.form.get("text") or "").strip()
        kind = request.form.get("kind", "social")
        if kind not in ("news", "social"):
            raise InputError("kind must be 'news' or 'social'.")
        img, img_name = _save_upload("image", config.ALLOWED_IMAGE_EXT, "image")
        vid, vid_name = _save_upload("video", config.ALLOWED_VIDEO_EXT, "video")
        if not text and not img and not vid:
            raise InputError("Provide text and/or an image and/or a video.")
        r = service().analyze_multimodal(text or None, str(img) if img else None, str(vid) if vid else None, kind)
        names = ", ".join(n for n in (img_name, vid_name) if n) or None
        return _respond(r, kind=kind if text else None, file_name=names, stored=img or vid)
    return _run(go)


@api.get("/history")
def api_history():
    limit = min(int(request.args.get("limit", 50)), 500)
    offset = int(request.args.get("offset", 0))
    return jsonify(database.list_detections(limit, offset, request.args.get("modality"), request.args.get("verdict")))


@api.get("/history/<int:det_id>")
def api_history_item(det_id):
    d = database.get_detection(det_id)
    return jsonify(d) if d else (jsonify(error="Not found"), 404)


@api.delete("/history/<int:det_id>")
def api_history_delete(det_id):
    d = database.get_detection(det_id)
    if not d:
        return jsonify(error="Not found"), 404
    if d.get("stored_file"):
        (config.ROOT / "app" / "static" / d["stored_file"]).unlink(missing_ok=True)
    database.delete_detection(det_id)
    return jsonify(deleted=det_id)
