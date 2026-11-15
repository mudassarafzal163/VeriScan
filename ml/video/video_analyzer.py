"""Video analysis by frame sampling.

What it does (all genuinely computed):
  * container metadata (fps, duration, resolution, codec FourCC)
  * uniform sampling of N key frames (OpenCV)
  * per-frame image analysis: forensic manipulation score, AI-generated
    score (if that model is installed), face count (OpenCV Haar cascade)
  * temporal consistency: frame-to-frame colour-histogram distance on densely
    sampled frames; abrupt outliers indicate hard cuts or spliced segments
  * aggregation into a video technical-suspicion score

What it does NOT do: there is no trained face-swap/deepfake-video model in
this build (FaceForensics++, DFDC and Celeb-DF require access requests and
GPU training). Audio is not analysed. These limits are reported in every
result. A face-swap detector can be dropped in via `face_model` later.
"""
from __future__ import annotations

import base64

import cv2
import numpy as np
from PIL import Image

N_KEYFRAMES = 12
N_TEMPORAL = 60
MAX_SECONDS_ANALYSED = 600


def _thumb_b64(frame_rgb: np.ndarray, max_side: int = 320) -> str:
    h, w = frame_rgb.shape[:2]
    s = max_side / max(h, w)
    if s < 1:
        frame_rgb = cv2.resize(frame_rgb, (int(w * s), int(h * s)))
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 80])
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()


class VideoAnalyzer:
    def __init__(self, image_analyzer):
        self.ia = image_analyzer
        self.face = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    def _read_frame(self, cap, idx):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, fr = cap.read()
        return cv2.cvtColor(fr, cv2.COLOR_BGR2RGB) if ok and fr is not None else None

    def analyze(self, path: str) -> dict:
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise ValueError("Could not open video (unsupported codec or corrupt file)")
        try:
            n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
            w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
            codec = "".join(chr((fourcc >> 8 * i) & 0xFF) for i in range(4)).strip("\x00") or "unknown"
            if n <= 0:
                raise ValueError("Video has no readable frames")
            duration = n / fps if fps > 0 else None
            last = min(n - 1, int(MAX_SECONDS_ANALYSED * fps) if fps > 0 else n - 1)
            warnings = ["No trained face-swap / deepfake-video model is installed; video results are based on "
                        "frame-level image forensics and temporal statistics only.",
                        "Audio track is not analysed."]
            if duration and duration > MAX_SECONDS_ANALYSED:
                warnings.append(f"Only the first {MAX_SECONDS_ANALYSED // 60} minutes were analysed.")
            if fourcc and codec.lower() in ("avc1", "h264", "hevc", "hvc1", "vp09", "av01"):
                warnings.append("Inter-frame video compression largely removes JPEG-style forensic traces, so "
                                "frame manipulation scores are weaker evidence than for photos.")

            # temporal consistency
            hist_d, prev = [], None
            for idx in np.linspace(0, last, min(N_TEMPORAL, last + 1)).astype(int):
                fr = self._read_frame(cap, idx)
                if fr is None:
                    continue
                small = cv2.resize(fr, (160, 90))
                hsv = cv2.cvtColor(small, cv2.COLOR_RGB2HSV)
                hist = cv2.calcHist([hsv], [0, 1], None, [16, 8], [0, 180, 0, 256])
                cv2.normalize(hist, hist)
                if prev is not None:
                    hist_d.append(float(cv2.compareHist(prev, hist, cv2.HISTCMP_BHATTACHARYYA)))
                prev = hist
            hist_d = np.array(hist_d) if hist_d else np.zeros(1)
            med = float(np.median(hist_d))
            mad = float(np.median(np.abs(hist_d - med))) + 1e-3
            cuts = int(((hist_d - med) / (1.4826 * mad) > 6).sum() if len(hist_d) > 3 else 0)

            # key frames
            frames = []
            for idx in np.linspace(0, last, min(N_KEYFRAMES, last + 1)).astype(int):
                fr = self._read_frame(cap, idx)
                if fr is None:
                    continue
                img = Image.fromarray(fr)
                r = self.ia.analyze(img, with_visuals=False)
                gray = cv2.cvtColor(fr, cv2.COLOR_RGB2GRAY)
                faces = self.face.detectMultiScale(gray, 1.1, 5, minSize=(40, 40))
                frames.append({
                    "frame_index": int(idx),
                    "time_s": round(idx / fps, 2) if fps > 0 else None,
                    "manipulation": r["manipulation"].get("score"),
                    "ai_generated": r["ai_generated"].get("score"),
                    "context": r["context"].get("score"),
                    "faces": int(len(faces)),
                    "thumb": _thumb_b64(fr),
                })
        finally:
            cap.release()
        if not frames:
            raise ValueError("No frames could be decoded")

        def agg(key):
            v = [f[key] for f in frames if f[key] is not None]
            return (round(float(np.mean(v)), 4), round(float(np.max(v)), 4)) if v else (None, None)

        man_mean, man_max = agg("manipulation")
        ai_mean, ai_max = agg("ai_generated")
        ctx_mean, _ = agg("context")
        technical = [s for s in (man_mean, ai_mean) if s is not None]
        tech_score = round(max(technical), 4) if technical else None
        reasons = []
        if man_mean is not None:
            reasons.append(f"Mean frame manipulation score {man_mean:.2f} (max {man_max:.2f}) over {len(frames)} key frames.")
        if ai_mean is not None:
            reasons.append(f"Mean AI-generated score {ai_mean:.2f} (max {ai_max:.2f}).")
        reasons.append(f"{cuts} abrupt scene discontinuities detected in {len(hist_d) + 1} sampled frames "
                       "(normal for edited news footage; many cuts in a 'single-shot' clip can indicate splicing).")
        face_frames = sum(1 for f in frames if f["faces"] > 0)
        if face_frames:
            warnings.append(f"Faces detected in {face_frames}/{len(frames)} key frames: face-swap deepfakes cannot be "
                            "ruled out by this system.")
        return {
            "metadata": {"frames": n, "fps": round(fps, 2), "duration_s": round(duration, 2) if duration else None,
                         "resolution": [w, h], "codec": codec},
            "technical_score": tech_score,
            "authenticity_score": round(1 - tech_score, 4) if tech_score is not None else None,
            "manipulation_mean": man_mean, "manipulation_max": man_max,
            "ai_generated_mean": ai_mean, "context_mean": ctx_mean,
            "temporal": {"scene_cuts": cuts, "median_hist_distance": round(med, 4),
                         "series": [round(x, 4) for x in hist_d.tolist()]},
            "frames": frames, "reasons": reasons, "warnings": warnings,
        }
