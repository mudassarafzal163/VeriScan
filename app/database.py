"""SQLite persistence for detection history (standard-library sqlite3)."""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path

import config

SCHEMA = (config.ROOT / "database" / "schema.sql").read_text()


def _path() -> Path:
    import os
    return Path(os.environ.get("FND_DB_PATH", config.DB_PATH))


@contextmanager
def connect():
    p = _path()
    p.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(p, timeout=10)
    con.row_factory = sqlite3.Row
    try:
        yield con
        con.commit()
    finally:
        con.close()


def init_db():
    with connect() as con:
        con.executescript(SCHEMA)


def _part_score(result, *keys):
    cur = result.get("parts", {})
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    return cur if isinstance(cur, (int, float)) else None


def save_detection(result: dict, text_kind=None, file_name=None, stored_file=None) -> int:
    with connect() as con:
        cur = con.execute(
            """INSERT INTO detections (modality, text_kind, input_summary, text_input, file_name, stored_file,
                   verdict, fused_score, text_score, image_score, video_score, n_warnings, elapsed_s, result_json)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (result["modality"], text_kind, (result.get("summary") or "")[:300], result.get("text_input"),
             file_name, stored_file, result["verdict"], result.get("score"),
             _part_score(result, "text", "fake_score"),
             _part_score(result, "image", "manipulation", "score"),
             _part_score(result, "video", "manipulation_mean"),
             len(result.get("warnings", [])), result.get("elapsed_s"), json.dumps(result)))
        return cur.lastrowid


def list_detections(limit: int = 50, offset: int = 0, modality: str | None = None, verdict: str | None = None):
    q = ("SELECT id, created_at, modality, text_kind, input_summary, file_name, verdict, fused_score, "
         "text_score, image_score, video_score, n_warnings FROM detections")
    cond, args = [], []
    if modality:
        cond.append("modality = ?")
        args.append(modality)
    if verdict:
        cond.append("verdict = ?")
        args.append(verdict)
    if cond:
        q += " WHERE " + " AND ".join(cond)
    q += " ORDER BY id DESC LIMIT ? OFFSET ?"
    args += [int(limit), int(offset)]
    with connect() as con:
        return [dict(r) for r in con.execute(q, args)]


def get_detection(det_id: int):
    with connect() as con:
        r = con.execute("SELECT * FROM detections WHERE id = ?", (det_id,)).fetchone()
    if not r:
        return None
    d = dict(r)
    d["result"] = json.loads(d.pop("result_json"))
    return d


def delete_detection(det_id: int) -> bool:
    with connect() as con:
        return con.execute("DELETE FROM detections WHERE id = ?", (det_id,)).rowcount > 0


def stats() -> dict:
    with connect() as con:
        total = con.execute("SELECT COUNT(*) FROM detections").fetchone()[0]
        by_mod = {r[0]: r[1] for r in con.execute("SELECT modality, COUNT(*) FROM detections GROUP BY modality")}
        by_ver = {r[0]: r[1] for r in con.execute("SELECT verdict, COUNT(*) FROM detections GROUP BY verdict")}
        recent = [dict(r) for r in con.execute(
            "SELECT id, created_at, modality, input_summary, verdict, fused_score FROM detections "
            "ORDER BY id DESC LIMIT 5")]
    return {"total": total, "by_modality": by_mod, "by_verdict": by_ver, "recent": recent}
