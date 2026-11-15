-- SQLite schema for the detection history.
-- Created automatically by app/database.py on first start.

CREATE TABLE IF NOT EXISTS detections (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at     TEXT    NOT NULL DEFAULT (datetime('now')),
    modality       TEXT    NOT NULL CHECK (modality IN ('text', 'image', 'video', 'multimodal')),
    text_kind      TEXT,                 -- 'news' | 'social' | NULL
    input_summary  TEXT,                 -- first characters of the text or the file name
    text_input     TEXT,                 -- full text input (if any)
    file_name      TEXT,                 -- original uploaded file name(s)
    stored_file    TEXT,                 -- path of the saved upload (relative to app/static)
    verdict        TEXT    NOT NULL,     -- LIKELY FAKE | LIKELY REAL | UNCERTAIN | INSUFFICIENT EVIDENCE
    fused_score    REAL,                 -- fused model score in [0,1] (NOT a probability of truth)
    text_score     REAL,
    image_score    REAL,                 -- image forensic manipulation score
    video_score    REAL,
    n_warnings     INTEGER NOT NULL DEFAULT 0,
    elapsed_s      REAL,
    result_json    TEXT    NOT NULL      -- full explanation payload
);

CREATE INDEX IF NOT EXISTS idx_detections_created ON detections (created_at DESC);
CREATE INDEX IF NOT EXISTS idx_detections_modality ON detections (modality);
CREATE INDEX IF NOT EXISTS idx_detections_verdict ON detections (verdict);
