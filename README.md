# VeriScan — Multimodal Deep Fake News Detection System

**BSCS Final Year Project:** *Multimodal Deep Fake News Detection System Using Artificial Intelligence and Deep Learning*

A complete, locally runnable web application. It analyses **news text, images and videos**, combines the modalities into one **explainable** assessment, and stores every analysis in a database. All models were trained and evaluated in this repository on public datasets. Every reported number comes from the included scripts (see [`reports/EVALUATION_REPORT.md`](reports/EVALUATION_REPORT.md)).

> **Scientific honesty notice.** The system scores *patterns* learned from labelled data. It does **not** verify facts, and its scores are **not** the probability that a story is true. Section [Limitations](#limitations) lists what the system cannot do.

---

## 1. Features

| Page | What it does |
|---|---|
| **Dashboard** | Usage statistics, model status, measured test performance, recent analyses |
| **Text detection** | TF-IDF + logistic regression (news-headline or social-post model). Exact word-level contributions, highlighted text, style indicators |
| **Image analysis** | Block-level forensic tampering detector with a heatmap, Error Level Analysis, EfficientNet-B0 content model with class-activation map, metadata/EXIF inspection (camera, editing software, AI-generator tags), optional AI-generated-image detector |
| **Video analysis** | Key-frame sampling, per-frame forensics, face detection, temporal discontinuity chart, codec/duration metadata |
| **Multimodal detection** | Text + image (+ video) analysed separately, then combined by **reliability-weighted late fusion**. Adds conflict detection, evidence flags and optional CLIP text–image consistency |
| **History** | Every analysis stored in SQLite. Filter, reopen the full explanation, delete |
| **About** | Architecture, datasets, full metric tables, limitations |

Every result shows the **final verdict** (LIKELY FAKE / UNCERTAIN / LIKELY REAL / INSUFFICIENT EVIDENCE), the **model score** and its decisiveness, **per-modality scores and fusion weights** (chart), **reasons**, **warnings**, **evidence flags**, visual evidence and a disclaimer.

## 2. Measured results (held-out test data)

| Model | Test set | Accuracy | F1 (fake) | ROC-AUC |
|---|---|---:|---:|---:|
| Text, news headlines (TF-IDF + LR) | FakeNewsNet test, 3 183 titles | 0.833 | 0.672 | 0.874 |
| Text, social posts (char TF-IDF + LR) | MediaEval 2015 test (unseen events) | 0.764 | 0.807 | 0.844 |
| Text, social posts | MediaEval 2016 test (unseen events) | 0.648 | 0.675 | 0.690 |
| Image content (EfficientNet-B0 + LR) | MediaEval 2016 test images | 0.567 | 0.602 | 0.658 |
| Image forensics, image level | Synthetic tampering test (426 images, unseen photos) | 0.688 | 0.714 | 0.793 |
| Image forensics, 32 px block localisation | same | — | — | 0.906 |
| **Multimodal fusion** | MediaEval 2016 test (1 117 posts, unseen events) | 0.602 | 0.619 | 0.635 |

Key findings, reported as measured:
* On unseen events, fusion did **not** beat the best single modality. With only 11 training events, modality reliabilities do not transfer well between event sets. The ablations are in the evaluation report.
* The forensic detector's false-positive rate on authentic held-out photos is **40 %**. Its AUC against MediaEval "misleading-use" labels is **0.48** (chance), because most such images are real photos used out of context.
* The learned text–image consistency model did not beat chance (AUC 0.34 / 0.57). It is therefore **not used** for decisions. The app uses CLIP when it is installed.

Regenerate everything with `python train_all.py` (≈6 min on a 4-core CPU, no GPU). A from-scratch run reproduced the numbers above exactly.

## 3. Architecture

```
            ┌────────────── Flask web app (app/) ─────────────┐
 Browser ──►│ pages + JSON API ──► DetectionService (ml/service.py)│──► SQLite (database/)
            └──────────────────────────┬──────────────────────┘
       ┌──────────────────┬────────────┴───────────┬──────────────────┐
  ml/text/            ml/image/                 ml/video/          ml/fusion/
  TF-IDF + LR         manipulation.py (GBDT,     frame sampler,     fusion.py (reliability
  (exact word         JPEG-ghost/noise blocks)   per-frame image    weights, conflicts, flags)
  contributions)      cnn_features.py (EffNet-B0) forensics, faces,  consistency.py (CLIP,
                      image_analyzer.py (+CAM,   temporal cuts      optional)
                      metadata, AI-gen head)
```

Fusion: `score = Σ w_m·s_m / Σ w_m` over the available modalities, with `w_m = max(AUC_val(m) − 0.5, 0)`. The weights are measured on a **validation** set of events that no base model saw. Verdict bands: ≥ 0.65 likely fake, ≤ 0.35 likely real, otherwise uncertain. Evidence flags (AI-generator metadata, a strong AI-detector score, low CLIP consistency) cannot be outweighed into a "LIKELY REAL" verdict.

### Project structure
```
fakenews_fyp/
├── app/                    Flask app: __init__.py (factory, error handlers), routes.py (pages + API),
│   ├── database.py         SQLite access layer
│   ├── templates/          7 pages + detail/error (Jinja2, Bootstrap 5)
│   └── static/             css, js (result renderer, Chart.js), vendor libs (offline), uploads/
├── ml/                     inference code
│   ├── preprocessing/      text cleaning
│   ├── text/               text model + explanations
│   ├── image/              forensics, block manipulation detector, CNN features, analyzer
│   ├── video/              video analyzer
│   ├── fusion/             fusion engine, text-image consistency
│   └── service.py          single entry point used by the app
├── data_prep/              download_datasets.py, prepare_datasets.py (leak-free splits), synthetic_tampering.py
├── training/               train_text.py, train_image.py, train_fusion.py, train_ai_generated.py
├── evaluation/             metrics.py, evaluate_all.py  → reports/
├── models/                 trained artefacts (+ EfficientNet-B0 ImageNet weights)
├── reports/                EVALUATION_REPORT.md, *.json, figures/ (ROC curves, confusion matrices)
├── database/schema.sql     DB schema (DB file is created automatically)
├── sample_inputs/          texts.json, images/, videos/, multimodal/
├── tests/                  pytest suite (13 tests)
├── config.py  run.py  train_all.py
└── *.bat                   Windows helpers
```

## 4. Installation and running on Windows

Requirements: **Windows 10/11, Python 3.10–3.13 (64-bit)** from python.org (tick *Add python.exe to PATH*), about 3 GB free disk space, and internet for the first install. No GPU needed.

**Option A: double-click**
1. Extract the project folder (avoid very long paths, e.g. use `C:\fyp\fakenews_fyp`).
2. Double-click **`setup_windows.bat`** (creates `.venv`, installs packages, 5–15 min).
3. Double-click **`run_windows.bat`**. The browser opens at http://127.0.0.1:5000.

**Option B: Command Prompt**

```bash
cd C:\fyp\fakenews_fyp
```

```bash
python -m venv .venv
```

```bash
.venv\Scripts\activate
```

```bash
pip install -r requirements.txt
```

```bash
python run.py
```

Then open http://127.0.0.1:5000. On Linux/macOS use `source .venv/bin/activate` instead.

The trained models are included, so the app works immediately. The first image or video request takes a few seconds while EfficientNet loads.

**Optional CLIP consistency** (internet needed once, ~600 MB):

```bash
pip install -r requirements-optional.txt
```

## 5. Retraining and evaluation

```bash
python train_all.py
```

This runs `download_datasets → prepare_datasets → train_text → train_image → train_fusion → evaluate_all`, writing `models/*` and `reports/*`. On Windows you can double-click `train_all_windows.bat`. Each step can also run alone, e.g. `python -m training.train_text`.

**AI-generated image detector (optional).** Download CIFAKE (Kaggle: *birdy654/cifake-real-and-ai-generated-synthetic-images*) and extract it, e.g. to `D:\data\cifake` with `train\REAL`, `train\FAKE`, `test\REAL`, `test\FAKE`. Then run:

```bash
python -m training.train_ai_generated --root D:\data\cifake --max-per-class 5000
```

The app picks up `models/ai_generated_effnet_lr.joblib` automatically on the next start, and `evaluate_all` adds its metrics to the report.

**Tests:**

```bash
python -m pytest -q
```

## 6. Datasets

| Dataset | Used for | Source / licence |
|---|---|---|
| FakeNewsNet (PolitiFact + GossipCop titles) | news-headline text model | github.com/KaiDMML/FakeNewsNet. Shu et al., *Big Data* 2020 |
| MediaEval Verifying Multimedia Use 2015/2016 | social-post text model, image content model, fusion validation/test, source photos for synthetic tampering | github.com/MKLab-ITI/image-verification-corpus. Boididou et al., *IJMIR* 2018 |
| Synthetic tampering (generated) | forensic detector | `data_prep/synthetic_tampering.py`, built from authentic MediaEval photos with pixel masks |
| CIFAKE (optional) | AI-generated image detector | Kaggle. Bird & Lotfi, *IEEE Access* 2024 |
| ImageNet-pretrained EfficientNet-B0 | image features | github.com/lukemelas/EfficientNet-PyTorch release weights |

**Leakage controls:** FakeNewsNet titles are normalised and de-duplicated *before* splitting, and conflicting duplicates are dropped. MediaEval train/validation/test sets have **disjoint events**. Synthetic train/test sets use disjoint source photos. Hyper-parameters and fusion weights are chosen only on training/validation data.

**Datasets investigated but not used:** Fakeddit and Weibo (hosted on Google Drive or behind request forms), FaceForensics++/DFDC/Celeb-DF (access request plus GPU training), CASIA/Columbia tampering sets. These hosts were unreachable from the build environment. `prepare_datasets.py` is modular, so a new loader can be added for any of them.

## 7. API

| Method | Endpoint | Body |
|---|---|---|
| POST | `/api/analyze/text` | JSON `{"text": "...", "kind": "news" \| "social"}` |
| POST | `/api/analyze/image` | multipart `image` |
| POST | `/api/analyze/video` | multipart `video` |
| POST | `/api/analyze/multimodal` | multipart `text`, `kind`, `image`, `video` (any subset, at least one) |
| GET | `/api/history?limit=&offset=&modality=&verdict=` | — |
| GET / DELETE | `/api/history/<id>` | — |
| GET | `/api/status` | model availability + stats |

Errors return JSON `{"error": ...}`: 400 for invalid input, 413 for too-large files (100 MB limit), 422 for unreadable media, 500 for unexpected errors (logged).

Example:

```bash
curl -X POST http://127.0.0.1:5000/api/analyze/text -H "Content-Type: application/json" -d "{\"text\":\"BREAKING: shocking secret revealed\",\"kind\":\"news\"}"
```

## 8. Database schema

`database/schema.sql` defines the table `detections`: id, created_at, modality, text_kind, input_summary, text_input, file_name, stored_file, verdict, fused_score, text_score, image_score, video_score, n_warnings, elapsed_s, result_json (the full explanation). It is indexed on time, modality and verdict. The DB file `database/detections.db` is created on first start. Set the env var `FND_DB_PATH` to relocate it.

## 9. Explainability

* **Text:** the linear model makes the contributions exact. Each word's push is tf-idf × coefficient; for the character-n-gram model, n-gram contributions are summed per word. Words are highlighted red/green in the original text.
* **Image:** per-block tamper-probability heatmap, ELA map, a class-activation map of the CNN head (exact for a linear head on pooled features), metadata findings, and recognised objects.
* **Video:** scored key frames, a temporal-distance chart and scene-cut count.
* **Fusion:** a per-modality score and weight chart, conflict warnings, evidence flags, and an insufficiency warning when only one modality contributed.

## 10. Limitations

1. Nothing here verifies facts. The text models learn the style and topic patterns of 2012–2018 data.
2. Pixel forensics cannot detect authentic photos reused in a false context. Social-media recompression weakens forensic traces, and the false-positive rate is high (40 % on held-out authentic photos).
3. The AI-generated image detector is **not trained** in the delivered build (dataset unreachable during the build). The training script is included and tested.
4. The video module has **no face-swap deepfake model** and does **not analyse audio**. It applies frame-level image forensics plus temporal statistics.
5. Text–image semantic consistency needs CLIP (optional install). It was not executable in the build environment, so its thresholds are untested heuristics.
6. MediaEval has few events, and post-level metrics are dominated by a few viral images. Fusion gains could not be demonstrated.

**Future work:** train on Fakeddit (more than 1M multimodal posts), fine-tune CLIP for consistency, add a FaceForensics++ video model, use a calibrated stacking model once more validation events are available, and add real tampering benchmarks (CASIA v2, IMD2020).

## 11. Demo walkthrough (for the viva)
1. **Text page:** click a *FakeNewsNet test · fake* sample and show the highlighted words. Then show a *real* sample.
2. **Image page:** upload `sample_inputs/images/synthetic_tampered_splice_5.jpg`. The heatmap lights up the pasted region; compare it with `..._groundtruth_mask.png`. Then upload an authentic sample and discuss false positives honestly.
3. **Video page:** upload `sample_inputs/videos/sample_two_scene_clip.mp4`. The cut at 4 s shows as a spike in the temporal chart, and the frames after it (the spliced image) score higher.
4. **Multimodal page:** use the text from `sample_inputs/multimodal/samples.json` with its image. Walk through the fusion weights chart and warnings.
5. **History and About:** show stored results and the evaluation tables.

## 12. Troubleshooting
* `pip` fails on torch: use 64-bit Python 3.10–3.13, then run `pip install torch torchvision` separately.
* Port in use: `python run.py --port 5050`.
* Face detection error with OpenCV 5: keep `opencv-python-headless<5` (as in requirements.txt).
* EfficientNet weights missing offline: they ship in `models/cnn_weights/`. If deleted, the first run re-downloads them from GitHub.
