"""Train the (optional) AI-generated image detector.

Works with any dataset in the folder layout
    <root>/train/REAL/*.jpg   <root>/train/FAKE/*.jpg
    <root>/test/REAL/*.jpg    <root>/test/FAKE/*.jpg
which is exactly the layout of CIFAKE (Bird & Lotfi, 2024; Kaggle:
'birdy654/cifake-real-and-ai-generated-synthetic-images'). Folder names are
case-insensitive; 'real'/'fake' and 'nature'/'ai' (GenImage) are accepted.

Model: frozen ImageNet EfficientNet-B0 embedding -> StandardScaler ->
LogisticRegression (C chosen on a validation split carved from train).
Saved to models/ai_generated_effnet_lr.joblib and picked up automatically by
the web app.

Usage:
    python -m training.train_ai_generated --root D:\\datasets\\cifake --max-per-class 5000
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

import config  # noqa: E402
from evaluation import metrics as M  # noqa: E402

REAL_NAMES = {"real", "nature", "authentic", "0_real"}
FAKE_NAMES = {"fake", "ai", "synthetic", "generated", "1_fake"}
EXT = (".jpg", ".jpeg", ".png", ".webp", ".bmp")


def list_split(root: Path, split: str, max_per_class: int, rng):
    base = root / split
    if not base.exists():
        cands = [d for d in root.iterdir() if d.is_dir() and d.name.lower() == split]
        if not cands:
            raise FileNotFoundError(f"{base} not found")
        base = cands[0]
    paths, labels = [], []
    for d in base.iterdir():
        if not d.is_dir():
            continue
        name = d.name.lower()
        lab = 0 if name in REAL_NAMES else 1 if name in FAKE_NAMES else None
        if lab is None:
            print(f"  skipping unknown class folder {d}")
            continue
        files = sorted(str(p) for p in d.rglob("*") if p.suffix.lower() in EXT)
        rng.shuffle(files)
        files = files[:max_per_class]
        paths += files
        labels += [lab] * len(files)
    if len(set(labels)) < 2:
        raise ValueError(f"{base} must contain both a real and a fake class folder")
    return paths, np.array(labels)


def embed(paths, batch=64):
    from ml.image.cnn_features import get_extractor, load_image_list
    ext = get_extractor()
    out = []
    for i in range(0, len(paths), batch):
        out.append(ext.embed(load_image_list(paths[i:i + batch]), batch))
        print(f"  embedded {min(i + batch, len(paths))}/{len(paths)}", end="\r")
    print()
    return np.concatenate(out)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="dataset root with train/ and test/")
    ap.add_argument("--max-per-class", type=int, default=5000)
    ap.add_argument("--out", default=str(config.AI_GENERATED_MODEL))
    ap.add_argument("--report-name", default="ai_generated_results.json")
    a = ap.parse_args(argv)
    rng = np.random.default_rng(config.RANDOM_SEED)
    root = Path(a.root)
    tr_p, tr_y = list_split(root, "train", a.max_per_class, rng)
    te_p, te_y = list_split(root, "test", a.max_per_class, rng)
    print(f"train {len(tr_p)} images, test {len(te_p)} images")
    Xtr, Xte = embed(tr_p), embed(te_p)
    Xa, Xv, ya, yv = train_test_split(Xtr, tr_y, test_size=0.15, stratify=tr_y, random_state=config.RANDOM_SEED)
    best = None
    for C in (0.01, 0.1, 1.0):
        pipe = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(C=C, max_iter=5000))])
        auc = M.compute(yv, pipe.fit(Xa, ya).predict_proba(Xv)[:, 1])["roc_auc"]
        print(f"  val C={C} AUC={auc}")
        if best is None or auc > best[1]:
            best = (C, auc)
    pipe = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(C=best[0], max_iter=5000))])
    pipe.fit(Xtr, tr_y)
    test = M.compute(te_y, pipe.predict_proba(Xte)[:, 1])
    print("TEST:", {k: test[k] for k in ("accuracy", "f1_fake", "roc_auc")})
    joblib.dump({"pipeline": pipe, "test_metrics": test, "dataset": str(root.name), "C": best[0]}, a.out, compress=3)
    M.save_json({"dataset_root": root.name, "n_train": len(tr_p), "n_test": len(te_p), "selected_C": best[0],
                 "val_auc": best[1], "test": test}, a.report_name)
    return test


if __name__ == "__main__":
    os.environ.setdefault("PYTHONWARNINGS", "ignore")
    main()
