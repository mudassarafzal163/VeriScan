"""Train and evaluate the image models.

A) Context ('misleading-use') model on MediaEval VMU images
   EfficientNet-B0 (ImageNet) embedding -> StandardScaler -> LogisticRegression.
   C chosen by leave-event-out CV on 2015 dev; tested on the event-disjoint
   2015 and 2016 test images.
B) Forensic manipulation detector on a synthetic tampering set built from
   AUTHENTIC ('real'-labelled) MediaEval photos. Train sources = dev photos,
   test sources = test2015/test2016 photos (no source photo in both).
   Also evaluated on the real MediaEval fake/real labels to show what it
   does NOT detect (out-of-context reuse).

Run:  python -m training.train_image
"""
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

import config  # noqa: E402
from data_prep import synthetic_tampering as st  # noqa: E402
from evaluation import metrics as M  # noqa: E402
from ml.image import forensics as fx  # noqa: E402
from ml.image import manipulation as mn  # noqa: E402

P = config.DATA_PROCESSED
EMB_CACHE = P / "mediaeval_image_embeddings.npz"


def get_embeddings(imgs_df):
    if EMB_CACHE.exists():
        z = np.load(EMB_CACHE, allow_pickle=True)
        if list(z["ids"]) == list(imgs_df.image_id):
            return z["emb"]
    from ml.image.cnn_features import get_extractor
    ext = get_extractor()
    pil = [fx.to_rgb(fx.load_rgb(config.ROOT / p)) for p in imgs_df.path]
    emb = ext.embed(pil)
    np.savez_compressed(EMB_CACHE, ids=np.array(imgs_df.image_id), emb=emb)
    return emb


def context_pipe(C):
    return Pipeline([("scaler", StandardScaler()),
                     ("clf", LogisticRegression(C=C, max_iter=5000, class_weight="balanced"))])


def train_context():
    imgs = pd.read_csv(P / "mediaeval_images.csv")
    emb = get_embeddings(imgs)
    dev = (imgs.split == "dev").values
    Xd, yd, gd = emb[dev], imgs.label.values[dev], imgs.event.values[dev]
    gkf = GroupKFold(n_splits=5)
    cv, best = {}, (None, -1, None)
    for C in (0.001, 0.01, 0.1):
        oof = np.zeros(len(yd))
        for tr, va in gkf.split(Xd, yd, gd):
            oof[va] = context_pipe(C).fit(Xd[tr], yd[tr]).predict_proba(Xd[va])[:, 1]
        cv[f"C{C}"] = M.compute(yd, oof)
        print(f"[context] leave-event-out CV C={C} AUC={cv[f'C{C}']['roc_auc']}")
        if cv[f"C{C}"]["roc_auc"] > best[1]:
            best = (C, cv[f"C{C}"]["roc_auc"], oof)
    C = best[0]
    final = context_pipe(C).fit(Xd, yd)
    res = {"dataset": "MediaEval VMU images", "selected_C": C, "leave_event_out_cv": cv, "test": {}}
    curves = {}
    for s in ("test2015", "test2016"):
        m = (imgs.split == s).values
        p = final.predict_proba(emb[m])[:, 1]
        res["test"][s] = M.compute(imgs.label.values[m], p)
        curves[s] = (imgs.label.values[m], p)
        print(f"[context] TEST {s}:", {k: res['test'][s][k] for k in ('accuracy', 'f1_macro', 'roc_auc')})
    M.plot_roc(curves, "Image context model (event-disjoint tests)", "roc_image_context.png")
    joblib.dump({"pipeline": final, "test_metrics": res["test"]}, config.IMAGE_CONTEXT_MODEL, compress=3)
    M.save_json(res, "image_context_results.json")
    return res


def build_synthetic(sources, seed, n_auth=3, n_tamp=3):
    """Returns per-image list of dicts with block features/labels."""
    rng = random.Random(seed)
    imgs = [fx.to_rgb(fx.load_rgb(config.ROOT / p)) for p in sources]
    imgs = [fx.limit_size(i, 1024) for i in imgs if min(i.size) >= 128]
    items, examples = [], []
    for si, img in enumerate(imgs):
        donors = imgs[:si] + imgs[si + 1:]
        for var, mask, lab, op in st.make_variants(img, donors, rng, n_auth, n_tamp):
            F, grid, prep = mn.block_features(var)
            bl = mn.block_labels(mask, grid, prep.size)
            items.append({"F": F, "grid": grid, "bl": bl, "label": lab, "op": op, "src": si})
            if si % 9 == 0 and len(examples) < 8:
                examples.append((var, mask, lab, op))
    return items, examples


def _block_xy(items):
    X = np.concatenate([it["F"] for it in items])
    frac = np.concatenate([it["bl"] for it in items])
    g = np.concatenate([np.full(len(it["bl"]), it["src"]) for it in items])
    keep = (frac <= 0.05) | (frac >= 0.5)  # drop ambiguous boundary blocks
    return X[keep], (frac[keep] >= 0.5).astype(int), g[keep]


def _block_model():
    return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08, max_leaf_nodes=31,
                                          l2_regularization=1.0, class_weight="balanced",
                                          random_state=config.RANDOM_SEED)


def train_manipulation():
    imgs = pd.read_csv(P / "mediaeval_images.csv")
    real = imgs[imgs.label == 0]
    tr_src = real[real.split == "dev"].path.tolist()
    te_src = real[real.split != "dev"].path.tolist()
    print(f"[manip] building synthetic set: {len(tr_src)} train sources, {len(te_src)} test sources")
    train_items, _ = build_synthetic(tr_src, config.RANDOM_SEED)
    test_items, examples = build_synthetic(te_src, config.RANDOM_SEED + 1)

    # stage-1 out-of-fold block probabilities (grouped by source photo) -> stage-2 training data
    src = np.array([it["src"] for it in train_items])
    oof_img_feats = np.zeros((len(train_items), 7), np.float32)
    for tr_s, va_s in GroupKFold(n_splits=4).split(src, groups=src):
        tr_set = set(src[tr_s])
        bm = _block_model().fit(*_block_xy([it for it in train_items if it["src"] in tr_set])[:2])
        for i in va_s:
            it = train_items[i]
            oof_img_feats[i] = mn.image_features(bm.predict_proba(it["F"])[:, 1], it["grid"])
    ytr = np.array([it["label"] for it in train_items])
    image_model = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=2000))])
    image_model.fit(oof_img_feats, ytr)
    Xb, yb, _ = _block_xy(train_items)
    block_model = _block_model().fit(Xb, yb)
    bundle = {"block_model": block_model, "image_model": image_model}

    # held-out evaluation on synthetic edits of unseen source photos
    yte = np.array([it["label"] for it in test_items])
    ops = np.array([it["op"] for it in test_items])
    bps = [block_model.predict_proba(it["F"])[:, 1] for it in test_items]
    p = image_model.predict_proba(np.array([mn.image_features(b, it["grid"]) for b, it in zip(bps, test_items)]))[:, 1]
    Xbt, ybt, _ = _block_xy(test_items)
    pbt = block_model.predict_proba(Xbt)[:, 1]
    res = {"dataset": "Synthetic tampering from authentic MediaEval photos",
           "train_images": int(len(ytr)), "test_images": int(len(yte)),
           "train_blocks": int(len(yb)), "test_blocks": int(len(ybt)),
           "test_image_level": M.compute(yte, p),
           "test_block_level_localisation": M.compute(ybt, pbt),
           "test_recall_by_operation": {}}
    for op in ("splice", "copy_move", "retouch"):
        res["test_recall_by_operation"][op] = round(float((p[ops == op] >= 0.5).mean()), 4)
    res["test_false_positive_rate_authentic"] = round(float((p[ops == "authentic"] >= 0.5).mean()), 4)
    # sanity check on the real MediaEval labels (misuse != pixel tampering)
    pm = np.array([mn.score(bundle, fx.load_rgb(config.ROOT / q))[0] for q in imgs.path])
    res["auc_on_mediaeval_misuse_labels"] = round(float(roc_auc_score(imgs.label, pm)), 4)
    print("[manip] TEST image-level:", {k: res["test_image_level"][k] for k in ("accuracy", "f1_fake", "roc_auc")},
          "| block AUC:", res["test_block_level_localisation"]["roc_auc"],
          "| recall by op:", res["test_recall_by_operation"], "| FPR:", res["test_false_positive_rate_authentic"],
          "| AUC on MediaEval misuse labels:", res["auc_on_mediaeval_misuse_labels"])
    M.plot_confusion(res["test_image_level"]["confusion_matrix"], "Manipulation (synthetic test)", "cm_manipulation.png")
    M.plot_roc({"image-level": (yte, p), "block-level": (ybt, pbt),
                "MediaEval misuse labels": (imgs.label.values, pm)},
               "Forensic manipulation detector", "roc_manipulation.png")
    bundle["test_metrics"] = res["test_image_level"]
    joblib.dump(bundle, config.MANIPULATION_MODEL, compress=3)
    out = config.ROOT / "sample_inputs" / "images"
    out.mkdir(parents=True, exist_ok=True)
    for i, (im, mask, lab, op) in enumerate(examples):
        name = f"synthetic_{'tampered_' + op if lab else 'authentic'}_{i}"
        im.save(out / f"{name}.jpg", quality=92)
        if lab:
            mask.save(out / f"{name}_groundtruth_mask.png")
    M.save_json(res, "manipulation_results.json")
    return res


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("all", "context"):
        train_context()
    if which in ("all", "manipulation"):
        train_manipulation()
