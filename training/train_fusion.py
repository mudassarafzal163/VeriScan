"""Fit and evaluate the multimodal fusion (and the experimental learned
text-image consistency model).

Protocol (no leakage, no tuning on the final test set):
  * Base models (social text, image context, manipulation) are trained on
    MediaEval 2015 DEV only.
  * MediaEval 2015 TEST serves as the fusion VALIDATION set: each modality's
    reliability (AUC) is measured there and turned into a fusion weight
    w = max(AUC - 0.5, 0).
  * MediaEval 2016 TEST is the final, untouched test set (events disjoint
    from both other sets).
  * Ablations on the same test posts: every single modality, an unweighted
    average, and a stacked logistic-regression fusion fitted on validation.
  * The learned consistency model is evaluated (matched vs mismatched pairs)
    and reported; it is not used at inference because it did not beat chance.
Run:  python -m training.train_fusion
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

import config  # noqa: E402
from evaluation import metrics as M  # noqa: E402
from ml.fusion import consistency as cons  # noqa: E402
from ml.image import forensics as fx  # noqa: E402
from ml.image import manipulation as mn  # noqa: E402
from ml.text.text_model import TextModel  # noqa: E402

P = config.DATA_PROCESSED
FEATURES = ["text", "image_context", "manipulation"]


def first_image(ids, valid):
    for i in str(ids).split(","):
        if i in valid:
            return i
    return None


def load_posts(split, valid):
    d = pd.read_csv(P / f"mediaeval_posts_{split}.csv").dropna(subset=["text"])
    d["image_id"] = d.image_ids.map(lambda s: first_image(s, valid))
    return d[d.image_id.notna()].reset_index(drop=True)


def mismatched_auc(bundle, posts, emb_of, rng):
    """AUC of consistency: true (text,image) pairs vs pairs with an image from another event."""
    imgs_by_event = posts.groupby("event").image_id.unique().to_dict()
    other = []
    for ev in posts.event:
        pool = np.concatenate([v for k, v in imgs_by_event.items() if k != ev])
        other.append(rng.choice(pool))
    s_true = cons.learned_scores(bundle, posts.text.tolist(), np.stack([emb_of[i] for i in posts.image_id]))
    s_false = cons.learned_scores(bundle, posts.text.tolist(), np.stack([emb_of[i] for i in other]))
    y = np.r_[np.ones(len(s_true)), np.zeros(len(s_false))]
    return float(roc_auc_score(y, np.r_[s_true, s_false])), s_true


def main():
    rng = np.random.default_rng(config.RANDOM_SEED)
    imgs = pd.read_csv(P / "mediaeval_images.csv")
    z = np.load(P / "mediaeval_image_embeddings.npz", allow_pickle=True)
    emb_of = dict(zip(z["ids"], z["emb"]))
    valid = set(imgs.image_id)
    dev = load_posts("dev", valid)
    t15, t16 = load_posts("test2015", valid), load_posts("test2016", valid)

    # ---------------------------------------------------------- consistency
    dev_bal = dev.drop_duplicates(["text", "image_id"]).groupby("image_id").head(25)
    bundle = cons.build_learned(dev_bal.text.tolist(), np.stack([emb_of[i] for i in dev_bal.image_id]))
    res = {"consistency": {"train_pairs": int(len(dev_bal))}}
    for name, d in (("test2015", t15), ("test2016", t16)):
        auc, s = mismatched_auc(bundle, d, emb_of, rng)
        d["consistency"] = s
        res["consistency"][f"{name}_matched_vs_mismatched_auc"] = round(auc, 4)
    bundle["test_auc"] = res["consistency"]
    joblib.dump(bundle, cons.CONSISTENCY_MODEL, compress=3)
    print("[consistency]", res["consistency"])

    # ---------------------------------------------------------- base scores
    text_model = TextModel(config.TEXT_SOCIAL_MODEL, "social")
    ctx = joblib.load(config.IMAGE_CONTEXT_MODEL)["pipeline"]
    manip_bundle = joblib.load(config.MANIPULATION_MODEL)
    # always recomputed from the current model file (a cache could go stale after retraining)
    manip_of = {i: mn.score(manip_bundle, fx.load_rgb(config.ROOT / p))[0] for i, p in zip(imgs.image_id, imgs.path)}
    for d in (t15, t16):
        d["text_raw"] = d["text"]
        d["text"] = text_model.predict_proba(d.text_raw.tolist())
        d["image_context"] = ctx.predict_proba(np.stack([emb_of[i] for i in d.image_id]))[:, 1]
        d["manipulation"] = d.image_id.map(manip_of)

    # ---------------------------------------------------------- fusion
    val, te = t15, t16
    yv, y = val.label.values, te.label.values
    val_auc = {f: float(roc_auc_score(yv, val[f])) for f in FEATURES}
    weights = {f: max(a - 0.5, 0.0) for f, a in val_auc.items()}
    tot = sum(weights.values()) or 1.0
    weights = {f: round(w / tot, 4) for f, w in weights.items()}

    def weighted(d):
        return sum(weights[f] * d[f].values for f in FEATURES)

    stack = Pipeline([("scaler", StandardScaler()),
                      ("clf", LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced"))])
    stack.fit(val[FEATURES].values, yv)
    r = {"validation_set": "test2015", "test_set": "test2016",
         "n_val_posts": int(len(val)), "n_test_posts": int(len(te)),
         "validation_auc": {k: round(v, 4) for k, v in val_auc.items()},
         "fusion_weights": weights,
         "reliability_weighted_fusion": M.compute(y, weighted(te)),
         "ablation_unweighted_average": M.compute(y, te[FEATURES].mean(axis=1).values),
         "ablation_stacked_lr": M.compute(y, stack.predict_proba(te[FEATURES].values)[:, 1])}
    for f in FEATURES:
        r[f"ablation_only_{f}"] = M.compute(y, te[f].values)
    r["note"] = ("Single-modality test results are shown for transparency only; choosing the modality "
                 "with the best TEST score would be test-set leakage, so weights come from validation.")
    res["fusion"] = r
    M.plot_roc({"reliability-weighted fusion": (y, weighted(te)), "stacked LR": (y, stack.predict_proba(te[FEATURES].values)[:, 1]),
                "unweighted average": (y, te[FEATURES].mean(axis=1).values),
                "text only": (y, te.text.values), "image context only": (y, te.image_context.values),
                "manipulation only": (y, te.manipulation.values)},
               "Multimodal fusion vs single modalities (MediaEval 2016 test)", "roc_fusion.png")
    M.plot_confusion(r["reliability_weighted_fusion"]["confusion_matrix"], "Fusion (MediaEval 2016 test)", "cm_fusion.png")
    print("[fusion] val AUC:", r["validation_auc"], "weights:", weights)
    print("[fusion] TEST 2016:", " | ".join(f"{k}: AUC={v['roc_auc']} F1m={v['f1_macro']}"
                                          for k, v in r.items() if isinstance(v, dict) and "roc_auc" in v))
    joblib.dump({"weights": weights, "validation_auc": val_auc, "features": FEATURES,
                 "test_metrics": r["reliability_weighted_fusion"]}, config.FUSION_MODEL, compress=3)
    M.save_json(res, "fusion_results.json")


if __name__ == "__main__":
    main()
