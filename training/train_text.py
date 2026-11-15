"""Train and evaluate the two text classifiers.

A) News-headline model on FakeNewsNet (PolitiFact + GossipCop titles)
   - model selection on the VALIDATION split only
   - the chosen configuration is refit on train+val and evaluated ONCE on test
   - baselines (majority class, Complement Naive Bayes, Linear SVM) reported
B) Social-post model on MediaEval VMU tweets (multilingual -> char n-grams)
   - hyper-parameter selection by leave-event-out GroupKFold on 2015 dev
   - evaluated on the event-disjoint 2015 test and 2016 test sets

Run:  python -m training.train_text
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.calibration import CalibratedClassifierCV  # noqa: E402
from sklearn.dummy import DummyClassifier  # noqa: E402
from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import f1_score  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402
from sklearn.naive_bayes import ComplementNB  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.svm import LinearSVC  # noqa: E402

import config  # noqa: E402
from evaluation import metrics as M  # noqa: E402
from ml.preprocessing.text_preprocess import clean_text  # noqa: E402

P = config.DATA_PROCESSED


def news_vectorizer():
    return TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.9, sublinear_tf=True,
                           max_features=100000, token_pattern=r"(?u)\b[\w#']+\b")


def social_vectorizer():
    return TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3, sublinear_tf=True,
                           max_features=150000)


def train_news():
    tr, va, te = (pd.read_csv(P / f"fakenewsnet_{s}.csv") for s in ("train", "val", "test"))
    Xtr, Xva, Xte = ([clean_text(t) for t in d.text] for d in (tr, va, te))
    ytr, yva, yte = tr.label.values, va.label.values, te.label.values

    candidates = {}
    for C in (0.5, 1, 2, 4, 8):
        candidates[f"logreg_C{C}"] = Pipeline([("tfidf", news_vectorizer()),
                                               ("clf", LogisticRegression(C=C, max_iter=3000, class_weight="balanced"))])
    candidates["complement_nb"] = Pipeline([("tfidf", news_vectorizer()), ("clf", ComplementNB(alpha=0.3))])
    candidates["linear_svm_calibrated"] = Pipeline([("tfidf", news_vectorizer()),
                                                    ("clf", CalibratedClassifierCV(LinearSVC(C=0.5, class_weight="balanced"), cv=3))])
    candidates["majority_baseline"] = Pipeline([("tfidf", news_vectorizer()), ("clf", DummyClassifier(strategy="prior"))])

    val_results = {}
    for name, pipe in candidates.items():
        pipe.fit(Xtr, ytr)
        p = pipe.predict_proba(Xva)[:, 1]
        val_results[name] = M.compute(yva, p)
        print(f"[news] val {name:24s} F1(fake)={val_results[name]['f1_fake']:.4f} AUC={val_results[name]['roc_auc']}")

    # deployed model must be linear (exact word explanations) -> best logistic regression on val
    best_lr = max((k for k in val_results if k.startswith("logreg")), key=lambda k: val_results[k]["f1_fake"])
    C = float(best_lr.split("C")[1])
    final = Pipeline([("tfidf", news_vectorizer()),
                      ("clf", LogisticRegression(C=C, max_iter=3000, class_weight="balanced"))])
    final.fit(Xtr + Xva, np.concatenate([ytr, yva]))
    p_te = final.predict_proba(Xte)[:, 1]
    test = M.compute(yte, p_te)

    # baselines on test, fit on train+val (same protocol)
    test_baselines, curves = {}, {"logreg (deployed)": (yte, p_te)}
    for name in ("complement_nb", "linear_svm_calibrated", "majority_baseline"):
        pipe = candidates[name]
        pipe.fit(Xtr + Xva, np.concatenate([ytr, yva]))
        pb = pipe.predict_proba(Xte)[:, 1]
        test_baselines[name] = M.compute(yte, pb)
        if name != "majority_baseline":
            curves[name] = (yte, pb)
    per_source = {src: M.compute(yte[te.source == src], p_te[(te.source == src).values])
                  for src in te.source.unique()}

    M.plot_confusion(test["confusion_matrix"], "Text (FakeNewsNet test)", "cm_text_news.png")
    M.plot_roc(curves, "Text model - FakeNewsNet test", "roc_text_news.png")
    joblib.dump({"pipeline": final, "selected": best_lr, "test_metrics": test,
                 "domain_note": "Trained on English news headlines (PolitiFact/GossipCop 2017-2018); "
                                "scores reflect writing-style patterns, not fact verification."},
                config.TEXT_NEWS_MODEL, compress=3)
    res = {"dataset": "FakeNewsNet titles", "selected_on_val": best_lr, "validation": val_results,
           "test_deployed": test, "test_baselines": test_baselines, "test_per_source": per_source}
    M.save_json(res, "text_news_results.json")
    print("[news] TEST deployed:", {k: test[k] for k in ("accuracy", "precision_fake", "recall_fake", "f1_fake", "roc_auc")})
    return res


def train_social():
    dev = pd.read_csv(P / "mediaeval_posts_dev.csv").dropna(subset=["text"])
    tests = {s: pd.read_csv(P / f"mediaeval_posts_{s}.csv").dropna(subset=["text"]) for s in ("test2015", "test2016")}
    X = [clean_text(t) for t in dev.text]
    y, groups = dev.label.values, dev.event.values
    gkf = GroupKFold(n_splits=5)

    def oof(C):
        out = np.zeros(len(y))
        for tr, va in gkf.split(X, y, groups):
            pipe = Pipeline([("tfidf", social_vectorizer()),
                             ("clf", LogisticRegression(C=C, max_iter=3000, class_weight="balanced"))])
            pipe.fit([X[i] for i in tr], y[tr])
            out[va] = pipe.predict_proba([X[i] for i in va])[:, 1]
        return out

    cv = {}
    best = (None, -1, None)
    for C in (0.1, 0.5, 2.0):
        o = oof(C)
        f = f1_score(y, o >= 0.5, average="macro")
        cv[f"C{C}"] = M.compute(y, o)
        print(f"[social] leave-event-out CV C={C} macroF1={f:.4f}")
        if f > best[1]:
            best = (C, f, o)
    C = best[0]
    final = Pipeline([("tfidf", social_vectorizer()),
                      ("clf", LogisticRegression(C=C, max_iter=3000, class_weight="balanced"))])
    final.fit(X, y)
    res = {"dataset": "MediaEval VMU tweets", "selected_C": C, "leave_event_out_cv": cv, "test": {}}
    curves = {}
    for s, d in tests.items():
        p = final.predict_proba([clean_text(t) for t in d.text])[:, 1]
        res["test"][s] = M.compute(d.label.values, p)
        res["test"][s + "_majority_baseline"] = M.compute(d.label.values, np.full(len(d), y.mean()))
        curves[s] = (d.label.values, p)
        M.plot_confusion(res["test"][s]["confusion_matrix"], f"Social text ({s})", f"cm_text_social_{s}.png")
        print(f"[social] TEST {s}:", {k: res['test'][s][k] for k in ('accuracy', 'f1_fake', 'f1_macro', 'roc_auc')})
    M.plot_roc(curves, "Social-post text model (event-disjoint tests)", "roc_text_social.png")
    joblib.dump({"pipeline": final, "test_metrics": res["test"].get("test2015", {}),
                 "domain_note": "Social-post model trained on 2012-2015 MediaEval tweets about 11 events; "
                                "it generalises poorly to new events."},
                config.TEXT_SOCIAL_MODEL, compress=3)
    M.save_json(res, "text_social_results.json")
    return res


if __name__ == "__main__":
    config.MODELS_DIR.mkdir(exist_ok=True)
    train_news()
    train_social()
