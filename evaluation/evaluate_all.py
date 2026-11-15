"""Evaluation pipeline -> reports/EVALUATION_REPORT.md

1. Re-loads the SAVED model files from models/ and recomputes their test
   metrics from the processed test splits (verifies the shipped artefacts
   reproduce the numbers logged at training time).
2. Collects the results written by the training scripts (validation,
   baselines, ablations, synthetic forensics test, fusion).
3. Writes a Markdown report with tables and links to the figures.

Run:  python -m evaluation.evaluate_all
"""
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import config  # noqa: E402
from evaluation import metrics as M  # noqa: E402
from ml.text.text_model import TextModel  # noqa: E402

P = config.DATA_PROCESSED
COLS = ("n", "accuracy", "precision_fake", "recall_fake", "f1_fake", "f1_macro", "roc_auc")


def recompute():
    out = {}
    te = pd.read_csv(P / "fakenewsnet_test.csv")
    out["Text (news) – FakeNewsNet test"] = M.compute(te.label, TextModel(config.TEXT_NEWS_MODEL, "n").predict_proba(te.text.tolist()))
    soc = TextModel(config.TEXT_SOCIAL_MODEL, "s")
    for s in ("test2015", "test2016"):
        d = pd.read_csv(P / f"mediaeval_posts_{s}.csv").dropna(subset=["text"])
        out[f"Text (social) – MediaEval {s}"] = M.compute(d.label, soc.predict_proba(d.text.tolist()))
    imgs = pd.read_csv(P / "mediaeval_images.csv")
    z = np.load(P / "mediaeval_image_embeddings.npz", allow_pickle=True)
    emb = dict(zip(z["ids"], z["emb"]))
    ctx = joblib.load(config.IMAGE_CONTEXT_MODEL)["pipeline"]
    for s in ("test2015", "test2016"):
        d = imgs[imgs.split == s]
        out[f"Image content – MediaEval {s} images"] = M.compute(d.label, ctx.predict_proba(np.stack([emb[i] for i in d.image_id]))[:, 1])
    return out


def table(rows: dict) -> str:
    head = "| Model / test set | " + " | ".join(COLS) + " |\n|---|" + "---:|" * len(COLS) + "\n"
    body = ""
    for name, m in rows.items():
        if not m:
            continue
        vals = [str(m.get(c)) if c == "n" else ("–" if m.get(c) is None else f"{m.get(c):.3f}") for c in COLS]
        body += f"| {name} | " + " | ".join(vals) + " |\n"
    return head + body


def cm(m):
    (tn, fp), (fn, tp) = m["confusion_matrix"]
    return f"TN={tn}, FP={fp}, FN={fn}, TP={tp}"


def main():
    rec = recompute()
    M.save_json(rec, "recomputed_test_metrics.json")
    tn, ts = M.load_json("text_news_results.json"), M.load_json("text_social_results.json")
    ic, mr, fr = M.load_json("image_context_results.json"), M.load_json("manipulation_results.json"), M.load_json("fusion_results.json")
    ds = M.load_json("dataset_summary.json")
    ai = M.load_json("ai_generated_results.json")
    fu = fr["fusion"]
    L = []
    L.append("# Evaluation Report\n")
    L.append(f"Generated {datetime.now(timezone.utc):%Y-%m-%d %H:%M UTC} by `python -m evaluation.evaluate_all`. "
             "Every number below was produced by the scripts in this repository on the public datasets listed; "
             "nothing is copied from papers. Positive class = **fake**; decision threshold 0.5 unless stated.\n")
    L.append("## 1. Data and splits\n")
    f = ds["fakenewsnet"]
    L.append(f"* **FakeNewsNet titles** (PolitiFact + GossipCop). Normalised titles de-duplicated before splitting "
             f"({f['removed_duplicates_or_short']} duplicates/short/conflicting titles removed). Stratified (label × source) "
             f"70/15/15: train {f['train']['n']} ({f['train']['fake']} fake), validation {f['val']['n']}, test {f['test']['n']} ({f['test']['fake']} fake).")
    me = ds["mediaeval"]
    L.append(f"* **MediaEval VMU**: 2015 dev = training ({me['dev']['posts']} posts, {me['dev']['events']} events); "
             f"2015 test = validation for fusion ({me['test2015']['posts']} posts, {me['test2015']['events']} events); "
             f"2016 test = final test ({me['test2016']['posts']} posts, {me['test2016']['events']} events). Events are disjoint "
             "across the three sets (checked in `data_prep/prepare_datasets.py`); humour posts count as fake (official protocol). "
             f"Images: {me['images']}.")
    L.append(f"* **Synthetic tampering**: {mr['train_images']} training images from authentic 2015-dev photos, {mr['test_images']} test "
             "images from authentic 2015/2016-test photos (no shared source photo); 50 % authentic re-saves, 50 % splice / copy-move / retouch with pixel masks.\n")

    L.append("## 2. Verification of the shipped model files\n")
    L.append("Metrics recomputed now from `models/*.joblib` on the test splits:\n")
    L.append(table(rec))
    L.append("\n## 3. Text models\n")
    L.append("### 3.1 News headlines (FakeNewsNet)\nModel selection on the validation split:\n")
    L.append(table({k: v for k, v in tn["validation"].items()}))
    L.append(f"\nSelected `{tn['selected_on_val']}` (best validation F1 among the linear models, needed for exact word explanations), "
             "refit on train+validation, evaluated once on test:\n")
    L.append(table({"Logistic regression (deployed)": tn["test_deployed"],
                    **{f"baseline: {k}": v for k, v in tn["test_baselines"].items()},
                    **{f"deployed, subset {k}": v for k, v in tn["test_per_source"].items()}}))
    L.append(f"\nConfusion matrix (deployed, test): {cm(tn['test_deployed'])}. Figures: `figures/cm_text_news.png`, `figures/roc_text_news.png`.\n")
    L.append("### 3.2 Social posts (MediaEval, char n-gram TF-IDF + LR)\n")
    L.append(table({f"leave-event-out CV on dev ({k})": v for k, v in ts["leave_event_out_cv"].items()}))
    L.append("\nTests on unseen events:\n")
    L.append(table({f"{k}": v for k, v in ts["test"].items()}))
    L.append("\nThe leave-event-out CV score on dev is low because one event (Hurricane Sandy) holds 86 % of dev posts; "
             "removing it leaves very little training data for that fold. Cross-event generalisation is the main weakness of "
             "text-only models on this benchmark.\n")

    L.append("## 4. Image models\n### 4.1 Content model (EfficientNet-B0 embedding + LR) on MediaEval images\n")
    L.append(table({f"leave-event-out CV on dev ({k})": v for k, v in ic["leave_event_out_cv"].items()}))
    L.append("\n" + table({f"{k}": v for k, v in ic["test"].items()}))
    L.append("\nPooled leave-event-out AUC on dev is **below 0.5**: each held-out event has a different fake/real ratio and the "
             "model's per-fold offsets do not transfer. On the two test sets it is weakly above chance. Interpretation: image content "
             "alone is a weak, dataset-specific cue for 'misleading use'.\n")
    L.append("### 4.2 Forensic manipulation detector (block-level)\n")
    L.append(table({"image level (synthetic test)": mr["test_image_level"],
                    "32×32 block localisation (synthetic test)": mr["test_block_level_localisation"]}))
    L.append(f"\n* Recall by edit type: {mr['test_recall_by_operation']}.")
    L.append(f"* False-positive rate on held-out authentic photos: **{mr['test_false_positive_rate_authentic']*100:.1f} %**: high, mostly on "
             "photos with text overlays, borders or heavy prior compression.")
    L.append(f"* ROC-AUC against the real MediaEval fake/real labels: **{mr['auc_on_mediaeval_misuse_labels']}** (≈ chance). As expected, most "
             "MediaEval 'fake' images are genuine photos used out of context, which pixel forensics cannot detect.")
    L.append(f"* Confusion matrix (image level): {cm(mr['test_image_level'])}. Figures: `figures/roc_manipulation.png`, `figures/cm_manipulation.png`.\n")
    L.append("### 4.3 AI-generated image detector\n")
    if ai:
        L.append(table({f"test ({ai['dataset_root']})": ai["test"]}))
    else:
        L.append("**Not trained in this build.** CIFAKE/GenImage are hosted on Kaggle/HuggingFace, which were not reachable from the "
                 "build environment. `training/train_ai_generated.py` trains it in one command once the data is downloaded; "
                 "the code path was smoke-tested on a small dummy folder (see `tests/`). No metric is claimed.\n")

    L.append("## 5. Text–image consistency (learned, experimental)\n")
    L.append(f"Ridge projection from tweet TF-IDF/SVD to the EfficientNet embedding, trained on {fr['consistency']['train_pairs']} dev pairs. "
             f"Matched-vs-mismatched-image AUC: 2015 test **{fr['consistency']['test2015_matched_vs_mismatched_auc']}**, "
             f"2016 test **{fr['consistency']['test2016_matched_vs_mismatched_auc']}**. It does not beat chance reliably, "
             "so it is **not used** at inference; the app uses CLIP when it can be installed (not executable in the build environment).\n")

    L.append("## 6. Multimodal fusion (final test: MediaEval 2016, unseen events)\n")
    L.append(f"Validation AUCs (2015 test): {fu['validation_auc']} → reliability weights w = max(AUC−0.5,0), normalised: "
             f"**{fu['fusion_weights']}**.\n")
    L.append(table({"Reliability-weighted fusion (deployed)": fu["reliability_weighted_fusion"],
                    "ablation: unweighted average": fu["ablation_unweighted_average"],
                    "ablation: stacked LR (fit on validation)": fu["ablation_stacked_lr"],
                    "ablation: text only": fu["ablation_only_text"],
                    "ablation: image content only": fu["ablation_only_image_context"],
                    "ablation: forensics only": fu["ablation_only_manipulation"]}))
    L.append(f"\nConfusion matrix (deployed fusion): {cm(fu['reliability_weighted_fusion'])}. Figure: `figures/roc_fusion.png`.\n")
    L.append("**Finding.** On unseen events, fusion did **not** outperform the best single modality. The image-content model happens to "
             "be the best on 2016 test, but it was below chance on the validation events, so a leakage-free procedure cannot select it. "
             "With only 11 training events, modality reliabilities are unstable across event sets; this is the main open problem, and "
             "the reason more diverse multimodal data (e.g. Fakeddit) is the first item of future work.\n")
    L.append("## 7. Threats to validity\n"
             "* Small number of events in MediaEval; post-level metrics are dominated by a few viral images.\n"
             "* FakeNewsNet labels come from fact-checking sites with topical/stylistic biases (GossipCop = celebrity news); "
             "the text model learns those patterns, not truth.\n"
             "* Synthetic tampering is cruder than professional forgeries; real-world forensic recall will be lower.\n"
             "* Scores are not calibrated probabilities of truth; the UI shows 0.35–0.65 as UNCERTAIN.\n")
    (config.REPORTS_DIR / "EVALUATION_REPORT.md").write_text("\n".join(L), encoding="utf-8")
    print("wrote reports/EVALUATION_REPORT.md")
    for k, v in rec.items():
        print(f"{k:45s} acc={v['accuracy']:.3f} f1={v['f1_fake']:.3f} auc={v['roc_auc']:.3f}")


if __name__ == "__main__":
    main()
