# Evaluation Report

Generated 2026-11-15 11:53 UTC by `python -m evaluation.evaluate_all`. Every number below was produced by the scripts in this repository on the public datasets listed; nothing is copied from papers. Positive class = **fake**; decision threshold 0.5 unless stated.

## 1. Data and splits

* **FakeNewsNet titles** (PolitiFact + GossipCop). Normalised titles de-duplicated before splitting (1421 duplicates/short/conflicting titles removed). Stratified (label × source) 70/15/15: train 14854 (3580 fake), validation 3183, test 3183 (767 fake).
* **MediaEval VMU**: 2015 dev = training (14483 posts, 11 events); 2015 test = validation for fusion (3781 posts, 6 events); 2016 test = final test (2228 posts, 36 events). Events are disjoint across the three sets (checked in `data_prep/prepare_datasets.py`); humour posts count as fake (official protocol). Images: {'dev': {'0': 176, '1': 182}, 'test2015': {'0': 17, '1': 31}, 'test2016': {'0': 54, '1': 50}}.
* **Synthetic tampering**: 1056 training images from authentic 2015-dev photos, 426 test images from authentic 2015/2016-test photos (no shared source photo); 50 % authentic re-saves, 50 % splice / copy-move / retouch with pixel masks.

## 2. Verification of the shipped model files

Metrics recomputed now from `models/*.joblib` on the test splits:

| Model / test set | n | accuracy | precision_fake | recall_fake | f1_fake | f1_macro | roc_auc |
|---|---:|---:|---:|---:|---:|---:|---:|
| Text (news) – FakeNewsNet test | 3183 | 0.833 | 0.637 | 0.711 | 0.672 | 0.780 | 0.874 |
| Text (social) – MediaEval test2015 | 3781 | 0.764 | 0.903 | 0.729 | 0.807 | 0.751 | 0.844 |
| Text (social) – MediaEval test2016 | 2228 | 0.648 | 0.689 | 0.661 | 0.675 | 0.646 | 0.690 |
| Image content – MediaEval test2015 images | 48 | 0.542 | 0.655 | 0.613 | 0.633 | 0.511 | 0.613 |
| Image content – MediaEval test2016 images | 104 | 0.567 | 0.540 | 0.680 | 0.602 | 0.564 | 0.658 |


## 3. Text models

### 3.1 News headlines (FakeNewsNet)
Model selection on the validation split:

| Model / test set | n | accuracy | precision_fake | recall_fake | f1_fake | f1_macro | roc_auc |
|---|---:|---:|---:|---:|---:|---:|---:|
| logreg_C0.5 | 3183 | 0.834 | 0.640 | 0.709 | 0.673 | 0.781 | 0.861 |
| logreg_C1 | 3183 | 0.835 | 0.641 | 0.713 | 0.675 | 0.782 | 0.868 |
| logreg_C2 | 3183 | 0.835 | 0.643 | 0.711 | 0.675 | 0.782 | 0.871 |
| logreg_C4 | 3183 | 0.835 | 0.645 | 0.699 | 0.671 | 0.780 | 0.869 |
| logreg_C8 | 3183 | 0.832 | 0.647 | 0.669 | 0.658 | 0.773 | 0.865 |
| complement_nb | 3183 | 0.819 | 0.604 | 0.725 | 0.659 | 0.768 | 0.869 |
| linear_svm_calibrated | 3183 | 0.846 | 0.767 | 0.516 | 0.617 | 0.760 | 0.868 |
| majority_baseline | 3183 | 0.759 | 0.000 | 0.000 | 0.000 | 0.431 | 0.500 |


Selected `logreg_C1` (best validation F1 among the linear models, needed for exact word explanations), refit on train+validation, evaluated once on test:

| Model / test set | n | accuracy | precision_fake | recall_fake | f1_fake | f1_macro | roc_auc |
|---|---:|---:|---:|---:|---:|---:|---:|
| Logistic regression (deployed) | 3183 | 0.833 | 0.637 | 0.711 | 0.672 | 0.780 | 0.874 |
| baseline: complement_nb | 3183 | 0.823 | 0.612 | 0.729 | 0.665 | 0.773 | 0.875 |
| baseline: linear_svm_calibrated | 3183 | 0.853 | 0.772 | 0.555 | 0.646 | 0.777 | 0.873 |
| baseline: majority_baseline | 3183 | 0.759 | 0.000 | 0.000 | 0.000 | 0.431 | 0.500 |
| deployed, subset gossipcop | 3041 | 0.838 | 0.632 | 0.725 | 0.675 | 0.784 | 0.878 |
| deployed, subset politifact | 142 | 0.718 | 0.739 | 0.548 | 0.630 | 0.701 | 0.796 |


Confusion matrix (deployed, test): TN=2106, FP=310, FN=222, TP=545. Figures: `figures/cm_text_news.png`, `figures/roc_text_news.png`.

### 3.2 Social posts (MediaEval, char n-gram TF-IDF + LR)

| Model / test set | n | accuracy | precision_fake | recall_fake | f1_fake | f1_macro | roc_auc |
|---|---:|---:|---:|---:|---:|---:|---:|
| leave-event-out CV on dev (C0.1) | 14483 | 0.607 | 0.641 | 0.908 | 0.751 | 0.406 | 0.438 |
| leave-event-out CV on dev (C0.5) | 14483 | 0.607 | 0.641 | 0.910 | 0.752 | 0.405 | 0.438 |
| leave-event-out CV on dev (C2.0) | 14483 | 0.606 | 0.641 | 0.906 | 0.751 | 0.408 | 0.437 |


Tests on unseen events:

| Model / test set | n | accuracy | precision_fake | recall_fake | f1_fake | f1_macro | roc_auc |
|---|---:|---:|---:|---:|---:|---:|---:|
| test2015 | 3781 | 0.764 | 0.903 | 0.729 | 0.807 | 0.751 | 0.844 |
| test2015_majority_baseline | 3781 | 0.678 | 0.678 | 1.000 | 0.808 | 0.404 | 0.500 |
| test2016 | 2228 | 0.648 | 0.689 | 0.661 | 0.675 | 0.646 | 0.690 |
| test2016_majority_baseline | 2228 | 0.552 | 0.552 | 1.000 | 0.711 | 0.356 | 0.500 |


The leave-event-out CV score on dev is low because one event (Hurricane Sandy) holds 86 % of dev posts; removing it leaves very little training data for that fold. Cross-event generalisation is the main weakness of text-only models on this benchmark.

## 4. Image models
### 4.1 Content model (EfficientNet-B0 embedding + LR) on MediaEval images

| Model / test set | n | accuracy | precision_fake | recall_fake | f1_fake | f1_macro | roc_auc |
|---|---:|---:|---:|---:|---:|---:|---:|
| leave-event-out CV on dev (C0.001) | 358 | 0.408 | 0.451 | 0.753 | 0.564 | 0.321 | 0.316 |
| leave-event-out CV on dev (C0.01) | 358 | 0.408 | 0.450 | 0.742 | 0.560 | 0.327 | 0.308 |
| leave-event-out CV on dev (C0.1) | 358 | 0.394 | 0.442 | 0.736 | 0.553 | 0.307 | 0.310 |


| Model / test set | n | accuracy | precision_fake | recall_fake | f1_fake | f1_macro | roc_auc |
|---|---:|---:|---:|---:|---:|---:|---:|
| test2015 | 48 | 0.542 | 0.655 | 0.613 | 0.633 | 0.511 | 0.613 |
| test2016 | 104 | 0.567 | 0.540 | 0.680 | 0.602 | 0.564 | 0.658 |


Pooled leave-event-out AUC on dev is **below 0.5**: each held-out event has a different fake/real ratio and the model's per-fold offsets do not transfer. On the two test sets it is weakly above chance. Interpretation: image content alone is a weak, dataset-specific cue for 'misleading use'.

### 4.2 Forensic manipulation detector (block-level)

| Model / test set | n | accuracy | precision_fake | recall_fake | f1_fake | f1_macro | roc_auc |
|---|---:|---:|---:|---:|---:|---:|---:|
| image level (synthetic test) | 426 | 0.688 | 0.659 | 0.779 | 0.714 | 0.685 | 0.793 |
| 32×32 block localisation (synthetic test) | 96793 | 0.910 | 0.200 | 0.733 | 0.314 | 0.633 | 0.906 |


* Recall by edit type: {'splice': 0.8056, 'copy_move': 0.6912, 'retouch': 0.8356}.
* False-positive rate on held-out authentic photos: **40.4 %**: high, mostly on photos with text overlays, borders or heavy prior compression.
* ROC-AUC against the real MediaEval fake/real labels: **0.4769** (≈ chance). As expected, most MediaEval 'fake' images are genuine photos used out of context, which pixel forensics cannot detect.
* Confusion matrix (image level): TN=127, FP=86, FN=47, TP=166. Figures: `figures/roc_manipulation.png`, `figures/cm_manipulation.png`.

### 4.3 AI-generated image detector

**Not trained in this build.** CIFAKE/GenImage are hosted on Kaggle/HuggingFace, which were not reachable from the build environment. `training/train_ai_generated.py` trains it in one command once the data is downloaded; the code path was smoke-tested on a small dummy folder (see `tests/`). No metric is claimed.

## 5. Text–image consistency (learned, experimental)

Ridge projection from tweet TF-IDF/SVD to the EfficientNet embedding, trained on 3733 dev pairs. Matched-vs-mismatched-image AUC: 2015 test **0.3448**, 2016 test **0.5651**. It does not beat chance reliably, so it is **not used** at inference; the app uses CLIP when it can be installed (not executable in the build environment).

## 6. Multimodal fusion (final test: MediaEval 2016, unseen events)

Validation AUCs (2015 test): {'text': 0.8, 'image_context': 0.336, 'manipulation': 0.5305} → reliability weights w = max(AUC−0.5,0), normalised: **{'text': 0.9076, 'image_context': 0.0, 'manipulation': 0.0924}**.

| Model / test set | n | accuracy | precision_fake | recall_fake | f1_fake | f1_macro | roc_auc |
|---|---:|---:|---:|---:|---:|---:|---:|
| Reliability-weighted fusion (deployed) | 1117 | 0.602 | 0.689 | 0.561 | 0.619 | 0.601 | 0.635 |
| ablation: unweighted average | 1117 | 0.626 | 0.728 | 0.558 | 0.632 | 0.626 | 0.665 |
| ablation: stacked LR (fit on validation) | 1117 | 0.488 | 0.718 | 0.182 | 0.290 | 0.445 | 0.677 |
| ablation: text only | 1117 | 0.604 | 0.691 | 0.566 | 0.622 | 0.603 | 0.645 |
| ablation: image content only | 1117 | 0.588 | 0.669 | 0.563 | 0.612 | 0.587 | 0.758 |
| ablation: forensics only | 1117 | 0.552 | 0.637 | 0.518 | 0.571 | 0.551 | 0.477 |


Confusion matrix (deployed fusion): TN=311, FP=163, FN=282, TP=361. Figure: `figures/roc_fusion.png`.

**Finding.** On unseen events, fusion did **not** outperform the best single modality. The image-content model happens to be the best on 2016 test, but it was below chance on the validation events, so a leakage-free procedure cannot select it. With only 11 training events, modality reliabilities are unstable across event sets; this is the main open problem, and the reason more diverse multimodal data (e.g. Fakeddit) is the first item of future work.

## 7. Threats to validity
* Small number of events in MediaEval; post-level metrics are dominated by a few viral images.
* FakeNewsNet labels come from fact-checking sites with topical/stylistic biases (GossipCop = celebrity news); the text model learns those patterns, not truth.
* Synthetic tampering is cruder than professional forgeries; real-world forensic recall will be lower.
* Scores are not calibrated probabilities of truth; the UI shows 0.35–0.65 as UNCERTAIN.
