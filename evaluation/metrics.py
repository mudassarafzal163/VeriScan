"""Metric computation and plots shared by every training/evaluation script."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,  # noqa: E402
                             precision_score, recall_score, roc_auc_score, roc_curve)

import config  # noqa: E402

FIG_DIR = config.REPORTS_DIR / "figures"


def compute(y_true, prob, threshold: float = 0.5) -> dict:
    y_true = np.asarray(y_true).astype(int)
    prob = np.asarray(prob, float)
    pred = (prob >= threshold).astype(int)
    m = {
        "n": int(len(y_true)), "n_fake": int(y_true.sum()), "n_real": int((1 - y_true).sum()),
        "accuracy": accuracy_score(y_true, pred),
        "precision_fake": precision_score(y_true, pred, zero_division=0),
        "recall_fake": recall_score(y_true, pred, zero_division=0),
        "f1_fake": f1_score(y_true, pred, zero_division=0),
        "f1_macro": f1_score(y_true, pred, average="macro", zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, pred, labels=[0, 1]).tolist(),
    }
    m["roc_auc"] = roc_auc_score(y_true, prob) if len(set(y_true)) == 2 else None
    return {k: (round(float(v), 4) if isinstance(v, (float, np.floating)) else v) for k, v in m.items()}


def plot_confusion(cm, title: str, fname: str):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    cm = np.array(cm)
    fig, ax = plt.subplots(figsize=(3.6, 3.2))
    ax.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, str(v), ha="center", va="center", color="white" if v > cm.max() / 2 else "black")
    ax.set_xticks([0, 1], ["real", "fake"])
    ax.set_yticks([0, 1], ["real", "fake"])
    ax.set_xlabel("predicted")
    ax.set_ylabel("actual")
    ax.set_title(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG_DIR / fname, dpi=130)
    plt.close(fig)


def plot_roc(curves: dict, title: str, fname: str):
    """curves: {label: (y_true, prob)}"""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(4.2, 3.8))
    for label, (y, p) in curves.items():
        if len(set(np.asarray(y))) < 2:
            continue
        fpr, tpr, _ = roc_curve(y, p)
        ax.plot(fpr, tpr, label=f"{label} (AUC={roc_auc_score(y, p):.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(title, fontsize=9)
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG_DIR / fname, dpi=130)
    plt.close(fig)


def save_json(obj, name: str):
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(config.REPORTS_DIR / name, "w") as f:
        json.dump(obj, f, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))


def load_json(name: str):
    p = Path(config.REPORTS_DIR / name)
    return json.loads(p.read_text()) if p.exists() else None
