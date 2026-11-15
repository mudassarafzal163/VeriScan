"""Build clean, leakage-free CSV splits from the raw public datasets.

Outputs (data/processed/):
  fakenewsnet_{train,val,test}.csv   text,label,source   (label 1 = fake)
  mediaeval_posts_{dev,test2015,test2016}.csv  text,label,image_ids,event,image_path
  mediaeval_images.csv               image_id,path,label,event,split

Leakage controls
  * FakeNewsNet titles are normalised and de-duplicated BEFORE splitting, and
    titles whose duplicates carry conflicting labels are dropped.
  * MediaEval: the official 2015 dev/test split is event-disjoint. The 2016
    test set is also restricted to events that do not appear in 2015 dev.
    Any image id that appears in more than one split is removed from the
    later split.
Run:  python -m data_prep.prepare_datasets
"""
import csv
import os
import re
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from ml.preprocessing.text_preprocess import clean_text  # noqa: E402

IMG_EXT = (".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp")


# ---------------------------------------------------------------- FakeNewsNet
def prepare_fakenewsnet() -> dict:
    frames = []
    for source in ("politifact", "gossipcop"):
        for lab_name, lab in (("fake", 1), ("real", 0)):
            f = config.FAKENEWSNET_DIR / f"{source}_{lab_name}.csv"
            if not f.exists():
                raise FileNotFoundError(f"{f} missing - run data_prep/download_datasets.py")
            df = pd.read_csv(f, usecols=["id", "title"])
            df["label"] = lab
            df["source"] = source
            frames.append(df)
    df = pd.concat(frames, ignore_index=True).rename(columns={"title": "text"})
    df["text"] = df["text"].astype(str)
    df["norm"] = df["text"].map(clean_text)
    df = df[df["norm"].str.split().str.len() >= 3]
    n_before = len(df)
    # drop titles whose duplicates disagree on the label, then dedupe
    conflicted = df.groupby("norm")["label"].nunique()
    df = df[~df["norm"].isin(conflicted[conflicted > 1].index)]
    df = df.drop_duplicates("norm")
    strat = df["label"].astype(str) + "_" + df["source"]
    train, tmp = train_test_split(df, test_size=0.30, stratify=strat, random_state=config.RANDOM_SEED)
    strat_tmp = tmp["label"].astype(str) + "_" + tmp["source"]
    val, test = train_test_split(tmp, test_size=0.50, stratify=strat_tmp, random_state=config.RANDOM_SEED)
    out = {}
    for name, part in (("train", train), ("val", val), ("test", test)):
        part[["text", "label", "source"]].to_csv(config.DATA_PROCESSED / f"fakenewsnet_{name}.csv", index=False)
        out[name] = {"n": len(part), "fake": int(part.label.sum()), "real": int((1 - part.label).sum())}
    out["removed_duplicates_or_short"] = n_before - len(df)
    return out


# ------------------------------------------------------------------ MediaEval
def event_of(image_id: str) -> str:
    """'sandyA_fake_46' -> 'sandy', 'boston_real_17' -> 'boston', 'nepal_01' -> 'nepal'."""
    s = image_id.strip()
    s = re.sub(r"_(fake|real).*$", "", s, flags=re.I)
    s = re.sub(r"_\d+$", "", s)
    s = re.sub(r"_video$", "", s, flags=re.I)
    s = re.sub(r"(?<=[a-z])[AB]$", "", s)  # sandyA / sandyB
    return s.lower()


def _index_images(root: Path) -> dict:
    idx = {}
    for dirpath, _, files in os.walk(root):
        for fn in files:
            stem, ext = os.path.splitext(fn)
            if ext.lower() in IMG_EXT:
                idx.setdefault(stem, os.path.relpath(os.path.join(dirpath, fn), config.ROOT))
    return idx


def _read_posts(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", quoting=csv.QUOTE_NONE, dtype=str, on_bad_lines="skip")
    df.columns = [c.lower() for c in df.columns]
    text_col = "tweettext" if "tweettext" in df.columns else "post_text"
    img_col = [c for c in df.columns if c.startswith("image")][0]
    df = df.rename(columns={text_col: "text", img_col: "image_ids"})
    df = df[df["label"].isin(["fake", "real", "humor"])].copy()
    # Official MediaEval VMU protocol: humour posts are treated as fake.
    df["label"] = (df["label"] != "real").astype(int)
    df["image_ids"] = df["image_ids"].fillna("").str.replace(" ", "")
    df["event"] = df["image_ids"].str.split(",").str[0].map(event_of)
    return df[["text", "label", "image_ids", "event"]]


def prepare_mediaeval() -> dict:
    me = config.MEDIAEVAL_DIR
    sets = {
        "dev": (me / "dev_tweets.txt", me / "img2015dev"),
        "test2015": (me / "test_tweets.txt", me / "img2015test"),
        "test2016": (me / "posts2016_test.txt", me / "img2016test"),
    }
    seen_events, seen_images = set(), set()
    img_rows, out = [], {}
    for split, (post_file, img_dir) in sets.items():
        posts = _read_posts(post_file)
        idx = _index_images(img_dir)
        if split != "dev":
            posts = posts[~posts["event"].isin(seen_events)]
        posts["image_path"] = posts["image_ids"].map(
            lambda s: next((idx[i] for i in s.split(",") if i in idx), ""))
        # image-level labels: an image is fake if the posts using it are fake
        exploded = posts.assign(image_id=posts["image_ids"].str.split(",")).explode("image_id")
        exploded = exploded[exploded["image_id"].isin(idx.keys())]
        img_lab = exploded.groupby("image_id").agg(label=("label", "max"), event=("event", "first"))
        for iid, r in img_lab.iterrows():
            if iid in seen_images:
                continue
            img_rows.append({"image_id": iid, "path": idx[iid], "label": int(r.label),
                             "event": r.event, "split": split})
        seen_images |= set(img_lab.index)
        seen_events |= set(posts["event"])
        posts.to_csv(config.DATA_PROCESSED / f"mediaeval_posts_{split}.csv", index=False)
        out[split] = {"posts": len(posts), "fake_posts": int(posts.label.sum()),
                      "events": int(posts.event.nunique()),
                      "posts_with_image_file": int((posts.image_path != "").sum())}
    imgs = pd.DataFrame(img_rows)
    imgs.to_csv(config.DATA_PROCESSED / "mediaeval_images.csv", index=False)
    out["images"] = imgs.groupby(["split", "label"]).size().unstack(fill_value=0).to_dict("index")
    return out


if __name__ == "__main__":
    import json
    config.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    summary = {"fakenewsnet": prepare_fakenewsnet(), "mediaeval": prepare_mediaeval()}
    (config.REPORTS_DIR).mkdir(exist_ok=True)
    with open(config.REPORTS_DIR / "dataset_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(json.dumps(summary, indent=2, default=str))
