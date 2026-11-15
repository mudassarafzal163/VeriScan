"""Download the public datasets used by this project into data/raw/.

  * FakeNewsNet (KaiDMML/FakeNewsNet, GitHub): 4 CSV files of news titles
  * MediaEval Verifying Multimedia Use 2015/2016 (MKLab-ITI/image-verification-corpus,
    GitHub): tweets/posts + image archives (RAR/ZIP)

RAR extraction uses `tar -xf`, which on Windows 10/11 is the built-in
libarchive tar (supports RAR). On Linux install `libarchive-tools` (bsdtar);
on macOS the system tar works. Alternatively `pip install rarfile` + unrar.

Usage:  python -m data_prep.download_datasets
Optional (not downloadable automatically - see README "Datasets"):
  CIFAKE (Kaggle) for the AI-generated image detector.
"""
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402

FNN = "https://raw.githubusercontent.com/KaiDMML/FakeNewsNet/master/dataset/{}.csv"
ME = "https://raw.githubusercontent.com/MKLab-ITI/image-verification-corpus/master/"
ME_FILES = {
    "dev_tweets.txt": "mediaeval2015/devset/tweets.txt",
    "test_tweets.txt": "mediaeval2015/testset/tweets.txt",
    "dev_images.txt": "mediaeval2015/devset/images.txt",
    "posts2016_dev.txt": "mediaeval2016/devset/posts.txt",
    "posts2016_test.txt": "mediaeval2016/testset/posts_groundtruth.txt",
    "set_images.txt": "set_images.txt",
    "README.md": "README.md",
}
ME_ARCHIVES = {
    "img2015dev": "mediaeval2015/devset/MediaEval2015_DevSet_Images.rar",
    "img2015test": "mediaeval2015/testset/MediaEval2015_TestSetImages.rar",
    "img2016test": "mediaeval2016/testset/Mediaeval2016_TestSet_Images.zip",
}


def fetch(url: str, dest: Path):
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  exists  {dest.name}")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  fetch   {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "fnd-fyp-downloader"})
    with urllib.request.urlopen(req, timeout=300) as r, open(dest.with_suffix(dest.suffix + ".part"), "wb") as f:
        shutil.copyfileobj(r, f)
    dest.with_suffix(dest.suffix + ".part").replace(dest)


def extract(archive: Path, out: Path):
    if out.exists() and any(out.rglob("*.jpg")):
        print(f"  exists  {out.name}/")
        return
    out.mkdir(parents=True, exist_ok=True)
    if archive.suffix == ".zip":
        with zipfile.ZipFile(archive) as z:
            z.extractall(out)
        return
    for tool in (["tar", "-xf"], ["bsdtar", "-xf"]):
        try:
            subprocess.run(tool + [str(archive), "-C", str(out)], check=True, capture_output=True)
            if any(out.rglob("*.*")):
                return
        except (OSError, subprocess.CalledProcessError):
            continue
    try:
        import rarfile
        rarfile.RarFile(archive).extractall(out)
        return
    except Exception as e:  # noqa: BLE001
        raise SystemExit(f"Could not extract {archive}. Install bsdtar/libarchive or 'pip install rarfile' + unrar. ({e})")


def main():
    print("FakeNewsNet:")
    for n in ("politifact_fake", "politifact_real", "gossipcop_fake", "gossipcop_real"):
        fetch(FNN.format(n), config.FAKENEWSNET_DIR / f"{n}.csv")
    print("MediaEval VMU:")
    for local, remote in ME_FILES.items():
        fetch(ME + remote, config.MEDIAEVAL_DIR / local)
    dl = config.MEDIAEVAL_DIR / "_archives"
    for folder, remote in ME_ARCHIVES.items():
        arc = dl / Path(remote).name
        fetch(ME + remote, arc)
        extract(arc, config.MEDIAEVAL_DIR / folder)
    print("Done. Next: python -m data_prep.prepare_datasets")


if __name__ == "__main__":
    main()
