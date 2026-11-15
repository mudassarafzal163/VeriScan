"""Full reproducible pipeline: download -> prepare -> train -> evaluate.

    python train_all.py              # everything
    python train_all.py --skip-download
Runtime on a 4-core CPU: ~10 minutes (no GPU needed).
"""
import argparse
import subprocess
import sys

STEPS = [
    ("download", "data_prep.download_datasets"),
    ("prepare", "data_prep.prepare_datasets"),
    ("text", "training.train_text"),
    ("image", "training.train_image"),
    ("fusion", "training.train_fusion"),
    ("evaluate", "evaluation.evaluate_all"),
]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-download", action="store_true")
    a = ap.parse_args()
    for name, mod in STEPS:
        if name == "download" and a.skip_download:
            continue
        print(f"\n===== {name}: python -m {mod} =====", flush=True)
        r = subprocess.run([sys.executable, "-W", "ignore", "-m", mod])
        if r.returncode != 0:
            sys.exit(f"Step '{name}' failed (exit code {r.returncode}).")
    print("\nAll steps finished. Start the app with:  python run.py")
