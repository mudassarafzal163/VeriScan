"""Deep image features from an ImageNet-pretrained EfficientNet-B0.

One forward pass gives
  * the 1280x7x7 last conv feature map (used for Class Activation Maps)
  * the 1280-d globally-average-pooled embedding (input to our classifiers)
  * ImageNet class logits (used to describe image content for the
    text-image consistency check)
Because our heads are linear on the pooled embedding, the CAM
  cam(x, y) = sum_k w_k * F_k(x, y)
is an exact spatial decomposition of the head's decision (Zhou et al., 2016).

Weights (~21 MB) are downloaded once from the efficientnet-pytorch GitHub
release into models/cnn_weights and reused offline afterwards.
"""
from __future__ import annotations

import threading

import numpy as np

import config

_LOCK = threading.Lock()
_INSTANCE = None


class CNNFeatureExtractor:
    def __init__(self, device: str | None = None):
        import torch
        from efficientnet_pytorch import EfficientNet
        from torchvision import transforms
        from torchvision.models import EfficientNet_B0_Weights

        config.CNN_WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
        torch.hub.set_dir(str(config.CNN_WEIGHTS_DIR))
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = EfficientNet.from_pretrained("efficientnet-b0").to(self.device).eval()
        self.categories = EfficientNet_B0_Weights.IMAGENET1K_V1.meta["categories"]
        self.tf = transforms.Compose([
            transforms.Resize(256), transforms.CenterCrop(config.IMAGE_SIZE), transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    def _prep(self, imgs):
        from ml.image.forensics import to_rgb
        return self.torch.stack([self.tf(to_rgb(im)) for im in imgs]).to(self.device)

    def forward(self, imgs, batch_size: int = 32):
        """Returns (pooled[N,1280], logits[N,1000], fmap[N,1280,7,7])."""
        pooled, logits, fmaps = [], [], []
        with self.torch.no_grad():
            for i in range(0, len(imgs), batch_size):
                x = self._prep(imgs[i:i + batch_size])
                fm = self.model.extract_features(x)
                p = fm.mean(dim=(2, 3))
                lg = self.model._fc(p)
                pooled.append(p.cpu().numpy())
                logits.append(lg.cpu().numpy())
                fmaps.append(fm.cpu().numpy())
        return np.concatenate(pooled), np.concatenate(logits), np.concatenate(fmaps)

    def embed(self, imgs, batch_size: int = 32) -> np.ndarray:
        return self.forward(imgs, batch_size)[0]

    def top_concepts(self, logits: np.ndarray, k: int = 5):
        e = np.exp(logits - logits.max())
        p = e / e.sum()
        idx = np.argsort(-p)[:k]
        return [{"label": self.categories[i], "prob": round(float(p[i]), 4)} for i in idx]

    @staticmethod
    def cam(fmap: np.ndarray, weights: np.ndarray, scaler_scale=None) -> np.ndarray:
        """fmap [1280,7,7], weights [1280] (in standardised space if a scaler
        was used: pass scaler.scale_ to undo it)."""
        w = weights / scaler_scale if scaler_scale is not None else weights
        return np.tensordot(w, fmap, axes=(0, 0))


def get_extractor() -> CNNFeatureExtractor:
    global _INSTANCE
    with _LOCK:
        if _INSTANCE is None:
            _INSTANCE = CNNFeatureExtractor()
        return _INSTANCE


def load_image_list(paths):
    from ml.image.forensics import load_rgb, to_rgb
    out = []
    for p in paths:
        out.append(to_rgb(load_rgb(p)))
    return out
