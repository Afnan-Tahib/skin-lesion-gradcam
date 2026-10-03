"""Grad-CAM implemented from scratch (no extra dependency).

Grad-CAM (Selvaraju et al., 2017): weight each feature map of the last conv
layer by the mean gradient of the target class score, sum, ReLU -> heatmap.
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import cm


class GradCAM:
    def __init__(self, model, target_layer=None):
        self.model = model.eval()
        self.target_layer = target_layer if target_layer is not None else model.features[-1]
        self.acts, self.grads = None, None
        self.target_layer.register_forward_hook(self._forward_hook)

    def _forward_hook(self, module, inp, out):
        self.acts = out
        if out.requires_grad:
            out.register_hook(lambda g: setattr(self, "grads", g))

    def __call__(self, x: torch.Tensor, class_idx=None):
        """x: (1,3,H,W) normalized. Returns (cam[H,W] in 0..1, class_idx, probs[C])."""
        with torch.enable_grad():
            logits = self.model(x)
            probs = F.softmax(logits, dim=1)
            if class_idx is None:
                class_idx = int(logits.argmax(1).item())
            self.model.zero_grad()
            logits[0, class_idx].backward()
        weights = self.grads.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * self.acts).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
        cam = cam.detach().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam, class_idx, probs[0].detach().cpu().numpy()


def overlay(img_uint8: np.ndarray, cam: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    heat = (cm.jet(cam)[..., :3] * 255).astype(np.uint8)
    return (img_uint8 * (1 - alpha) + heat * alpha).astype(np.uint8)


def main():
    from .dataset import HAMDataset, denormalize, get_transforms
    from .hamdata import CLASSES, build_dataframe, split_dataframe
    from .model import load_checkpoint

    ap = argparse.ArgumentParser(description="Grad-CAM gallery on the test set")
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--ckpt", default="models/best_model.pt")
    ap.add_argument("--out", default="results")
    ap.add_argument("--n", type=int, default=6, help="examples per group")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, ckpt = load_checkpoint(args.ckpt, device)
    size = ckpt.get("img_size", 224)
    _, _, test_df = split_dataframe(build_dataframe(args.data_dir), seed=args.seed)
    ds = HAMDataset(test_df, get_transforms(size, train=False), cache_size=size)
    cam_engine = GradCAM(model)

    correct, wrong = [], []
    for i in np.random.RandomState(0).permutation(len(ds)):
        if len(correct) >= args.n and len(wrong) >= args.n:
            break
        x, y = ds[int(i)]
        cam, pred, probs = cam_engine(x.unsqueeze(0).to(device))
        item = (denormalize(x), cam, y, pred, float(probs[pred]))
        if pred == y and len(correct) < args.n:
            correct.append(item)
        elif pred != y and len(wrong) < args.n:
            wrong.append(item)

    os.makedirs(args.out, exist_ok=True)
    for name, items in (("correct", correct), ("wrong", wrong)):
        if not items:
            continue
        fig, axes = plt.subplots(2, len(items), figsize=(3 * len(items), 6.4), squeeze=False)
        for j, (img, cam, y, pred, conf) in enumerate(items):
            axes[0, j].imshow(img); axes[0, j].axis("off")
            axes[0, j].set_title(f"True: {CLASSES[y]}", fontsize=10)
            axes[1, j].imshow(overlay(img, cam)); axes[1, j].axis("off")
            axes[1, j].set_title(f"Pred: {CLASSES[pred]} ({conf:.0%})", fontsize=10,
                                 color="green" if pred == y else "red")
        fig.suptitle(f"Grad-CAM - {name} predictions", fontsize=13)
        plt.tight_layout()
        path = os.path.join(args.out, f"gradcam_{name}.png")
        plt.savefig(path, dpi=130)
        plt.close()
        print("saved", path)


if __name__ == "__main__":
    main()
