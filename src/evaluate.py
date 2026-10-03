"""Evaluate the best checkpoint on the held-out test split.

Example:
    python -m src.evaluate --data-dir data/ham10000
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader

from .dataset import HAMDataset, get_transforms
from .hamdata import CLASSES, build_dataframe, split_dataframe
from .model import load_checkpoint


def plot_confusion(cm, path):
    cmn = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(CLASSES))); ax.set_xticklabels(CLASSES)
    ax.set_yticks(range(len(CLASSES))); ax.set_yticklabels(CLASSES)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title("Normalized Confusion Matrix (Test)")
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            ax.text(j, i, f"{cm[i, j]}", ha="center", va="center",
                    color="white" if cmn[i, j] > 0.5 else "black", fontsize=9)
    fig.colorbar(im); plt.tight_layout(); plt.savefig(path, dpi=140); plt.close()


def plot_curves(hist_path, path):
    with open(hist_path) as f:
        h = json.load(f)
    ep = range(1, len(h["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, key, title in zip(axes, ("loss", "acc", "f1"), ("Loss", "Accuracy", "Macro F1")):
        ax.plot(ep, h[f"train_{key}"], label="train", marker="o")
        ax.plot(ep, h[f"val_{key}"], label="val", marker="o")
        ax.set_title(title); ax.set_xlabel("Epoch"); ax.grid(alpha=.3); ax.legend()
    plt.tight_layout(); plt.savefig(path, dpi=140); plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--ckpt", default="models/best_model.pt")
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--subset", type=int, default=0)
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, ckpt = load_checkpoint(args.ckpt, device)
    size = ckpt.get("img_size", 224)

    df = build_dataframe(args.data_dir)
    if args.subset:
        df = df.sample(n=min(args.subset, len(df)), random_state=args.seed).reset_index(drop=True)
    _, _, test_df = split_dataframe(df, seed=args.seed)
    ds = HAMDataset(test_df, get_transforms(size, False), cache_size=size)
    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=args.workers)

    preds, targets = [], []
    with torch.no_grad():
        for x, y in loader:
            preds += model(x.to(device)).argmax(1).cpu().tolist()
            targets += y.tolist()

    os.makedirs(args.results_dir, exist_ok=True)
    labels = list(range(len(CLASSES)))
    report = classification_report(targets, preds, labels=labels, target_names=CLASSES,
                                   zero_division=0, digits=3)
    rep_dict = classification_report(targets, preds, labels=labels, target_names=CLASSES,
                                     zero_division=0, output_dict=True)
    cm = confusion_matrix(targets, preds, labels=labels)
    print(report)

    with open(os.path.join(args.results_dir, "classification_report.txt"), "w") as f:
        f.write(report)
    metrics = {
        "test_images": len(targets),
        "accuracy": rep_dict["accuracy"],
        "macro_f1": rep_dict["macro avg"]["f1-score"],
        "weighted_f1": rep_dict["weighted avg"]["f1-score"],
        "melanoma_recall": rep_dict["mel"]["recall"],
        "melanoma_precision": rep_dict["mel"]["precision"],
    }
    with open(os.path.join(args.results_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    plot_confusion(cm, os.path.join(args.results_dir, "confusion_matrix.png"))
    hist = os.path.join(args.results_dir, "history.json")
    if os.path.exists(hist):
        plot_curves(hist, os.path.join(args.results_dir, "training_curves.png"))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
