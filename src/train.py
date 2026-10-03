"""Train EfficientNet-B0 on HAM10000.

Example:
    python -m src.train --data-dir data/ham10000 --epochs 12
"""
import argparse
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from .dataset import HAMDataset, get_transforms
from .hamdata import CLASSES, build_dataframe, split_dataframe
from .model import build_model


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run_epoch(model, loader, criterion, device, optimizer=None, scaler=None):
    train = optimizer is not None
    model.train(train)
    total_loss, preds, targets = 0.0, [], []
    use_amp = device == "cuda"
    with torch.set_grad_enabled(train):
        for x, y in tqdm(loader, leave=False, desc="train" if train else "val"):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", enabled=use_amp):
                out = model(x)
                loss = criterion(out, y)
            if train:
                optimizer.zero_grad(set_to_none=True)
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
            total_loss += loss.item() * x.size(0)
            preds += out.argmax(1).cpu().tolist()
            targets += y.cpu().tolist()
    n = len(targets)
    acc = float(np.mean(np.array(preds) == np.array(targets)))
    f1 = float(f1_score(targets, preds, average="macro", zero_division=0))
    return total_loss / n, acc, f1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out-dir", default="models")
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--subset", type=int, default=0, help="use only N images (quick smoke test)")
    ap.add_argument("--no-pretrained", action="store_true", help="skip ImageNet weights (offline test)")
    args = ap.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    df = build_dataframe(args.data_dir)
    if args.subset:
        df = df.sample(n=min(args.subset, len(df)), random_state=args.seed).reset_index(drop=True)
    train_df, val_df, test_df = split_dataframe(df, seed=args.seed)
    print(f"train={len(train_df)}  val={len(val_df)}  test={len(test_df)}")
    print("train class counts:", train_df["dx"].value_counts().to_dict())

    cache = args.img_size + 32
    train_ds = HAMDataset(train_df, get_transforms(args.img_size, True), cache)
    val_ds = HAMDataset(val_df, get_transforms(args.img_size, False), cache)
    train_loader = DataLoader(train_ds, args.batch_size, shuffle=True, num_workers=args.workers,
                              pin_memory=(device == "cuda"), drop_last=len(train_ds) > args.batch_size)
    val_loader = DataLoader(val_ds, args.batch_size, shuffle=False, num_workers=args.workers)

    # class-weighted loss to handle the heavy imbalance (nv ~ 67% of data)
    counts = np.bincount(train_df["label"], minlength=len(CLASSES)).astype(float)
    weights = counts.sum() / (len(CLASSES) * np.maximum(counts, 1))
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32).to(device))

    model = build_model(len(CLASSES), pretrained=not args.no_pretrained).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda") if device == "cuda" else None

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)
    history = {k: [] for k in ("train_loss", "val_loss", "train_acc", "val_acc", "train_f1", "val_f1")}
    best_f1, best_path = -1.0, os.path.join(args.out_dir, "best_model.pt")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tl, ta, tf = run_epoch(model, train_loader, criterion, device, optimizer, scaler)
        vl, va, vf = run_epoch(model, val_loader, criterion, device)
        scheduler.step()
        for k, v in zip(history, (tl, vl, ta, va, tf, vf)):
            history[k].append(v)
        flag = ""
        if vf > best_f1:
            best_f1 = vf
            torch.save({"model": model.state_dict(), "classes": CLASSES,
                        "img_size": args.img_size, "val_macro_f1": vf, "epoch": epoch}, best_path)
            flag = "  <- saved best"
        print(f"Epoch {epoch:02d}/{args.epochs} | train loss {tl:.3f} acc {ta:.3f} | "
              f"val loss {vl:.3f} acc {va:.3f} macroF1 {vf:.3f} | {time.time()-t0:.0f}s{flag}")

    with open(os.path.join(args.results_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)
    print(f"Done. Best val macro-F1 = {best_f1:.3f}. Checkpoint: {best_path}")


if __name__ == "__main__":
    main()
