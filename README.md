# 🔬 Explainable Skin Lesion Classifier (HAM10000 + Grad-CAM)

Deep Learning mini project — B.Tech AI & Data Science.
A transfer-learning **EfficientNet-B0** classifies dermoscopic images into **7 skin-lesion types**, and **Grad-CAM** highlights *which part of the image* drove each prediction — making the model's decision inspectable instead of a black box.

> ⚠️ Educational project only. Not a medical device. Not a diagnosis.

## Why this is more than "another classifier"
| Challenge | How it is handled |
|---|---|
| Heavy class imbalance (nv ≈ 67 %) | Class-weighted cross-entropy + **macro-F1** as the model-selection metric |
| Data leakage (same lesion, several photos) | **Lesion-level split** using `lesion_id` (no lesion in two splits) |
| Black-box predictions | **Grad-CAM** implemented from scratch (`src/gradcam.py`) |
| Accuracy hides rare-class failure | Per-class precision/recall/F1, confusion matrix, **melanoma recall** reported |
| Not usable by others | **Streamlit web demo** (upload image → prediction + heatmap) |

## Dataset
HAM10000 — 10,015 dermoscopic images, 7 classes:
`akiec, bcc, bkl, df, mel, nv, vasc`.
Kaggle: https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000

## Project structure
```
src/
  hamdata.py   metadata loading + lesion-level train/val/test split
  dataset.py   PyTorch Dataset, augmentations, normalization
  model.py     EfficientNet-B0 with new 7-class head
  train.py     training loop (AdamW, cosine LR, AMP, class weights)
  evaluate.py  test metrics, confusion matrix, training curves
  gradcam.py   Grad-CAM + gallery of correct/wrong predictions
app/streamlit_app.py   web demo
notebooks/colab_train.ipynb   one-click free-GPU run
scripts/make_dummy_data.py    fake data to smoke-test the pipeline
```

## Run (free GPU) — recommended: Google Colab
1. Push this repo to GitHub, open `notebooks/colab_train.ipynb` in Colab, set **Runtime → GPU (T4)**.
2. Run all cells (downloads data via Kaggle API, trains ~12 epochs ≈ 20–30 min, evaluates, makes Grad-CAM figures).
3. Download `models/best_model.pt` and the `results/` folder, commit them to the repo.

### Or from the command line
```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python -m src.train    --data-dir data/ham10000 --epochs 12
python -m src.evaluate --data-dir data/ham10000
python -m src.gradcam  --data-dir data/ham10000
streamlit run app/streamlit_app.py
```
Quick pipeline check without the real dataset:
```bash
python scripts/make_dummy_data.py data/dummy 140
python -m src.train --data-dir data/dummy --epochs 1 --no-pretrained --batch-size 8 --workers 0
```

## Method
- **Model:** ImageNet-pretrained EfficientNet-B0, classifier replaced by `Dropout(0.3) → Linear(1280, 7)`, all layers fine-tuned.
- **Training:** AdamW (lr 2e-4, wd 1e-4), cosine annealing, mixed precision, batch 32, 224×224 inputs.
- **Augmentation:** random resized crop, H/V flips, rotation ±25°, colour jitter.
- **Selection:** best epoch by **validation macro-F1**.
- **Grad-CAM:** gradients of the predicted-class score w.r.t. the last conv block (`features[-1]`) → channel weights → weighted sum → ReLU → upsample → overlay.

## Results
Fill in after training (from `results/metrics.json`):

| Metric | Value |
|---|---|
| Test accuracy | _ |
| Macro F1 | _ |
| Weighted F1 | _ |
| Melanoma recall | _ |

Figures in `results/`: `training_curves.png`, `confusion_matrix.png`, `gradcam_correct.png`, `gradcam_wrong.png`.

## Limitations & future work
- Single public dataset; no external validation → may not generalize to other devices/skin tones.
- Grad-CAM shows *where* the model looked, not *why* it is right; it can highlight artefacts (rulers, ink marks).
- Future: focal loss, test-time augmentation, ensemble, calibration, Grad-CAM++ / Score-CAM.

## References
- Tschandl et al., *The HAM10000 dataset*, Scientific Data, 2018.
- Tan & Le, *EfficientNet*, ICML 2019.
- Selvaraju et al., *Grad-CAM*, ICCV 2017.
