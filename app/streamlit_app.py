"""Streamlit demo: upload a dermoscopic image -> prediction + Grad-CAM heatmap.

Run:  streamlit run app/streamlit_app.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import streamlit as st
import torch
from PIL import Image

from src.dataset import get_transforms
from src.gradcam import GradCAM, overlay
from src.hamdata import CLASS_NAMES
from src.model import load_checkpoint

CKPT = os.environ.get("CKPT_PATH", "models/best_model.pt")

st.set_page_config(page_title="Explainable Skin Lesion Classifier", page_icon="🔬", layout="wide")
st.title("🔬 Explainable Skin Lesion Classifier")
st.caption("EfficientNet-B0 (transfer learning) trained on HAM10000 + Grad-CAM explanations")
st.warning("Educational project only. NOT a medical device and NOT a diagnosis. "
           "Consult a dermatologist for any skin concern.")


@st.cache_resource
def get_model():
    model, ckpt = load_checkpoint(CKPT, "cpu")
    return model, GradCAM(model), ckpt


if not os.path.exists(CKPT):
    st.error(f"Model checkpoint not found at '{CKPT}'. Train first (see README) and place best_model.pt in /models.")
    st.stop()

model, cam_engine, ckpt = get_model()
classes, size = ckpt["classes"], ckpt.get("img_size", 224)

file = st.file_uploader("Upload a dermoscopic skin lesion image", type=["jpg", "jpeg", "png"])
if file:
    img = Image.open(file).convert("RGB")
    x = get_transforms(size, train=False)(img).unsqueeze(0)
    cam, pred, probs = cam_engine(x)
    shown = img.resize((size, size))
    heat = overlay(np.asarray(shown), cam)

    c1, c2, c3 = st.columns(3)
    c1.image(shown, caption="Input")
    c2.image(heat, caption=f"Grad-CAM for '{classes[pred]}'")
    with c3:
        st.subheader(CLASS_NAMES[classes[pred]])
        st.metric("Confidence", f"{probs[pred]:.1%}")
        if probs[pred] < 0.5:
            st.info("Low confidence - the model is unsure about this image.")
        df = pd.DataFrame({"class": classes, "probability": probs}).set_index("class")
        st.bar_chart(df)
    st.caption("Red/yellow regions = areas that influenced the prediction most.")
else:
    st.info("Upload an image to start. Sample images are in the HAM10000 dataset.")
