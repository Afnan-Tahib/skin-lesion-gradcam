"""HAM10000 metadata handling (no torch dependency).

Key point: the SAME lesion can appear in several images, so we split by
`lesion_id` (not by image) to avoid data leakage between train/val/test.
"""
import glob
import os

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
CLASS_NAMES = {
    "akiec": "Actinic keratosis / Bowen's disease",
    "bcc": "Basal cell carcinoma",
    "bkl": "Benign keratosis-like lesion",
    "df": "Dermatofibroma",
    "mel": "Melanoma",
    "nv": "Melanocytic nevus (mole)",
    "vasc": "Vascular lesion",
}
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}


def build_dataframe(data_dir: str) -> pd.DataFrame:
    """Find metadata CSV + images anywhere under data_dir."""
    metas = glob.glob(os.path.join(data_dir, "**", "HAM10000_metadata*"), recursive=True)
    metas = [m for m in metas if m.lower().endswith(".csv")]
    if not metas:
        raise FileNotFoundError(f"HAM10000_metadata.csv not found under '{data_dir}'")
    df = pd.read_csv(metas[0])

    paths = {}
    for ext in ("jpg", "jpeg", "png"):
        for p in glob.glob(os.path.join(data_dir, "**", f"*.{ext}"), recursive=True):
            paths[os.path.splitext(os.path.basename(p))[0]] = p
    df["path"] = df["image_id"].map(paths)
    missing = int(df["path"].isna().sum())
    if missing == len(df):
        raise FileNotFoundError(f"No images from the metadata were found under '{data_dir}'")
    if missing:
        print(f"[warn] {missing} images listed in metadata were not found; dropping them")
        df = df.dropna(subset=["path"])
    df["label"] = df["dx"].map(CLASS_TO_IDX).astype(int)
    return df.reset_index(drop=True)


def split_dataframe(df: pd.DataFrame, seed: int = 42, val_frac: float = 0.15, test_frac: float = 0.15):
    """Lesion-level (group) split -> train / val / test with no lesion overlap."""
    gss = GroupShuffleSplit(n_splits=1, test_size=val_frac + test_frac, random_state=seed)
    train_idx, rest_idx = next(gss.split(df, groups=df["lesion_id"]))
    train_df, rest_df = df.iloc[train_idx], df.iloc[rest_idx]
    gss2 = GroupShuffleSplit(n_splits=1, test_size=test_frac / (val_frac + test_frac), random_state=seed)
    val_idx, test_idx = next(gss2.split(rest_df, groups=rest_df["lesion_id"]))
    val_df, test_df = rest_df.iloc[val_idx], rest_df.iloc[test_idx]
    return (train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True))
