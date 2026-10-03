"""Create a tiny FAKE dataset in HAM10000 layout (for smoke-testing the pipeline only)."""
import os
import sys

import numpy as np
import pandas as pd
from PIL import Image

out = sys.argv[1] if len(sys.argv) > 1 else "data/dummy"
n = int(sys.argv[2]) if len(sys.argv) > 2 else 120
os.makedirs(os.path.join(out, "images"), exist_ok=True)
rng = np.random.RandomState(0)
classes = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
rows = []
for i in range(n):
    dx = classes[i % 7]
    arr = (rng.rand(96, 96, 3) * 255).astype("uint8")
    arr[..., i % 3] = np.clip(arr[..., i % 3] + (i % 7) * 20, 0, 255)
    Image.fromarray(arr).save(os.path.join(out, "images", f"ISIC_{i:07d}.jpg"))
    rows.append({"lesion_id": f"HAM_{i // 2:07d}", "image_id": f"ISIC_{i:07d}", "dx": dx,
                 "dx_type": "histo", "age": 50, "sex": "male", "localization": "back"})
pd.DataFrame(rows).to_csv(os.path.join(out, "HAM10000_metadata.csv"), index=False)
print(f"dummy dataset with {n} images at {out}")
