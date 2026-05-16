"""
Create out/latentspaces.csv from latent .npy files in out/latentspaces.

Run from project root:
python3 validation/generate_latentspaces_csv.py
"""

import glob
import os
from os.path import basename

import numpy as np
import pandas as pd


columns = ["id"]
for layer in range(1, 19):
    for col in range(1, 513):
        columns.append(f"{layer}_{col}")

df = pd.DataFrame(columns=columns)

for f in sorted(glob.glob("out/latentspaces/*.npy")):
    image_id = basename(f).split("_")[0]
    np_matrix = np.load(f)
    values = [image_id]
    values.extend(list(np_matrix.flatten()))
    df.loc[len(df)] = values

os.makedirs("out", exist_ok=True)
df.to_csv("out/latentspaces.csv", index=False)
