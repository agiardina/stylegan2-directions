#!/usr/bin/env python3
"""
Generate L1/L2 norms summary files from latent spaces.

Run from project root:
python3 analysis/generate_norms.py

Inputs:
- out/latentspaces/*.npy

Outputs:
- out/analysis/norms.csv
- out/analysis/stats.csv
"""

import glob
import os
from os.path import basename

import numpy as np
import pandas as pd


data = []
for f in glob.glob("out/latentspaces/*.npy"):
    image_id = basename(f).split("_")[0]
    np_matrix = np.load(f)
    v = np_matrix.flatten()
    norm_l2 = np.linalg.norm(v)
    norm_l1 = np.linalg.norm(v, ord=1)
    data.append({"id": image_id, "norm_l1": norm_l1, "norm_l2": norm_l2})

df = pd.DataFrame(data)
df = df.iloc[df["id"].astype(float).argsort()]

os.makedirs("out/analysis", exist_ok=True)
df.to_csv("out/analysis/norms.csv", index=False, encoding="utf-8", sep=",")
df.describe().to_csv("out/analysis/stats.csv", sep=",")
