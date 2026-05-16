#!/usr/bin/env python
"""
Generate the 68-point dlib landmark table used by the color validation scripts.

Run this script from the project root:
python3 validation/generate_landmarks68.py

Before running, download the dlib landmark model:
mkdir -p models
curl -L -o models/shape_predictor_68_face_landmarks.dat.bz2 http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
bzip2 -dk models/shape_predictor_68_face_landmarks.dat.bz2

The out/images, out/measurements, and models paths are resolved relative to the
project root.
"""

import glob
import os

import dlib
import numpy as np
import pandas as pd
from PIL import Image


input_dir = "out/images"
output_path = "out/measurements/landmarks68.xlsx"
predictor_path = "models/shape_predictor_68_face_landmarks.dat"

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

rows = []

for image_path in sorted(glob.glob(input_dir + "/*.jpg")):
    filename = os.path.basename(image_path)
    img = Image.open(image_path).convert("RGB")
    img_np = np.array(img)
    detections = detector(img_np, 1)
    shape = predictor(img_np, detections[0])

    row = {"Filename": filename}
    i = 1
    for point in shape.parts():
        row[str(i)] = point.x
        row[str(i + 1)] = point.y
        i = i + 2

    rows.append(row)
    print(filename)

df = pd.DataFrame(rows, columns=["Filename"] + [str(i) for i in range(1, 137)])
os.makedirs(os.path.dirname(output_path), exist_ok=True)
df.to_excel(output_path, index=False)
