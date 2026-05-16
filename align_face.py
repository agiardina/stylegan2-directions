#!/usr/bin/env python
"""
Align an input face image for StyleGAN2/FFHQ.

Before running, download the dlib landmark model:
mkdir -p models
curl -L -o models/shape_predictor_68_face_landmarks.dat.bz2 http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
bzip2 -dk models/shape_predictor_68_face_landmarks.dat.bz2
"""

import os
import sys

import dlib
import numpy as np
import scipy.ndimage
from PIL import Image


predictor_path = "models/shape_predictor_68_face_landmarks.dat"
input_path = sys.argv[1]
output_path = sys.argv[2]

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

img = Image.open(input_path).convert("RGB")
img_np = np.array(img)
detections = detector(img_np, 1)

shape = predictor(img_np, detections[0])
lm = np.array([[point.x, point.y] for point in shape.parts()])

lm_eye_left = lm[36:42]
lm_eye_right = lm[42:48]
lm_mouth_outer = lm[48:60]

eye_left = np.mean(lm_eye_left, axis=0)
eye_right = np.mean(lm_eye_right, axis=0)
eye_avg = (eye_left + eye_right) * 0.5
eye_to_eye = eye_right - eye_left
mouth_left = lm_mouth_outer[0]
mouth_right = lm_mouth_outer[6]
mouth_avg = (mouth_left + mouth_right) * 0.5
eye_to_mouth = mouth_avg - eye_avg

x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]
x /= np.hypot(*x)
x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)
y = np.flipud(x) * [-1, 1]
c = eye_avg + eye_to_mouth * 0.1
quad = np.stack([c - x - y, c - x + y, c + x + y, c + x - y])
qsize = np.hypot(*x) * 2

transform_size = 4096
output_size = 1024
enable_padding = True

shrink = int(np.floor(qsize / output_size * 0.5))
if shrink > 1:
    rsize = (
        int(np.rint(float(img.size[0]) / shrink)),
        int(np.rint(float(img.size[1]) / shrink)),
    )
    img = img.resize(rsize, Image.Resampling.LANCZOS)
    quad /= shrink
    qsize /= shrink

border = max(int(np.rint(qsize * 0.1)), 3)
crop = (
    int(np.floor(min(quad[:, 0]))),
    int(np.floor(min(quad[:, 1]))),
    int(np.ceil(max(quad[:, 0]))),
    int(np.ceil(max(quad[:, 1]))),
)
crop = (
    max(crop[0] - border, 0),
    max(crop[1] - border, 0),
    min(crop[2] + border, img.size[0]),
    min(crop[3] + border, img.size[1]),
)
if crop[2] - crop[0] < img.size[0] or crop[3] - crop[1] < img.size[1]:
    img = img.crop(crop)
    quad -= crop[0:2]

pad = (
    int(np.floor(min(quad[:, 0]))),
    int(np.floor(min(quad[:, 1]))),
    int(np.ceil(max(quad[:, 0]))),
    int(np.ceil(max(quad[:, 1]))),
)
pad = (
    max(-pad[0] + border, 0),
    max(-pad[1] + border, 0),
    max(pad[2] - img.size[0] + border, 0),
    max(pad[3] - img.size[1] + border, 0),
)
if enable_padding and max(pad) > border - 4:
    pad = np.maximum(pad, int(np.rint(qsize * 0.3)))
    img = np.pad(
        np.float32(img),
        ((pad[1], pad[3]), (pad[0], pad[2]), (0, 0)),
        "reflect",
    )
    h, w, _ = img.shape
    y_grid, x_grid, _ = np.ogrid[:h, :w, :1]
    mask = np.maximum(
        1.0 - np.minimum(np.float32(x_grid) / pad[0], np.float32(w - 1 - x_grid) / pad[2]),
        1.0 - np.minimum(np.float32(y_grid) / pad[1], np.float32(h - 1 - y_grid) / pad[3]),
    )
    blur = qsize * 0.02
    img += (scipy.ndimage.gaussian_filter(img, [blur, blur, 0]) - img) * np.clip(
        mask * 3.0 + 1.0, 0.0, 1.0
    )
    img += (np.median(img, axis=(0, 1)) - img) * np.clip(mask, 0.0, 1.0)
    img = Image.fromarray(np.uint8(np.clip(np.rint(img), 0, 255)), "RGB")
    quad += pad[:2]

img = img.transform(
    (transform_size, transform_size),
    Image.Transform.QUAD,
    (quad + 0.5).flatten(),
    Image.Resampling.BILINEAR,
)

if output_size < transform_size:
    img = img.resize((output_size, output_size), Image.Resampling.LANCZOS)

os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
img.save(output_path)
