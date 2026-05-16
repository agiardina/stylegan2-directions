#!/usr/bin/env python
"""
Generate image variations by applying latent directions.

Age and gender directions are loaded from the original external repository.
Before running:
git clone https://github.com/a312863063/generators-with-stylegan2.git
"""

import sys

sys.path.insert(0, "./stylegan2-ada-pytorch")

import glob
import os
import shutil

import numpy as np
import torch
from legacy import load_network_pkl
from PIL import Image


network_pkl = "models/ffhq.pkl"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
output_dir = "out/images"

if os.path.exists(output_dir):
    raise SystemExit("The output image folder already exists. Remove it before running this script.")

with open(network_pkl, "rb") as f:
    G = load_network_pkl(f)["G_ema"].to(device)


def generate_image_from_w(w):
    w_tensor = torch.tensor(w).float().to(device)

    with torch.no_grad():
        image = G.synthesis(w_tensor, noise_mode="const")

    image = (image.clamp(-1, 1) + 1) * 127.5
    image = image.permute(0, 2, 3, 1).cpu().numpy()[0]
    return Image.fromarray(image.astype(np.uint8))


v = {}
v["age"] = np.load("generators-with-stylegan2/latent_directions/age.npy")
v["age"] = v["age"] / np.linalg.norm(v["age"])
v["gender"] = np.load("generators-with-stylegan2/latent_directions/gender.npy")
v["gender"] = v["gender"] / np.linalg.norm(v["gender"])
v["blueeyes"] = np.load("directions/blueeyes.npy")
v["honeyeyes"] = np.load("directions/honeyeyes.npy")
v["eyebrow"] = np.load("directions/eyebrown.npy")
v["nose"] = np.load("directions/nose.npy")
v["lips"] = np.load("directions/lips.npy")
v["chin"] = np.load("directions/chin.npy")

b = {}
b["age"] = [-20, 20]
b["gender"] = [-20, 20]
b["blueeyes"] = [-20, 20]
b["honeyeyes"] = [-30, 30]
b["eyebrow"] = [-40, 40]
b["nose"] = [-20, 20]
b["lips"] = [-20, 20]
b["chin"] = [-30, 30]


os.makedirs("out", exist_ok=True)
os.mkdir(output_dir)


def process_image(image_number):
    original = "data/test_images/{}.jpg".format(image_number)
    target = "{}/{}_aligned.jpg".format(output_dir, image_number)
    shutil.copyfile(original, target)

    w = np.load("out/latentspaces/{}_latentspace_restyle.npy".format(image_number))
    if w.shape == (18, 512):
        w = np.expand_dims(w, axis=0)

    image = generate_image_from_w(w)
    image.save("{}/{}_projected.jpg".format(output_dir, image_number), quality=90)

    n = 5
    for key in v.keys():
        for v_magn in np.linspace(b[key][0], b[key][1], n):
            if v_magn != 0:
                w_mod = w + v_magn * v[key]
                if v_magn > 0:
                    pref = "pos"
                else:
                    pref = "neg"

                image = generate_image_from_w(w_mod)
                filename = "{}_{}_{}{}.jpg".format(
                    image_number, key, pref, int(abs(v_magn))
                )
                filepath = output_dir + "/" + filename
                image.save(filepath)

                print(filename)


for file in glob.glob("data/test_images/*.jpg"):
    n = file.split("/")[-1].split(".")[0]
    process_image(n)
