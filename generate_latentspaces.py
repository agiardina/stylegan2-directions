#!/usr/bin/env python
"""
Project all aligned test images into StyleGAN2 W+ space with ReStyle-pSp.

Before running:
git clone https://github.com/yuval-alaluf/restyle-encoder.git

Download the FFHQ ReStyle-pSp checkpoint from:
https://drive.google.com/file/d/1sw6I2lRIB0MpuJkpc8F5BJiSZrc0hjfE/view?usp=sharing

Save it as:
models/restyle_psp_ffhq_encode.pt
"""

import glob
import os
import sys
from argparse import Namespace

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image


input_dir = "data/test_images"
output_dir = "out/latentspaces"
model_path = "models/restyle_psp_ffhq_encode.pt"
device = "cuda" if torch.cuda.is_available() else "cpu"

if os.path.exists(output_dir):
    raise SystemExit("The latentspaces folder already exists. Remove it before running this script.")

os.makedirs("out", exist_ok=True)
os.mkdir(output_dir)

sys.path.insert(0, "restyle-encoder")

from models.psp import pSp
from utils.inference_utils import run_on_batch


img_transforms = transforms.Compose(
    [
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ]
)

ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
opts = ckpt["opts"]
opts["checkpoint_path"] = model_path
opts = Namespace(**opts)
net = pSp(opts)

net.eval()
net.to(device)

opts.n_iters_per_batch = 5
opts.resize_outputs = False


def get_avg_image(net):
    avg_image = net(
        net.latent_avg.unsqueeze(0),
        input_code=True,
        randomize_noise=False,
        return_latents=False,
        average_code=True,
    )[0]
    avg_image = avg_image.to(device).float().detach()
    return avg_image


def project_image(image_path):
    input_image = Image.open(image_path).convert("RGB")
    transformed_image = img_transforms(input_image)

    with torch.no_grad():
        avg_image = get_avg_image(net)
        _, result_latents = run_on_batch(
            transformed_image.unsqueeze(0).to(device), net, opts, avg_image
        )

    return result_latents[0][4]


for image_path in glob.glob(input_dir + "/*.jpg"):
    image_id = os.path.splitext(os.path.basename(image_path))[0]
    latent_space = project_image(image_path)
    latent_space_path = "{}/{}_latentspace_restyle.npy".format(output_dir, image_id)
    np.save(latent_space_path, latent_space)
    print(latent_space_path)
