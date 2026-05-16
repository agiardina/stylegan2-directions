#!/usr/bin/env python

 """
  Example usage.

  Before running this file:
  1. Clone the StyleGAN2-ADA PyTorch repository into this project folder:
     https://github.com/NVlabs/stylegan2-ada-pytorch

  2. Download the pre-trained FFHQ StyleGAN2 model and place it in the
     `models/` directory, for example:
     models/ffhq.pkl. You can download the model with curl:
     curl -L -o models/ffhq.pkl https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl


  3. Install the Python dependencies required by StyleGAN2-ADA PyTorch,
     including PyTorch, NumPy, and Pillow.

  This example loads a projected latent vector, applies the `nose` direction
  in W+ latent space, and saves the resulting image to the output folder.
  """


import sys

sys.path.insert(0, "./stylegan2-ada-pytorch")

import numpy as np
import torch
from legacy import load_network_pkl
from PIL import Image


network_pkl = "models/ffhq.pkl"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Loading StyleGAN2 model...")
with open(network_pkl, "rb") as f:
    G = load_network_pkl(f)["G_ema"].to(device)


def generate_image_from_w(w_vector, G):
    w_tensor = w_vector.clone().detach().to(device)

    with torch.no_grad():
        img = G.synthesis(w_tensor, noise_mode="const")

    # Convert StyleGAN output from CHW tensors in [-1, 1] to a uint8 RGB image.
    img = (img.clamp(-1, 1) + 1) * 127.5
    img = img.permute(0, 2, 3, 1).cpu().numpy()[0]
    return Image.fromarray(img.astype(np.uint8))


w_source = np.load("latentspaces/91_latentspace_restyle.npy").reshape((1, 18, 512))
w_source = torch.tensor(w_source)

generate_image_from_w(w_source, G)

w_nose = np.load("directions/nose.npy").reshape((1, 18, 512))
w_nose = torch.tensor(w_nose)

img = generate_image_from_w(w_source + (40 * w_nose), G)
img = img.resize((512, 512), Image.Resampling.LANCZOS)
img.save("output/big_nose.png")
