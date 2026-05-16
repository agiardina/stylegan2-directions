#!/usr/bin/env python
"""
Create a StyleGAN2 direction from two W+ latent codes.

The input latent codes can be generated with project_face.py.
Usage:
python3 create_direction.py original_latent.npy edited_latent.npy directions/new_direction.npy
"""

import os
import sys

import numpy as np


original_latent_path = sys.argv[1]
edited_latent_path = sys.argv[2]
output_path = sys.argv[3]

original_latent = np.load(original_latent_path)
edited_latent = np.load(edited_latent_path)

if original_latent.shape == (18, 512):
    original_latent = np.expand_dims(original_latent, axis=0)

if edited_latent.shape == (18, 512):
    edited_latent = np.expand_dims(edited_latent, axis=0)

direction = edited_latent - original_latent
direction = direction / np.linalg.norm(direction)

os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
np.save(output_path, direction.astype(np.float32))
