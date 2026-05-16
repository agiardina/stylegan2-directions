# StyleGAN2 Directions Repository

[![DOI](https://zenodo.org/badge/447692854.svg)](https://doi.org/10.5281/zenodo.20245919)

This repository contains code, data, latent directions, generated outputs, and
analysis files for a manuscript in preparation. The final article title and
journal are not fixed yet.

All paths below are relative to the repository root.

## Repository Organization

```text
.
├── align_face.py              # Align one input image for StyleGAN2/FFHQ
├── project_face.py            # Project one aligned image into W+ with ReStyle-pSp
├── create_direction.py        # Create a direction from two projected latent codes
├── test_direction.py          # Minimal example of applying directions
├── data/                      # Input images and small metadata tables
├── directions/                # Direction vectors created for this project
├── validation/                # Scripts that generate validation outputs
├── analysis/                  # Scripts that generate analysis tables and figures
└── out/                       # Generated outputs used by validation and analysis
```

## Directory Conventions

`data/` contains project inputs that are tracked in Git, including test images,
direction source images, gender labels, and the Face++/dlib landmark protocol.

`directions/` contains the direction vectors produced for this project, such as
eye color, eyebrow, nose, lips, and chin directions. Age and gender directions
are not produced here; they are read from an external repository listed below.

`validation/` contains scripts that generate intermediate outputs used to
validate the method. These scripts write to `out/`, mainly:

- `out/latentspaces/`
- `out/latentspaces.csv`
- `out/images/`
- `out/measurements/`

`analysis/` contains scripts that consume tracked inputs and files in `out/` to
produce paper-oriented figures and summary tables. Analysis outputs are written
to `out/analysis/`.

`out/` contains generated material included for reproducibility: projected and
edited images, latent spaces, measurements, figures, and summary CSV files.

`models/` is not a source-code directory. It is the expected local destination
for downloaded model files.

## External Repositories and Model Files

Some scripts require external code or model files that are not authored in this
repository.

### StyleGAN2-ADA PyTorch

Required by scripts that synthesize images from latent codes.

```bash
git clone https://github.com/NVlabs/stylegan2-ada-pytorch.git
mkdir -p models
curl -L -o models/ffhq.pkl https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl
```

Expected paths:

```text
stylegan2-ada-pytorch/
models/ffhq.pkl
```

### ReStyle Encoder

Required by `project_face.py` and `validation/generate_latentspaces.py`.

```bash
git clone https://github.com/yuval-alaluf/restyle-encoder.git
```

Download the FFHQ ReStyle-pSp checkpoint from:

```text
https://drive.google.com/file/d/1sw6I2lRIB0MpuJkpc8F5BJiSZrc0hjfE/view?usp=sharing
```

Expected path:

```text
models/restyle_psp_ffhq_encode.pt
```

### Original Direction Repository

Required for the external age and gender directions used by
`validation/generate_variations.py` and `analysis/latents_hist.py`.

```bash
git clone https://github.com/a312863063/generators-with-stylegan2.git
```

Expected paths:

```text
generators-with-stylegan2/latent_directions/age.npy
generators-with-stylegan2/latent_directions/gender.npy
```

### dlib Landmark Model

Required by `align_face.py` and `validation/generate_landmarks68.py`.

```bash
mkdir -p models
curl -L -o models/shape_predictor_68_face_landmarks.dat.bz2 http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
bzip2 -dk models/shape_predictor_68_face_landmarks.dat.bz2
```

Expected path:

```text
models/shape_predictor_68_face_landmarks.dat
```

### Iris Segmentation Model

Required by `validation/color_eyes.py`.

The iris segmentation model comes from the KartalOl repository:

```text
https://github.com/Jalilnkh/KartalOl-NIR-ISL2021031301
```

Download the "Trained weights for Iris segmentation" file from:

```text
https://drive.google.com/file/d/1kJZcUX5lDqc7BiU7jSj0GTZuZwepbns8/view?usp=sharing
```

Expected path:

```text
models/MobileNetV2_Iris_Seg_10May.h5
```

## Main Scripts

Typical method scripts:

```bash
python3 align_face.py input.jpg aligned.jpg
python3 project_face.py aligned.jpg out/latentspaces/example_latentspace_restyle.npy
python3 create_direction.py original.npy edited.npy directions/new_direction.npy
```

Validation scripts:

```bash
python3 validation/generate_latentspaces.py
python3 validation/generate_latentspaces_csv.py
python3 validation/generate_variations.py
python3 validation/generate_landmarks68.py
python3 validation/color_skin.py
python3 validation/color_eyes.py
```

Analysis scripts:

```bash
python3 analysis/generate_norms.py
python3 analysis/latents_hist.py
Rscript analysis/pca.R
Rscript analysis/landmarks.R
```

## License and Third-Party Materials

Code authored for this repository is released under the MIT License. Copyright
is held collectively by the project authors. See `LICENSE` for the license text
and `CITATION.cff` for citation metadata.

The MIT License does not apply to third-party datasets, external repositories,
pre-trained models, checkpoints, or generated materials derived from restricted
third-party data. Those materials remain subject to their original licenses and
terms of use.

The images in `data/test_images/` are a subset of CelebA-HQ, which is derived
from the CelebA dataset. CelebA is made available for non-commercial research
purposes and is subject to restrictions on copying, publishing, and
redistribution under its original terms:

```text
https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html
```

Generated outputs derived from these images, including files under `out/images/`,
may also be subject to the same source-data restrictions and should not be
treated as MIT-licensed data.

External repositories and model files listed above retain their own upstream
licenses and terms. In particular, users should review the licenses and terms
for StyleGAN2-ADA, ReStyle, the original direction repository, the dlib landmark
model, and the iris segmentation model before redistribution.
