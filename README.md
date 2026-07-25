# Wavelet-Driven GAN with Attention Mechanism for Improved Raindrop Removal

### Manuscript ID: IEEE LATAM Submission ID: 10735
**Code repository accompanying the submitted journal manuscript**

#### Authors and Affiliations

- **Muthukumar Balamurugan** (Department of Electronics and Communication Engineering, National Institute of Technology Tiruchirappalli, Tamil Nadu, India; Valeo India Private Limited, Tamil Nadu, India)
- **Shivarama K. Holla** (AUMOVIO Autonomous Mobility India Private Limited, Bengaluru, Karnataka, India)
- **Varun P. Gopi** (Department of Electronics and Communication Engineering, National Institute of Technology Tiruchirappalli, Tamil Nadu, India)

---

This repository contains the complete implementation of **D2W-GAN**, a wavelet-driven Generative Adversarial Network (GAN) with attention mechanisms for single-image raindrop removal.

The repository includes:

- Training pipeline for D2W-GAN
- Testing and inference scripts
- DT-CWT-based multi-scale feature extraction
- Attention-guided U-Net generator
- Pix2Pix PatchGAN discriminator
- Multi-GPU training support
- Automatic checkpoint saving and loading
- Inference time measurement
- Model parameter evaluation
- Restoration result visualization

---

# Project Objective

The objective of this work is to restore high-quality clean images from raindrop-degraded inputs by integrating the **Dual-Tree Complex Wavelet Transform (DT-CWT)** with an attention-guided Generative Adversarial Network.

The proposed framework aims to:

- Remove adherent raindrops from a single image
- Preserve structural details and fine textures
- Improve perceptual image quality
- Enhance downstream computer vision applications such as object detection and autonomous driving

---

# Dataset

The proposed framework is evaluated on publicly available raindrop removal datasets.

| Dataset | Purpose |
|----------|----------|
| RainDrop Dataset | Model training and evaluation |
| RainDS Dataset | Generalization evaluation on real-world raindrop images |

---

# Installation

## Clone Repository

```bash
git clone https://github.com/<username>/D2W-GAN.git

cd D2W-GAN
```

---

## Create Virtual Environment

```bash
python -m venv venv
```

---

## Activate Environment

### Windows

```bash
venv\Scripts\activate
```

### Linux

```bash
source venv/bin/activate
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Dataset Organization

Organize the dataset as follows.

```text
data/
├── train/
│
├── test_a/
│   └── data/
│       ├── image001.png
│       ├── image002.png
│       └── ...
│
└── test_b/
    └── data/
        ├── image001.png
        ├── image002.png
        └── ...
```

> **Note:** Training image pairs and testing images should follow the dataset format expected by the data loader implemented in `model/find_model.py`.

---

# Training

Run

```bash
python train.py
```

Example

```bash
python train.py \
    --model proposed \
    --dataset_dir ./data \
    --batch_size 8 \
    --max_epochs 2000 \
    --lr 0.0002
```

---

# Testing

Evaluate a trained model using

```bash
python test.py
```

Example

```bash
python test.py \
    --model proposed \
    --ckpt_epoch 400 \
    --dataset_dir ./data
```

Available resize modes

- original
- square
- expand

---

# Expected Outputs

Model checkpoints are saved automatically in

```text
checkpoints/
└── proposed/
```

The restored images are saved in

```text
results/
└── proposed/
    └── test_b/
        └── 400_original/
            ├── output/
            ├── image001.png
            ├── image002.png
            └── ...
```

The testing script additionally reports:

- Total trainable parameters
- Average inference time
- Per-image inference time
- Restored output images

---

# Experimental Settings

The proposed model was trained using:

- Optimizer: Adam
- Learning rate: 2 × 10⁻⁴
- Batch size: 4 (paper) / configurable in the code
- Training epochs: 1700 (paper)
- Multi-scale DT-CWT decomposition
- Attention-guided U-Net generator
- PatchGAN discriminator

---

# Requirements

```text
Python >= 3.8

torch>=1.12
torchvision>=0.13
opencv-python
numpy
Pillow
thop
tqdm
```

Install all dependencies using

```bash
pip install -r requirements.txt
```

---

# Citation

If you find this repository useful in your research, please cite:

```
Citation information will be updated once the manuscript is published
```

---

# License

This project is released for academic and research purposes. If you use this repository in your research, please cite the corresponding publication.