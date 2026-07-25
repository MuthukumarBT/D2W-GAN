# D2W-GAN

Official implementation of **DeRainDrop**, a deep learning framework for raindrop removal from single images.

---

## Overview

This repository provides the implementation for training and evaluating the proposed DeRainDrop model. The framework supports both training from scratch and inference using pretrained checkpoints.

---

## Requirements

### Hardware
- NVIDIA GPU (recommended)
- CUDA-enabled environment

### Software
- Python >= 3.8
- PyTorch >= 1.12
- CUDA >= 11.3 (recommended)

### Python Packages

Install the required dependencies using:

```bash
pip install -r requirements.txt
```

or install manually:

```bash
pip install torch torchvision
pip install opencv-python
pip install numpy
pip install thop
```

---

## Repository Structure

```
DeRainDrop/
│
├── data/
│   ├── train/
│   ├── test_a/
│   └── test_b/
│
├── model/
├── utils.py
├── train.py
├── test.py
├── requirements.txt
├── checkpoints/
├── results/
└── README.md
```

---

# Dataset Preparation

## Training Dataset

Place the training dataset inside

```
data/
└── train/
```

Example:

```
data/
└── train/
    ├── image001.png
    ├── image002.png
    ├── image003.png
    └── ...
```

If your dataset contains paired rainy and clean images, organize them according to the dataset loader implemented in `find_model()`.

---

## Testing Dataset

Testing images should be placed under

```
data/
└── test_b/
    └── data/
```

Example:

```
data/
└── test_b/
    └── data/
        ├── test001.png
        ├── test002.png
        ├── test003.png
        └── ...
```

Additional test datasets can be added following the same structure.

---

# Training

Run

```bash
python train.py
```

Common options

```bash
python train.py \
    --model proposed \
    --dataset_dir ./data \
    --batch_size 8 \
    --max_epochs 2000 \
    --lr 0.0002
```

### Training Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--model` | Model name | proposed |
| `--dataset_dir` | Dataset directory | ./data |
| `--batch_size` | Batch size | 8 |
| `--max_epochs` | Number of epochs | 2000 |
| `--lr` | Learning rate | 0.0002 |
| `--ckpt_dir` | Checkpoint directory | ./checkpoints |
| `--save_dir` | Output directory | ./results |

---

# Testing

Run

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

### Testing Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--model` | Model name | proposed |
| `--ckpt_epoch` | Checkpoint epoch | 400 |
| `--dataset_dir` | Dataset directory | ./data |
| `--resize` | Resize mode (`original`, `square`, `expand`) | original |

---

# Output

Results are saved to

```
results/
└── proposed/
    └── test_b/
        └── 400_original/
            ├── output/
            ├── image001.png
            ├── image002.png
            └── ...
```

---

# Checkpoints

Model checkpoints are stored in

```
checkpoints/
└── proposed/
```

To evaluate a trained model, specify the checkpoint epoch:

```bash
python test.py --ckpt_epoch 400
```

---

# Citation

If you find this work useful, please cite our paper:

```bibtex
@article{yourpaper2026,
  title={Title of Your Paper},
  author={Author One and Author Two},
  journal={Journal Name},
  year={2026}
}
```

---

# License

This project is released for research purposes. Please cite the corresponding paper if you use this repository in your research.
=======
# DewdropNet - Raindrop Removal Evaluation

This repository contains the code for evaluating the performance of a raindrop removal model using various image quality metrics. The evaluation includes metrics such as PSNR, SSIM and NIQE on a set of raindrop-degraded images and their corresponding ground truth (clean) images.

## Prerequisites

Make sure you have the following libraries installed:

- Python 3.6.2
- OpenCV 4.2.0.34
- NumPy 1.19.5
- Scikit-image 0.17.2
- scipy 1.5.4

You can install the required libraries using the following command:

```bash
pip install opencv-python numpy scikit-image niqe

python evaluate.py --output_folder "path/to/output_folder" --gt_folder "path/to/gt_folder"   
```

## Dataset 

The whole dataset can be find in ATTGAN author pages(https://github.com/rui1996/DeRaindrop)
The whole dataset can be find here as well in drive directly (https://drive.google.com/open?id=1e7R76s6vwUJxILOcAsthgDLPSnOrQ49K)

####Training Set:

861 image pairs for training.

####Testing Set A:

For quantitative evaluation where the alignment of image pairs is good. A subset of testing set B.

####Testing Set B:

239 image pairs for testing.

Replace the following placeholders with your actual folder paths:

"path/to/output_folder": The folder containing the raindrop-degraded images (model outputs).  
"path/to/gt_folder": The folder containing the corresponding ground truth (clean) images.

## Acknowledgements

This code is based on the implementations of [Raindrop-Removal](https://github.com/Hyukju/Raindrop-Removal).
<<<<<<< Updated upstream
=======

>>>>>>> Stashed changes
