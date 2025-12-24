# GenFaD: Generalized Fault Detection

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.0+-green.svg)](https://opencv.org/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

**Generalized Crack Detection** (GenFaD) is a comprehensive framework for automated infrastructure crack detection that combines classical computer vision techniques with deep learning approaches. This project investigates cross-domain generalization challenges and proposes hybrid solutions for robust crack detection across diverse infrastructure types.

![Pipeline Overview](crack.jpg)

## 🎯 Key Features

- **Dual Approach Implementation**: Both traditional CV and deep learning pipelines
- **Cross-Domain Evaluation**: Rigorous testing across multiple datasets to assess generalization
- **Hybrid Architecture**: Combines SIFT-based proposal generation with ResNet classification
- **Geometric Quantification**: Automated crack measurement including width, length, and area
- **Multi-Dataset Support**: Compatible with 18+ crack detection datasets
- **Production-Ready**: Modular codebase with clear separation of concerns

## 📊 Project Highlights

Our research demonstrates:

- ✅ **99.73% accuracy** on in-domain test data (ResNet18)
- ⚠️ **50% accuracy drop** when models face out-of-distribution data
- 🔬 **Systematic cross-domain analysis** revealing generalization challenges
- 🛠️ **Hybrid pipeline** combining strengths of classical and modern approaches

## 📋 Table of Contents

- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Project Structure](#-project-structure)
- [Methodology](#-methodology)
- [Datasets](#-datasets)
- [Usage](#-usage)
- [Results](#-results)
- [Contributing](#-contributing)
- [Citation](#-citation)
- [Team](#-team)
- [License](#-license)

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended for training)
- 8GB+ RAM

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/vivekjyotibanerjee/GenFaD.git
cd GenFaD
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

### Additional Requirements for Deep Learning

```bash
# PyTorch (adjust CUDA version as needed)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# OpenCV for computer vision
pip install opencv-python opencv-contrib-python

# Additional ML libraries
pip install scikit-learn matplotlib pillow tqdm tensorboard
```

## ⚡ Quick Start

### 1. Classical CV Pipeline

Detect and measure cracks using traditional computer vision:

```bash
python finalPipeline.py --image crack.jpg --output results/
```

For crack area measurement with physical dimensions:

```bash
python final_with_area_in_cm.py --image crack.jpg --reference-width 10.0
```

### 2. Deep Learning Training

Train ResNet model on crack detection:

```bash
python training_script.py \
    --dataset-paths /path/to/CCIC /path/to/SAHighway \
    --model resnet18 \
    --epochs 50 \
    --batch-size 32 \
    --output-dir models/
```

### 3. Download Datasets

```bash
# Download classification datasets
python utils/download_cls_datasets.py --output data/

# Download GAPs datasets (requires authentication)
python utils/download_gaps.py --output data/gaps/
```

## 📁 Project Structure

```
GenFaD/
├── data/                          # Dataset links and configurations
│   └── cls_dataset_links.csv      # 18 curated dataset sources
├── utils/                         # Utility functions
│   ├── download_cls_datasets.py   # Dataset downloader
│   ├── download_gaps.py           # GAPs dataset handler
│   └── misc_utils.py              # Helper functions
├── training_script.py             # Deep learning training pipeline
├── finalPipeline.py               # Classical CV crack detection
├── finalPipeline_Area.py          # Area-enhanced detection
├── final_with_area_in_cm.py       # Physical measurement pipeline
├── requirements.txt               # Python dependencies
├── setup.py                       # Package installation
├── crack.jpg                      # Example crack image
├── noCrack.png                    # Example non-crack image
└── README.md                      # This file
```

## 🔬 Methodology

### Classical Computer Vision Pipeline

Our traditional CV approach uses a three-stage process:

#### Stage 1: Feature-Based Proposal Generation
1. **SIFT Feature Extraction**: Detect scale-invariant keypoints
2. **K-Means Clustering**: Cluster keypoints into N regions (default: 20)
3. **Proposal Generation**: Extract 150×150px windows around cluster centers

#### Stage 2: Crack Detection
1. **HSV Color Masking**: Threshold for dark crack regions
2. **Canny Edge Detection**: Identify crack boundaries
3. **Contour Extraction**: Isolate crack contours

#### Stage 3: Geometric Measurement
1. **Skeletonization**: Extract crack centerline
2. **Distance Transform**: Compute perpendicular widths
3. **Metric Extraction**: Calculate width, length, and area

### Deep Learning Architecture

**ResNet-Based Transfer Learning**:
- Base architectures: ResNet18 (11M params) and ResNet50 (25M params)
- ImageNet pre-trained initialization
- Two-phase training: frozen backbone → full fine-tuning
- Binary classification: Crack vs. No-Crack

**Training Strategy**:
```
Epoch 1-5:   Freeze backbone, train classifier only (lr=0.001)
Epoch 6+:    Fine-tune entire network (lr=0.0001)
```

### Cross-Domain Evaluation

We employ **leave-one-dataset-out** evaluation:
```
Train on Dataset A → Test on Dataset B
Domain Gap = Accuracy(A) - Accuracy(B)
```

This protocol reveals true generalization capability beyond in-domain performance.

## 📦 Datasets

We support 18+ crack detection datasets spanning multiple infrastructure types:

| Dataset | Type | Images | Domains |
|---------|------|--------|---------|
| **CCIC** | Classification | 40,000 | Concrete structures |
| **South African Highway** | Classification | 14,000 | Road pavements |
| **GAPs v1** | Classification | 9.0M | Multi-domain |
| **GAPs v2** | Classification | 4.3M | Multi-domain |
| **Railway Track Fault** | Detection | Various | Railway infrastructure |
| **SDNET-2018** | Classification | 56,000 | Bridge, pavement, walls |

### Dataset Access

All dataset links are maintained in `data/cls_dataset_links.csv`. Use our download utilities:

```bash
# List available datasets
python utils/download_cls_datasets.py --list

# Download specific dataset
python utils/download_cls_datasets.py --dataset CCIC --output data/
```

### Data Format

Expected directory structure:
```
dataset/
├── crack/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── no_crack/
    ├── image1.jpg
    ├── image2.jpg
    └── ...
```

## 💻 Usage

### Classical CV Detection

**Basic crack detection:**
```python
import cv2
from finalPipeline import get_proposals, detect_cracks

# Load image
image = cv2.imread('crack.jpg')

# Generate proposals
centers, kp, kp_loc, labels = get_proposals(image, n_clusters=20)

# Detect cracks in proposals
results = detect_cracks(image, centers)
```

**With area measurement:**
```python
from final_with_area_in_cm import measure_crack_area

# Measure crack with known reference
area_cm2, width_avg = measure_crack_area(
    image_path='crack.jpg',
    reference_width_cm=10.0,  # Known dimension in image
    reference_pixels=500       # Corresponding pixels
)
```

### Deep Learning Training

**Single dataset training:**
```python
python training_script.py \
    --dataset-paths /data/CCIC \
    --model resnet18 \
    --epochs 50 \
    --batch-size 32 \
    --lr 0.001 \
    --warmup-epochs 5 \
    --output-dir models/ccic_resnet18
```

**Multi-dataset training:**
```python
python training_script.py \
    --dataset-paths /data/CCIC /data/SAHighway \
    --model resnet50 \
    --epochs 100 \
    --batch-size 64 \
    --output-dir models/multidomain_resnet50
```

**Cross-domain evaluation:**
```python
python training_script.py \
    --dataset-paths /data/CCIC \
    --test-dataset-path /data/SAHighway \
    --load-checkpoint models/ccic_resnet18/best_model.pth \
    --eval-only
```

### Hyperparameter Configuration

Key training parameters in `training_script.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--model` | resnet18 | Architecture (resnet18/resnet50) |
| `--epochs` | 50 | Total training epochs |
| `--batch-size` | 32 | Training batch size |
| `--lr` | 0.001 | Initial learning rate |
| `--warmup-epochs` | 5 | Epochs with frozen backbone |
| `--weight-decay` | 1e-4 | L2 regularization |
| `--num-workers` | 4 | DataLoader workers |

## 📈 Results

### In-Domain Performance

| Model | Dataset | Val Acc | Test Acc | F1 Score |
|-------|---------|---------|----------|----------|
| ResNet18 | CCIC + SAHighway | 99.70% | 99.73% | 0.9987 |
| ResNet50 | CCIC + SAHighway | 95.80% | 97.00% | 0.9848 |

### Cross-Domain Generalization

| Train Dataset | Test Dataset | Accuracy | Domain Gap |
|---------------|--------------|----------|------------|
| CCIC | SA Highway | 49.86% | **50.11%** |
| SA Highway | CCIC | ~50% | **~50%** |

**Key Finding**: Models achieve near-perfect accuracy on in-domain data but fail catastrophically on out-of-distribution datasets, highlighting severe overfitting to dataset-specific features.

### Traditional CV Performance

| Dataset | Total Images | Crack Detection Rate | False Positive Rate | Overall Accuracy |
|---------|--------------|---------------------|---------------------|------------------|
| CCIC | 8,000 | 99.33% | 7.45% | 95.94% |
| SA Highway | 1,400 | 99.14% | **82.43%** | 58.36% |

**Key Finding**: Classical methods provide consistent performance but struggle with high-texture backgrounds, generating excessive false positives on asphalt surfaces.

### Visualization Examples

**SIFT Feature Clustering:**
![SIFT Clustering](crack_area.png)

**Crack Measurement:**
- Minimum width: 2.00 pixels
- Maximum width: 32.56 pixels  
- Average width: 16.38 pixels
- Total area: 8,856 pixels

## 🔧 Proposed Hybrid Pipeline

Based on our findings, we recommend a three-stage hybrid approach:

### Stage 1: Classical CV Proposal (Training-Free)
```
SIFT + K-Means → Region Proposals
- Fast, interpretable
- No training required
- Effective crack localization
```

### Stage 2: Deep Learning Refinement
```
ResNet/U-Net → Crack Segmentation
- Multi-domain trained
- Domain-adversarial losses
- Heavy augmentation
```

### Stage 3: Post-Processing
```
CRF + Morphology → Final Masks
- Reconnect thin segments
- Remove false positives
- Geometric validation
```

This hybrid approach leverages the complementary strengths of both methodologies.

## 🐛 Known Issues & Limitations

### Deep Learning Challenges
- ❌ **Severe domain shift**: 50% accuracy drop on unseen datasets
- ❌ **Dataset bias**: Models overfit to surface textures and lighting
- ❌ **Thin structure loss**: CNN pooling disconnects fine cracks
- ❌ **Limited context**: Local patches lack global scene understanding

### Classical CV Challenges  
- ❌ **High false positives**: 82% FP rate on textured surfaces
- ❌ **Manual tuning**: HSV thresholds require per-dataset adjustment
- ❌ **Lighting sensitivity**: Performance degrades under variable illumination
- ❌ **No semantic understanding**: Cannot distinguish crack-like patterns

### Computational Constraints
- ⚠️ Limited GPU resources prevented large-scale GAPs training
- ⚠️ Domain adaptation techniques remain unexplored
- ⚠️ Segmentation models (U-Net, DeepLab) not yet implemented

## 🛣️ Future Work

### Immediate Next Steps
- [ ] Implement domain adaptation (DANN, ADDA)
- [ ] Large-scale training on GAPs (13M images)
- [ ] Transition to segmentation (U-Net, DeepLabv3+)
- [ ] Enhanced evaluation (confusion matrices, ROC curves)
- [ ] Implement uncertainty quantification

### Advanced Directions
- [ ] Multi-modal fusion (RGB + thermal/depth)
- [ ] Temporal modeling for video-based inspection
- [ ] Active learning for efficient labeling
- [ ] Real-time edge deployment optimization
- [ ] 3D crack reconstruction from multiple views

## 🤝 Contributing

## 👥 Team

This project was developed as part of EN.601.661 Computer Vision (Fall 2024) at Johns Hopkins University.

**Team Members:**

- **Vivekjyoti Banerjee** (vbanerj3)
- **Vivek Reddy Nalla Chandrasekharreddy** (vreddyn1)  
- **Venkata Harshavardhan Bontalakoti** (vbontal1)
  
## 📚 References

1. D.G. Lowe, "Object recognition from local scale-invariant features," ICCV, 1999.
2. Xinan Zhang et al., "Deep Learning for Crack Detection: A Review," arXiv:2508.10256, 2025.
3. Zhengyun Xu et al., "Application of Deep Convolution Neural Network in Crack Identification," Applied AI, 2022.
4. Drew Linsley et al., "Recurrent neural circuits for contour detection," arXiv:2010.15314, 2020.
5. Hermann Tapamo et al., "CNNs for Crack Detection on Flexible Road Pavements," SoCPaR, 2023.
6. Ç.F. Özgenel, "Concrete Crack Images for Classification," 2019.

## 🙏 Acknowledgments

- Johns Hopkins University Computer Vision course (EN.601.661.01.FA25)
- Dataset creators and maintainers
- Open-source community for tools and libraries

---

**Last Updated**: December 2025
