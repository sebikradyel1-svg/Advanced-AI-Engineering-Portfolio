# 🖼️ Universal Image Classifier

> **VGG16 Transfer Learning with Grad-CAM Explainability — Production deployed on HuggingFace Spaces**

[![Live Demo](https://img.shields.io/badge/🤗%20Live%20Demo-Try%20Now-blue)](https://huggingface.co/spaces/KradyelSebi/animal-image-classifier)
[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.20-orange.svg)](https://tensorflow.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-Latest-red.svg)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 🎯 Project Overview

Production-ready image classification system using **VGG16 Transfer Learning** with integrated **Grad-CAM visual explainability**. The model classifies images across 12 categories with 95%+ accuracy and shows *where* it's looking to make predictions.

### Key Highlights

- **95%+ Accuracy** on multi-class classification across 12 categories (10,000+ images)
- **Grad-CAM Explainability** — visual heatmaps showing model attention regions per prediction
- **Top-k Class Comparison** — Grad-CAM for multiple predicted classes to debug misclassifications
- **Data Augmentation** — rotation, flip, zoom, brightness reducing overfitting by 30%
- **Production Deployed** on HuggingFace Spaces with real-time inference

---

## 🚀 Live Demo

**Try it live:** [https://huggingface.co/spaces/KradyelSebi/animal-image-classifier](https://huggingface.co/spaces/KradyelSebi/animal-image-classifier)

---

## 🔍 Grad-CAM Explainability

Integrated **Grad-CAM (Gradient-weighted Class Activation Mapping)** based on [Selvaraju et al., ICCV 2017](https://arxiv.org/abs/1610.02391) for visual model explanations:

| Original | Heatmap | Overlay |
|----------|---------|---------|
| Input image | Attention regions (red=high, blue=low) | Blended visualization |

### How It Works

```
1. Forward pass → get target class score
2. Backward pass → gradients w.r.t. last conv layer (block5_conv3)
3. Global average pool gradients → channel importance weights
4. Weighted sum of feature maps → heatmap
5. ReLU → only positive influence regions
6. Normalize + JET colormap → overlay on original image
```

### Why Grad-CAM?

- **Debugging**: Understand *why* the model misclassifies — is it looking at background instead of the subject?
- **Trust**: Show users which image regions drive predictions
- **Production ML**: Explainability is critical in regulated domains (healthcare, finance)

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                    Universal Image Classifier                     │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐        │
│  │   Streamlit  │───▶│  TensorFlow  │───▶│    VGG16     │        │
│  │   Frontend   │    │   Backend    │    │   Backbone   │        │
│  └──────────────┘    └──────────────┘    └──────────────┘        │
│         │                   │                   │                 │
│         ▼                   ▼                   ▼                 │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐        │
│  │ Image Upload │    │ Preprocessing│    │  Dense Head  │        │
│  │  & Preview   │    │  & Scaling   │    │  (512→256→N) │        │
│  └──────────────┘    └──────────────┘    └──────────────┘        │
│         │                                       │                 │
│         ▼                                       ▼                 │
│  ┌──────────────┐                       ┌──────────────┐         │
│  │   Grad-CAM   │◀──────────────────────│  Predictions │         │
│  │  Heatmap +   │                       │  Top-3 +     │         │
│  │  Overlay     │                       │  Confidence  │         │
│  └──────────────┘                       └──────────────┘         │
│                                                                   │
├──────────────────────────────────────────────────────────────────┤
│  Input: 224×224×3  │  Output: Top-3 predictions + Grad-CAM maps  │
└──────────────────────────────────────────────────────────────────┘
```

---

## 📊 Model Performance

| Metric | Value |
|--------|-------|
| **Base Model** | VGG16 (ImageNet pretrained) |
| **Input Size** | 224 × 224 |
| **Training Strategy** | 2-Phase (Frozen → Fine-tuned Block5) |
| **Test Accuracy** | 95%+ |
| **Categories** | 12 classes |
| **Training Images** | 10,000+ |
| **Inference Time** | ~50-100ms (CPU) |
| **Overfitting Reduction** | 30% via data augmentation |

### Transfer Learning Process

```
Phase 1: Feature Extraction (Frozen VGG16)
├── Learning Rate: 0.001
├── Epochs: 10
├── VGG16 Layers: All frozen
└── Training: Only custom dense head

Phase 2: Fine-tuning (Unfrozen Block5)
├── Learning Rate: 0.0001
├── Epochs: 15
├── VGG16 Layers: Block5 unfrozen
└── Training: End-to-end refinement
```

---

## 🚀 Quick Start

### Local Development

```bash
# Clone repository
git clone https://github.com/sebikradyel1-svg/Advanced-AI-Engineering-Portfolio.git
cd Advanced-AI-Engineering-Portfolio/image-classifier-project

# Install dependencies
pip install -r requirements.txt

# Run Streamlit app
python -m streamlit run app/streamlit_app.py
```

Visit `http://localhost:8501` — upload an image or use sample gallery.

### Grad-CAM Standalone

```bash
python app/grad_cam.py \
    --model models/image_classifier_model.h5 \
    --config models/image_classifier_config.json \
    --image test.jpg \
    --output gradcam_output.png
```

---

## 📁 Project Structure

```
image-classifier-project/
├── app/
│   ├── streamlit_app.py              # Streamlit UI with Grad-CAM integration
│   └── grad_cam.py                   # Grad-CAM module (standalone + importable)
├── models/
│   ├── image_classifier_model.h5     # Trained VGG16 model (120 MB)
│   └── image_classifier_config.json  # Class names & config
├── sample_images/                    # Example images for demo
├── universal_classifier.py           # Training script (2-phase)
├── requirements.txt                  # Python dependencies
└── README.md
```

---

## 🔧 Requirements

```
tensorflow==2.20.0
streamlit>=1.40.0
Pillow>=10.0.0
numpy>=1.24.0
matplotlib>=3.7.0
opencv-python-headless>=4.8.0
```

---

## 🎓 Skills Demonstrated

- **Transfer Learning** — VGG16 with custom classification head and 2-phase training
- **Grad-CAM / XAI** — Model explainability from scratch (Selvaraju et al., ICCV 2017)
- **Computer Vision** — Data augmentation, preprocessing, multi-class classification
- **Production Deployment** — HuggingFace Spaces with Streamlit
- **TensorFlow/Keras** — Model building, training, and inference optimization

---

## 👤 Author

**Paul Sebastian Kradyel** — AI Engineer & ML Specialist

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue)](https://www.linkedin.com/in/paul-sebastian-kradyel)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-black)](https://github.com/sebikradyel1-svg)
[![HuggingFace](https://img.shields.io/badge/🤗%20HuggingFace-Profile-yellow)](https://huggingface.co/KradyelSebi)

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.


---

<p align="center">
  Built with ❤️ using TensorFlow & Streamlit
</p>
