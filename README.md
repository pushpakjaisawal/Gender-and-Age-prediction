# 👤 Age & Gender Prediction

A deep learning application that predicts **age group** and **gender** from face images using a **MobileNetV2** multi-task model trained on the **UTKFace** dataset. Deployed as a web app with **live webcam** support via Streamlit.

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15+-orange.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

---

## 📸 Demo

### Upload Image Mode
| Input | Prediction |
|-------|------------|
| ![sample face](assets/sample1.jpg) | Gender: **Female** (94%) · Age: **23-30** (87%) |
| ![sample face](assets/sample2.jpg) | Gender: **Male** (91%) · Age: **41-50** (82%) |

### Live Webcam Mode
- Real-time face detection and prediction through your browser
- Bounding boxes with gender and age group labels
- Confidence scores displayed on each frame

---

## 🏗 Architecture

```
┌──────────────────────────────────────────────┐
│              Input: 224 × 224 × 3            │
└──────────────────┬───────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────┐
│         MobileNetV2 Backbone                 │
│       (Pre-trained on ImageNet)              │
│            ~2.3M parameters                 │
│                                              │
│   Phase 1: Frozen  → Train heads only        │
│   Phase 2: Unfreeze last 40 layers            │
└──────────┬──────────────────┬────────────────┘
           │                  │
     ┌─────┴─────┐     ┌─────┴─────┐
     │           │     │           │
     ▼           ▼     ▼           ▼
┌─────────┐ ┌─────────┐ ┌─────────┐ ┌──────────┐
│ Dense   │ │ Dense   │ │ Dense   │ │ Dense    │
│ 128     │ │ 256     │ │ 64      │ │ 128      │
│ BN+Drop │ │ BN+Drop │ │ BN+Drop │ │ BN+Drop  │
└────┬────┘ └────┬────┘ └────┬────┘ └────┬─────┘
     │           │          │           │
     ▼           │          ▼           │
┌─────────┐      │    ┌──────────┐      │
│ Dense 1 │      │    │ Dense 9  │      │
│(sigmoid)│      │    │(softmax) │      │
└────┬────┘      │    └────┬─────┘      │
     │           │         │            │
     ▼           ▼         ▼            ▼
  Gender      Age Group (9 classes)
 (Male/Female)
```

---

## 📂 Project Structure

```
age-gender-app/
│
├── app.py                           # Streamlit web application
├── requirements.txt                 # Python dependencies
├── .gitignore                       # Git ignore rules
├── README.md                        # This file
│
├── models/
│   └── best_model_finetuned.keras   # Trained model (~15 MB)
│
├── training/
│   └── age-gender-improved.ipynb    # Training notebook (Kaggle)
│
└── assets/
    └── (demo images, screenshots)
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10 or higher
- A working webcam (for live prediction mode)
- Downloaded model file from [Kaggle](#model-download)

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/YOUR-USERNAME/age-gender-app.git
cd age-gender-app

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate      # Linux/Mac
# venv\Scripts\activate       # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download model (see Model Download section below)
mkdir -p models
# Place best_model_finetuned.keras inside models/
```

### Run Locally

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 🧠 Model Details

### Training Approach

| Feature | Details |
|---------|---------|
| **Backbone** | MobileNetV2 (ImageNet pre-trained) |
| **Training Strategy** | 2-Phase Transfer Learning |
| **Phase 1** | Train classification heads (backbone frozen) |
| **Phase 2** | Fine-tune last 40 backbone layers (LR ÷ 10) |
| **Loss** | Binary Cross-Entropy (gender) + Categorical Cross-Entropy (age) |
| **Loss Weights** | Gender: 1.0, Age: 1.5 |
| **Regularization** | BatchNormalization + Dropout (0.2–0.3) |
| **Pooling** | GlobalAveragePooling2D |
| **Augmentation** | Rotation, flips, brightness, zoom, shifts, shear |

### Age Groups (9 Classes)

| Class | Range | Description |
|-------|-------|-------------|
| 0 | 0–3 | Infants / Toddlers |
| 1 | 4–8 | Young Children |
| 2 | 9–15 | Older Children / Early Teens |
| 3 | 16–22 | Late Teens / Young Adults |
| 4 | 23–30 | Adults |
| 5 | 31–40 | Adults |
| 6 | 41–50 | Middle-Aged |
| 7 | 51–60 | Mature Adults |
| 8 | 61+ | Seniors |

### Dataset

- **UTKFace**: ~23,000 labeled face images
- **Labels**: Age (0–116), Gender (Male=0, Female=1)
- **Source**: [UTKFace on Kaggle](https://www.kaggle.com/datasets/jangedoo/utkface-new)

---

## 📥 Model Download

The trained model is too large for GitHub. Download it from Kaggle:

### Option 1: Direct Download
1. Open the [training notebook on Kaggle](https://www.kaggle.com/)
2. Go to **Output** tab → find `best_model_finetuned.keras`
3. Click **Download**
4. Place it in `models/` folder

### Option 2: Kaggle API
```bash
pip install kaggle
kaggle kernels output YOUR-USERNAME/age-gender-training -p ./models/
```

---

## 🌐 Deploy to Streamlit Cloud

### Step 1: Push to GitHub

```bash
# Initialize
git init
git add .
git commit -m "Initial commit"

# Setup Git LFS for model file
git lfs install
git lfs track "*.keras"
git add .gitattributes
git add models/best_model_finetuned.keras
git commit -m "Add trained model"

# Push to GitHub
git remote add origin https://github.com/YOUR-USERNAME/age-gender-app.git
git branch -M main
git push -u origin main
```

### Step 2: Deploy

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Click **"New app"**
3. Select your repository, branch `main`, and file `app.py`
4. Click **"Deploy"**
5. Your app will be live at: `https://age-gender-app.streamlit.app`

---

## 🛠 Tech Stack

| Component | Technology |
|-----------|-----------|
| Deep Learning | TensorFlow / Keras |
| Pre-trained Model | MobileNetV2 (ImageNet) |
| Web Framework | Streamlit |
| Webcam Support | Streamlit-WebRTC |
| Face Detection | OpenCV (Haar Cascade) |
| Image Processing | OpenCV, Pillow, NumPy |
| Deployment | Streamlit Cloud / GitHub |

---

## 📊 Performance

| Metric | Value |
|--------|-------|
| Gender Accuracy | ~93–96% |
| Age Group Accuracy | ~75–85% |
| Input Size | 224 × 224 × 3 |
| Inference Time | ~50ms per face (GPU) |
| Model Size | ~15 MB |

> ⚠️ Performance may vary depending on image quality, lighting, and face angle.

---

## ⚠️ Limitations

- Works best with **frontal faces** (Haar Cascade detector)
- **Side profiles**, **occluded faces**, and **extreme angles** may reduce accuracy
- Performance depends on **lighting conditions** and **image resolution**
- Age prediction is grouped into ranges, not exact years
- The model was trained primarily on faces from the UTKFace dataset (diverse but not universally representative)

---

## 🔧 Troubleshooting

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` |
| Model not found | Download `best_model_finetuned.keras` to `models/` folder |
| Webcam not working | Use Chrome/Edge; allow camera permissions |
| No faces detected | Ensure good lighting and frontal face |
| OOM error during training | Use the memory-efficient generator in the notebook |
| `np.digitize` IndexError | Ensure `- 1` is added: `np.digitize(ages, age_bins) - 1` |

---

## 📜 License

This project is licensed under the **MIT License**. See `LICENSE` for details.

---

## 🙏 Acknowledgments

- **UTKFace Dataset** — [Zhang et al.](https://www.kaggle.com/datasets/jangedoo/utkface-new)
- **MobileNetV2** — [Howard et al., 2019](https://arxiv.org/abs/1801.04381)
- **TensorFlow** — [Google](https://www.tensorflow.org/)
- **Streamlit** — [Streamlit Inc.](https://streamlit.io/)
