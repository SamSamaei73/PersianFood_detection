# 🍛 Persian Food Detection

A deep-learning image classifier that recognizes traditional Persian dishes from photos, built with **TensorFlow / Keras** and a custom Convolutional Neural Network (CNN).

The model currently distinguishes between three classic dishes:

| Class | Dish | Description |
|-------|------|-------------|
| `Gheymeh` | Khoresh Gheymeh (قیمه) | Split-pea and meat stew, often topped with fried potatoes |
| `Ghormesabzi` | Ghormeh Sabzi (قورمه‌سبزی) | Herb stew with kidney beans and dried lime |
| `Kabab` | Kabab (کباب) | Grilled skewered meat, usually served with rice |

---

## ✨ Features

- **Custom CNN**: five convolutional blocks trained from scratch on 300×300 RGB images
- **Data augmentation**: rotation, shifting, shearing, zoom and horizontal flips to reduce overfitting on a small dataset
- **Training visualization**: accuracy and loss curves for the training and validation sets
- **Inference script**: predicts the dish in a test image and shows the image with its predicted label and confidence score
- **Extra experiments**: an MNIST digit classifier and early-stage Persian-alphabet and animal-image scripts

---

## 📁 Project Structure

```
.
├── Food.py                     # Train the Persian food CNN → saves Food.h5
├── Show.py                     # Predict a random test image with the trained model
├── Test.py                     # MNIST handwritten-digit classifier (experiment)
├── Testmodel.py                # Run inference with the MNIST model
├── number_model.h5             # Pre-trained MNIST model
├── animal.py                   # Animal image loading / visualization (experiment)
├── Alphabet/
│   ├── PersianAlphabet.py      # Persian alphabet classifier (work in progress)
│   └── archive/labels.csv      # Labels for the alphabet dataset
└── Food/                       # Dataset (not included in the repo, see below)
    ├── Train/
    │   ├── Gheymeh/
    │   ├── Ghormesabzi/
    │   └── Kabab/
    ├── Validation/
    │   ├── Gheymeh/
    │   ├── Ghormesabzi/
    │   └── Kabab/
    └── Test/
```

---

## 🧠 Model Architecture

```
Input (300 × 300 × 3)
│
├── Conv2D(16,  3×3, ReLU) → MaxPool(2×2)
├── Conv2D(32,  3×3, ReLU) → MaxPool(2×2)
├── Conv2D(64,  3×3, ReLU) → MaxPool(2×2)
├── Conv2D(128, 3×3, ReLU) → MaxPool(2×2)
├── Conv2D(256, 3×3, ReLU) → MaxPool(2×2)
│
├── Flatten
├── Dense(512, ReLU)
└── Dense(3, Softmax)
```

| Setting | Value |
|---------|-------|
| Loss | Categorical cross-entropy |
| Optimizer | Adam |
| Batch size | 10 |
| Epochs | 10 |
| Input size | 300 × 300 px |

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/SamSamaei73/PersianFood_detection.git
cd PersianFood_detection
```

### 2. Create a virtual environment and install dependencies

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install tensorflow matplotlib numpy opencv-python
```

### 3. Prepare the dataset

The image dataset is **not included** because of its size. Put your images in the folder layout below, with one sub-folder per class:

```
Food/
├── Train/        # ~20 images per class
│   ├── Gheymeh/
│   ├── Ghormesabzi/
│   └── Kabab/
├── Validation/   # ~10 images per class
│   ├── Gheymeh/
│   ├── Ghormesabzi/
│   └── Kabab/
└── Test/         # unlabeled .jpeg images for inference
```

### 4. Train the model

```bash
python Food.py
```

This shows sample augmented images, trains the CNN, saves the model as `Food.h5`, and plots the accuracy and loss curves.

### 5. Run a prediction

```bash
python Show.py
```

This picks a random `.jpeg` from `Food/Test/`, classifies it, and shows the image with the predicted dish and confidence score.

---

## 📦 Trained Model

The trained food model (`Food.h5`, ~132 MB) is over GitHub's file-size limit, so it isn't in this repository. Run `python Food.py` to generate it locally.

---

## 🧪 Other Experiments

| Script | Description |
|--------|-------------|
| `Test.py` | Trains a fully connected network on the MNIST handwritten-digit dataset and saves `number_model.h5` |
| `Testmodel.py` | Loads `number_model.h5` and predicts a random MNIST test digit |
| `Alphabet/PersianAlphabet.py` | Starting point for a Persian alphabet classifier (data loader still to be implemented) |
| `animal.py` | Loads and displays grayscale images from an animal dataset |

---

## 🛣️ Roadmap

- [ ] Add more Persian dishes (e.g. Tahdig, Fesenjan, Zereshk Polo)
- [ ] Use transfer learning (MobileNetV2 / EfficientNet) for higher accuracy
- [ ] Add a `requirements.txt` and command-line arguments for image paths
- [ ] Build a simple web demo (Streamlit / Gradio)
- [ ] Finish the Persian alphabet classifier

---

## 🛠️ Tech Stack

- Python 3
- TensorFlow / Keras
- NumPy
- Matplotlib
- OpenCV

---

## 👤 Author

**Sam Samaei**, [@SamSamaei73](https://github.com/SamSamaei73)

If you find this project useful, please consider giving it a ⭐!
