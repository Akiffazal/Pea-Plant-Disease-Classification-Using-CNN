# Pea‑Plant Disease Classification Using CNN 🌱

## 📄 Project Description

This project uses a Convolutional Neural Network (CNN) to classify pea plant leaves into four categories. Given an image of a pea leaf, the model predicts its health status — helping automate disease detection for better crop management. The model achieves **95% accuracy** on the validation dataset.

## 🧠 Motivation

* Manual inspection of plant diseases is slow and error-prone, especially on large farms.
* Automating disease detection with deep learning helps farmers identify problems early, reduce crop loss, and improve yield.
* This project explores practical applications of CNNs for agricultural image classification.

## 🗂 Dataset

* **Custom dataset collected manually**: All leaf images were collected and labeled personally to ensure data quality.
* Dataset classes:

  1. **DOWNY_MILDEW_LEAF** – Leaf affected by downy mildew
  2. **FRESH_LEAF** – Healthy, fresh leaf
  3. **LEAFMINER_LEAF** – Leaf affected by leafminer infestation
  4. **POWDER_MILDEW_LEAF** – Leaf affected by powdery mildew

> Dataset structure:

```
pea_plant_dataset/
├── DOWNY_MILDEW_LEAF/
├── FRESH_LEAF/
├── LEAFMINER_LEAF/
└── POWDER_MILDEW_LEAF/
```

## ✅ Features

* CNN trained on the custom pea leaf dataset.
* Data preprocessing (resizing, normalization) and augmentation applied.
* Achieved **95% accuracy** on the validation set.
* Ready-to-use trained model for inference on new images.

## 📁 Repository Structure

```
/
├── pea_plant_dataset/        ← Custom dataset collected manually  
├── training/                 ← Training scripts & notebooks  
├── saved_models/             ← Trained model weights / checkpoints  
├── api/                      ← (Optional) API for prediction  
├── frontend/                 ← (Optional) front-end UI  
├── .gitignore  
└── README.md                 ← This file  
```

## 🛠️ Installation & Setup

1. Clone the repository:

```bash
git clone https://github.com/Akiffazal/Pea-Plant-Disease-Classification-Using-CNN.git
cd Pea-Plant-Disease-Classification-Using-CNN
```

2. Create a virtual environment and install dependencies:

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. Ensure the dataset is in `pea_plant_dataset/` as shown above.
4. Run training scripts/notebooks in `training/` to train the CNN.
5. Trained model weights will be saved in `saved_models/`.


## 📊 Model & Training Details

* CNN architecture with convolutional, pooling, and dense layers.
* Data preprocessing and optional augmentation for better generalization.
* Achieved **95% validation accuracy**.
* Dataset split into training and validation sets for evaluation.

## 🤝 Contributions & Future Work

* Add more leaf diseases or extend dataset further.
* Improve CNN architecture or apply transfer learning.
* Add visualizations like confusion matrix, accuracy/loss plots.
* Build an API or UI for deployment.

## 📜 License

Open-source. Use, modify, or extend freely.
