# 🇫🇷 FR 🇫🇷

# 🎯 Reconnaissance en temps réel d'éléments via Webcam
Modèle CNN entraîné from scratch

---

## 🧠 Projet
Ce projet vise à créer une IA capable de reconnaître plusieurs éléments via webcam, en temps réel.  
L'idée est de, contrairement aux approches classiques basées sur du pré-entraînement (OpenCV, MobileNet, YOLO), **le modèle est construit et entraîné from scratch**. 
(Des premières versions seront possibles via du pré-entraînement avant personnalisation complète)

---

## 🚀 Objectifs
- Développement d’un modèle CNN personnalisé
- Détection simultanée
- Optimisation pour exécution en temps réel sur GPU
- Préparation pour intégration dans un site web

---

## 📂 Structure du projet
ai-webcam-vision/

│

├── data/ # Datasets d'entraînement

├── models/ # Architectures et versions

├── train.py # Script d'entraînement

├── inference.py # Script temps réel (webcam)

├── utils/ # Fonctions auxiliaires

└── README.md # Ce fichier

---

## 🔬 Architecture IA

- CNN personnalisé
- Plusieurs blocs convolutionnels + BatchNorm + ReLU
- Extraction de landmarks
- Optimisation par Adam
- Loss adaptée à landmarks/coordonnées

---

## 💾 Dataset
Pour le moment, le dataset provient de :
- Datasets personnalisés (annoté manuellement)
- Datasets Kaggle

Dataset en cours d’augmentation :
- Rotation
- Zoom
- Luminosité
- Occlusion

---

## ⚙️ Entraînement

### ⚡ GPU recommandé

### 🖥️ Inference (Webcam)
inference.py

Affiche en live la position détectée des objets

#### 📊 Performance (en cours)

| Élément | Précision   |
| ------- | ----------- |
| Yeux    | 🔄 Training |
| Nez     | 🔄 Training |
| Bouche  | 🔄 Training |

---

# 👤 Auteur

Michael Lapeyrere

  Ingénieur IA & Big Data
  Spécialiste IA sur mesure
  Expert Power BI

# 📬 Contact

💼 LinkedIn : www.linkedin.com/in/michaël-lapeyrère-465a93203

✉️ Email pro : michaellapeyrere.ml@gmail.com


---


# 🇬🇧 EN 🇬🇧

# 🎯 Real-Time Recognition AI through webcam
CNN model trained from scratch

---

## 🧠 Project
This project aims to build an AI system capable of recognizing several elements through a webcam, in real time.
Unlike traditional approaches based on pretrained models (OpenCV, MobileNet, YOLO), the model here is designed and trained from scratch.
(Initial versions may rely on pretrained models before moving to a fully customized architecture.)

---

## 🚀 Goals
Development of a custom CNN model
Simultaneous detection of multiple features
Optimization for real-time GPU execution
Web-ready integration

---

## 📂 Project structure
ai-webcam-vision/

│

├── data/         # Training datasets

├── models/       # Architectures and model versions

├── train.py      # Training script

├── inference.py  # Real-time webcam inference

├── utils/        # Utility functions

└── README.md     # This file

---

## 🔬 AI Architecture

  - Custom CNN
  - Multiple convolutional blocks + BatchNorm + ReLU
  - Landmark extraction
  - Adam optimization
  - Custom loss for landmark/coordinate outputs

---

## 💾 Dataset
Currently, the dataset comes from:
  - Custom manually annotated dataset
  - Kaggle datasets

Dataset augmentation includes:
  - Rotation
  - Zoom
  - Brightness variation
  - Occlusion simulation

---

## ⚙️ Training

### ⚡ GPU recommended

### 🖥️ Inference (Webcam)
inference.py

Displays detected features in real time.

### 📊 Performance (in progress)

| Element | Precision   |
| ------- | ----------- |
| Eyes    | 🔄 Training |
| Nose    | 🔄 Training |
| Mouth   | 🔄 Training |

---

# 👤 Author

Michael Lapeyrere

  AI & Big Data Engineer
  Custom AI development specialist
  Power BI Expert

# 📬 Contact

💼 LinkedIn : www.linkedin.com/in/michaël-lapeyrère-465a93203

✉️ Professional email : michaellapeyrere.ml@gmail.com
