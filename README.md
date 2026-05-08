# Multimodal Emotion Recognition System

## Overview

This project is a Deep Learning based Multimodal Emotion Recognition System developed using:

* Speech modality
* Text modality
* Fusion of speech and text

The system recognizes human emotions from speech audio and corresponding textual content using Transformer-based architectures.

The project uses:

* Wav2Vec2 for speech feature extraction
* BERT for text understanding
* Multimodal fusion architecture for final emotion classification

---

# Features

* Speech Emotion Recognition
* Text Emotion Recognition
* Multimodal Fusion Model
* Confusion Matrix Visualization
* Classification Report
* GPU Training Support
* Google Colab Compatible
* PyTorch + Transformers based implementation

---

# Dataset

Dataset Used:

TESS (Toronto Emotional Speech Set)

The dataset contains emotional speech recordings with multiple emotion categories.

Emotions:

* Angry
* Disgust
* Fear
* Happy
* Neutral
* Pleasant Surprise
* Sad

Dataset Structure:

```text
TESS Toronto emotional speech set data/
    OAF_angry/
    OAF_happy/
    YAF_sad/
    ...
```

---

# Project Structure

```text
Multimodal_Emotion_Recognition/
│
├── data/
│   ├── raw/
│   └── csv/
│
├── models/
│   ├── speech_pipeline/
│   ├── text_pipeline/
│   └── fusion_pipeline/
│
├── utils/
│
├── Results/
│   ├── models/
│   └── plots/
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

# Technologies Used

* Python
* PyTorch
* Hugging Face Transformers
* Wav2Vec2
* BERT
* Librosa
* Scikit-learn
* Matplotlib
* Seaborn
* Google Colab

---

# Model Architectures

## 1. Speech Pipeline

Architecture:

```text
Audio → Wav2Vec2 → BiLSTM → Dense Layers → Emotion
```

Used for extracting emotional information from speech signals.

---

## 2. Text Pipeline

Architecture:

```text
Text → BERT → Dense Layers → Emotion
```

Used for extracting semantic information from spoken words.

---

## 3. Fusion Pipeline

Architecture:

```text
Speech Embedding + Text Embedding
                ↓
           Fusion Layer
                ↓
         Emotion Prediction
```

Combines both modalities for improved emotion recognition.

---

# Installation

## Clone Repository

```bash
git clone https://github.com/nithya996/Multimodal_Emotion_Recognition
cd Multimodal_Emotion_Recognition
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Requirements

```text
torch
torchaudio
transformers
librosa
soundfile
pandas
numpy
scikit-learn
matplotlib
seaborn
tqdm
sentencepiece
accelerate
```

---

# Running the Project

## 1. Create Dataset CSV

```bash
python utils/create_csv.py
```

---

# Speech Pipeline

## Train

```bash
python -m models.speech_pipeline.train
```

## Test

```bash
python -m models.speech_pipeline.test
```

---

# Text Pipeline

## Train

```bash
python -m models.text_pipeline.train
```

## Test

```bash
python -m models.text_pipeline.test
```

---

# Fusion Pipeline

## Train

```bash
python -m models.fusion_pipeline.train
```

## Test

```bash
python -m models.fusion_pipeline.test
```

---

# Results

## Text Model

* Lower accuracy due to isolated lexical tokens in TESS dataset.
* Text alone contains limited emotional information.

Approximate Accuracy:

```text
~14%
```

---

## Speech Model

Speech modality captures:

* Tone
* Pitch
* Intensity
* Emotional prosody

Approximate Accuracy:

```text
~99%
```

---

## Fusion Model

Fusion combines:

* Acoustic features
* Semantic features

Approximate Accuracy:

```text
~98-100%
```

---

# Observations

* Speech modality contributes most to emotion recognition.
* Text modality alone performs poorly on isolated words.
* Multimodal fusion improves robustness and representation learning.
* Transformer-based models perform effectively for emotion classification.

---

# Future Improvements

* Real-time microphone emotion prediction
* Streamlit/Flask web application
* Attention-based fusion
* Data augmentation
* Cross-dataset evaluation
* Real-world conversational emotion recognition
* Model optimization for deployment

---

# Output Examples

Generated Outputs:

* Confusion Matrix
* Classification Report
* Accuracy Metrics
* Saved Trained Models (.pth)

Saved inside:

```text
Results/
```

---

# Google Colab Support

This project is fully compatible with Google Colab GPU training.

Recommended GPU:

* Tesla T4

---

# Drive Link

https://drive.google.com/drive/folders/10UrcOIp8l2AyX8YoquXTE11Mz13hAHqg

---

# Author

Nithya

---

# License

This project is developed for academic and internship purposes.
