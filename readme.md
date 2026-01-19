# ✋ Sign Language to Text Conversion

A **Computer Vision and Deep Learning–based system** that converts **sign language hand gestures into readable text** in real time using a webcam.  
This project aims to bridge the communication gap between **deaf / hard-of-hearing individuals** and non-sign language users.

---

## 📌 Project Overview

Sign language is a visual language that uses **hand gestures, orientation, and movement** to convey meaning.  
This project uses **Computer Vision (CV)** and **Convolutional Neural Networks (CNNs)** to recognize **American Sign Language (ASL) alphabets (A–Z)** and convert them into text.

The system captures live video, detects hands, preprocesses the gesture image, and predicts the corresponding alphabet using a trained deep learning model.

---

## 🎯 Objectives

- Detect and track hand gestures in real time  
- Recognize ASL alphabet gestures accurately  
- Convert detected gestures into text output  
- Build an accessible real-time sign language recognition system  

---

## 🛠️ Tech Stack

| Category | Technology |
|--------|------------|
| Programming Language | Python |
| Computer Vision | OpenCV |
| Hand Tracking | MediaPipe, CVZone |
| Deep Learning Framework | TensorFlow, Keras |
| Numerical Computing | NumPy |
| Model Type | Convolutional Neural Network (CNN) |
| Input Device | Webcam |

---

## 🧠 How It Works (Workflow)

1. **Video Capture** – Live frames captured using a webcam  
2. **Hand Detection & Tracking** – MediaPipe + CVZone detect hand landmarks  
3. **Image Preprocessing** – Cropping, resizing, background normalization  
4. **Gesture Recognition** – CNN model predicts the sign  
5. **Text Output** – Predicted alphabet displayed on screen  

---

## ✨ Features

- 🔴 Real-time webcam-based detection  
- ✋ Accurate hand landmark tracking  
- 🧠 CNN-based gesture classification  
- 🔤 Supports ASL alphabets (A–Z)  
- 📸 Dataset collection using live camera  
- ⚡ Fast and responsive predictions  

---

## 📂 Project Structure

```bash
Sign-Language-To-Text/
├── Data/                  # Collected gesture images
├── Model/
│   └── keras_model.h5     # Trained CNN model
├── datacollection.py      # Dataset collection script
├── test.py                # Real-time prediction script
├── requirements.txt       # Project dependencies
└── README.md
```

---
## ⚙️ Setup & Installation

### 1️⃣ Clone the repository
```bash
git clone https://github.com/bhoomikagoel24/sign-language-to-text.git
cd sign-language-to-text
```

### 2️⃣ Create virtual environment (Python 3.10 recommended)
```bash
py -3.10 -m venv venv
venv\Scripts\activate
```

### 3️⃣ Install dependencies
```bash
pip install -r requirements.txt
```

### ▶️ Running the Project
- **Dataset Collection**
```bash
python datacollection.py
```
Press S to save gesture images.

- **Gesture Recognition**
```bash
python test.py
```
Press ESC / Q to exit.

## 📊 Dataset

- Custom dataset collected using webcam
- Images of hand gestures representing ASL alphabets
- Preprocessed and normalized for CNN training

### Sample Dataset Preview

<p align="center">
  <img src="assets/dataPreview_1.jpg" width="280"/>
  <img src="assets/dataPreview_2.jpg" width="280"/>
  <img src="assets/dataPreview_3.jpeg" width="280"/>
</p>

## ⚠️ Limitations

- Performance depends on lighting conditions
- Supports static alphabet gestures only
- Continuous sentence recognition not implemented

## 🚀 Future Enhancements

- Word and sentence-level recognition
- Continuous gesture detection
- Multilingual sign language support (ISL, BSL, etc.)
- Text-to-Speech integration

## 👩‍💻 Author
~ Bhoomika Goel