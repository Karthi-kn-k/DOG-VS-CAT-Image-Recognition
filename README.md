
## 🐶🐱 Dogs vs Cats Image Classifier

A simple Convolutional Neural Network (CNN)-based image classifier built with **TensorFlow** and **Keras** to distinguish between dog and cat images. This project uses the **PetImages dataset** and supports data cleaning, training, validation, and image prediction.

---

### 📁 Project Structure

```
DOGS-VS-CATS-CLASSIFIER/
├── PetImages/                # Raw image dataset (Cat and Dog folders)
│   ├── Cat/
│   └── Dog/
├── image-recognization.py   # Main Python script (model + training + prediction)
├── requirements.txt         # Python dependencies
└── README.md                # Project documentation
```

---

### 🧠 Features

* Cleans corrupted images automatically
* Splits dataset into training and validation sets
* Uses a simple CNN model to classify cats vs dogs
* Supports custom image prediction via upload

---

### 🧪 Model Overview

* **Model Type:** CNN (Convolutional Neural Network)
* **Framework:** TensorFlow + Keras
* **Layers:** Conv2D → MaxPooling → Dense → Dropout
* **Loss Function:** Binary Crossentropy
* **Optimizer:** Adam
* **Input Size:** 128x128x3

---
### 🔥 Future Improvements

* Save and load trained models
* Visualize training/validation accuracy
* Add GUI with Streamlit or Tkinter
* Support batch predictions

---

### 🙌 Credits

* Dataset: [Microsoft PetImages](https://www.kaggle.com/datasets/bhavikjikadara/dog-and-cat-classification-dataset)
* Frameworks: TensorFlow, Keras, Python

---

