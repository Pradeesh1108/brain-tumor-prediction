# 🧠 Brain Tumor Segmentation using ResNet50 + Flask

This project is a deep learning-based brain tumor segmentation system trained using ResNet50. It supports JPEG images and includes a Flask-based web application for real-time tumor segmentation.

---

## ✨ Features

- Brain tumor segmentation using ResNet50
- JPEG image support
- Deep learning with TensorFlow and Keras
- Flask web application for easy interaction
- Compatible with M1 Mac using MPS backend

---

## 📁 Folder Structure

```
brain-tumor-segmentation/
├── static/                 # Static files (CSS, JS, outputs)
│   └── style.css
│   └── script.js        
├── templates/              # HTML templates
│   ├── index.html
├── data/                   # Dataset folder
│   ├── images/             # Input MRI images (.jpg)
├── model/                  # Saved trained models
│   └── model
├── app.py                  # Flask application file
├── training.ipynb          # Jupyter notebook for model training
├── requirements.txt        # Python dependencies
└── README.md               # This documentation file
```

---

## 🛠️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/Pradeesh1108/brain-tumor-prediction.git
cd brain-tumor-prediction
```

### 2. Create and activate a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install the dependencies

```bash
pip install -r requirements.txt
```

---

## 📦 requirements.txt

```text
tensorflow
flask
opencv-python
numpy
pillow
matplotlib
scikit-learn
jupyter
```

For M1 Mac users:

```text
tensorflow-macos
tensorflow-metal
```

---

## 🧪 Model Training

Train the ResNet50 model using the provided Jupyter notebook.

### 1. Launch the notebook

```bash
jupyter notebook training.ipynb
```

### 2. Dataset structure

Make sure your image and mask folders look like this:

```
data/
├── images/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
```

### 3. Run the notebook

- Load and preprocess the data
- Train the model
- Save the trained model to `model`

---

## 🚀 Running the Web App

Once the model is trained and saved, launch the Flask app:

```bash
python3 app.py
```

The app will run at:

```
http://127.0.0.1:5000
```

### 🖼️ Using the App

- Upload a brain MRI image
- View the predicted tumor result
---

## 🧠 Credits

- Model Architecture: ResNet50
- Frameworks: TensorFlow, Flask
- Frontend: HTML + JS

---
