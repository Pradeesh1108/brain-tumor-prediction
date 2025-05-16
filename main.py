from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
from io import BytesIO
import os
import requests
import shutil
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

model_url = 'https://huggingface.co/spaces/pradeesh11/Brain-tumor-prediction/resolve/main/brain_tumor_model_savedmodel.zip'
model_zip = '/tmp/model.zip'
model_path = '/tmp/brain_tumor_model_savedmodel'

try:
    if not os.path.exists(model_path):
        os.makedirs(os.path.dirname(model_zip), exist_ok=True)
        print("📦 Downloading model from Hugging Face...")

        response = requests.get(model_url)
        if response.status_code != 200:
            raise Exception(f"Failed to download model. Status code: {response.status_code}")

        with open(model_zip, 'wb') as f:
            f.write(response.content)

        shutil.unpack_archive(model_zip, '/tmp')
        print("✅ Model downloaded and extracted.")
except Exception as e:
    print(f"❌ Error downloading or unzipping model: {e}")
    raise

# ----------------------
# Load the TensorFlow model
# ----------------------

try:
    model = tf.saved_model.load(model_path)
    infer = model.signatures["serving_default"]
    print("✅ Model loaded successfully.")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    raise


# ----------------------
# Image preprocessing and prediction
# ----------------------

def preprocess_image(image_data, target_size=(256, 256)):
    img = Image.open(BytesIO(image_data)).convert('RGB')
    img = img.resize(target_size)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0).astype(np.float32)
    return img_array


def predict_image(image_data):
    try:
        img_array = preprocess_image(image_data)
        predictions = infer(tf.constant(img_array))
        output_key = list(predictions.keys())[0]
        prob = predictions[output_key].numpy()[0][0]
        label = "yes" if prob >= 0.5 else "no"
        return float(prob), label
    except Exception as e:
        return None, f"Prediction error: {str(e)}"


# ----------------------
# Routes
# ----------------------

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    image = request.files['image'].read()
    prob, label = predict_image(image)

    if prob is None:
        return jsonify({'error': label}), 500

    return jsonify({
        'probability_of_tumor_yes': round(prob * 100, 2),  # Convert to percentage
        'predicted_class': label
    })


@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get("message")
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        return jsonify({"error": "API key not configured"}), 500

    headers = {
        "Content-Type": "application/json"
    }

    data = {
        "contents": [
            {
                "parts": [{"text": user_input}]
            }
        ]
    }

    try:
        response = requests.post(
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent",
            headers=headers,
            params={"key": api_key},
            json=data
        )
        response.raise_for_status()
        content = response.json()["candidates"][0]["content"]["parts"][0]["text"]
    except Exception as e:
        content = f"Sorry, I couldn't process that. Error: {str(e)}"

    return jsonify({"response": content})


# ----------------------
# Start server
# ----------------------

if __name__ == '__main__':
    app.run(debug=True)
