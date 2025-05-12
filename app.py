from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
from io import BytesIO
import os
import requests
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

# Load the TensorFlow model
model = tf.saved_model.load("brain_tumor_model_savedmodel")
infer = model.signatures["serving_default"]

def preprocess_image(image_data, target_size=(256, 256)):
    img = Image.open(BytesIO(image_data)).convert('RGB')
    img = img.resize(target_size)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0).astype(np.float32)
    return img_array

def predict_image(image_data):
    img_array = preprocess_image(image_data)
    predictions = infer(tf.constant(img_array))
    output_key = list(predictions.keys())[0]
    prob = predictions[output_key].numpy()[0][0]
    label = "yes" if prob >= 0.5 else "no"
    return float(prob), label

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    image = request.files['image'].read()
    prob, label = predict_image(image)

    return jsonify({
        'probability_of_tumor_yes': round(prob * 100, 2),  # Convert to percentage
        'predicted_class': label
    })

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get("message")
    api_key = os.getenv("GEMINI_API_KEY")

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

    response = requests.post(
        "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent",
        headers=headers,
        params={"key": api_key},
        json=data
    )

    try:
        content = response.json()["candidates"][0]["content"]["parts"][0]["text"]
    except Exception as e:
        content = "Sorry, I couldn't process that. Please try again."

    return jsonify({"response": content})

if __name__ == '__main__':
    app.run(debug=True)