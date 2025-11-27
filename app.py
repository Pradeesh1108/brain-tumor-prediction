from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
from io import BytesIO
import os
import requests
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

if not os.getenv("GOOGLE_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_API_KEY")

app = Flask(__name__)

# Load Vector Store
DB_FAISS_PATH = "vectorstore/db_faiss"
vector_store = None
try:
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    vector_store = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
    print("Vector store loaded successfully.")
except Exception as e:
    print(f"Warning: Could not load vector store: {e}")

# Load the TensorFlow model
try:
    model = tf.saved_model.load("brain_tumor_model_savedmodel")
    infer = model.signatures["serving_default"]
    model_loaded = True
    print("Model loaded successfully.")
except Exception as e:
    print(f"Warning: Could not load model: {e}")
    print("Running in demo mode with mock predictions.")
    model_loaded = False
    infer = None

def preprocess_image(image_data, target_size=(256, 256)):
    img = Image.open(BytesIO(image_data)).convert('RGB')
    img = img.resize(target_size)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0).astype(np.float32)
    return img_array

def predict_image(image_data):
    if not model_loaded:
        # Mock prediction for demo purposes
        import random
        prob = random.random()
        label = "yes" if prob >= 0.5 else "no"
        return float(prob), label

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

    # Retrieve context
    context = ""
    if vector_store:
        try:
            docs = vector_store.similarity_search(user_input, k=3)
            context = "\n\n".join([d.page_content for d in docs])
            print(f"Retrieved {len(docs)} documents for context.")
        except Exception as e:
            print(f"Error retrieving documents: {e}")

    # Construct prompt
    prompt = f"""You are a helpful medical assistant specializing in brain tumors. Use the following context to answer the user's question.

Context:
{context}

Question:
{user_input}

Important Instructions:
1. Answer ONLY using the provided context.
2. If the answer is not contained in the context, say "I can only provide answers for brain tumor related queries."
3. Do not make up information or use outside knowledge.
"""

    headers = {
        "Content-Type": "application/json"
    }

    data = {
        "contents": [
            {
                "parts": [{"text": prompt}]
            }
        ]
    }

    response = requests.post(
        "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent",
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
    app.run(debug=True, port=8080)