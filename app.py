from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
from io import BytesIO
import os
import requests
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

app = Flask(__name__)

# Load Vector Store
DB_FAISS_PATH = "vectorstore/db_faiss"
vector_store = None
try:
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={'device': 'cpu'})
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
    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        return jsonify({"response": "Error: GROQ_API_KEY not found in environment."})

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
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    data = {
        "model": "llama-3.3-70b-versatile",
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": 0.5,
        "max_completion_tokens": 1024,
    }

    try:
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers=headers,
            json=data
        )
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"Error calling Groq API: {e}")
        if 'response' in locals() and hasattr(response, 'text'):
            print(f"Response text: {response.text}")
        content = "Sorry, I couldn't process that. Please try again."

    return jsonify({"response": content})

if __name__ == '__main__':
    app.run(debug=True, port=8080)