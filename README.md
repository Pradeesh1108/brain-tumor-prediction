# 🧠 Brain Tumor Prediction & RAG Chat Assistant

This project is a Flask-based web application that provides two robust features:
1. **Brain Tumor Segmentation/Prediction & Analysis** using a trained Deep Learning model (ResNet50).
2. **Interactive RAG Chat Assistant** which leverages a local vector store built on your medical documents and the high-speed Groq API (Llama 3) to answer queries contextually.

---

## ✨ Features

- **Brain Tumor Prediction:** Upload a JPEG MRI image to get real-time tumor predictions.
- **RAG Chatbot:** Ask questions about brain tumors, and the bot will answer strictly based on the provided PDF/DOCX contextual documents.
- **Local Document Embeddings:** Uses the fast, local Hugging Face `all-MiniLM-L6-v2` model for privacy and zero API costs on ingestion.
- **Lightning Fast LLM:** Uses the Groq API (`llama-3.3-70b-versatile`) for instant chat text generation.
- **Flask Web Application:** An easy-to-use GUI for both image prediction and chat interaction.

---

## 📁 Folder Structure

```
brain-tumor-prediction/
├── static/                 # Static files (CSS, JS)
├── templates/              # HTML templates (index.html)
├── documents/              # Place your PDF and DOCX files here for the RAG context
├── vectorstore/            # Generated FAISS vector database
├── brain_tumor_model_savedmodel/ # Saved TensorFlow model for image prediction
├── app.py                  # Flask web server and API endpoints
├── ingest.py               # Script to convert documents into local embeddings
├── requirements.txt        # Python dependencies
├── .env                    # Environment variables (API Keys)
└── README.md               # Project documentation
```

---

## 🛠️ Installation & Setup

### 1. Clone the repository

```bash
git clone https://github.com/Pradeesh1108/brain-tumor-prediction.git
cd brain-tumor-prediction
```

### 2. Create and activate a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

### 3. Install the dependencies

```bash
pip install -r requirements.txt
```

### 4. Setup Environment Variables

Create a `.env` file in the root directory and add your API keys:

```text
GROQ_API_KEY=your_groq_api_key_here
```

---

## 📚 Building the Knowledge Base (RAG)

Before chatting with the assistant, you need to ingest the knowledge documents:

1. Place your target `.pdf` or `.docx` files in the `documents/` folder.
2. Run the ingestion script to create local embeddings:

```bash
python3 ingest.py
```
This will create a `vectorstore/db_faiss` directory containing the local FAISS database.

---

## 🚀 Running the Web App

Once your vector store is created and your `.env` is set up, launch the Flask app:

```bash
python3 app.py
```

The app will become available at:
```
http://127.0.0.1:8080
```

### 🖼️ Using the Application

- **Image Prediction:** Upload a brain MRI image under the upload tab to view the predicted tumor result.
- **Chat Assistant:** Use the chat interface to ask context-aware questions. The AI strictly answers based on the files ingested into the `documents/` folder.

---

## 🧠 Technology Stack

- **Model Architecture:** ResNet50 (TensorFlow)
- **Framework:** Flask (Python)
- **Embeddings:** Hugging Face (`sentence-transformers/all-MiniLM-L6-v2`) via LangChain
- **LLM API:** Groq (`llama-3.3-70b-versatile`) via REST
- **Vector Store:** FAISS
- **Frontend:** HTML, Vanilla CSS, Vanilla JS
