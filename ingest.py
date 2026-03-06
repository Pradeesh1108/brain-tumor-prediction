from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
import glob
import os

load_dotenv()

if not os.getenv("GOOGLE_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_API_KEY")

DATA_PATH = "documents"
DB_FAISS_PATH = "vectorstore/db_faiss"

def create_vector_db():
    print(f"Loading documents from {DATA_PATH}...")
    documents = []
    
    # Load PDFs
    pdf_files = glob.glob(os.path.join(DATA_PATH, "*.pdf"))
    for pdf_file in pdf_files:
        print(f"Loading {pdf_file}...")
        loader = PyPDFLoader(pdf_file)
        documents.extend(loader.load())

    # Load DOCX
    docx_files = glob.glob(os.path.join(DATA_PATH, "*.docx"))
    for docx_file in docx_files:
        print(f"Loading {docx_file}...")
        loader = Docx2txtLoader(docx_file)
        documents.extend(loader.load())
    
    if not documents:
        print(f"No documents found in {DATA_PATH}/")
        return

    print(f"Loaded {len(documents)} document pages.")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)
    print(f"Split into {len(texts)} chunks.")

    print("Generating embeddings with local Hugging Face model (all-MiniLM-L6-v2)...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={'device': 'cpu'})
    
    print("Creating vector store...")
    db = FAISS.from_documents(texts, embeddings)
    db.save_local(DB_FAISS_PATH)
    print(f"Vector store saved to {DB_FAISS_PATH}")

if __name__ == "__main__":
    create_vector_db()
