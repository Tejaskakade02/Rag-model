import os
import uuid
import chromadb
import nltk

from PyPDF2 import PdfReader
from nltk.tokenize import sent_tokenize
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from chromadb.config import Settings

# ==============================
# INITIAL SETUP
# ==============================

nltk.download("punkt")
nltk.download("punkt_tab")

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CHROMA_COLLECTION = "rag_chunks"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMA_PATH = os.path.join(BASE_DIR, "chroma_db")


# ==============================
# PDF LOADER
# ==============================

def load_pdf(pdf_path):
    """
    Load PDF and return page-wise text with metadata
    """
    reader = PdfReader(pdf_path)
    documents = []

    for page_num, page in enumerate(reader.pages):
        text = page.extract_text()
        if text and text.strip():
            documents.append({
                "text": text,
                "metadata": {
                    "source": os.path.basename(pdf_path),
                    "page": page_num + 1
                }
            })
    return documents


# ==============================
# CHUNKING STRATEGIES
# ==============================

def fixed_chunking(text, chunk_size=500, overlap=50):
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap

    return chunks


def sentence_chunking(text, max_sentences=5):
    sentences = sent_tokenize(text, language="english")
    chunks = []

    for i in range(0, len(sentences), max_sentences):
        chunk = " ".join(sentences[i:i + max_sentences])
        chunks.append(chunk)

    return chunks


def paragraph_chunking(text):
    return [p.strip() for p in text.split("\n\n") if p.strip()]


def recursive_chunking(text, chunk_size=500, overlap=50):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap
    )
    return splitter.split_text(text)
# ==============================
# EMBEDDING MODEL
# ==============================
def load_embedding_model():
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )
# ==============================
# CHROMADB INITIALIZATION
# ==============================

def init_chroma():
    """
    Initialize ChromaDB with explicit disk persistence.
    Ensures chroma_db folder is created and used.
    """

    # 1️⃣ Force-create chroma_db folder
    os.makedirs(CHROMA_PATH, exist_ok=True)

    # 2️⃣ Initialize Chroma client with persistence
    client = chromadb.PersistentClient(
        path=CHROMA_PATH
    )
    

    # 3️⃣ Create / load collection
    collection = client.get_or_create_collection(
        name=CHROMA_COLLECTION
    )

    return client, collection


# ==============================
# INGESTION PIPELINE
# ==============================

def ingest_pdf(pdf_path, strategy, collection, embedder):
    documents = load_pdf(pdf_path)
    total_chunks = 0

    for doc in documents:
        text = doc["text"]
        metadata = doc["metadata"]

        if strategy == "fixed":
            chunks = fixed_chunking(text)

        elif strategy == "sentence":
            chunks = sentence_chunking(text)

        elif strategy == "paragraph":
            chunks = paragraph_chunking(text)

        else:
            chunks = recursive_chunking(text)

        embeddings = embedder.embed_documents(chunks)

        for chunk, emb in zip(chunks, embeddings):
            collection.add(
                documents=[chunk],
                embeddings=[emb],
                metadatas=[metadata],
                ids=[str(uuid.uuid4())]
            )
            total_chunks += 1

    return total_chunks


# ==============================
# ENTRY POINT
# ==============================

if __name__ == "__main__":
    # 🔴 Change these
    pdf_path = "demo pdf (2).pdf"       # Place your PDF in same folder
    strategy = "recursive"        # fixed | sentence | paragraph | recursive

    print("🚀 Initializing ChromaDB...")
    client, collection = init_chroma()

    print("🧠 Loading embedding model...")
    embedder = load_embedding_model()

    print(f"📄 Ingesting PDF using '{strategy}' chunking...")
    total = ingest_pdf(pdf_path, strategy, collection, embedder)

    print("💾 Persisting database to disk...")
    # client.persist()

    print("\n✅ INGESTION COMPLETE")
    print(f"📄 PDF: {pdf_path}")
    print(f"✂️ Chunking Strategy: {strategy}")
    print(f"📦 Total Chunks Stored: {total}")
    print(f"🗄️ Stored at: {CHROMA_PATH}\n")
    print("🔢 Total vectors in DB:", collection.count())
