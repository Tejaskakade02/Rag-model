
```md
# 📚 Multimodal RAG Ingestion Engine

A **production-ready, multimodal Retrieval-Augmented Generation (RAG) ingestion pipeline** that can ingest, understand, chunk, embed, and store data from **documents, tables, images, and scanned PDFs** into multiple vector databases.

This system supports **OCR, image captioning, intelligent chunking strategies, multiple embedding models, and pluggable vector stores**, making it suitable for **enterprise-grade RAG systems**.

---

## 🚀 Key Capabilities

### ✅ Multimodal Document Support
- **PDFs** (native text + OCR + image captioning)
- **Images** (OCR + image description)
- **Word (.docx)**
- **PowerPoint (.pptx)**
- **Text & Markdown**
- **CSV / Excel tables**

### 🧠 Intelligent Chunking Strategies
- Fixed chunking
- Sentence-based chunking
- Paragraph chunking
- Recursive chunking (LangChain)
- Semantic chunking (embedding-aware)
- Content-aware chunking (section detection)
- Agentic chunking (goal-driven filtering & merging)

### 🔎 High-Quality Embeddings
Supports **state-of-the-art embedding models**:
- BGE (Base, M3)
- E5 (Large, Multilingual)
- GTE Large
- Nomic Embed v1.5
- Cohere (English & Multilingual)

### 🗄️ Multiple Vector Databases
- ChromaDB
- FAISS
- Qdrant
- Pinecone
- Weaviate  
(*Milvus scaffold included and ready to enable*)

### 🖼️ Image Understanding
- OCR via **Tesseract**
- Image captioning via **BLIP (Salesforce)**

---

## 🏗️ Architecture Overview

```
```
Document / Image
        │
        ▼
┌──────────────────────┐
│   DocumentLoader     │
└──────────────────────┘
        │
        ├── PDF Text Extraction
        ├── OCR (Images / Scanned PDFs)
        ├── Image Captioning (BLIP)
        │
        ▼
┌──────────────────────┐
│   Chunking Engine    │
│──────────────────────│
│ • Fixed              │
│ • Semantic           │
│ • Agentic            │
│ • Content-aware      │
└──────────────────────┘
        │
        ▼
┌──────────────────────┐
│  Embedding Engine    │
│──────────────────────│
│ • HuggingFace        │
│ • Cohere             │
└──────────────────────┘
        │
        ▼
┌──────────────────────────────────────────┐
│              Vector Store                │
│──────────────────────────────────────────│
│ • Chroma                                 │
│ • FAISS                                  │
│ • Qdrant                                 │
│ • Pinecone                               │
│ • Weaviate                               │
└──────────────────────────────────────────┘
```

```

---

## 📂 Project Structure
.
├── ingestion.py              # Main ingestion pipeline (this file)
├── chroma_db/                # Persistent Chroma storage
├── qdrant_db/                # Local Qdrant storage
├── faiss_store__*.pkl        # FAISS index files
└── README.md
```
```

---

## 📥 Supported File Types

| Type | Features |
|----|----|
| PDF | Text extraction, OCR, image captioning |
| Image | OCR + visual description |
| DOCX | Paragraph extraction |
| PPTX | Slide text extraction |
| CSV / Excel | Row-wise semantic ingestion |
| TXT / MD | Plain text ingestion |

---

## 🧠 Chunking Strategies Explained

| Strategy | Description |
|-------|------------|
| fixed | Character-based chunking with overlap |
| sentence | Groups N sentences per chunk |
| paragraph | Splits by blank lines |
| recursive | LangChain recursive splitter |
| semantic | Embedding similarity based |
| content_aware | Section + sub-chunking |
| agentic | Goal-driven keep/drop/merge logic |

Agentic chunking uses **rules and intent filtering** to remove irrelevant content like pricing, legal, or company info.

---

## 🔍 Embedding Models

Configured via `EMBEDDING_REGISTRY`:

- **BAAI/bge-base-en-v1.5**
- **BAAI/bge-m3**
- **intfloat/e5-large-v2**
- **intfloat/multilingual-e5-large**
- **thenlper/gte-large**
- **nomic-ai/nomic-embed-text-v1.5**
- **Cohere embed-english-v3.0**
- **Cohere embed-multilingual-v3.0**

E5 models automatically apply:
```
```
query: ...
passage: ...
```
```

---

## 🗄️ Vector Database Support

| DB | Mode |
|----|-----|
| Chroma | Persistent local |
| FAISS | File-based |
| Qdrant | Local persistent |
| Pinecone | Cloud serverless |
| Weaviate | Cloud (v4 client) |

Each embedding model gets its **own isolated collection/index**.

---

## 🖼️ Image Captioning

Powered by:
```

Salesforce/blip-image-captioning-base

````

Used for:
- Standalone images
- Embedded PDF images
- Scanned documents

Captions are embedded like normal text for RAG retrieval.

---

## ⚙️ Configuration

Update API keys and paths at the top of the file:

```python
PINECONE_API_KEY
WEAVIATE_URL
WEAVIATE_API_KEY
COHERE_API_KEY
````

GPU support is automatically enabled if CUDA is available.

---

## ▶️ How to Run

```bash
python ingestion.py
```

Default settings:

```python
pdf_path = "ai demo.jpg"
strategy = "recursive"
vector_db = "qdrant"
embedding_key = "bge-base"
```

---

## 📊 Output

At completion, the pipeline prints:

* Total chunks generated
* Total vectors stored
* Database statistics
* Safe shutdown of vector store

---

## 🔒 Production-Ready Features

* UUID-based chunk IDs
* Metadata-rich storage
* Fault-tolerant OCR & captioning
* Embedding isolation per model
* Cloud & local DB support
* Extensible design

---

## 🧠 Ideal Use Cases

* Enterprise RAG systems
* Multimodal search engines
* Knowledge base ingestion
* AI assistants with document understanding
* OCR-heavy industries (legal, healthcare, finance)

---

## 📌 Next Enhancements (Optional)

* Async ingestion
* Batch embedding
* Hybrid (BM25 + Vector) search
* Metadata filtering at query time
* Vision-language re-ranking

---

## 🏁 Conclusion

This ingestion engine is a **complete, enterprise-grade foundation for Multimodal RAG systems**, designed to scale from local experimentation to cloud production environments.

You can directly plug this into:

* LangChain / LlamaIndex
* Custom RAG APIs
* Chatbots & AI assistants

---

**Author:** You
**Status:** Production-ready 🚀

```

---

If you want, I can also:
- 🔹 Add **query-time retriever code**
- 🔹 Convert this into a **FastAPI ingestion service**
- 🔹 Create a **separate README for vector DB comparison**
- 🔹 Add **diagram images for GitHub**

Just tell me 👍
```
