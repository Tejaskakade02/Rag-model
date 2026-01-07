# import os
# import uuid
# import chromadb
# import nltk
# from sklearn.metrics.pairwise import cosine_similarity


# from PyPDF2 import PdfReader
# from nltk.tokenize import sent_tokenize
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_huggingface import HuggingFaceEmbeddings
# from chromadb.config import Settings

# # ==============================
# # INITIAL SETUP
# # ==============================

# nltk.download("punkt")
# nltk.download("punkt_tab")

# EMBEDDING_MODEL = "all-MiniLM-L6-v2"
# CHROMA_COLLECTION = "rag_chunks"
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# CHROMA_PATH = os.path.join(BASE_DIR, "chroma_db")


# # ==============================
# # PDF LOADER
# # ==============================

# def load_pdf(pdf_path):
#     """
#     Load PDF and return page-wise text with metadata
#     """
#     reader = PdfReader(pdf_path)
#     documents = []

#     for page_num, page in enumerate(reader.pages):
#         text = page.extract_text()
#         if text and text.strip():
#             documents.append({
#                 "text": text,
#                 "metadata": {
#                     "source": os.path.basename(pdf_path),
#                     "page": page_num + 1
#                 }
#             })
#     return documents


# # ==============================
# # CHUNKING STRATEGIES
# # ==============================

# def fixed_chunking(text, chunk_size=500, overlap=50): #---------------- fixed chunking function
#     chunks = []
#     start = 0

#     while start < len(text):
#         end = start + chunk_size
#         chunks.append(text[start:end])
#         start = end - overlap

#     return chunks


# def sentence_chunking(text, max_sentences=5): #---------------- sentence chunking function
#     sentences = sent_tokenize(text, language="english")
#     chunks = []

#     for i in range(0, len(sentences), max_sentences):
#         chunk = " ".join(sentences[i:i + max_sentences])
#         chunks.append(chunk)

#     return chunks


# def paragraph_chunking(text): #-------------------------paragharph chunking function
#     return [p.strip() for p in text.split("\n\n") if p.strip()]


# def recursive_chunking(text, chunk_size=500, overlap=50): #------------------ recursive chunking function
#     splitter = RecursiveCharacterTextSplitter(
#         chunk_size=chunk_size,
#         chunk_overlap=overlap
#     )
#     return splitter.split_text(text)

# def semantic_chunking( 
#     text,
#     embedder,
#     similarity_threshold=0.75,
#     max_chunk_chars=800
#                         ):                                      #------------------ semantic chunking function
#     """
#     Semantic (meaning-based) chunking.
#     Groups sentences until semantic similarity drops.
#     """

#     # 1️⃣ Split text into sentences
#     sentences = sent_tokenize(text, language="english")

#     # Edge case
#     if len(sentences) <= 1:
#         return sentences

#     # 2️⃣ Embed sentences
#     sentence_embeddings = embedder.embed_documents(sentences)

#     chunks = []
#     current_chunk = [sentences[0]]
#     current_length = len(sentences[0])

#     # 3️⃣ Compare adjacent sentences
#     for i in range(1, len(sentences)):
#         prev_emb = sentence_embeddings[i - 1]
#         curr_emb = sentence_embeddings[i]

#         similarity = cosine_similarity(
#             [prev_emb],
#             [curr_emb]
#         )[0][0]

#         # 4️⃣ Decide whether to keep grouping
#         if (
#             similarity >= similarity_threshold
#             and current_length < max_chunk_chars
#         ):
#             current_chunk.append(sentences[i])
#             current_length += len(sentences[i])
#         else:
#             # Finalize current chunk
#             chunks.append(" ".join(current_chunk))
#             current_chunk = [sentences[i]]
#             current_length = len(sentences[i])

#     # 5️⃣ Add last chunk
#     if current_chunk:
#         chunks.append(" ".join(current_chunk))

#     return chunks

# # ==============================
# # EMBEDDING MODEL
# # ==============================
# def load_embedding_model():
#     return HuggingFaceEmbeddings(
#         model_name=EMBEDDING_MODEL
#     )
# # ==============================
# # CHROMADB INITIALIZATION
# # ==============================

# def init_chroma():
#     """
#     Initialize ChromaDB with explicit disk persistence.
#     Ensures chroma_db folder is created and used.
#     """

#     # 1️⃣ Force-create chroma_db folder
#     os.makedirs(CHROMA_PATH, exist_ok=True)

#     # 2️⃣ Initialize Chroma client with persistence
#     client = chromadb.PersistentClient(
#         path=CHROMA_PATH
#     )
    

#     # 3️⃣ Create / load collection
#     collection = client.get_or_create_collection(
#         name=CHROMA_COLLECTION
#     )

#     return client, collection


# # ==============================
# # INGESTION PIPELINE
# # ==============================

# def ingest_pdf(pdf_path, strategy, collection, embedder):
#     documents = load_pdf(pdf_path)
#     total_chunks = 0

#     for doc in documents:
#         text = doc["text"]
#         metadata = doc["metadata"]

#         if strategy == "fixed":
#             chunks = fixed_chunking(text)

#         elif strategy == "sentence":
#             chunks = sentence_chunking(text)

#         elif strategy == "paragraph":
#             chunks = paragraph_chunking(text)

#         elif strategy == "semantic":
#             chunks = semantic_chunking(
#                  text,
#                 embedder,
#                 similarity_threshold=0.75,
#                 max_chunk_chars=800
#             )
#             # if total_chunks < 2:
#             #     print("\n--- SEMANTIC CHUNK ---")
#             #     print(chunks)
#         else:
#             chunks = recursive_chunking(text)

#         embeddings = embedder.embed_documents(chunks)

#         for chunk, emb in zip(chunks, embeddings):
#             collection.add(
#                 documents=[chunk],
#                 embeddings=[emb],
#                 metadatas=[metadata],
#                 ids=[str(uuid.uuid4())]
#             )
#             total_chunks += 1

#     return total_chunks


# # ==============================
# # ENTRY POINT
# # ==============================

# if __name__ == "__main__":
#     # 🔴 Change these
#     pdf_path = "demo pdf (2).pdf"       # Place your PDF in same folder
#     strategy = "fixed"        # fixed | sentence | paragraph | recursive

#     print("🚀 Initializing ChromaDB...")
#     client, collection = init_chroma()

#     print("🧠 Loading embedding model...")
#     embedder = load_embedding_model()

#     print(f"📄 Ingesting PDF using '{strategy}' chunking...")
#     total = ingest_pdf(pdf_path, strategy, collection, embedder)

#     print("💾 Persisting database to disk...")
#     # client.persist()

#     print("\n✅ INGESTION COMPLETE")
#     print(f"📄 PDF: {pdf_path}")
#     print(f"✂️ Chunking Strategy: {strategy}")
#     print(f"📦 Total Chunks Stored: {total}")
#     print(f"🗄️ Stored at: {CHROMA_PATH}\n")
#     print("🔢 Total vectors in DB:", collection.count())
################################################################################################################################
# import os
# import uuid
# import re
# import chromadb
# import nltk
# from sklearn.metrics.pairwise import cosine_similarity

# from PyPDF2 import PdfReader
# from nltk.tokenize import sent_tokenize
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_huggingface import HuggingFaceEmbeddings

# # ==============================
# # INITIAL SETUP
# # ==============================

# nltk.download("punkt")
# nltk.download("punkt_tab")

# EMBEDDING_MODEL = "all-MiniLM-L6-v2"
# CHROMA_COLLECTION = "rag_chunks"

# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# CHROMA_PATH = os.path.join(BASE_DIR, "chroma_db")

# # ==============================
# # AGENT RULES (CONFIG)
# # ==============================

# AGENT_RULES = {
#     "technical_assistant": {
#         "keep_keywords": [
#             "architecture", "design", "deployment", "installation",
#             "configuration", "setup", "implementation"
#         ],
#         "drop_keywords": [
#             "pricing", "contact", "about", "company", "legal", "copyright"
#         ],
#         "merge_keywords": [
#             "setup", "installation", "deployment"
#         ],
#         "max_section_length": 1500
#     }
# }

# # ==============================
# # PDF LOADER
# # ==============================

# def load_pdf(pdf_path):
#     reader = PdfReader(pdf_path)
#     documents = []

#     for page_num, page in enumerate(reader.pages):
#         text = page.extract_text()
#         if text and text.strip():
#             documents.append({
#                 "text": text,
#                 "metadata": {
#                     "source": os.path.basename(pdf_path),
#                     "page": page_num + 1
#                 }
#             })
#     return documents

# # ==============================
# # BASIC CHUNKING STRATEGIES
# # ==============================

# def fixed_chunking(text, chunk_size=500, overlap=50): # ----------------- fixed chunking function
#     chunks, start = [], 0
#     while start < len(text):
#         end = start + chunk_size
#         chunks.append(text[start:end])
#         start = end - overlap
#     return chunks


# def sentence_chunking(text, max_sentences=5): # ------------------ sentence chunking function
#     sentences = sent_tokenize(text, language="english")
#     return [
#         " ".join(sentences[i:i + max_sentences])
#         for i in range(0, len(sentences), max_sentences)
#     ]


# def paragraph_chunking(text):          # ------------------ paragraph chunking function
#     return [p.strip() for p in text.split("\n\n") if p.strip()]


# def recursive_chunking(text, chunk_size=500, overlap=50):  # ------------------ recursive chunking function
#     splitter = RecursiveCharacterTextSplitter(
#         chunk_size=chunk_size,
#         chunk_overlap=overlap
#     )
#     return splitter.split_text(text)

# # ==============================
# # SEMANTIC CHUNKING
# # ==============================

# def semantic_chunking(
#     text,
#     embedder,
#     similarity_threshold=0.75,
#     max_chunk_chars=800
# ):                                                             # ------------------ semantic chunking function
#     sentences = sent_tokenize(text, language="english")
#     if len(sentences) <= 1:
#         return sentences

#     sentence_embeddings = embedder.embed_documents(sentences)

#     chunks = []
#     current_chunk = [sentences[0]]
#     current_len = len(sentences[0])

#     for i in range(1, len(sentences)):
#         sim = cosine_similarity(
#             [sentence_embeddings[i - 1]],
#             [sentence_embeddings[i]]
#         )[0][0]

#         if sim >= similarity_threshold and current_len < max_chunk_chars:
#             current_chunk.append(sentences[i])
#             current_len += len(sentences[i])
#         else:
#             chunks.append(" ".join(current_chunk))
#             current_chunk = [sentences[i]]
#             current_len = len(sentences[i])

#     if current_chunk:
#         chunks.append(" ".join(current_chunk))

#     return chunks

# # ==============================
# # CONTENT-AWARE CHUNKING
# # ==============================

# def is_heading(line: str) -> bool:
#     line = line.strip()
#     if not line:
#         return False
#     if line.isupper() and len(line) < 100:
#         return True
#     if re.match(r"^\d+(\.\d+)*\s+", line):
#         return True
#     if line.endswith(":") and len(line) < 80:
#         return True
#     return False


# def detect_sections(text: str):
#     lines = text.split("\n")
#     sections = []

#     current = {
#         "title": "INTRODUCTION",
#         "content": [],
#         "index": 0
#     }

#     section_index = 1

#     for line in lines:
#         if is_heading(line):
#             if current["content"]:
#                 sections.append(current)

#             current = {
#                 "title": line.strip(),
#                 "content": [],
#                 "index": section_index
#             }
#             section_index += 1
#         else:
#             current["content"].append(line)

#     if current["content"]:
#         sections.append(current)

#     return sections


# def content_aware_chunking(text, embedder, inner_strategy="recursive"):
#     sections = detect_sections(text)
#     chunks = []

#     for section in sections:
#         section_text = "\n".join(section["content"]).strip()
#         if not section_text:
#             continue

#         if inner_strategy == "paragraph":
#             inner_chunks = paragraph_chunking(section_text)
#         elif inner_strategy == "semantic":
#             inner_chunks = semantic_chunking(section_text, embedder)
#         else:
#             inner_chunks = recursive_chunking(section_text)

#         for idx, chunk in enumerate(inner_chunks):
#             chunks.append({
#                 "text": chunk,
#                 "section_title": section["title"],
#                 "section_index": section["index"],
#                 "chunk_index_in_section": idx
#             })

#     return chunks

# # ==============================
# # AGENTIC CHUNKING
# # ==============================

# def evaluate_section(section, agent_goal):
#     rules = AGENT_RULES[agent_goal]
#     title = section["title"].lower()
#     content = " ".join(section["content"]).lower()

#     for kw in rules["drop_keywords"]:
#         if kw in title:
#             return "DROP", f"Irrelevant section: {section['title']}"

#     for kw in rules["merge_keywords"]:
#         if kw in title:
#             return "MERGE", f"Merged operational section: {section['title']}"

#     if len(content) > rules["max_section_length"]:
#         return "SPLIT", "Section too large"

#     for kw in rules["keep_keywords"]:
#         if kw in title or kw in content:
#             return "KEEP", "Relevant to agent goal"

#     return "KEEP", "Default keep"


# def agentic_chunking(
#     text,
#     embedder,
#     agent_goal="technical_assistant",
#     inner_strategy="recursive"
# ):
#     sections = detect_sections(text)
#     chunks = []
#     merge_buffer = []

#     for section in sections:
#         decision, reason = evaluate_section(section, agent_goal)
#         section_text = "\n".join(section["content"]).strip()

#         if not section_text:
#             continue

#         if decision == "DROP":
#             continue

#         if decision == "MERGE":
#             merge_buffer.append(section)
#             continue

#         if merge_buffer:
#             merged_text = "\n\n".join(
#                 ["\n".join(s["content"]) for s in merge_buffer]
#             )
#             chunks.append({
#                 "text": merged_text,
#                 "decision": "MERGE",
#                 "decision_reason": "Merged related sections",
#                 "agent_goal": agent_goal,
#                 "source_sections": [s["title"] for s in merge_buffer]
#             })
#             merge_buffer = []

#         if decision == "SPLIT":
#             sub_chunks = (
#                 semantic_chunking(section_text, embedder)
#                 if inner_strategy == "semantic"
#                 else recursive_chunking(section_text)
#             )
#             for idx, sub in enumerate(sub_chunks):
#                 chunks.append({
#                     "text": sub,
#                     "decision": "SPLIT",
#                     "decision_reason": reason,
#                     "agent_goal": agent_goal,
#                     "source_sections": [section["title"]],
#                     "chunk_index_in_section": idx
#                 })

#         if decision == "KEEP":
#             chunks.append({
#                 "text": section_text,
#                 "decision": "KEEP",
#                 "decision_reason": reason,
#                 "agent_goal": agent_goal,
#                 "source_sections": [section["title"]]
#             })

#     return chunks

# # ==============================
# # EMBEDDING MODEL
# # ==============================

# def load_embedding_model():
#     return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

# # ==============================
# # CHROMADB INITIALIZATION
# # ==============================

# def init_chroma():
#     os.makedirs(CHROMA_PATH, exist_ok=True)
#     client = chromadb.PersistentClient(path=CHROMA_PATH)
#     collection = client.get_or_create_collection(name=CHROMA_COLLECTION)
#     return client, collection

# # ==============================
# # INGESTION PIPELINE
# # ==============================

# def ingest_pdf(pdf_path, strategy, collection, embedder):
#     documents = load_pdf(pdf_path)
#     total_chunks = 0

#     for doc in documents:
#         text = doc["text"]
#         base_metadata = doc["metadata"]

#         if strategy == "agentic":
#             chunks = agentic_chunking(text, embedder)
#             texts = [c["text"] for c in chunks]
#         elif strategy == "content_aware":
#             chunks = content_aware_chunking(text, embedder)
#             texts = [c["text"] for c in chunks]
#         elif strategy == "semantic":
#             chunks = semantic_chunking(text, embedder)
#             texts = chunks
#         elif strategy == "sentence":
#             chunks = sentence_chunking(text)
#             texts = chunks
#         elif strategy == "paragraph":
#             chunks = paragraph_chunking(text)
#             texts = chunks
#         elif strategy == "fixed":
#             chunks = fixed_chunking(text)
#             texts = chunks
#         else:
#             chunks = recursive_chunking(text)
#             texts = chunks

#         embeddings = embedder.embed_documents(texts)

#         for i, emb in enumerate(embeddings):
#             if strategy == "agentic":
#                 meta = {
#                     **base_metadata,
#                     "chunk_type": "agentic",
#                     "agent_goal": chunks[i]["agent_goal"],
#                     "decision": chunks[i]["decision"],
#                     "decision_reason": chunks[i]["decision_reason"],
#                     "source_sections": " | ".join(chunks[i]["source_sections"])
#                 }
#                 chunk_text = chunks[i]["text"]
#             elif strategy == "content_aware":
#                 meta = {
#                     **base_metadata,
#                     "chunk_type": "content_aware",
#                     "section_title": chunks[i]["section_title"],
#                     "section_index": chunks[i]["section_index"],
#                     "chunk_index_in_section": chunks[i]["chunk_index_in_section"]
#                 }
#                 chunk_text = chunks[i]["text"]
#             else:
#                 meta = {**base_metadata, "chunk_type": strategy}
#                 chunk_text = texts[i]

#             collection.add(
#                 documents=[chunk_text],
#                 embeddings=[emb],
#                 metadatas=[meta],
#                 ids=[str(uuid.uuid4())]
#             )
#             total_chunks += 1

#     return total_chunks

# # ==============================
# # ENTRY POINT
# # ==============================

# if __name__ == "__main__":

#     pdf_path = "demo pdf (2).pdf"
#     strategy = "agentic"
#     # fixed | sentence | paragraph | recursive | semantic | content_aware | agentic

#     print("🚀 Initializing ChromaDB...")
#     client, collection = init_chroma()

#     print("🧠 Loading embedding model...")
#     embedder = load_embedding_model()

#     print(f"📄 Ingesting PDF using '{strategy}' chunking...")
#     total = ingest_pdf(pdf_path, strategy, collection, embedder)

#     print("\n✅ INGESTION COMPLETE")
#     print(f"📄 PDF: {pdf_path}")
#     print(f"✂️ Chunking Strategy: {strategy}")
#     print(f"📦 Total Chunks Stored: {total}")
#     print(f"🗄️ Stored at: {CHROMA_PATH}")
#     print("🔢 Total vectors in DB:", collection.count())
#############################################################################################################################################################################################
# import os
# import uuid
# import re
# import pickle
# import numpy as np
# import chromadb
# import nltk
# import faiss
 
# import weaviate
# from weaviate.auth import AuthApiKey
# from weaviate.classes.config import Property
# from weaviate.classes.config import DataType as WeaviateDataType
# from weaviate.classes.data import DataObject
# from sklearn.metrics.pairwise import cosine_similarity
# from pypdf import PdfReader
# from nltk.tokenize import sent_tokenize
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_huggingface import HuggingFaceEmbeddings

# from qdrant_client import QdrantClient
# from qdrant_client.models import PointStruct, VectorParams, Distance
# from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection,utility
# from pinecone import Pinecone, ServerlessSpec

# # ==============================
# # INITIAL SETUP
# # ==============================

# nltk.download("punkt")
# nltk.download("punkt_tab")

# EMBEDDING_MODEL = "all-MiniLM-L6-v2"
# CHROMA_COLLECTION = "rag_chunks"

# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# CHROMA_PATH = os.path.join(BASE_DIR, "chroma_db")       # ----- chroma path
# FAISS_PATH = os.path.join(BASE_DIR, "faiss_store.pkl")     # ----- faiss path
# QDRANT_PATH = os.path.join(BASE_DIR, "qdrant_db")       # ----- qdrant path
# QDRANT_COLLECTION = "rag_chunks"                          # ----- qdrant collection name
# MILVUS_COLLECTION = "rag_chunks"                           # ----- milvus collection name
# PINECONE_API_KEY = "pcsk_36kLSk_KyRnADaFcY1xF4dS7gyTKjCqBu5qHj8rYMmmr7RP6anZb56YmGZw1Z5BaM1Gznv"      # pinecone api key
# PINECONE_ENV = "us-east-1"                                                                   # ---------------- pinecone environment
# PINECONE_INDEX = "rag-chunks"                                            
# WEAVIATE_URL = "https://pmh8cdktsmeloccbguom5a.c0.asia-southeast1.gcp.weaviate.cloud"                  # ---------------- weaviate url
# WEAVIATE_API_KEY = "UGd4eU0yeC9sbWhNZm53Nl9STDIvSTY0K2puOTRUY05KZ1hNVEhYajhIbk9PTUVMKzluc3RERVI0WC9NPV92MjAw"   # ---------------- weaviate api key
# WEAVIATE_CLASS = "RagChunks"
# # ==============================
# # AGENT RULES (CONFIG)
# # ==============================

# AGENT_RULES = {
#     "technical_assistant": {
#         "keep_keywords": [
#             "architecture", "design", "deployment", "installation",
#             "configuration", "setup", "implementation"
#         ],
#         "drop_keywords": [
#             "pricing", "contact", "about", "company", "legal", "copyright"
#         ],
#         "merge_keywords": [
#             "setup", "installation", "deployment"
#         ],
#         "max_section_length": 1500
#     }
# }

# # ==============================
# # PDF LOADER
# # ==============================

# def load_pdf(pdf_path):
#     reader = PdfReader(pdf_path)
#     documents = []

#     for page_num, page in enumerate(reader.pages):
#         text = page.extract_text()
#         if text and text.strip():
#             documents.append({
#                 "text": text,
#                 "metadata": {
#                     "source": os.path.basename(pdf_path),
#                     "page": page_num + 1
#                 }
#             })
#     return documents

# # ==============================
# # BASIC CHUNKING STRATEGIES
# # ==============================

# def fixed_chunking(text, chunk_size=500, overlap=50): # ----------------- fixed chunking function
#     chunks, start = [], 0
#     while start < len(text):
#         end = start + chunk_size
#         chunks.append(text[start:end])
#         start = end - overlap
#     return chunks


# def sentence_chunking(text, max_sentences=5): # ------------------ sentence chunking function
#     sentences = sent_tokenize(text, language="english")
#     return [
#         " ".join(sentences[i:i + max_sentences])
#         for i in range(0, len(sentences), max_sentences)
#     ]


# def paragraph_chunking(text):          # ------------------ paragraph chunking function
#     return [p.strip() for p in text.split("\n\n") if p.strip()]


# def recursive_chunking(text, chunk_size=500, overlap=50):  # ------------------ recursive chunking function
#     splitter = RecursiveCharacterTextSplitter(
#         chunk_size=chunk_size,
#         chunk_overlap=overlap
#     )
#     return splitter.split_text(text)

# # ==============================
# # SEMANTIC CHUNKING
# # ==============================

# def semantic_chunking(
#     text,
#     embedder,
#     similarity_threshold=0.75,
#     max_chunk_chars=800
# ):                                                             # ------------------ semantic chunking function
#     sentences = sent_tokenize(text, language="english")
#     if len(sentences) <= 1:
#         return sentences

#     sentence_embeddings = embedder.embed_documents(sentences)

#     chunks = []
#     current_chunk = [sentences[0]]
#     current_len = len(sentences[0])

#     for i in range(1, len(sentences)):
#         sim = cosine_similarity(
#             [sentence_embeddings[i - 1]],
#             [sentence_embeddings[i]]
#         )[0][0]

#         if sim >= similarity_threshold and current_len < max_chunk_chars:
#             current_chunk.append(sentences[i])
#             current_len += len(sentences[i])
#         else:
#             chunks.append(" ".join(current_chunk))
#             current_chunk = [sentences[i]]
#             current_len = len(sentences[i])

#     if current_chunk:
#         chunks.append(" ".join(current_chunk))

#     return chunks

# # ==============================
# # CONTENT-AWARE CHUNKING
# # ==============================

# def is_heading(line: str) -> bool:
#     line = line.strip()
#     if not line:
#         return False
#     if line.isupper() and len(line) < 100:
#         return True
#     if re.match(r"^\d+(\.\d+)*\s+", line):
#         return True
#     if line.endswith(":") and len(line) < 80:
#         return True
#     return False


# def detect_sections(text: str):
#     lines = text.split("\n")
#     sections = []

#     current = {
#         "title": "INTRODUCTION",
#         "content": [],
#         "index": 0
#     }

#     section_index = 1

#     for line in lines:
#         if is_heading(line):
#             if current["content"]:
#                 sections.append(current)

#             current = {
#                 "title": line.strip(),
#                 "content": [],
#                 "index": section_index
#             }
#             section_index += 1
#         else:
#             current["content"].append(line)

#     if current["content"]:
#         sections.append(current)

#     return sections


# def content_aware_chunking(text, embedder, inner_strategy="recursive"):
#     sections = detect_sections(text)
#     chunks = []

#     for section in sections:
#         section_text = "\n".join(section["content"]).strip()
#         if not section_text:
#             continue

#         if inner_strategy == "paragraph":
#             inner_chunks = paragraph_chunking(section_text)
#         elif inner_strategy == "semantic":
#             inner_chunks = semantic_chunking(section_text, embedder)
#         else:
#             inner_chunks = recursive_chunking(section_text)

#         for idx, chunk in enumerate(inner_chunks):
#             chunks.append({
#                 "text": chunk,
#                 "section_title": section["title"],
#                 "section_index": section["index"],
#                 "chunk_index_in_section": idx
#             })

#     return chunks

# # ==============================
# # AGENTIC CHUNKING
# # ==============================

# def evaluate_section(section, agent_goal):
#     rules = AGENT_RULES[agent_goal]
#     title = section["title"].lower()
#     content = " ".join(section["content"]).lower()

#     for kw in rules["drop_keywords"]:
#         if kw in title:
#             return "DROP", f"Irrelevant section: {section['title']}"

#     for kw in rules["merge_keywords"]:
#         if kw in title:
#             return "MERGE", f"Merged operational section: {section['title']}"

#     if len(content) > rules["max_section_length"]:
#         return "SPLIT", "Section too large"

#     for kw in rules["keep_keywords"]:
#         if kw in title or kw in content:
#             return "KEEP", "Relevant to agent goal"

#     return "KEEP", "Default keep"


# def agentic_chunking(
#     text,
#     embedder,
#     agent_goal="technical_assistant",
#     inner_strategy="recursive"
# ):
#     sections = detect_sections(text)
#     chunks = []
#     merge_buffer = []

#     for section in sections:
#         decision, reason = evaluate_section(section, agent_goal)
#         section_text = "\n".join(section["content"]).strip()

#         if not section_text:
#             continue

#         if decision == "DROP":
#             continue

#         if decision == "MERGE":
#             merge_buffer.append(section)
#             continue

#         if merge_buffer:
#             merged_text = "\n\n".join(
#                 ["\n".join(s["content"]) for s in merge_buffer]
#             )
#             chunks.append({
#                 "text": merged_text,
#                 "decision": "MERGE",
#                 "decision_reason": "Merged related sections",
#                 "agent_goal": agent_goal,
#                 "source_sections": [s["title"] for s in merge_buffer]
#             })
#             merge_buffer = []

#         if decision == "SPLIT":
#             sub_chunks = (
#                 semantic_chunking(section_text, embedder)
#                 if inner_strategy == "semantic"
#                 else recursive_chunking(section_text)
#             )
#             for idx, sub in enumerate(sub_chunks):
#                 chunks.append({
#                     "text": sub,
#                     "decision": "SPLIT",
#                     "decision_reason": reason,
#                     "agent_goal": agent_goal,
#                     "source_sections": [section["title"]],
#                     "chunk_index_in_section": idx
#                 })

#         if decision == "KEEP":
#             chunks.append({
#                 "text": section_text,
#                 "decision": "KEEP",
#                 "decision_reason": reason,
#                 "agent_goal": agent_goal,
#                 "source_sections": [section["title"]]
#             })

#     return chunks

# # ==============================
# # EMBEDDING MODEL
# # ==============================

# def load_embedding_model():
#     return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

# # ==============================
# # VECTOR STORE ADAPTERS
# # ==============================

# class VectorStore:
#     def add(self, texts, embeddings, metadatas, ids):
#         raise NotImplementedError

#     def count(self):
#         raise NotImplementedError


# class ChromaStore(VectorStore):
#     def __init__(self):
#         os.makedirs(CHROMA_PATH, exist_ok=True)
#         self.client = chromadb.PersistentClient(path=CHROMA_PATH)
#         self.collection = self.client.get_or_create_collection(CHROMA_COLLECTION)

#     def add(self, texts, embeddings, metadatas, ids):
#         self.collection.add(
#             documents=texts,
#             embeddings=embeddings,
#             metadatas=metadatas,
#             ids=ids
#         )

#     def count(self):
#         return self.collection.count()


# class FaissStore(VectorStore):
#     def __init__(self, dim):
#         self.index = faiss.IndexFlatL2(dim)
#         self.data = []

#         if os.path.exists(FAISS_PATH):
#             with open(FAISS_PATH, "rb") as f:
#                 saved = pickle.load(f)
#                 self.index = saved["index"]
#                 self.data = saved["data"]

#     def add(self, texts, embeddings, metadatas, ids):
#         vectors = np.array(embeddings).astype("float32")
#         self.index.add(vectors)

#         for i in range(len(texts)):
#             self.data.append({
#                 "id": ids[i],
#                 "text": texts[i],
#                 "metadata": metadatas[i]
#             })

#         with open(FAISS_PATH, "wb") as f:
#             pickle.dump({"index": self.index, "data": self.data}, f)

#     def count(self):
#         return self.index.ntotal


# class QdrantStore(VectorStore):
#     def __init__(self, dim):
#         self.client = QdrantClient(path=QDRANT_PATH)

#         existing = [c.name for c in self.client.get_collections().collections]
#         if QDRANT_COLLECTION not in existing:
#             self.client.create_collection(
#                 collection_name=QDRANT_COLLECTION,
#                 vectors_config=VectorParams(
#                     size=dim,
#                     distance=Distance.COSINE
#                 )
#             )

#     def add(self, texts, embeddings, metadatas, ids):
#         points = []
#         for i in range(len(texts)):
#             payload = {
#                 "text": texts[i],
#                 **metadatas[i]
#             }
#             points.append(
#                 PointStruct(
#                     id=ids[i],
#                     vector=embeddings[i],
#                     payload=payload
#                 )
#             )

#         self.client.upsert(
#             collection_name=QDRANT_COLLECTION,
#             points=points
#         )

#     def count(self):
#         return self.client.count(QDRANT_COLLECTION).count

#     def close(self):
#         self.client.close()



# # # ==============================
# # # ✅ MILVUS STORE (ADDED)
# # # ==============================

# # class MilvusStore(VectorStore):
# #     def __init__(self, dim):
# #         connections.connect(
# #             alias="default",
# #             host="localhost",
# #             port="19530"
# #         )

# #         if not utility.has_collection(MILVUS_COLLECTION):
# #             fields = [
# #                 FieldSchema(
# #                     name="id",
# #                     dtype=DataType.VARCHAR,
# #                     is_primary=True,
# #                     max_length=64
# #                 ),
# #                 FieldSchema(
# #                     name="embedding",
# #                     dtype=DataType.FLOAT_VECTOR,
# #                     dim=dim
# #                 ),
# #                 FieldSchema(
# #                     name="text",
# #                     dtype=DataType.VARCHAR,
# #                     max_length=65535
# #                 ),
# #                 FieldSchema(
# #                     name="metadata",
# #                     dtype=DataType.JSON
# #                 )
# #             ]

# #             schema = CollectionSchema(
# #                 fields=fields,
# #                 description="RAG chunks"
# #             )

# #             self.collection = Collection(
# #                 name=MILVUS_COLLECTION,
# #                 schema=schema
# #             )

# #             self.collection.create_index(
# #                 field_name="embedding",
# #                 index_params={
# #                     "index_type": "HNSW",
# #                     "metric_type": "COSINE",
# #                     "params": {"M": 8, "efConstruction": 64}
# #                 }
# #             )
# #         else:
# #             self.collection = Collection(MILVUS_COLLECTION)

# #         self.collection.load()

# #     def add(self, texts, embeddings, metadatas, ids):
# #         self.collection.insert([
# #             ids,
# #             embeddings,
# #             texts,
# #             metadatas
# #         ])

# #     def count(self):
# #         return self.collection.num_entities

# #     def close(self):
# #         self.collection.release()
# #         connections.disconnect("default")


# # ==============================
# # ✅ Pine-cone
# # ==============================
# class PineconeStore(VectorStore):
#     def __init__(self, dim):
#         self.pc = Pinecone(api_key=PINECONE_API_KEY)

#         existing_indexes = [idx["name"] for idx in self.pc.list_indexes()]
#         if PINECONE_INDEX not in existing_indexes:
#             self.pc.create_index(
#                 name=PINECONE_INDEX,
#                 dimension=dim,
#                 metric="cosine",
#                 spec=ServerlessSpec(
#                     cloud="aws",
#                     region=PINECONE_ENV
#                 )
#             )

#         self.index = self.pc.Index(PINECONE_INDEX)

#     def add(self, texts, embeddings, metadatas, ids):
#         vectors = []
#         for i in range(len(texts)):
#             vectors.append((
#                 ids[i],
#                 embeddings[i],
#                 {
#                     "text": texts[i],
#                     **metadatas[i]
#                 }
#             ))

#         self.index.upsert(vectors=vectors)

#     def count(self):
#         stats = self.index.describe_index_stats()
#         return stats["total_vector_count"]
    


# # ==============================
# # ✅ Weaviate STORE
# # ==============================
# class WeaviateStore(VectorStore):
#     def __init__(self, dim):
#         # 1️⃣ Connect to Weaviate Cloud (v4)
#         self.client = weaviate.connect_to_weaviate_cloud(
#             cluster_url=WEAVIATE_URL.replace("https://", ""),
#             auth_credentials=AuthApiKey(WEAVIATE_API_KEY),
#         )

#         print("🔍 [Weaviate] Client created")

#         ready = self.client.is_ready()
#         print("🔍 [Weaviate] is_ready():", ready)

#         if not ready:
#             raise RuntimeError("Weaviate is not ready")
#         existing = self.client.collections.list_all()      
#         if WEAVIATE_CLASS not in existing:
#             self.client.collections.create(
#                       name=WEAVIATE_CLASS,
#                          properties=[
#                             Property(name="text", data_type=WeaviateDataType.TEXT),
#                             Property(name="source", data_type=WeaviateDataType.TEXT),
#                             Property(name="page", data_type=WeaviateDataType.INT),
#                         ],
#                         )
#         else:
#             print(f"✅ [Weaviate] Collection already exists: {WEAVIATE_CLASS}")
#         # 5️⃣ Get collection handle
#         self.collection = self.client.collections.get(WEAVIATE_CLASS)
#     def add(self, texts, embeddings, metadatas, ids):
        
#         objects = []
#         for i in range(len(texts)):
#              objects.append(
#             DataObject(
#                 properties={
#                     "text": texts[i],
#                     **metadatas[i]
#                 },
#                 vector=embeddings[i],
#                 uuid=ids[i]
#             )
#         )
#         self.collection.data.insert_many(objects)
#     def count(self):
#         count = self.collection.aggregate.over_all(
#             total_count=True
#         ).total_count

#         print("🔢 [Weaviate] Total vectors:", count)
#         return count
    
#     def close(self):
#         print("🔒 [Weaviate] Closing client")
#         self.client.close()



# def init_vector_store(db_type, embedder):
#     dim = len(embedder.embed_query("test"))

#     if db_type == "chroma":
#         return ChromaStore()
#     elif db_type == "faiss":
#         return FaissStore(dim)
#     elif db_type == "qdrant":
#         return QdrantStore(dim)
#     # elif db_type == "milvus":          
#     #     return MilvusStore(dim)
#     elif db_type == "pinecone":      
#         return PineconeStore(dim)
#     elif db_type == "weaviate":
#         return WeaviateStore(dim)
#     else:
#         raise ValueError("Unsupported DB")

# # ==============================
# # INGESTION PIPELINE
# # ==============================

# def ingest_pdf(pdf_path, strategy, collection, embedder):
#     documents = load_pdf(pdf_path)
#     total_chunks = 0

#     for doc in documents:
#         text = doc["text"]
#         base_metadata = doc["metadata"]

#         if strategy == "agentic":
#             chunks = agentic_chunking(text, embedder)
#             texts = [c["text"] for c in chunks]
#         elif strategy == "content_aware":
#             chunks = content_aware_chunking(text, embedder)
#             texts = [c["text"] for c in chunks]
#         elif strategy == "semantic":
#             chunks = semantic_chunking(text, embedder)
#             texts = chunks
#         elif strategy == "sentence":
#             chunks = sentence_chunking(text)
#             texts = chunks
#         elif strategy == "paragraph":
#             chunks = paragraph_chunking(text)
#             texts = chunks
#         elif strategy == "fixed":
#             chunks = fixed_chunking(text)
#             texts = chunks
#         else:
#             chunks = recursive_chunking(text)
#             texts = chunks

#         embeddings = embedder.embed_documents(texts)

#         for i, emb in enumerate(embeddings):
#             if strategy == "agentic":
#                 meta = {
#                     **base_metadata,
#                     "chunk_type": "agentic",
#                     "agent_goal": chunks[i]["agent_goal"],
#                     "decision": chunks[i]["decision"],
#                     "decision_reason": chunks[i]["decision_reason"],
#                     "source_sections": " | ".join(chunks[i]["source_sections"])
#                 }
#                 chunk_text = chunks[i]["text"]
#             elif strategy == "content_aware":
#                 meta = {
#                     **base_metadata,
#                     "chunk_type": "content_aware",
#                     "section_title": chunks[i]["section_title"],
#                     "section_index": chunks[i]["section_index"],
#                     "chunk_index_in_section": chunks[i]["chunk_index_in_section"]
#                 }
#                 chunk_text = chunks[i]["text"]
#             else:
#                 meta = {**base_metadata, "chunk_type": strategy}
#                 chunk_text = texts[i]

#             collection.add(
#                  texts=[chunk_text],   
#                 embeddings=[emb],
#                 metadatas=[meta],
#                 ids=[str(uuid.uuid4())]
#             )       
#             total_chunks += 1

#     return total_chunks

# # ==============================
# # ENTRY POINT
# # ==============================

# if __name__ == "__main__":

#     pdf_path = "demo pdf (2).pdf"
#     strategy = "agentic"  # fixed | sentence | paragraph | recursive | semantic | agentic | content_aware
#     vector_db = "qdrant"   # chroma | faiss | qdrant | milvus | pinecone | weaviate

#     print("🧠 Loading embedding model...")
#     embedder = load_embedding_model()

#     print(f"🗄️ Initializing Vector DB: {vector_db}")
#     vector_store = init_vector_store(vector_db, embedder)

#     print(f"📄 Ingesting PDF using '{strategy}' chunking...")
#     total = ingest_pdf(pdf_path, strategy, vector_store, embedder)

#     print("\n✅ INGESTION COMPLETE")
#     print(f"📄 PDF: {pdf_path}")
#     print(f"✂️ Chunking Strategy: {strategy}")
#     print(f"🗄️ Vector DB: {vector_db}")
#     print(f"📦 Total Chunks Stored: {total}")
#     print("🔢 Total vectors in DB:", vector_store.count())
#     print("🔒 Closing vector store...")
#     if hasattr(vector_store, "close"):
#         vector_store.close()        
###############################################################################################################################
import os
import uuid
import re
import pickle
import numpy as np
import chromadb
import nltk
import faiss
 
import weaviate
from weaviate.auth import AuthApiKey
from weaviate.classes.config import Property
from weaviate.classes.config import DataType as WeaviateDataType
from weaviate.classes.data import DataObject
from sklearn.metrics.pairwise import cosine_similarity
from pypdf import PdfReader
from nltk.tokenize import sent_tokenize
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection,utility
from pinecone import Pinecone, ServerlessSpec

# ==============================
# INITIAL SETUP
# ==============================

nltk.download("punkt")
nltk.download("punkt_tab")

# EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CHROMA_COLLECTION = "rag-chunks"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMA_PATH = os.path.join(BASE_DIR, "chroma_db")       # ----- chroma path
FAISS_PATH = os.path.join(BASE_DIR, "faiss_store.pkl")     # ----- faiss path
QDRANT_PATH = os.path.join(BASE_DIR, "qdrant_db")       # ----- qdrant path
QDRANT_COLLECTION = "rag_chunks"                          # ----- qdrant collection name
MILVUS_COLLECTION = "rag_chunks"                           # ----- milvus collection name
PINECONE_API_KEY = "pcsk_36kLSk_KyRnADaFcY1xF4dS7gyTKjCqBu5qHj8rYMmmr7RP6anZb56YmGZw1Z5BaM1Gznv"      # pinecone api key
PINECONE_ENV = "us-east-1"                                                                   # ---------------- pinecone environment
PINECONE_INDEX = "rag-chunks"                                            
WEAVIATE_URL = "https://pmh8cdktsmeloccbguom5a.c0.asia-southeast1.gcp.weaviate.cloud"                  # ---------------- weaviate url
WEAVIATE_API_KEY = "UGd4eU0yeC9sbWhNZm53Nl9STDIvSTY0K2puOTRUY05KZ1hNVEhYajhIbk9PTUVMKzluc3RERVI0WC9NPV92MjAw"   # ---------------- weaviate api key
WEAVIATE_CLASS = "RagChunks"
# ==============================
# AGENT RULES (CONFIG)
# ==============================

AGENT_RULES = {
    "technical_assistant": {
        "keep_keywords": [
            "architecture", "design", "deployment", "installation",
            "configuration", "setup", "implementation"
        ],
        "drop_keywords": [
            "pricing", "contact", "about", "company", "legal", "copyright"
        ],
        "merge_keywords": [
            "setup", "installation", "deployment"
        ],
        "max_section_length": 1500
    }
}

# ==============================
# EMBEDDING REGISTRY (PRODUCTION)
# ==============================

EMBEDDING_REGISTRY = {
    "bge-base": {
        "model_name": "BAAI/bge-base-en-v1.5",
        "provider": "huggingface",
        "type": "default"
    },
    "bge-m3": {
        "model_name": "BAAI/bge-m3",
        "provider": "huggingface",
        "type": "multilingual"
    },
    "e5-large": {
        "model_name": "intfloat/e5-large-v2",
        "provider": "huggingface",
        "type": "high_accuracy",
        "requires_prefix": True
    },
    "e5-multilingual": {
        "model_name": "intfloat/multilingual-e5-large",
        "provider": "huggingface",
        "type": "multilingual",
        "requires_prefix": True
    },
    "gte-large": {
        "model_name": "thenlper/gte-large",
        "provider": "huggingface",
        "type": "instruction"
    },
    "jina-v3": {
        "model_name": "jinaai/jina-embeddings-v3",
        "provider": "huggingface",
        "type": "infra_scale"
    },
    "nomic-v1.5": {
        "model_name": "nomic-ai/nomic-embed-text-v1.5",
        "provider": "huggingface",
        "type": "open_source"
    }
}

# ==============================
# PDF LOADER
# ==============================

def load_pdf(pdf_path):
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
# BASIC CHUNKING STRATEGIES
# ==============================

def fixed_chunking(text, chunk_size=500, overlap=50): # ----------------- fixed chunking function
    chunks, start = [], 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks


def sentence_chunking(text, max_sentences=5): # ------------------ sentence chunking function
    sentences = sent_tokenize(text, language="english")
    return [
        " ".join(sentences[i:i + max_sentences])
        for i in range(0, len(sentences), max_sentences)
    ]


def paragraph_chunking(text):          # ------------------ paragraph chunking function
    return [p.strip() for p in text.split("\n\n") if p.strip()]


def recursive_chunking(text, chunk_size=500, overlap=50):  # ------------------ recursive chunking function
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap
    )
    return splitter.split_text(text)

# ==============================
# SEMANTIC CHUNKING
# ==============================

def semantic_chunking(
    text,
    embedder,
    similarity_threshold=0.75,
    max_chunk_chars=800
):                                                             # ------------------ semantic chunking function
    sentences = sent_tokenize(text, language="english")
    if len(sentences) <= 1:
        return sentences

    sentence_embeddings = embedder.embed_documents(sentences)

    chunks = []
    current_chunk = [sentences[0]]
    current_len = len(sentences[0])

    for i in range(1, len(sentences)):
        sim = cosine_similarity(
            [sentence_embeddings[i - 1]],
            [sentence_embeddings[i]]
        )[0][0]

        if sim >= similarity_threshold and current_len < max_chunk_chars:
            current_chunk.append(sentences[i])
            current_len += len(sentences[i])
        else:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentences[i]]
            current_len = len(sentences[i])

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

# ==============================
# CONTENT-AWARE CHUNKING
# ==============================

def is_heading(line: str) -> bool:
    line = line.strip()
    if not line:
        return False
    if line.isupper() and len(line) < 100:
        return True
    if re.match(r"^\d+(\.\d+)*\s+", line):
        return True
    if line.endswith(":") and len(line) < 80:
        return True
    return False


def detect_sections(text: str):
    lines = text.split("\n")
    sections = []

    current = {
        "title": "INTRODUCTION",
        "content": [],
        "index": 0
    }

    section_index = 1

    for line in lines:
        if is_heading(line):
            if current["content"]:
                sections.append(current)

            current = {
                "title": line.strip(),
                "content": [],
                "index": section_index
            }
            section_index += 1
        else:
            current["content"].append(line)

    if current["content"]:
        sections.append(current)

    return sections


def content_aware_chunking(text, embedder, inner_strategy="recursive"):
    sections = detect_sections(text)
    chunks = []

    for section in sections:
        section_text = "\n".join(section["content"]).strip()
        if not section_text:
            continue

        if inner_strategy == "paragraph":
            inner_chunks = paragraph_chunking(section_text)
        elif inner_strategy == "semantic":
            inner_chunks = semantic_chunking(section_text, embedder)
        else:
            inner_chunks = recursive_chunking(section_text)

        for idx, chunk in enumerate(inner_chunks):
            chunks.append({
                "text": chunk,
                "section_title": section["title"],
                "section_index": section["index"],
                "chunk_index_in_section": idx
            })

    return chunks

# ==============================
# AGENTIC CHUNKING
# ==============================

def evaluate_section(section, agent_goal):
    rules = AGENT_RULES[agent_goal]
    title = section["title"].lower()
    content = " ".join(section["content"]).lower()

    for kw in rules["drop_keywords"]:
        if kw in title:
            return "DROP", f"Irrelevant section: {section['title']}"

    for kw in rules["merge_keywords"]:
        if kw in title:
            return "MERGE", f"Merged operational section: {section['title']}"

    if len(content) > rules["max_section_length"]:
        return "SPLIT", "Section too large"

    for kw in rules["keep_keywords"]:
        if kw in title or kw in content:
            return "KEEP", "Relevant to agent goal"

    return "KEEP", "Default keep"


def agentic_chunking(
    text,
    embedder,
    agent_goal="technical_assistant",
    inner_strategy="recursive"
):
    sections = detect_sections(text)
    chunks = []
    merge_buffer = []

    for section in sections:
        decision, reason = evaluate_section(section, agent_goal)
        section_text = "\n".join(section["content"]).strip()

        if not section_text:
            continue

        if decision == "DROP":
            continue

        if decision == "MERGE":
            merge_buffer.append(section)
            continue

        if merge_buffer:
            merged_text = "\n\n".join(
                ["\n".join(s["content"]) for s in merge_buffer]
            )
            chunks.append({
                "text": merged_text,
                "decision": "MERGE",
                "decision_reason": "Merged related sections",
                "agent_goal": agent_goal,
                "source_sections": [s["title"] for s in merge_buffer]
            })
            merge_buffer = []

        if decision == "SPLIT":
            sub_chunks = (
                semantic_chunking(section_text, embedder)
                if inner_strategy == "semantic"
                else recursive_chunking(section_text)
            )
            for idx, sub in enumerate(sub_chunks):
                chunks.append({
                    "text": sub,
                    "decision": "SPLIT",
                    "decision_reason": reason,
                    "agent_goal": agent_goal,
                    "source_sections": [section["title"]],
                    "chunk_index_in_section": idx
                })

        if decision == "KEEP":
            chunks.append({
                "text": section_text,
                "decision": "KEEP",
                "decision_reason": reason,
                "agent_goal": agent_goal,
                "source_sections": [section["title"]]
            })

    return chunks

# ==============================
# EMBEDDING MODEL
# ==============================

def load_embedding_model(embedding_key: str):
    if embedding_key not in EMBEDDING_REGISTRY:
        raise ValueError(f"Unsupported embedding model: {embedding_key}")

    config = EMBEDDING_REGISTRY[embedding_key]

    if config["provider"] == "huggingface":
        return HuggingFaceEmbeddings(
            model_name=config["model_name"]
        )

    raise ValueError("Unsupported embedding provider")

# ==============================
# helper function for E5
# ==============================
def prepare_texts_for_embedding(texts, embedding_key, is_query=False):
    config = EMBEDDING_REGISTRY[embedding_key]

    if config.get("requires_prefix"):
        prefix = "query: " if is_query else "passage: "
        return [prefix + t for t in texts]

    return texts


# ==============================
# VECTOR STORE ADAPTERS
# ==============================

class VectorStore:
    def add(self, texts, embeddings, metadatas, ids):
        raise NotImplementedError

    def count(self):
        raise NotImplementedError


class ChromaStore(VectorStore):
    def __init__(self,suffix):
        os.makedirs(CHROMA_PATH, exist_ok=True)
        self.client = chromadb.PersistentClient(path=CHROMA_PATH)
        self.collection = self.client.get_or_create_collection(name=f"rag_chunks__{suffix}")

    def add(self, texts, embeddings, metadatas, ids):
        self.collection.add(
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )

    def count(self):
        return self.collection.count()


class FaissStore(VectorStore):
    def __init__(self, dim, suffix):
        self.index = faiss.IndexFlatL2(dim)
        self.path = os.path.join(BASE_DIR, f"faiss_store__{suffix}.pkl")
        self.data = []

        if os.path.exists(self.path):
            with open(self.path, "rb") as f:
                saved = pickle.load(f)
                self.index = saved["index"]
                self.data = saved["data"]

    def add(self, texts, embeddings, metadatas, ids):
        vectors = np.array(embeddings).astype("float32")
        self.index.add(vectors)

        for i in range(len(texts)):
            self.data.append({
                "id": ids[i],
                "text": texts[i],
                "metadata": metadatas[i]
            })

        with open(self.path, "wb") as f:
            pickle.dump({"index": self.index, "data": self.data}, f)

    def count(self):
        return self.index.ntotal


class QdrantStore(VectorStore):
    def __init__(self, dim, suffix):
        self.client = QdrantClient(path=QDRANT_PATH)
        self.collection_name = f"rag_chunks__{suffix}"

        existing = [c.name for c in self.client.get_collections().collections]
        if self.collection_name not in existing:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=dim,
                    distance=Distance.COSINE
                )
            )

    def add(self, texts, embeddings, metadatas, ids):
        points = []
        for i in range(len(texts)):
            points.append(
                PointStruct(
                    id=ids[i],
                    vector=embeddings[i],
                    payload={
                        "text": texts[i],
                        **metadatas[i]
                    }
                )
            )

        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )

    def count(self):
        return self.client.count(self.collection_name).count


    def close(self):
        self.client.close()



# # ==============================
# # ✅ MILVUS STORE (ADDED)
# # ==============================

# class MilvusStore(VectorStore):
#     def __init__(self, dim,suffix):
#         connections.connect(
#             alias="default",
#             host="localhost",
#             port="19530"
#         )

#         if not utility.has_collection(MILVUS_COLLECTION):
#             fields = [
#                 FieldSchema(
#                     name="id",
#                     dtype=DataType.VARCHAR,
#                     is_primary=True,
#                     max_length=64
#                 ),
#                 FieldSchema(
#                     name="embedding",
#                     dtype=DataType.FLOAT_VECTOR,
#                     dim=dim
#                 ),
#                 FieldSchema(
#                     name="text",
#                     dtype=DataType.VARCHAR,
#                     max_length=65535
#                 ),
#                 FieldSchema(
#                     name="metadata",
#                     dtype=DataType.JSON
#                 )
#             ]

#             schema = CollectionSchema(
#                 fields=fields,
#                 description="RAG chunks"
#             )

#             self.collection = Collection(
#                 name=MILVUS_COLLECTION,
#                 schema=schema
#             )

#             self.collection.create_index(
#                 field_name="embedding",
#                 index_params={
#                     "index_type": "HNSW",
#                     "metric_type": "COSINE",
#                     "params": {"M": 8, "efConstruction": 64}
#                 }
#             )
#         else:
#             self.collection = Collection(MILVUS_COLLECTION)

#         self.collection.load()

#     def add(self, texts, embeddings, metadatas, ids):
#         self.collection.insert([
#             ids,
#             embeddings,
#             texts,
#             metadatas
#         ])

#     def count(self):
#         return self.collection.num_entities

#     def close(self):
#         self.collection.release()
#         connections.disconnect("default")


# ==============================
# ✅ Pine-cone
# ==============================
class PineconeStore(VectorStore):
    def __init__(self, dim, suffix):
        self.pc = Pinecone(api_key=PINECONE_API_KEY)
        index_name = f"rag-chunks-{suffix}"

        existing_indexes = [idx["name"] for idx in self.pc.list_indexes()]
        if index_name not in existing_indexes:
            self.pc.create_index(
                name=index_name,
                dimension=dim,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region=PINECONE_ENV
                )
            )

        self.index = self.pc.Index(index_name)

    def add(self, texts, embeddings, metadatas, ids):
        vectors = []
        for i in range(len(texts)):
            vectors.append((
                ids[i],
                embeddings[i],
                {
                    "text": texts[i],
                    **metadatas[i]
                }
            ))

        self.index.upsert(vectors=vectors)

    def count(self):
        stats = self.index.describe_index_stats()
        return stats["total_vector_count"]
    


# ==============================
# ✅ Weaviate STORE
# ==============================
class WeaviateStore(VectorStore):
    def __init__(self, dim, suffix):
        # 1️⃣ Connect to Weaviate Cloud (v4)
        self.client = weaviate.connect_to_weaviate_cloud(
            cluster_url=WEAVIATE_URL.replace("https://", ""),
            auth_credentials=AuthApiKey(WEAVIATE_API_KEY),
        )
        class_name = f"RagChunks_{suffix.replace('-', '_')}"
        print("🔍 [Weaviate] Client created")

        ready = self.client.is_ready()
        print("🔍 [Weaviate] is_ready():", ready)

        if not ready:
            raise RuntimeError("Weaviate is not ready")
        existing = self.client.collections.list_all()      
        if class_name not in existing:
            self.client.collections.create(
                      name=class_name,
                         properties=[
                            Property(name="text", data_type=WeaviateDataType.TEXT),
                            Property(name="source", data_type=WeaviateDataType.TEXT),
                            Property(name="page", data_type=WeaviateDataType.INT),
                        ],
                        )
        else:
            print(f"✅ [Weaviate] Collection already exists: {class_name}")
        # 5️⃣ Get collection handle
        self.collection = self.client.collections.get(class_name)

    def add(self, texts, embeddings, metadatas, ids):
        
        objects = []
        for i in range(len(texts)):
             objects.append(
            DataObject(
                properties={
                    "text": texts[i],
                    **metadatas[i]
                },
                vector=embeddings[i],
                uuid=ids[i]
            )
        )
        self.collection.data.insert_many(objects)
    def count(self):
        count = self.collection.aggregate.over_all(
            total_count=True
        ).total_count

        print("🔢 [Weaviate] Total vectors:", count)
        return count
    
    def close(self):
        print("🔒 [Weaviate] Closing client")
        self.client.close()



def init_vector_store(db_type, embedder, embedding_key):
    dim = len(embedder.embed_query("test"))
    suffix = embedding_key
    if db_type == "chroma":
        return ChromaStore(suffix)
    elif db_type == "faiss":
        return FaissStore(dim, suffix)
    elif db_type == "qdrant":
        return QdrantStore(dim, suffix)
    # elif db_type == "milvus":          
    #     return MilvusStore(dim,suffix)
    elif db_type == "pinecone":      
        return PineconeStore(dim, suffix)
    elif db_type == "weaviate":
        return WeaviateStore(dim, suffix)
    else:
        raise ValueError("Unsupported DB")

# ==============================
# INGESTION PIPELINE
# ==============================

def ingest_pdf(pdf_path, strategy, collection, embedder, embedding_key):
    documents = load_pdf(pdf_path)
    total_chunks = 0

    for doc in documents:
        text = doc["text"]
        base_metadata = doc["metadata"]

        if strategy == "agentic":
            chunks = agentic_chunking(text, embedder)
            texts = [c["text"] for c in chunks]
        elif strategy == "content_aware":
            chunks = content_aware_chunking(text, embedder)
            texts = [c["text"] for c in chunks]
        elif strategy == "semantic":
            chunks = semantic_chunking(text, embedder)
            texts = chunks
        elif strategy == "sentence":
            chunks = sentence_chunking(text)
            texts = chunks
        elif strategy == "paragraph":
            chunks = paragraph_chunking(text)
            texts = chunks
        elif strategy == "fixed":
            chunks = fixed_chunking(text)
            texts = chunks
        else:
            chunks = recursive_chunking(text)
            texts = chunks

        prepared_texts = prepare_texts_for_embedding(
             texts,
             embedding_key,
            is_query=False
            )

        embeddings = embedder.embed_documents(prepared_texts)
        for i, emb in enumerate(embeddings):
            if strategy == "agentic":
                meta = {
                    **base_metadata,
                    "chunk_type": "agentic",
                    "agent_goal": chunks[i]["agent_goal"],
                    "decision": chunks[i]["decision"],
                    "decision_reason": chunks[i]["decision_reason"],
                    "source_sections": " | ".join(chunks[i]["source_sections"])
                }
                chunk_text = chunks[i]["text"]
            elif strategy == "content_aware":
                meta = {
                    **base_metadata,
                    "chunk_type": "content_aware",
                    "section_title": chunks[i]["section_title"],
                    "section_index": chunks[i]["section_index"],
                    "chunk_index_in_section": chunks[i]["chunk_index_in_section"]
                }
                chunk_text = chunks[i]["text"]
            else:
                meta = {**base_metadata, "chunk_type": strategy}
                chunk_text = texts[i]

            collection.add(
                 texts=[chunk_text],   
                embeddings=[emb],
                metadatas=[meta],
                ids=[str(uuid.uuid4())]
            )       
            total_chunks += 1

    return total_chunks

# ==============================
# ENTRY POINT
# ==============================

if __name__ == "__main__":

    pdf_path = "demo pdf (2).pdf"
    strategy = "agentic"         # fixed | sentence | paragraph | recursive | semantic | agentic | content_aware
    vector_db = "qdrant"         # chroma | faiss | qdrant | milvus | pinecone | weaviate
    embedding_key = "bge-m3"#    # bge-base | bge-m3 | e5-large | e5-multilingual | gte-large | jina-v3 | nomic-v1.5 
    print("🧠 Loading embedding model...")
    embedder = load_embedding_model(embedding_key)
    print(f"🧠 Embedding model loaded: {embedding_key}")
    print(f"🗄️ Initializing Vector DB: {vector_db}")
    vector_store = init_vector_store(vector_db, embedder,embedding_key)

    print(f"📄 Ingesting PDF using '{strategy}' chunking...")
    total = ingest_pdf(pdf_path, strategy, vector_store, embedder,embedding_key)

    print("\n✅ INGESTION COMPLETE")
    print(f"📄 PDF: {pdf_path}")
    print(f"✂️ Chunking Strategy: {strategy}")
    print(f"🗄️ Vector DB: {vector_db}")
    print(f"📦 Total Chunks Stored: {total}")
    print("🔢 Total vectors in DB:", vector_store.count())
    print("🔒 Closing vector store...")
    if hasattr(vector_store, "close"):
        vector_store.close()        