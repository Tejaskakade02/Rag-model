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
import os
import uuid
import re
import chromadb
import nltk
from sklearn.metrics.pairwise import cosine_similarity

from PyPDF2 import PdfReader
from nltk.tokenize import sent_tokenize
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

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

def load_embedding_model():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

# ==============================
# CHROMADB INITIALIZATION
# ==============================

def init_chroma():
    os.makedirs(CHROMA_PATH, exist_ok=True)
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_or_create_collection(name=CHROMA_COLLECTION)
    return client, collection

# ==============================
# INGESTION PIPELINE
# ==============================

def ingest_pdf(pdf_path, strategy, collection, embedder):
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

        embeddings = embedder.embed_documents(texts)

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
                documents=[chunk_text],
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
    strategy = "agentic"
    # fixed | sentence | paragraph | recursive | semantic | content_aware | agentic

    print("🚀 Initializing ChromaDB...")
    client, collection = init_chroma()

    print("🧠 Loading embedding model...")
    embedder = load_embedding_model()

    print(f"📄 Ingesting PDF using '{strategy}' chunking...")
    total = ingest_pdf(pdf_path, strategy, collection, embedder)

    print("\n✅ INGESTION COMPLETE")
    print(f"📄 PDF: {pdf_path}")
    print(f"✂️ Chunking Strategy: {strategy}")
    print(f"📦 Total Chunks Stored: {total}")
    print(f"🗄️ Stored at: {CHROMA_PATH}")
    print("🔢 Total vectors in DB:", collection.count())
