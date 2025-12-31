# import ollama

# # Load dataset
# dataset = []
# with open('cat-facts.txt', 'r', encoding="utf-8") as file:
#     dataset = file.readlines()
#     print(f"Loaded {len(dataset)} Entries")

# # Define models
# EMBEDDING_MODEL = "nomic-embed-text"
# LANGUAGE_MODEL = 'llama3.2:1b'

# # Vector database
# VECTOR_DB = []

# def add_chunks_to_database(chunk):
#     embedding = ollama.embed(model=EMBEDDING_MODEL, input=chunk)['embeddings'][0]
#     VECTOR_DB.append((chunk, embedding))

# for i, chunk in enumerate(dataset):
#     add_chunks_to_database(chunk)
# print(f"Added {i + 1}/{len(dataset)} to database")

# # Cosine similarity
# def cosine_similarity(a, b):
#     dot_product = sum([x * y for x, y in zip(a, b)])
#     norm_a = sum([x ** 2 for x in a]) ** 0.5
#     norm_b = sum([y ** 2 for y in b]) ** 0.5
#     return dot_product / (norm_a * norm_b)

# # Retrieval function
# def retrieve(query, top_n=3):
#     query_embedding = ollama.embed(model=EMBEDDING_MODEL, input=query)['embeddings'][0]
#     similarities = []
#     for chunk, embedding in VECTOR_DB:
#         similarity = cosine_similarity(query_embedding, embedding)
#         similarities.append((chunk, similarity))
#     similarities.sort(key=lambda x: x[1], reverse=True)
#     return similarities[:top_n]

# # === 🔁 Chat loop ===
# print("\nYou can now ask questions about cats! (Press Ctrl+C to exit)\n")

# while True:
#     try:
#         # Chatbot input
#         input_query = input("Ask your question: ").strip()
#         if input_query == "":
#             print("Please enter a question.\n")
#             continue

#         # Retrieve relevant knowledge
#         retrieved_knowledge = retrieve(input_query)
#         print("\nRetrieved knowledge:")
#         for chunk, similarity in retrieved_knowledge:
#             print(f" - (similarity: {similarity:.2f}) {chunk}")

#         # Build prompt
#         context_text = '\n'.join([f' - {chunk.strip()}' for chunk, _ in retrieved_knowledge])
#         instruction_prompt = f'''You are a helpful chatbot.
# Use only the following pieces of context to answer the question. Don't make up any new information:
# {context_text}
# '''

#         # Chat with Ollama
#         stream = ollama.chat(
#             model=LANGUAGE_MODEL,
#             messages=[
#                 {'role': 'system', 'content': instruction_prompt},
#                 {'role': 'user', 'content': input_query}
#             ],
#             stream=True
#         )

#         print("\nChatbot Response:")
#         for chunk in stream:
#             print(chunk['message']['content'], end='', flush=True)

#     except KeyboardInterrupt:
#         print("\n\n🔚 Chat ended by user. Goodbye!")
#         break
##############################################################################################################################
import ollama
import pdfplumber
import math

# ==============================
# CONFIG
# ==============================
PDF_PATH = "demo pdf (2).pdf"
EMBEDDING_MODEL = "nomic-embed-text"
LANGUAGE_MODEL = "llama3.2:1b"

CHUNK_SIZE = 500     # characters
CHUNK_OVERLAP = 100

# ==============================
# STEP 1: LOAD PDF SAFELY
# ==============================
documents = []

with pdfplumber.open(PDF_PATH) as pdf:
    for page_no, page in enumerate(pdf.pages):
        text = page.extract_text()
        if text and text.strip():
            documents.append({
                "page": page_no + 1,
                "text": text.strip()
            })

print(f"Loaded {len(documents)} valid pages")

# ==============================
# STEP 2: CHUNKING FUNCTION
# ==============================
def chunk_text(text, size=500, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        start += size - overlap
    return chunks

chunks = []

for doc in documents:
    page_chunks = chunk_text(doc["text"])
    for c in page_chunks:
        if c.strip():
            chunks.append({
                "page": doc["page"],
                "text": c.strip()
            })

print(f"Created {len(chunks)} chunks")

# ==============================
# STEP 3: VECTOR DATABASE
# ==============================
VECTOR_DB = []

def embed_text(text):
    response = ollama.embed(
        model=EMBEDDING_MODEL,
        input=text
    )
    embeddings = response.get("embeddings", [])
    if not embeddings:
        return None
    return embeddings[0]

for chunk in chunks:
    embedding = embed_text(chunk["text"])
    if embedding is not None:
        VECTOR_DB.append({
            "page": chunk["page"],
            "text": chunk["text"],
            "embedding": embedding,
            "is_metadata": chunk["page"] <= 6 
        })

print(f"Stored {len(VECTOR_DB)} embeddings")

# ==============================
# STEP 4: COSINE SIMILARITY
# ==============================
def cosine_similarity(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)

# ==============================
# STEP 5: RETRIEVAL
# ==============================
def retrieve(query, top_k=3):
    query_embedding = embed_text(query)
    scores = []

    for item in VECTOR_DB:
        score = cosine_similarity(query_embedding, item["embedding"])

        # 🔥 Boost for publishing / edition questions
        if any(k in query.lower() for k in ["publish", "edition", "year", "isbn", "press"]):
            if item["is_metadata"]:
                score += 0.15

        scores.append((item, score))

    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_k]
# ==============================
# STEP 6: CHAT LOOP
# ==============================
print("\n📘 Ask questions about the PDF (Ctrl+C to exit)\n")

while True:
    try:
        query = input("❓ Question: ").strip()
        if not query:
            continue

        retrieved = retrieve(query)

        print("\n🔍 Retrieved context:")
        context_blocks = []

        for item, score in retrieved:
            print(f"Page {item['page']} | score: {score:.2f}")
            context_blocks.append(item["text"][:400])

        context_text = "\n\n".join(context_blocks)

        system_prompt = f"""
You are a document-grounded assistant.

Rules:
- Use ONLY the provided context.
- You MAY summarize or combine ideas that are clearly stated across the context.
- Do NOT add new facts or external knowledge.
- Answer by explicitly stating the editor’s observations or claims, not later scholars or external references.
- If the context does not provide enough information even after summarizing, say:
  "Not found in the document."
".

Context:
{context_text}
"""

        stream = ollama.chat(
            model=LANGUAGE_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ],
            stream=True
        )

        print("\n🤖 Answer:")
        for chunk in stream:
            print(chunk["message"]["content"], end="", flush=True)
        print("\n")

    except KeyboardInterrupt:
        print("\n👋 Chat ended.")
        break
