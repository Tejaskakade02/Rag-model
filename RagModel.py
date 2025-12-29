import ollama

# Load dataset
dataset = []
with open('cat-facts.txt', 'r', encoding="utf-8") as file:
    dataset = file.readlines()
    print(f"Loaded {len(dataset)} Entries")

# Define models
EMBEDDING_MODEL = "nomic-embed-text"
LANGUAGE_MODEL = 'llama3.2:1b'

# Vector database
VECTOR_DB = []

def add_chunks_to_database(chunk):
    embedding = ollama.embed(model=EMBEDDING_MODEL, input=chunk)['embeddings'][0]
    VECTOR_DB.append((chunk, embedding))

for i, chunk in enumerate(dataset):
    add_chunks_to_database(chunk)
print(f"Added {i + 1}/{len(dataset)} to database")

# Cosine similarity
def cosine_similarity(a, b):
    dot_product = sum([x * y for x, y in zip(a, b)])
    norm_a = sum([x ** 2 for x in a]) ** 0.5
    norm_b = sum([y ** 2 for y in b]) ** 0.5
    return dot_product / (norm_a * norm_b)

# Retrieval function
def retrieve(query, top_n=3):
    query_embedding = ollama.embed(model=EMBEDDING_MODEL, input=query)['embeddings'][0]
    similarities = []
    for chunk, embedding in VECTOR_DB:
        similarity = cosine_similarity(query_embedding, embedding)
        similarities.append((chunk, similarity))
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_n]

# === 🔁 Chat loop ===
print("\nYou can now ask questions about cats! (Press Ctrl+C to exit)\n")

while True:
    try:
        # Chatbot input
        input_query = input("Ask your question: ").strip()
        if input_query == "":
            print("Please enter a question.\n")
            continue

        # Retrieve relevant knowledge
        retrieved_knowledge = retrieve(input_query)
        print("\nRetrieved knowledge:")
        for chunk, similarity in retrieved_knowledge:
            print(f" - (similarity: {similarity:.2f}) {chunk}")

        # Build prompt
        context_text = '\n'.join([f' - {chunk.strip()}' for chunk, _ in retrieved_knowledge])
        instruction_prompt = f'''You are a helpful chatbot.
Use only the following pieces of context to answer the question. Don't make up any new information:
{context_text}
'''

        # Chat with Ollama
        stream = ollama.chat(
            model=LANGUAGE_MODEL,
            messages=[
                {'role': 'system', 'content': instruction_prompt},
                {'role': 'user', 'content': input_query}
            ],
            stream=True
        )

        print("\nChatbot Response:")
        for chunk in stream:
            print(chunk['message']['content'], end='', flush=True)

    except KeyboardInterrupt:
        print("\n\n🔚 Chat ended by user. Goodbye!")
        break
