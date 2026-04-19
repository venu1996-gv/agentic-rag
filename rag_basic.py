from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from qdrant_client import QdrantClient
from dotenv import load_dotenv
import os

load_dotenv()

# --- Setup ---
client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY"),
)

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=os.getenv("GROQ_API_KEY"))

# --- Retrieve function ---
def retrieve(query, k=3):
    query_vector = embeddings.embed_query(query)
    results = client.query_points(
        collection_name="financial_docs",
        query=query_vector,
        limit=k,
    ).points
    return results

# --- Format context ---
def format_context(results):
    context = ""
    for i, r in enumerate(results):
        context += f"[{r.payload['source']}]: {r.payload['text']}\n\n"
    return context

# --- RAG function ---
def ask(question):
    print(f"\nQuestion: {question}")
    print("-" * 50)

    # Step 1: Retrieve
    results = retrieve(question)
    context = format_context(results)
    print(f"Retrieved {len(results)} chunks")

    # Step 2: Generate
    prompt = f"""You are a financial analyst assistant.
Answer the question using ONLY the context below.
If the context doesn't contain the answer, say "I don't have enough information."
Always mention which source you used.

Context:
{context}

Question: {question}

Answer:"""

    response = llm.invoke(prompt)
    print(f"\nAnswer:\n{response.content}")
    return response.content

# --- Test it ---
ask("Which cloud company had the best Q1 2024 performance?")
ask("How did Tesla perform in Q1 2024?")