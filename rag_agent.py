from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from qdrant_client import QdrantClient
from dotenv import load_dotenv
import os

load_dotenv()

client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY"),
)

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=os.getenv("GROQ_API_KEY"))

# --- Retrieve ---
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
    for r in results:
        context += f"[{r.payload['source']}]: {r.payload['text']}\n\n"
    return context

# --- Self-Evaluator ---
def evaluate(question, answer, context):
    eval_prompt = f"""You are an evaluator. Given a question, context, and answer, rate the answer quality.

Question: {question}
Context: {context}
Answer: {answer}

Rate on two things:
1. Is the answer grounded in the context? (yes/no)
2. Does the context contain enough info to answer? (yes/no)

Reply in exactly this format:
GROUNDED: yes/no
SUFFICIENT: yes/no
SCORE: 0-10"""

    response = llm.invoke(eval_prompt)
    text = response.content
    print(f"\n--- Evaluator says ---\n{text}")

    grounded = "yes" in text.lower().split("grounded:")[1].split("\n")[0]
    sufficient = "yes" in text.lower().split("sufficient:")[1].split("\n")[0]
    try:
        score = int(text.lower().split("score:")[1].strip().split("\n")[0])
    except:
        score = 5

    return grounded, sufficient, score

# --- Rewrite query ---
def rewrite_query(original_query):
    prompt = f"""Rewrite this search query to be more specific and detailed:
Original: {original_query}
Rewritten:"""
    response = llm.invoke(prompt)
    return response.content.strip()

# --- Agentic RAG with self-correction ---
def agentic_ask(question, max_retries=2):
    print(f"\n{'='*60}")
    print(f"Question: {question}")
    print('='*60)

    current_query = question
    attempt = 0

    while attempt <= max_retries:
        print(f"\n[Attempt {attempt + 1}] Retrieving...")

        # Retrieve
        results = retrieve(current_query)
        context = format_context(results)

        # Generate answer
        prompt = f"""You are a financial analyst assistant.
Answer the question using ONLY the context below.
If the context doesn't contain the answer, say "I don't have enough information."
Always mention which source you used.

Context:
{context}

Question: {question}

Answer:"""
        response = llm.invoke(prompt)
        answer = response.content

        print(f"\n[Draft Answer]\n{answer}")

        # Evaluate
        grounded, sufficient, score = evaluate(question, answer, context)
        print(f"\nScore: {score}/10 | Grounded: {grounded} | Sufficient: {sufficient}")

        # Decision
        if score >= 7 and grounded:
            print(f"\n✅ Answer accepted after {attempt + 1} attempt(s)")
            break
        elif attempt < max_retries:
            print(f"\n🔄 Low confidence — rewriting query and retrying...")
            current_query = rewrite_query(question)
            print(f"New query: {current_query}")
        else:
            print(f"\n⚠️ Max retries reached — returning best answer")

        attempt += 1

    print(f"\n{'='*60}")
    print(f"FINAL ANSWER:\n{answer}")
    print('='*60)
    return answer

# --- Test ---
agentic_ask("Which cloud company had the best Q1 2024 performance?")