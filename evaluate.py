from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from qdrant_client import QdrantClient
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy
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
    return [r.payload["text"] for r in results]

# --- Generate answer ---
def generate(question, contexts):
    context_text = "\n\n".join(contexts)
    prompt = f"""You are a financial analyst assistant.
Answer the question using ONLY the context below.
If the context doesn't contain the answer, say "I don't have enough information."

Context:
{context_text}

Question: {question}

Answer:"""
    response = llm.invoke(prompt)
    return response.content

# --- Test questions ---
test_questions = [
    "What were Tesla's total revenues for three months ended March 31 2024?",
    "What was Tesla's net income in Q2 2024?",
    "What risks did Tesla mention in their 2024 filings?",
    "What was Apple's iPhone revenue in Q1 2024?",
    "How did Tesla's automotive revenue change in 2024?",
]

# --- Build evaluation dataset ---
print("Building evaluation dataset...\n")

questions = []
answers = []
contexts = []

for question in test_questions:
    print(f"Processing: {question[:60]}...")
    retrieved = retrieve(question)
    answer = generate(question, retrieved)
    questions.append(question)
    answers.append(answer)
    contexts.append(retrieved)
    print(f"Answer: {answer[:100]}...\n")

# --- Run RAGAS evaluation ---
print("\nRunning RAGAS evaluation...")

data = {
    "question": questions,
    "answer": answers,
    "contexts": contexts,
}

dataset = Dataset.from_dict(data)

results = evaluate(
    dataset=dataset,
    metrics=[faithfulness, answer_relevancy],
    llm=llm,
    embeddings=embeddings,
)

# --- Display clean results ---
import numpy as np

faithfulness_scores = list(results['faithfulness'])
relevancy_scores = list(results['answer_relevancy'])

f_scores_clean = [float(s) for s in faithfulness_scores if str(s) != 'nan']
r_scores_clean = [float(s) for s in relevancy_scores if str(s) != 'nan']

f_avg = sum(f_scores_clean) / len(f_scores_clean) if f_scores_clean else 0
r_avg = sum(r_scores_clean) / len(r_scores_clean) if r_scores_clean else 0

print("\n" + "="*50)
print("EVALUATION RESULTS")
print("="*50)
print(f"Faithfulness Score:  {f_avg:.2%}")
print(f"Answer Relevancy:    {r_avg:.2%}")
print("="*50)
print("\nPer question faithfulness:")
for i, (q, s) in enumerate(zip(test_questions, faithfulness_scores)):
    print(f"  Q{i+1}: {s:.2%} — {q[:50]}")
print("\nResume bullet point:")
print(f"Achieved {f_avg:.0%} faithfulness score (RAGAS) on real Tesla and Apple SEC 10-Q filings")