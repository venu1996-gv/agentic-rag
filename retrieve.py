from langchain_huggingface import HuggingFaceEmbeddings
from qdrant_client import QdrantClient
from dotenv import load_dotenv
import os

load_dotenv()

client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY"),
)

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

query = "How did cloud companies perform in Q1 2024?"
query_vector = embeddings.embed_query(query)

results = client.query_points(
    collection_name="financial_docs",
    query=query_vector,
    limit=3,
).points

print(f"\nQuery: {query}\n")
print("Top results:")
for i, result in enumerate(results):
    print(f"\n[{i+1}] Score: {result.score:.4f}")
    print(f"    {result.payload['text']}")