from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from dotenv import load_dotenv
import os
import uuid

load_dotenv()

# --- Documents ---
raw_docs = [
    "Apple reported revenue of $89.5 billion in Q1 2024, driven by strong iPhone sales.",
    "Tesla's Q1 2024 earnings showed a decline in margins due to price cuts across all models.",
    "Microsoft Azure grew 28% year-over-year, contributing significantly to cloud revenue.",
    "Google's advertising revenue rebounded strongly in Q1 2024 after a slow 2023.",
    "Amazon Web Services posted $25 billion in Q1 2024, beating analyst expectations.",
]

documents = [Document(page_content=text, metadata={"source": f"doc_{i}"}) for i, text in enumerate(raw_docs)]

splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
chunks = splitter.split_documents(documents)
print(f"Total chunks: {len(chunks)}")

# --- Embeddings ---
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
texts = [chunk.page_content for chunk in chunks]
vectors = embeddings.embed_documents(texts)
print(f"Embeddings created: {len(vectors)}")

# --- Connect to Qdrant Cloud ---
client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY"),
)

# --- Create collection ---
client.recreate_collection(
    collection_name="financial_docs",
    vectors_config=VectorParams(size=384, distance=Distance.COSINE),
)

# --- Upload vectors ---
points = [
    PointStruct(
        id=str(uuid.uuid4()),
        vector=vectors[i],
        payload={"text": chunks[i].page_content, "source": chunks[i].metadata["source"]}
    )
    for i in range(len(chunks))
]

client.upsert(collection_name="financial_docs", points=points)
print("Documents ingested into Qdrant Cloud successfully!")