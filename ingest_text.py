from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from dotenv import load_dotenv
import os
import uuid

load_dotenv()

client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY"),
)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

all_chunks = []

for filename in os.listdir("docs_text"):
    filepath = os.path.join("docs_text", filename)
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()

    if len(text) < 1000:
        print(f"Skipping {filename} — too short ({len(text)} chars)")
        continue

    chunks = splitter.split_text(text)
    for chunk in chunks:
        all_chunks.append({"text": chunk, "source": filename})
    print(f"✅ {filename} — {len(chunks)} chunks")

print(f"\nTotal chunks: {len(all_chunks)}")

# Embed and upload
texts = [c["text"] for c in all_chunks]
vectors = embeddings.embed_documents(texts)

client.recreate_collection(
    collection_name="financial_docs",
    vectors_config=VectorParams(size=384, distance=Distance.COSINE),
)

points = [
    PointStruct(
        id=str(uuid.uuid4()),
        vector=vectors[i],
        payload={"text": all_chunks[i]["text"], "source": all_chunks[i]["source"]}
    )
    for i in range(len(all_chunks))
]

for i in range(0, len(points), 100):
    client.upsert(collection_name="financial_docs", points=points[i:i+100])
    print(f"Uploaded batch {i//100 + 1}")

print(f"\n✅ Done! {len(all_chunks)} chunks ingested!")