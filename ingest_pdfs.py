from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from pypdf import PdfReader
from dotenv import load_dotenv
import os
import uuid

load_dotenv()

# --- Connect ---
client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY"),
)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# --- Read all PDFs from docs/ folder ---
def read_pdfs(folder="docs"):
    all_chunks = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

    for filename in os.listdir(folder):
        if filename.endswith(".pdf"):
            filepath = os.path.join(folder, filename)
            print(f"Reading: {filename}")

            reader = PdfReader(filepath)
            full_text = ""
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    full_text += text + "\n"

            chunks = splitter.split_text(full_text)
            for chunk in chunks:
                all_chunks.append({
                    "text": chunk,
                    "source": filename
                })
            print(f"  → {len(chunks)} chunks extracted")

    return all_chunks

# --- Ingest into Qdrant ---
def ingest(chunks):
    print(f"\nEmbedding {len(chunks)} chunks...")
    texts = [c["text"] for c in chunks]
    vectors = embeddings.embed_documents(texts)

    # Recreate collection fresh
    client.recreate_collection(
        collection_name="financial_docs",
        vectors_config=VectorParams(size=384, distance=Distance.COSINE),
    )

    points = [
        PointStruct(
            id=str(uuid.uuid4()),
            vector=vectors[i],
            payload={
                "text": chunks[i]["text"],
                "source": chunks[i]["source"]
            }
        )
        for i in range(len(chunks))
    ]

    # Upload in batches of 100
    batch_size = 100
    for i in range(0, len(points), batch_size):
        batch = points[i:i+batch_size]
        client.upsert(collection_name="financial_docs", points=batch)
        print(f"  Uploaded batch {i//batch_size + 1}")

    print(f"\n✅ {len(chunks)} chunks ingested successfully!")

# --- Run ---
chunks = read_pdfs("docs")
print(f"\nTotal chunks from all PDFs: {len(chunks)}")
ingest(chunks)