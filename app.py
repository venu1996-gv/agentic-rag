import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from qdrant_client import QdrantClient
from dotenv import load_dotenv
import os

os.getenv()

st.set_page_config(page_title="Agentic RAG", page_icon="🤖")
st.title("🤖 Agentic RAG — Financial Assistant")
st.caption("Ask questions about Q1 2024 company earnings")

# --- Setup ---
@st.cache_resource
def setup():
    client = QdrantClient(
        url=os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY"),
    )
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=os.getenv("GROQ_API_KEY"))
    return client, embeddings, llm

client, embeddings, llm = setup()

# --- Functions ---
def retrieve(query, k=3):
    query_vector = embeddings.embed_query(query)
    results = client.query_points(
        collection_name="financial_docs",
        query=query_vector,
        limit=k,
    ).points
    return results

def ask(question):
    results = retrieve(question)
    context = ""
    sources = []
    for r in results:
        context += f"[{r.payload['source']}]: {r.payload['text']}\n\n"
        sources.append(r.payload['source'])

    prompt = f"""You are a financial analyst assistant.
Answer the question using ONLY the context below.
If the context doesn't contain the answer, say "I don't have enough information."
Always mention which source you used.

Context:
{context}

Question: {question}

Answer:"""

    response = llm.invoke(prompt)
    return response.content, sources

# --- Chat UI ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask about Q1 2024 earnings..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer, sources = ask(prompt)
        st.markdown(answer)
        st.caption(f"Sources: {', '.join(sources)}")

    st.session_state.messages.append({"role": "assistant", "content": answer})