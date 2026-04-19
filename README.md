# Agentic RAG with Self-Correction

I built this project to learn how modern AI search systems work under the hood. Instead of just calling an API and hoping for the best, I wanted the system to actually *think* about whether its answer was good enough — and try again if it wasn't.

## What does it do?

You ask a question. The system searches through a set of documents to find relevant information, generates an answer, then evaluates its own answer. If the answer isn't confident enough, it rewrites the search query and tries again — automatically. Every answer comes with a source citation so you know where the information came from.

## Why I built it this way

Most RAG tutorials stop at "retrieve + generate." I kept asking — what happens when the retrieved context isn't good enough? What if the question was too vague? That's what led me to the self-correction loop. The agent now acts more like a researcher who double-checks their work before submitting it.

## Tools I used

I wanted to keep this completely free so anyone can run it:

| What | Tool |
|---|---|
| Storing and searching documents | Qdrant Cloud (free tier) |
| Turning text into vectors | all-MiniLM-L6-v2 from HuggingFace |
| Generating answers | Llama 3.3 70B via Groq (free) |
| Chat interface | Streamlit |
| Language | Python |

No OpenAI, no paid APIs. Everything here is free.

## How the self-correction loop works

This is the part I'm most proud of. Here's the exact flow:

1. You type a question
2. The system embeds your question and searches the vector database for the 3 most relevant chunks
3. Those chunks get passed to the LLM along with your question
4. The LLM generates an answer
5. A second LLM call evaluates the answer — is it grounded in the context? Is the context sufficient?
6. If the score is 7 or above, the answer gets returned
7. If the score is below 7, the query gets rewritten and the whole process repeats
8. It stops after 2 retries to avoid infinite loops

The interesting part is step 5 — using the LLM to evaluate itself. It's not perfect, but it catches a lot of cases where the retrieval missed the point of the question.

## Project structure

```
agentic-rag/
├── ingest.py       — loads documents, creates embeddings, stores in Qdrant
├── retrieve.py     — tests semantic search independently
├── rag_basic.py    — basic retrieve + generate pipeline
├── rag_agent.py    — full agentic loop with self-correction
└── app.py          — Streamlit chat UI
```

I built it in this order on purpose. Each file taught me something new before adding complexity in the next one.

## How to run it yourself

**1. Install dependencies**
```bash
pip install langchain-text-splitters langchain-community langchain-groq
pip install langchain-huggingface qdrant-client python-dotenv
pip install streamlit sentence-transformers groq
```

**2. Create a `.env` file**
```env
QDRANT_URL=your-qdrant-cluster-url
QDRANT_API_KEY=your-qdrant-api-key
GROQ_API_KEY=your-groq-api-key
```

You can get all three for free:
- Qdrant: https://cloud.qdrant.io
- Groq: https://console.groq.com

**3. Ingest your documents**
```bash
python ingest.py
```

**4. Run the app**
```bash
streamlit run app.py --server.port 8502
```

## Things I learned building this

- LangChain's abstractions break a lot between versions — sometimes it's easier to talk directly to the client library
- The self-evaluator prompt matters a lot. Vague instructions give inconsistent scores
- Qdrant's newer client API changed several method names from what most tutorials show
- Semantic search feels like magic until you understand it's just cosine similarity between vectors

## What I want to add next

- Ingest real PDF documents instead of hardcoded sample text
- Add RAGAS evaluation to measure retrieval and generation quality properly
- Rewrite the agent loop using LangGraph
- Deploy to Hugging Face Spaces so it runs without a local setup

## Author

Venu Gopal — building in public while learning AI/ML engineering