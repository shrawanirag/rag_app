# Notes RAG Chat

A simple RAG (Retrieval-Augmented Generation) app. Upload your notes/PDFs, ask questions,
and get answers grounded in your own documents — not the model's general knowledge.

## How it works

1. You upload PDF or TXT files.
2. The text is split into overlapping chunks (~300 words each).
3. Each chunk is embedded locally using `sentence-transformers` (no API cost for this step).
4. Embeddings are stored in a FAISS index (in-memory vector search).
5. When you ask a question, the app embeds your question, finds the most similar chunks,
   and sends them + your question to Gemini, which answers using only that context.

## Setup

1. **Install Python 3.11 or 3.12** (recommended — some dependencies here don't yet have
   prebuilt wheels for the very latest Python versions on Windows).

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate      # on Windows: venv\Scripts\Activate.ps1
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Get a free Gemini API key** from https://aistudio.google.com/apikey (you'll paste this
   into the app's sidebar when you run it — it's not stored anywhere by the app itself).

## Run it

```bash
streamlit run app.py
```

This opens the app in your browser (usually at `http://localhost:8501`).

## Using the app

1. Paste your Gemini API key into the sidebar.
2. Upload one or more PDF/TXT files (e.g. your DL lecture notes, lab reports).
3. Click **"Build knowledge base"** — this reads, chunks, and embeds your files.
4. Ask questions in the chat box at the bottom. Each answer shows an expandable
   section with the exact chunks that were retrieved, so you can see *why* it
   answered the way it did.

## Deploying it

See `DEPLOYMENT.md` for step-by-step instructions on deploying to Streamlit Community
Cloud, including how to store your API key as a secret instead of typing it in each time.

## Things to try / extend

- **Change `TOP_K`** in `app.py` to retrieve more or fewer chunks per question.
- **Change `CHUNK_SIZE`/`CHUNK_OVERLAP`** to see how chunking affects answer quality —
  this is a great way to build intuition about retrieval tradeoffs.
- **Swap the embedding model** (`EMBED_MODEL_NAME`) for a different `sentence-transformers`
  model to compare retrieval quality.
- **Add a re-ranking step** after retrieval for more precise context selection.
- **Persist the FAISS index to disk** instead of rebuilding it every session.

## Notes

- Everything except the final answer generation runs locally (embeddings + retrieval)
  — only the retrieved chunks + your question are sent to the Gemini API.
- The app currently keeps the index in memory only; it resets if you restart Streamlit.