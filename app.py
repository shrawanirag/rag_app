"""
Simple RAG (Retrieval-Augmented Generation) App
------------------------------------------------
Upload your notes/PDFs, ask questions, get answers grounded in your own documents.

Pipeline:
  1. Upload PDF/TXT files
  2. Split into overlapping text chunks
  3. Embed chunks with a local sentence-transformer model
  4. Store embeddings in a FAISS index (in-memory)
  5. On each question: embed the query, retrieve top-k similar chunks,
     pass them + the question to Gemini, and display the grounded answer.

Run with:  streamlit run app.py
"""

import streamlit as st
import numpy as np
import faiss
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from google import genai

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"   # small, fast, runs locally (CPU is fine)
CHUNK_SIZE = 300                        # words per chunk
CHUNK_OVERLAP = 50                      # words of overlap between chunks
TOP_K = 4                               # how many chunks to retrieve per question
GEMINI_MODEL = "gemini-3.6-flash"       # free-tier friendly; change if you want a different Gemini model

st.set_page_config(page_title="Notes RAG Chat", page_icon="📚")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
@st.cache_resource
def load_embedder():
    """Load the embedding model once and cache it across reruns."""
    return SentenceTransformer(EMBED_MODEL_NAME)


def extract_text(uploaded_file) -> str:
    """Extract raw text from an uploaded PDF or TXT file."""
    if uploaded_file.name.lower().endswith(".pdf"):
        reader = PdfReader(uploaded_file)
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    else:
        return uploaded_file.read().decode("utf-8", errors="ignore")


def chunk_text(text: str, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Split text into overlapping word-based chunks."""
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        if chunk.strip():
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


def build_index(chunks, embedder):
    """Embed all chunks and build a FAISS index (cosine similarity via normalized vectors)."""
    embeddings = embedder.encode(chunks, show_progress_bar=False, normalize_embeddings=True)
    embeddings = np.array(embeddings, dtype="float32")
    index = faiss.IndexFlatIP(embeddings.shape[1])  # inner product == cosine on normalized vectors
    index.add(embeddings)
    return index


def retrieve(query, embedder, index, chunks, top_k=TOP_K):
    """Embed the query and return the top_k most similar chunks."""
    q_emb = embedder.encode([query], normalize_embeddings=True)
    q_emb = np.array(q_emb, dtype="float32")
    scores, idxs = index.search(q_emb, top_k)
    results = [chunks[i] for i in idxs[0] if i != -1]
    return results


def ask_gemini(client, question, retrieved_chunks):
    """Send the question + retrieved context to Gemini and return the answer."""
    context = "\n\n---\n\n".join(retrieved_chunks)
    system_prompt = (
        "You are a study assistant. Answer the user's question using ONLY the "
        "context provided below. If the answer isn't in the context, say so clearly "
        "instead of guessing. Keep answers clear and well-explained, like a tutor."
    )
    user_message = f"Context:\n{context}\n\nQuestion: {question}"

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=user_message,
        config={"system_instruction": system_prompt, "max_output_tokens": 1000},
    )
    return response.text


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------
st.title("📚 Notes RAG Chat")
st.caption("Upload your notes or PDFs, then ask questions grounded in them.")

with st.sidebar:
    st.header("Setup")
    # If you deploy this yourself and add GEMINI_API_KEY to Streamlit secrets,
    # visitors won't need to paste their own key. Otherwise they enter their own below.
    default_key = st.secrets.get("GEMINI_API_KEY", "") if hasattr(st, "secrets") else ""
    api_key = st.text_input("Gemini API key", type="password", value=default_key)
    st.markdown("[Get a free key from Google AI Studio](https://aistudio.google.com/apikey)")

    st.divider()
    uploaded_files = st.file_uploader(
        "Upload PDF or TXT files", type=["pdf", "txt"], accept_multiple_files=True
    )
    build_button = st.button("Build knowledge base", type="primary")

# Session state
if "index" not in st.session_state:
    st.session_state.index = None
    st.session_state.chunks = []
    st.session_state.messages = []

# Build the index when the user clicks the button
if build_button:
    if not uploaded_files:
        st.sidebar.error("Upload at least one file first.")
    else:
        with st.spinner("Reading files and building index..."):
            embedder = load_embedder()
            all_chunks = []
            for f in uploaded_files:
                text = extract_text(f)
                all_chunks.extend(chunk_text(text))

            if not all_chunks:
                st.sidebar.error("Couldn't extract any text from the uploaded files.")
            else:
                st.session_state.index = build_index(all_chunks, embedder)
                st.session_state.chunks = all_chunks
                st.session_state.messages = []
                st.sidebar.success(f"Indexed {len(all_chunks)} chunks from {len(uploaded_files)} file(s).")

# Chat interface
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

question = st.chat_input("Ask a question about your uploaded documents...")

if question:
    if st.session_state.index is None:
        st.error("Build the knowledge base first (upload files + click 'Build knowledge base').")
    elif not api_key:
        st.error("Enter your Gemini API key in the sidebar.")
    else:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Retrieving relevant chunks and generating answer..."):
                embedder = load_embedder()
                retrieved = retrieve(question, embedder, st.session_state.index, st.session_state.chunks)
                client = genai.Client(api_key=api_key)
                answer = ask_gemini(client, question, retrieved)

                st.markdown(answer)
                with st.expander("Retrieved context used for this answer"):
                    for i, chunk in enumerate(retrieved, 1):
                        st.markdown(f"**Chunk {i}:**\n\n{chunk}")

        st.session_state.messages.append({"role": "assistant", "content": answer})
