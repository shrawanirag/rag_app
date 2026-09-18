"""
Simple RAG (Retrieval-Augmented Generation) App
------------------------------------------------
Upload your notes/PDFs, ask questions, get answers grounded in your own documents.
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
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
CHUNK_SIZE = 300
CHUNK_OVERLAP = 50
TOP_K = 4
GEMINI_MODEL = "gemini-3.6-flash"

st.set_page_config(page_title="Notes RAG Chat", page_icon="📚")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
@st.cache_resource
def load_embedder():
    return SentenceTransformer(EMBED_MODEL_NAME)


def extract_text(uploaded_file) -> str:
    if uploaded_file.name.lower().endswith(".pdf"):
        reader = PdfReader(uploaded_file)
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    else:
        return uploaded_file.read().decode("utf-8", errors="ignore")


def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
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
    embeddings = embedder.encode(chunks, show_progress_bar=False, normalize_embeddings=True)
    embeddings = np.array(embeddings, dtype="float32")
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index


def retrieve(query, embedder, index, chunks, top_k=TOP_K):
    """Returns a list of (chunk_number, chunk_text, similarity_score), ranked best first.
    chunk_number is 1-indexed to match how you'd refer to 'chunk 5' when inspecting your data."""
    q_emb = embedder.encode([query], normalize_embeddings=True)
    q_emb = np.array(q_emb, dtype="float32")
    scores, idxs = index.search(q_emb, top_k)
    results = [
        (int(i) + 1, chunks[i], float(score))
        for i, score in zip(idxs[0], scores[0])
        if i != -1
    ]
    return results


def ask_gemini(client, question, retrieved):
    """retrieved is a list of (chunk_number, chunk_text, score)."""
    context = "\n\n---\n\n".join(chunk_text for _, chunk_text, _ in retrieved)
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
    default_key = st.secrets.get("GEMINI_API_KEY", "") if hasattr(st, "secrets") else ""
    api_key = st.text_input("Gemini API key", type="password", value=default_key)
    st.markdown("[Get a free key from Google AI Studio](https://aistudio.google.com/apikey)")

    st.divider()
    uploaded_files = st.file_uploader(
        "Upload PDF or TXT files", type=["pdf", "txt"], accept_multiple_files=True
    )
    build_button = st.button("Build knowledge base", type="primary")

if "index" not in st.session_state:
    st.session_state.index = None
    st.session_state.chunks = []
    st.session_state.messages = []

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

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        # Re-show retrieval info for past messages too, if this message has it attached
        if "retrieved" in msg:
            with st.expander(f"Sources used ({len(msg['retrieved'])} chunks)"):
                for chunk_num, chunk_txt, score in msg["retrieved"]:
                    st.markdown(f"**Chunk {chunk_num}** — similarity score: `{score:.3f}`")
                    st.text(chunk_txt[:400] + ("..." if len(chunk_txt) > 400 else ""))
                    st.divider()

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

                # This is the key visibility panel: chunk number + similarity score for every
                # chunk that was actually handed to Gemini for this specific answer.
                with st.expander(f"Sources used ({len(retrieved)} chunks)"):
                    for chunk_num, chunk_txt, score in retrieved:
                        st.markdown(f"**Chunk {chunk_num}** — similarity score: `{score:.3f}`")
                        st.text(chunk_txt[:400] + ("..." if len(chunk_txt) > 400 else ""))
                        st.divider()

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "retrieved": retrieved,
        })