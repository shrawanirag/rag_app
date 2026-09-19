"""
Simple RAG (Retrieval-Augmented Generation) App — improved version
--------------------------------------------------------------------
Upload your notes/PDFs, ask questions, get answers grounded in your own documents.

Key design choices:

1. CHUNK_SIZE = 180 words, not 300.
   all-MiniLM-L6-v2 truncates at 256 tokens (~200 words). 180 words stays safely
   under that so no chunk gets silently truncated before embedding.

2. Sentence-aware chunking instead of a blind word-count cut.
   Only ends a chunk at a sentence boundary, so no sentence gets sliced in half.

3. TOP_K = 6, so retrieval has more room to catch genuinely relevant chunks.

4. Retrieval confidence checking.
   If the best similarity score is low, the app warns instead of silently
   treating a weak match as a good one.

5. Retry logic around the Gemini call.
   Gemini's free tier occasionally returns a transient 503 "model overloaded"
   error that has nothing to do with your question - it's Google's servers
   being briefly busy. Retrying after a short wait almost always succeeds.
   Only transient server errors (5xx) are retried; a genuine problem like an
   invalid API key (4xx) fails immediately instead of retrying uselessly.
"""

import re
import time
import streamlit as st
import numpy as np
import faiss
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from google import genai
from google.genai import errors as genai_errors

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
CHUNK_SIZE = 180              # words per chunk (token-safe for this embedder)
CHUNK_OVERLAP_WORDS = 30      # approx words of overlap between chunks
TOP_K = 6
CONFIDENCE_THRESHOLD = 0.35   # below this, warn that retrieval confidence is low
GEMINI_MODEL = "gemini-3.6-flash"
MAX_RETRIES = 3                # how many times to retry on a transient server error
RETRY_BASE_DELAY_SECONDS = 2   # doubles each retry: 2s, 4s, 8s

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


def split_sentences(text: str):
    """Naive sentence splitter: breaks after ., !, or ? followed by whitespace."""
    text = re.sub(r"\s+", " ", text).strip()
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [s for s in sentences if s.strip()]

def split_front_matter(text: str):
    """Splits off the title/author/affiliation block that sits before the
    Abstract, so it can become its own chunk instead of being diluted inside
    a 180-word block dominated by abstract content."""
    match = re.search(r"\b(abstract)\b", text, re.IGNORECASE)
    if match and match.start() < 800:  # only trust this if it's near the top of the doc
        front_matter = text[:match.start()].strip()
        rest = text[match.start():].strip()
        return front_matter, rest
    return "", text  # no reliable marker found — treat the whole thing as one body


def chunk_text(text, chunk_size=CHUNK_SIZE, overlap_words=CHUNK_OVERLAP_WORDS):
    """Sentence-aware chunking: packs whole sentences into each chunk until
    chunk_size words is reached, then starts the next chunk carrying over the
    last few sentences (up to overlap_words) for continuity."""
    sentences = split_sentences(text)
    chunks = []
    current = []
    current_words = 0

    for sent in sentences:
        sent_word_count = len(sent.split())

        if current_words + sent_word_count > chunk_size and current:
            chunks.append(" ".join(current))
            carry, carry_words = [], 0
            for s in reversed(current):
                sw = len(s.split())
                if carry_words + sw > overlap_words:
                    break
                carry.insert(0, s)
                carry_words += sw
            current, current_words = carry, carry_words

        current.append(sent)
        current_words += sent_word_count

    if current:
        chunks.append(" ".join(current))

    return chunks


def build_index(chunks, embedder):
    embeddings = embedder.encode(chunks, show_progress_bar=False, normalize_embeddings=True)
    embeddings = np.array(embeddings, dtype="float32")
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index


def retrieve(query, embedder, index, chunks, top_k=TOP_K):
    """Returns a list of (chunk_number, chunk_text, similarity_score), ranked best first."""
    q_emb = embedder.encode([query], normalize_embeddings=True)
    q_emb = np.array(q_emb, dtype="float32")
    scores, idxs = index.search(q_emb, top_k)
    results = [
        (int(i) + 1, chunks[i], float(score))
        for i, score in zip(idxs[0], scores[0])
        if i != -1
    ]
    return results


def ask_gemini(client, question, retrieved, low_confidence: bool):
    """Calls Gemini with retry logic for transient server errors (5xx).
    Raises the original exception if all retries are exhausted, or immediately
    for non-retryable errors (e.g. bad API key)."""
    doc_metadata = "\n\n".join(st.session_state.get("front_matters", []))
    retrieved_context = "\n\n---\n\n".join(chunk_text for _, chunk_text, _ in retrieved)
    context = f"Document metadata (title/author/etc.):\n{doc_metadata}\n\n---\n\nRetrieved passages:\n{retrieved_context}"

    base_instructions = (
        "You are a study assistant. Answer the user's question using ONLY the "
        "context provided below. If the answer isn't in the context, say so clearly "
        "instead of guessing. Keep answers clear and well-explained, like a tutor."
    )
    if low_confidence:
        base_instructions += (
            " IMPORTANT: none of the retrieved passages closely match this question. "
            "Be extra cautious - if you can't find a clear, direct answer in the context, "
            "say so explicitly rather than piecing together a guess."
        )

    user_message = f"Context:\n{context}\n\nQuestion: {question}"

    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=user_message,
                config={"system_instruction": base_instructions, "max_output_tokens": 1000},
            )
            return response.text
        except genai_errors.ServerError as e:
            # Transient issue on Google's side (e.g. 503 "high demand") - worth retrying.
            last_error = e
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_BASE_DELAY_SECONDS * (2 ** (attempt - 1)))
            continue
        except genai_errors.ClientError:
            # e.g. 400 invalid API key, invalid request - retrying won't help, fail fast.
            raise

    # All retries exhausted - raise the last error so the caller can show a clean message.
    raise last_error


def render_sources(retrieved, low_confidence: bool):
    if low_confidence:
        st.warning(
            "Low retrieval confidence - the best-matching chunk scored below "
            f"{CONFIDENCE_THRESHOLD}. This document may not actually contain a clear "
            "answer to this question."
        )
    with st.expander(f"Sources used ({len(retrieved)} chunks)"):
        for chunk_num, chunk_txt, score in retrieved:
            st.markdown(f"**Chunk {chunk_num}** - similarity score: `{score:.3f}`")
            st.text(chunk_txt[:400] + ("..." if len(chunk_txt) > 400 else ""))
            st.divider()


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
                front_matter, body = split_front_matter(text)
                if front_matter:
                    st.session_state.setdefault("front_matters", []).append(front_matter)
                all_chunks.extend(chunk_text(body))

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
        if "retrieved" in msg:
            render_sources(msg["retrieved"], msg.get("low_confidence", False))

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
                low_confidence = (not retrieved) or (retrieved[0][2] < CONFIDENCE_THRESHOLD)

                client = genai.Client(api_key=api_key)

                try:
                    answer = ask_gemini(client, question, retrieved, low_confidence)
                except genai_errors.ServerError:
                    answer = None
                    st.error(
                        "Gemini's servers are currently overloaded and didn't respond after "
                        f"{MAX_RETRIES} attempts. This is temporary - please try asking again "
                        "in a moment."
                    )
                except genai_errors.ClientError as e:
                    answer = None
                    st.error(
                        "Gemini rejected the request - this usually means the API key is "
                        f"invalid or missing permissions. Details: {e}"
                    )

                if answer is not None:
                    st.markdown(answer)
                    render_sources(retrieved, low_confidence)

        if answer is not None:
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "retrieved": retrieved,
                "low_confidence": low_confidence,
            })