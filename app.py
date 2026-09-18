"""
Simple RAG (Retrieval-Augmented Generation) App — improved version
--------------------------------------------------------------------
Upload your notes/PDFs, ask questions, get answers grounded in your own documents.

Changes from the first version, and why:

1. CHUNK_SIZE dropped from 300 to 180 words.
   all-MiniLM-L6-v2 truncates at 256 tokens. English averages ~1.3 tokens/word,
   so 300 words (~390 tokens) was silently getting cut off before embedding.
   180 words (~235 tokens) stays safely under the limit.

2. Chunking is now sentence-aware instead of a blind word-count cut.
   The old version could slice a sentence in half at exactly word 300, regardless
   of where that fell. This version only ends a chunk at a sentence boundary,
   so every chunk is made of whole, coherent sentences.

3. TOP_K raised from 4 to 6.
   Gives retrieval a bit more room to catch relevant chunks that don't score
   as the single highest match but are still genuinely useful.

4. Retrieval confidence checking.
   If the best similarity score for a question is low (below CONFIDENCE_THRESHOLD),
   that means nothing in the document is a strong match - FAISS still returns its
   top-k regardless of quality. The app now surfaces this instead of silently
   treating a weak match like a good one.
"""

import re
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
CHUNK_SIZE = 180              # words per chunk (token-safe for this embedder)
CHUNK_OVERLAP_WORDS = 30      # approx words of overlap between chunks
TOP_K = 6
CONFIDENCE_THRESHOLD = 0.35   # below this, warn that retrieval confidence is low
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


def split_sentences(text: str):
    """Naive sentence splitter: breaks after ., !, or ? followed by whitespace."""
    text = re.sub(r"\s+", " ", text).strip()
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [s for s in sentences if s.strip()]


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
    context = "\n\n---\n\n".join(chunk_text for _, chunk_text, _ in retrieved)

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

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=user_message,
        config={"system_instruction": base_instructions, "max_output_tokens": 1000},
    )
    return response.text


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
                answer = ask_gemini(client, question, retrieved, low_confidence)

                st.markdown(answer)
                render_sources(retrieved, low_confidence)

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "retrieved": retrieved,
            "low_confidence": low_confidence,
        })