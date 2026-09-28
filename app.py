"""
Notes RAG Chat — improved version

Features:
- Streamlit interface for PDF/TXT uploads and document-grounded Q&A
- Sentence-aware, tokenizer-aware chunking
- Chunk metadata (filename and PDF page where available)
- FAISS cosine-similarity retrieval
- Configurable retrieval threshold and top-k
- Gemini retry handling for transient server errors
- Source display with filenames, page numbers, chunk IDs, and scores

Install:
pip install streamlit numpy faiss-cpu pypdf sentence-transformers google-genai
Run:
streamlit run app.py

Set GEMINI_API_KEY in .streamlit/secrets.toml or enter it in the sidebar.
"""

import re
import time
from typing import Dict, List, Tuple, Any

import streamlit as st
import numpy as np
import faiss
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from google import genai
from google.genai import errors as genai_errors


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"

# The embedding model has a 256-token input limit. Use a conservative budget
# below that limit to leave room for special tokens.
MAX_CHUNK_TOKENS = 220
OVERLAP_TARGET_TOKENS = 40

TOP_K = 6
CONFIDENCE_THRESHOLD = 0.35

GEMINI_MODEL = "gemini-3.6-flash"
MAX_RETRIES = 3
RETRY_BASE_DELAY_SECONDS = 2

st.set_page_config(page_title="Notes RAG Chat", page_icon="📚")
st.title("📚 Notes RAG Chat")
st.caption("Upload your notes or PDFs, then ask questions grounded in them.")


# ---------------------------------------------------------------------------
# Model loading and document extraction
# ---------------------------------------------------------------------------
@st.cache_resource
def load_embedder():
    """Load and cache the embedding model for the Streamlit process."""
    return SentenceTransformer(EMBED_MODEL_NAME)


def extract_documents(uploaded_file) -> List[Dict[str, Any]]:
    """
    Return page/document records.

    PDF: one record per page, preserving page numbers.
    TXT: one record for the complete text.
    """
    filename = uploaded_file.name

    if filename.lower().endswith(".pdf"):
        reader = PdfReader(uploaded_file)
        documents = []
        for page_number, page in enumerate(reader.pages, start=1):
            page_text = page.extract_text() or ""
            if page_text.strip():
                documents.append({
                    "text": page_text,
                    "source": filename,
                    "page": page_number,
                })
        return documents

    text = uploaded_file.read().decode("utf-8", errors="ignore")
    return [{
        "text": text,
        "source": filename,
        "page": None,
    }] if text.strip() else []


def normalize_text(text: str) -> str:
    """Normalize whitespace while preserving sentence punctuation."""
    return re.sub(r"\s+", " ", text).strip()


def split_sentences(text: str) -> List[str]:
    """
    Simple sentence splitter. This is intentionally lightweight; abbreviations
    such as 'Dr.' and decimal numbers may need a more advanced NLP splitter.
    """
    text = normalize_text(text)
    if not text:
        return []
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [sentence.strip() for sentence in sentences if sentence.strip()]


def split_front_matter(text: str) -> Tuple[str, str]:
    """Separate a leading title/author block when Abstract appears near top."""
    match = re.search(r"\babstract\b", text, re.IGNORECASE)
    if match and match.start() < 800:
        return text[:match.start()].strip(), text[match.start():].strip()
    return "", text


def token_count(text: str, embedder) -> int:
    """Count tokens with the same tokenizer used by the embedding model."""
    return len(
        embedder.tokenizer.encode(
            text,
            add_special_tokens=True,
            truncation=False,
        )
    )


def chunk_document(
    text: str,
    source: str,
    page: int | None,
    embedder,
    max_tokens: int = MAX_CHUNK_TOKENS,
    overlap_tokens: int = OVERLAP_TARGET_TOKENS,
) -> List[Dict[str, Any]]:
    """
    Pack whole sentences into chunks under a tokenizer-based token budget.
    Chunks overlap by carrying whole trailing sentences forward.

    A single sentence longer than max_tokens is kept intact rather than
    silently split; it is flagged in metadata so it can be reviewed.
    """
    sentences = split_sentences(text)
    chunks = []
    current_sentences = []
    current_text = ""

    for sentence in sentences:
        candidate = f"{current_text} {sentence}".strip()
        candidate_tokens = token_count(candidate, embedder)

        if current_sentences and candidate_tokens > max_tokens:
            chunk_text = " ".join(current_sentences).strip()
            chunks.append({
                "text": chunk_text,
                "source": source,
                "page": page,
                "oversize": token_count(chunk_text, embedder) > max_tokens,
            })

            # Carry complete trailing sentences for context overlap.
            carry = []
            carry_tokens = 0
            for old_sentence in reversed(current_sentences):
                sentence_tokens = token_count(old_sentence, embedder)
                if carry and carry_tokens + sentence_tokens > overlap_tokens:
                    break
                if not carry and sentence_tokens > overlap_tokens:
                    # Keep at least one trailing sentence, even if it is
                    # larger than the nominal overlap target.
                    carry.insert(0, old_sentence)
                    carry_tokens += sentence_tokens
                    break
                carry.insert(0, old_sentence)
                carry_tokens += sentence_tokens

            current_sentences = carry
            current_text = " ".join(current_sentences)

        current_sentences.append(sentence)
        current_text = " ".join(current_sentences).strip()

    if current_sentences:
        chunk_text = " ".join(current_sentences).strip()
        chunks.append({
            "text": chunk_text,
            "source": source,
            "page": page,
            "oversize": token_count(chunk_text, embedder) > max_tokens,
        })

    return chunks


# ---------------------------------------------------------------------------
# Embedding, indexing, and retrieval
# ---------------------------------------------------------------------------
def build_index(chunks: List[Dict[str, Any]], embedder):
    """Build a FAISS inner-product index over normalized embeddings."""
    texts = [chunk["text"] for chunk in chunks]
    embeddings = embedder.encode(
        texts,
        show_progress_bar=False,
        normalize_embeddings=True,
    )
    embeddings = np.asarray(embeddings, dtype=np.float32)

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index


def retrieve(
    query: str,
    embedder,
    index,
    chunks: List[Dict[str, Any]],
    top_k: int = TOP_K,
):
    """Return matching chunk metadata and similarity scores, best first."""
    if index is None or not chunks:
        return []

    query_embedding = embedder.encode(
        [query],
        normalize_embeddings=True,
    )
    query_embedding = np.asarray(query_embedding, dtype=np.float32)

    k = min(top_k, len(chunks))
    scores, indices = index.search(query_embedding, k)

    results = []
    for idx, score in zip(indices[0], scores[0]):
        if idx == -1:
            continue

        result = dict(chunks[int(idx)])
        result["chunk_id"] = int(idx) + 1
        result["score"] = float(score)
        results.append(result)

    return results


# ---------------------------------------------------------------------------
# Gemini generation with retry logic
# ---------------------------------------------------------------------------
def ask_gemini(client, question: str, retrieved, low_confidence: bool) -> str:
    """Generate an answer using retrieved passages and retry server errors."""
    context_parts = []
    for item in retrieved:
        page_label = (
            f", page {item['page']}" if item.get("page") is not None else ""
        )
        context_parts.append(
            f"[Source: {item['source']}{page_label}; "
            f"chunk {item['chunk_id']}]\n{item['text']}"
        )

    retrieved_context = "\n\n---\n\n".join(context_parts)

    if not retrieved_context:
        retrieved_context = "No relevant passages were retrieved."

    system_instruction = (
        "You are a study assistant. Answer using only the provided document "
        "context. If the answer is not clearly supported by the context, say "
        "that the uploaded documents do not contain enough information. "
        "Do not invent citations or facts. Explain clearly like a tutor."
    )

    if low_confidence:
        system_instruction += (
            " Retrieval similarity is low. Be especially cautious and do not "
            "infer an answer from weakly related passages."
        )

    user_message = (
        f"Document context:\n{retrieved_context}\n\n"
        f"Question: {question}\n\n"
        "When possible, mention the source filename and page number that "
        "supports the answer."
    )

    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=user_message,
                config={
                    "system_instruction": system_instruction,
                    "max_output_tokens": 1000,
                },
            )
            if not response.text:
                raise ValueError("Gemini returned an empty response.")
            return response.text

        except genai_errors.ServerError as exc:
            last_error = exc
            if attempt < MAX_RETRIES:
                delay = RETRY_BASE_DELAY_SECONDS * (2 ** (attempt - 1))
                time.sleep(delay)

        except genai_errors.ClientError:
            # Invalid key/request/permissions usually need user action.
            raise

    raise last_error


# ---------------------------------------------------------------------------
# Source rendering
# ---------------------------------------------------------------------------
def render_sources(retrieved, low_confidence: bool):
    if low_confidence:
        st.warning(
            "The best retrieved similarity score is below "
            f"{CONFIDENCE_THRESHOLD:.2f}. Check the sources carefully."
        )

    with st.expander(f"Sources used ({len(retrieved)} chunks)"):
        if not retrieved:
            st.write("No matching passages were found.")
        for item in retrieved:
            page_label = (
                f" | Page {item['page']}"
                if item.get("page") is not None
                else ""
            )
            st.markdown(
                f"**{item['source']}{page_label} — "
                f"Chunk {item['chunk_id']}** "
                f"| Similarity: `{item['score']:.3f}`"
            )
            if item.get("oversize"):
                st.caption(
                    "Note: this chunk exceeds the configured token budget "
                    "because a sentence was kept intact."
                )
            text = item["text"]
            st.text(text[:800] + ("..." if len(text) > 800 else ""))
            st.divider()


# ---------------------------------------------------------------------------
# Streamlit state and sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Setup")

    try:
        default_key = st.secrets.get("GEMINI_API_KEY", "")
    except Exception:
        default_key = ""

    api_key = st.text_input(
        "Gemini API key",
        type="password",
        value=default_key,
    )
    st.markdown(
        "[Get a key from Google AI Studio]"
        "(https://aistudio.google.com/apikey)"
    )
    st.divider()

    uploaded_files = st.file_uploader(
        "Upload PDF or TXT files",
        type=["pdf", "txt"],
        accept_multiple_files=True,
    )
    build_button = st.button(
        "Build knowledge base",
        type="primary",
    )

if "index" not in st.session_state:
    st.session_state.index = None
    st.session_state.chunks = []
    st.session_state.messages = []
    st.session_state.front_matters = []


# ---------------------------------------------------------------------------
# Build/rebuild the knowledge base
# ---------------------------------------------------------------------------
if build_button:
    if not uploaded_files:
        st.sidebar.error("Upload at least one file first.")
    else:
        with st.spinner("Reading files and building index..."):
            embedder = load_embedder()
            all_chunks = []
            front_matters = []

            # Reset all document-specific data before rebuilding. This avoids
            # carrying metadata or chunks over from a previous upload batch.
            st.session_state.index = None
            st.session_state.chunks = []
            st.session_state.messages = []
            st.session_state.front_matters = []

            for uploaded_file in uploaded_files:
                documents = extract_documents(uploaded_file)

                for document in documents:
                    front_matter, body = split_front_matter(
                        document["text"]
                    )

                    if front_matter:
                        front_matters.append({
                            "source": document["source"],
                            "page": document["page"],
                            "text": front_matter,
                        })

                    all_chunks.extend(
                        chunk_document(
                            text=body,
                            source=document["source"],
                            page=document["page"],
                            embedder=embedder,
                        )
                    )

            if not all_chunks:
                st.sidebar.error(
                    "Couldn't extract usable text from the uploaded files. "
                    "Scanned PDFs may require OCR."
                )
            else:
                st.session_state.index = build_index(
                    all_chunks,
                    embedder,
                )
                st.session_state.chunks = all_chunks
                st.session_state.front_matters = front_matters

                st.sidebar.success(
                    f"Indexed {len(all_chunks)} chunks "
                    f"from {len(uploaded_files)} file(s)."
                )

                oversize_count = sum(
                    1 for chunk in all_chunks if chunk["oversize"]
                )
                if oversize_count:
                    st.sidebar.warning(
                        f"{oversize_count} chunk(s) exceed the token budget "
                        "because a sentence was kept intact."
                    )


# ---------------------------------------------------------------------------
# Conversation display
# ---------------------------------------------------------------------------
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "retrieved" in message:
            render_sources(
                message["retrieved"],
                message.get("low_confidence", False),
            )


# ---------------------------------------------------------------------------
# Question handling
# ---------------------------------------------------------------------------
question = st.chat_input(
    "Ask a question about your uploaded documents..."
)

if question:
    if st.session_state.index is None:
        st.error(
            "Build the knowledge base first "
            "(upload files and click 'Build knowledge base')."
        )
    elif not api_key:
        st.error("Enter your Gemini API key in the sidebar.")
    else:
        st.session_state.messages.append({
            "role": "user",
            "content": question,
        })

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner(
                "Retrieving relevant chunks and generating answer..."
            ):
                embedder = load_embedder()
                retrieved = retrieve(
                    question,
                    embedder,
                    st.session_state.index,
                    st.session_state.chunks,
                )

                low_confidence = (
                    not retrieved
                    or retrieved[0]["score"] < CONFIDENCE_THRESHOLD
                )

                client = genai.Client(api_key=api_key)
                answer = None

                try:
                    answer = ask_gemini(
                        client,
                        question,
                        retrieved,
                        low_confidence,
                    )
                except genai_errors.ServerError:
                    st.error(
                        "Gemini's servers did not respond after "
                        f"{MAX_RETRIES} attempts. Please try again shortly."
                    )
                except genai_errors.ClientError as exc:
                    st.error(
                        "Gemini rejected the request. Check your API key, "
                        f"permissions, and request. Details: {exc}"
                    )
                except ValueError as exc:
                    st.error(str(exc))

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
