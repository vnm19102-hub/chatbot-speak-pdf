import streamlit as st
import fitz  # PyMuPDF
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from groq import Groq
import os

# =========================
# CONFIG
# =========================
# Get API key from environment (best practice for Streamlit Cloud)
# Get Groq API key securely
try:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
except Exception:
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    st.error("Groq API key is not configured.")
    st.stop()

client = Groq(api_key=GROQ_API_KEY)
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# =========================
# PDF PROCESSING
# =========================
def extract_text_from_pdfs(uploaded_files):
    docs = []
    for file in uploaded_files:
        pdf = fitz.open(stream=file.read(), filetype="pdf")
        for page_num, page in enumerate(pdf, start=1):
            text = page.get_text("text")
            if text.strip():
                docs.append((text, page_num, file.name))
    return docs

def create_faiss_index(docs):
    texts = [doc[0] for doc in docs]
    embeddings = embedder.encode(texts, convert_to_numpy=True)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index, docs

def search_docs(query, index, docs, top_k=3):
    query_emb = embedder.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_emb, top_k)
    results = []
    for idx in indices[0]:
        if idx < len(docs):
            results.append(docs[idx])
    return results

# =========================
# ASK LLM
# =========================
def ask_llm(question, context, chat_history):
    history_text = ""

    for message in chat_history:
        history_text += f"{message['role'].capitalize()}: {message['content']}\n"

    prompt = f"""
You are a helpful assistant that answers questions using the uploaded PDF documents.

Previous conversation:
{history_text}

Relevant information from the PDF:
{context}

Current question:
{question}

Answer clearly and accurately based on the PDF.
If the question refers to something from the previous conversation,
use the conversation history to understand what the user means.
"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "user", "content": prompt}
        ],
        max_tokens=600,
        temperature=0.7
    )

    return response.choices[0].message.content
# =========================
# STREAMLIT UI
# =========================
st.set_page_config(page_title="PDF Chatbot", layout="wide")
st.title("📘 AI PDF Chatbot (Groq LLaMA-3)")

uploaded_files = st.file_uploader(
    "Upload your PDFs",
    type="pdf",
    accept_multiple_files=True
)

if uploaded_files:

    st.success("PDFs uploaded successfully. You can now ask questions.")

    docs = extract_text_from_pdfs(uploaded_files)
    index, doc_store = create_faiss_index(docs)

    # Create chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()



    # Display previous conversation
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # Chat input
    user_question = st.chat_input("Ask a question about your PDFs...")

    if user_question:

        # Show user's question
        with st.chat_message("user"):
            st.write(user_question)

        # Search the PDF
        results = search_docs(
            user_question,
            index,
            doc_store
        )

        context = "\n\n".join(
            [
                f"[{doc[2]} - Page {doc[1]}]\n{doc[0]}"
                for doc in results
            ]
        )

        # Generate answer
        try:
            answer = ask_llm(
                user_question,
                context,
                st.session_state.messages
            )

        except Exception as e:
            st.error(
                "Unable to generate an answer. Please check the Groq API configuration."
            )
            st.stop()

        # Show AI answer
        with st.chat_message("assistant"):
            st.write(answer)

        # Save conversation
        st.session_state.messages.append(
            {
                "role": "user",
                "content": user_question
            }
        )

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer
            }
        )

        # Show sources
        with st.expander("Sources"):
            for r in results:
                st.markdown(
                    f"- **{r[2]} — Page {r[1]}**"
                )
