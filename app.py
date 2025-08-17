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
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "gsk_your_api_key_here")
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
def ask_llm(question, context):
    prompt = f"""
You are a helpful assistant. Answer in detail.

Context from documents:
{context}

Question:
{question}

Answer with clear explanation and examples where possible.
    """
    response = client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=600,
        temperature=0.7
    )
    return response.choices[0].message.content

# =========================
# STREAMLIT UI
# =========================
st.set_page_config(page_title="PDF Chatbot", layout="wide")
st.title("📘 AI PDF Chatbot (Groq LLaMA-3)")

uploaded_files = st.file_uploader("Upload your PDFs", type="pdf", accept_multiple_files=True)

if uploaded_files:
    st.success("✅ PDFs uploaded successfully. You can now ask questions.")
    docs = extract_text_from_pdfs(uploaded_files)
    index, doc_store = create_faiss_index(docs)

    user_question = st.text_input("Ask a question from the PDFs:")

    if st.button("Get Answer") and user_question:
        results = search_docs(user_question, index, doc_store)
        context = "\n\n".join([f"[{doc[2]} - Page {doc[1]}]\n{doc[0]}" for doc in results])
        answer = ask_llm(user_question, context)

        st.subheader("📖 Answer")
        st.write(answer)

        st.subheader("🔍 Sources")
        for r in results:
            st.markdown(f"- **{r[2]} (Page {r[1]})**")
