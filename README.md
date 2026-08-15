# AI PDF Chatbot

An AI-powered PDF question-answering application that allows users to upload PDF documents and ask questions about their content.

## Live Demo

[Try the AI PDF Chatbot](https://vnm19102-hub-chatbot-speak-pdf-app-q1ugxz.streamlit.app/)

## Features

- Upload one or multiple PDF files
- Extract text from PDF documents
- Semantic search using Sentence Transformers
- FAISS-based document retrieval
- AI-generated answers using Groq LLaMA
- Conversational chat history
- Source and page references

## Technologies Used

- Python
- Streamlit
- PyMuPDF
- Sentence Transformers
- FAISS
- Groq API
- LLaMA

## How It Works

1. User uploads one or more PDF documents.
2. Text is extracted from the uploaded PDFs.
3. The extracted text is converted into numerical embeddings.
4. FAISS searches for the most relevant sections of the documents.
5. The relevant content is provided to the LLaMA model.
6. The model generates an answer based on the retrieved information.
7. The conversation history is maintained for follow-up questions.

## Deployment

The application is deployed using Streamlit Community Cloud.

## Project Repository

The source code for this project is available in this repository.
