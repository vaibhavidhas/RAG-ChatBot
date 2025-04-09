# RAG-ChatBot
A Streamlit-based RAG chatbot that answers questions from both digital and scanned PDFs using local LLMs and FAISS-powered semantic search.

RAG Chatbot: Retrieval-Augmented Generation for PDF-Based QA
This project is an interactive chatbot built with Streamlit that uses Retrieval-Augmented Generation (RAG) to answer user queries based on the content of a collection of PDF documents. It supports both digital PDFs and scanned documents using OCR (Optical Character Recognition), and leverages FAISS for vector similarity search and Ollama for running local LLMs and embeddings.

🔍 Key Features:
PDF Parsing: Automatically distinguishes between digital and scanned PDFs.

Digital PDFs: Uses PyPDF2 to extract text.

Scanned PDFs: Uses pdf2image + pytesseract to perform OCR.

Chunking and Embedding:

Text is split into manageable chunks using RecursiveCharacterTextSplitter.

Chunks are embedded using OllamaEmbeddings with nomic-embed-text.

Vector Search with FAISS:

FAISS is used to store and retrieve relevant text chunks based on semantic similarity to the user query.

Conversational QA Chain:

Uses langchain's QA chain with a custom prompt to generate natural language responses.

LLM used: gemma:2b via Ollama.

Interactive UI with Streamlit:

Allows users to submit a query and view chat history.

One-click processing of documents via a "Submit & Process" button.

💬 Workflow Overview:
User clicks "Submit & Process":

All PDFs in the specified folder are parsed.

Text is extracted, chunked, embedded, and stored in a FAISS index.

User asks a question:

The FAISS index retrieves the top relevant chunks.

These chunks are fed into the LLM with the user’s question.

The chatbot responds with a generated answer.

Chat History is maintained for better UX and conversation tracking.


