import streamlit as st
from PyPDF2 import PdfReader
from pdf2image import convert_from_path
import pytesseract
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain

import os

st.set_page_config(page_title="RagChatBot", layout="wide")

st.markdown("""
## Rag Chat Bot
       Retrieval Augmented Generation extracts information from external sources and passes to llm's to genarate Answer     
""")

pdf_folder_path = "ALL_pdfs"  # replace with actual path
pdf_docs = [os.path.join(pdf_folder_path, f) for f in os.listdir(pdf_folder_path) if f.endswith('.pdf')]
#pdf_docs=['attention-is-all-you-need-Paper.pdf']

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "faiss_ready" not in st.session_state:
    st.session_state["faiss_ready"] = False

def is_scanned_pdf(pdf_path):
    try:
        reader = PdfReader(pdf_path)
        for page in reader.pages:
            if page.extract_text() and page.extract_text().strip():
                return False
        return True
    except Exception:
        return True  # if reading fails, treat it as scanned

def extract_text_from_scanned(pdf_path):
    try:
        images = convert_from_path(pdf_path)
        text = ""
        for image in images:
            text += pytesseract.image_to_string(image)
        return text
    except Exception as e:
        st.error(f"OCR failed: {e}")
        return ""
    

def get_pdf_text(pdf_docs):
    full_text = ""
    for pdf in pdf_docs:
        if is_scanned_pdf(pdf):
            st.warning(f"OCR processing scanned PDF: {os.path.basename(pdf)}")
            full_text += extract_text_from_scanned(pdf)
        else:
            st.info(f"Extracting text from digital PDF: {os.path.basename(pdf)}")
            reader = PdfReader(pdf)
            for page in reader.pages:
                full_text += page.extract_text() or ""
    return full_text

def get_chunk_text(text):
    text_splitter =  RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    st.session_state["faiss_ready"] = True
    return chunks


def get_vector_stores(chunks):
    embeddings = OllamaEmbeddings(model="nomic-embed-text") 
    vector_db = FAISS.from_texts(chunks,embeddings)
    vector_db.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = "Here is some relevant information:\n{context}\n\nNow, answer the question: {question}"
    llm = Ollama(model='gemma:2b', temperature=0.7)
    prompt = PromptTemplate(template=prompt_template,input_variables=["context", "question"])
    chain = load_qa_chain(llm, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_input):
    embeddings = OllamaEmbeddings(model="nomic-embed-text") 

    if not os.path.exists("faiss_index"):
        st.error("FAISS index not found. Please click 'Submit & Process' first.")
        return

    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

    docs = new_db.similarity_search(user_input,k=3)
    chain = get_conversational_chain()

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    response = chain.run(input_documents=docs, question=user_input)


    st.session_state.chat_history.append({
        "user": user_input,
        "bot": response
    })

    st.write("Reply: ", response)


def main():
    st.header("AI clone chatbot💁")

    query = st.text_input("Ask a question:")
    if query:
        user_input(query)

    if st.button("Submit & Process", key="process_button") : 
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_chunk_text(raw_text)
                get_vector_stores(text_chunks)
                st.success("Done")


    if st.session_state.chat_history:
        st.markdown("### Chat History")
        for pair in st.session_state.chat_history:
            st.markdown(f"**You:** {pair['user']}")
            st.markdown(f"**Bot:** {pair['bot']}")


if __name__ == "__main__":
    main()