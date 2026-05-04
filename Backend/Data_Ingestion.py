import os,sys
from langchain_community.document_loaders import DirectoryLoader, UnstructuredFileLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
import streamlit as st
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from Data.Vector_DB_Pine import get_index_name
from Models.Model import get_embedings

BASE_DIR    = Path(__file__).resolve().parent.parent
DOCS_DIR    = BASE_DIR / "Data" / "Docs"

def get_vectorstore():
    pc = Pinecone(api_key=st.secrets["PINE_API_KEY"])
    index_name = get_index_name()

    return PineconeVectorStore(
        index=pc.Index(index_name),
        embedding=get_embedings()
    )

def process_initial_document_to_pinecone():
    loader = DirectoryLoader(
        path = str(DOCS_DIR) ,
        glob = "./*.pdf",
        loader_cls = UnstructuredFileLoader
    )

    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size =2000,
        chunk_overlap = 200
    )

    texts = text_splitter.split_documents(documents)
    vector_db = get_vectorstore()
    text_contents = [doc.page_content for doc in texts]
    metadata = [doc.metadata for doc in texts]
    vector_db.add_texts(texts=text_contents, metadatas=metadata)
    
    return 0

def get_processed_files():
    processed_file_path = DOCS_DIR / "processed.txt"
    if processed_file_path.exists():
        return set(processed_file_path.read_text().splitlines())
    return set()

def save_processed_file(filename: str):
    processed_files_path = DOCS_DIR / "processed.txt"
    with open(processed_files_path, "a") as f:
        f.write(filename + "\n")

def process_new_documents_to_pinecone(uploaded_files):
    processed = get_processed_files()
    new_files = []

    for uploaded_file in uploaded_files:
        if uploaded_file.name not in processed:
            file_path = DOCS_DIR / uploaded_file.name
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            new_files.append(file_path)

    if not new_files:
        st.info("No new documents to process.")
        return
    
    loader = DirectoryLoader(
        path=str(DOCS_DIR),
        glob=[f"./{f.name}" for f in new_files],  
        loader_cls=UnstructuredFileLoader
    )
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=200
    )
    texts = text_splitter.split_documents(documents)
    vector_db = get_vectorstore()
    text_contents = [doc.page_content for doc in texts]
    metadata = [doc.metadata for doc in texts]
    vector_db.add_texts(texts=text_contents, metadatas=metadata)

    for f in new_files:
        save_processed_file(f.name)

    get_vectorstore.clear()

    st.success(f"{len(new_files)} new document(s) embedded successfully.")


process_initial_document_to_pinecone()