from pathlib import Path
import os, sys
import streamlit as st
from langchain_classic.chains.retrieval_qa.base import RetrievalQA

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from Models.Model import get_embedings, get_llm
from Backend.Data_Ingestion import get_vectorstore


BASE_DIR    = Path(__file__).resolve().parent.parent
DOCS_DIR    = BASE_DIR / "Data" / "Docs"


def process_answer(question):
    vector_db = get_vectorstore()
    retriever = vector_db.as_retriever()

    qa_chain = RetrievalQA.from_chain_type(
        llm=get_llm(),
        chain_type = "stuff",
        retriever = retriever,
        return_source_documents=True
    )

    response = qa_chain.invoke({"query": question})
    answer = response["result"]
    sources = response["sources"]

    return answer, sources