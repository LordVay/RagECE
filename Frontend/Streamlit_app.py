import streamlit as st
import os, sys
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from Models.Model import get_llm
from Backend.Data_Retrieval import process_answer
from Backend.Data_Ingestion import process_new_documents_to_pinecone

llm = get_llm()

st.set_page_config(
    page_title="RagTRONICS",
    page_icon="",
    layout="centered"
)

st.title("RagTRONICS")

def chat_history():
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

def upload_file():
    if "upload_file" not in st.session_state:
        uploaded_file = st.file_uploader("Upload a PDF File", type=["pdf"])
        if uploaded_file is not None:
            process_new_documents_to_pinecone(uploaded_files=uploaded_file)

def user_prompt():
    user_prompt = st.chat_input("Ask Electronics Stuff")
    if user_prompt:
        with st.chat_message("user"):
            st.markdown(user_prompt)
        st.session_state.chat_history.append({"role":"user", "content":user_prompt})

        history = [{"role":"system", "content":"You are a helpful Chatbot Assistant"}, *st.session_state.chat_history]
        answer, sources = process_answer(history)

        st.session_state.chat_history.append({"role":"assistant", "content":answer})

        with st.chat_message("assistant"):
            st.markdown(answer)
            st.markdown(sources)