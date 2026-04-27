from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
import streamlit as st

def get_embedings():
    embedding = HuggingFaceEmbeddings()
    return embedding

def get_llm():
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key = st.secrets["GEMINI_API_KEY"],
        temperature = 0.1
    )
    return llm