from pinecone import Pinecone, ServerlessSpec
import streamlit as st

def get_index_name():
    index_name = "ragtronics"
    return index_name

def create_index():
    pc = Pinecone(api_key=st.secrets["PINE_API_KEY"])
    index_name = get_index_name()

    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=384,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region = "us-east-1"
            )
        )

    index = pc.Index(index_name)

    return index

create_index()


