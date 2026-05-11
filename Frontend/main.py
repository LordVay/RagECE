import streamlit as st
import os, sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from Models.Model import get_llm
from Backend.Data_Retrieval import process_answer
from Backend.Data_Ingestion import process_new_documents_to_pinecone

# ── Initialize session state immediately ──────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "file_uploaded" not in st.session_state:
    st.session_state.file_uploaded = False

# ── Load LLM safely ───────────────────────────────────────────────────────────
try:
    llm = get_llm()
except Exception as e:
    st.error(f"Failed to load LLM: {e}")
    st.stop()

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RagTRONICS",
    page_icon="",
    layout="centered"
)
st.title("RagTRONICS")


def upload_file():
        uploaded_file = st.file_uploader("Upload a PDF File", type=["pdf"])
        if uploaded_file is not None:
            file_key = f"processed_{uploaded_file.name}"
            if file_key not in st.session_state:
                try:
                    process_new_documents_to_pinecone(uploaded_files=uploaded_file)
                    st.session_state[file_key] = True
                    st.session_state.file_uploaded = True
                    st.success(f"'{uploaded_file.name}' processed successfully!")
                except Exception as e:
                    st.error(f"Failed to process file: {e}")
            else:
                st.info(f"'{uploaded_file.name}' has already been processed.")


def chat_history():
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


def user_prompt():
    prompt = st.chat_input("Ask Electronics Stuff")
    if prompt:
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.chat_history.append({"role": "user", "content": prompt})

        # ── Build pairs safely ────────────────────────────────────────────────
        chat_history_pairs = []
        msgs = st.session_state.chat_history
        for i in range(0, len(msgs) - 1, 2):
            if msgs[i]["role"] == "user" and msgs[i + 1]["role"] == "assistant":
                chat_history_pairs.append(
                    (msgs[i]["content"], msgs[i + 1]["content"])
                )

        # ── Get answer ────────────────────────────────────────────────────────
        try:
            answer, sources = process_answer(prompt, chat_history_pairs)
        except Exception as e:
            answer = f"Error getting answer: {e}"
            sources = []

        st.session_state.chat_history.append({"role": "assistant", "content": answer})

        with st.chat_message("assistant"):
            st.markdown(answer)
            for doc in sources:
                st.markdown(f"Sources: {doc.metadata}")



with st.sidebar:
    upload_file()
    
chat_history()
user_prompt()


