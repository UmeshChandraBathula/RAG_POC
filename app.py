from data.employees import generate_employee_data
import json
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import logging
from assistant import Assistant
from prompts import SYSTEM_PROMPT, WELCOME_MESSAGE
from langchain_groq import ChatGroq

if __name__ == "__main__":
    load_dotenv()
    logging.basicConfig(level=logging.info)

    st.set_page_config(page_title= "Umbrella Onboarding", page_icon="-", layout = "wide")

    @st.cache_data(ttl = 3600, show_spinner= "Loading Employee Data...")
    def get_user_data():
        return generate_employee_data(1)[0]

    @st.cache_resource(ttl = 3600, show_spinner= "Loading Vector Store...")
    def init_vector_store(pdf_path):
        try:
            loader = PyPDFLoader(pdf_path)
            docs = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size =  3000, chunk_overlap = 300
            )
            splitz = text_splitter.split_documents(docs)

            embedding_fuction = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

            persistent_path = "./data/vectorstore"
            vectorstore = Chroma.from_documents(
                documents = splitz,
                embedding = embedding_fuction,
                persist_directory = persistent_path)
            
            return vectorstore

        except Exception as e:

            logging.error(f"Error Initializing vector store: {str(e)}")
            st.error(f"Failed to initialize vector store: {str(e)}")

            return None
    
    vector_store = init_vector_store("RAG_POC\data\umbrella_corp_policies.pdf")
    llm = ChatGroq()

    user_data = get_user_data()
    if "user" not in st.session_state:
        st.session_state.user =  user_data

    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "ai", "content": WELCOME_MESSAGE}]

    assistant = Assistant(
        system_prompt = SYSTEM_PROMPT,
        llm = llm,
        message_history=st.session_state.messages,
        employee_information=st.session_state.user,
        vector_store= vector_store
    )

    







    

