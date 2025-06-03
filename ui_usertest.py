import streamlit as st
from rag_module.rag import RAG
from dotenv import load_dotenv
load_dotenv()
import os

# Settings
os.environ["embedding_path"] = "./embeddings/guidance_framework_2/"
show_sidebar = True 
rag_SA = RAG(explanation='SA')
rag_Default = RAG(explanation=None)

#st.title("What can I help with?")

# Only define sidebar content if show_sidebar is True
if show_sidebar:
    sidebar = st.sidebar
    sidebar.title("Documents")
else:
    sidebar = None

def render_chunk(i, chunk):
    text = chunk['text']

    if sidebar:
        with sidebar.expander(f"Document {i+1}", expanded=True):
            st.write(text)

def generate_response(input_text):
    response_SA = rag_SA.query(input_text)
    response_Default = rag_Default.query(input_text)

    for i, chunk in enumerate(response_SA['chunks']):
        render_chunk(i, chunk)

    st.markdown("### Answer A:")
    st.info(response_Default['answer'])

    st.markdown("### Answer B:")
    st.info(response_SA['answer'])

with st.form("my_form"):
    st.markdown("### Query:")
    text = st.text_area("", "")
    submitted = st.form_submit_button("Submit")
    if submitted:
        generate_response(text)
