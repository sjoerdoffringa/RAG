import streamlit as st
from rag_module.rag import RAG
from dotenv import load_dotenv
load_dotenv()
import os

# Settings
os.environ["embedding_path"] = "./embeddings/ELOQ_news/"
show_sidebar = True 

rag = RAG()

st.title("What can I help with?")

# Only define sidebar content if show_sidebar is True
if show_sidebar:
    sidebar = st.sidebar
    sidebar.title("Documents")
    show_scores = sidebar.checkbox("Relevance scores", value=True)
    sidebar.markdown("---")
else:
    # Fallback if sidebar is hidden
    show_scores = False  # Or whatever default you want
    sidebar = None  # For safety

def render_chunk(i, chunk, show_scores):
    text = chunk['text']
    score = int(chunk['score'] * 100)

    if sidebar:
        if show_scores:
            if score > 80:
                label, status, icon = f"Highly relevant ({score}%)", st.success, "🟢"
            elif score > 60:
                label, status, icon = f"Relevant ({score}%)", st.warning, "🟡"
            elif score > 40:
                label, status, icon = f"Less relevant ({score}%)", st.error, "🟠"
            else:
                label, status, icon = f"Not relevant ({score}%)", st.error, "🔴"

            with sidebar.expander(f"Document {i+1}", expanded=True, icon=icon):
                status(label)
                st.write(text)
        else:
            with sidebar.expander(f"Document {i+1}", expanded=True):
                st.write(text)

def generate_response(input_text):
    response = rag.query(input_text)

    for i, chunk in enumerate(response['chunks']):
        render_chunk(i, chunk, show_scores)

    st.info(response['answer'])

with st.form("my_form"):
    text = st.text_area("Enter text:", "What are the responsibilities of a scrum master?")
    submitted = st.form_submit_button("Submit")
    if submitted:
        generate_response(text)
