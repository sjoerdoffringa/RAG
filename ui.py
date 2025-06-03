import streamlit as st
from rag_module.rag import RAG
from dotenv import load_dotenv
load_dotenv()
import os

# Settings
os.environ["embedding_path"] = "./embeddings/guidance_framework_2/"
show_sidebar = True 

rag = RAG(scope_model_id='73e8819c680c49b2a511a5efbe04876b', LLM_name='mistral-small-latest')
st.title("What can I help with?")

# add toggle button to set RAG explanation to SD, SA or None
#st.markdown("### RAG explanation")
rag_explanation = st.selectbox("Explanation method:", ["Scope Detection", "Source Attribution", "Default RAG", "Default LLM"])
if rag_explanation == "Scope Detection":
    rag.explanation = "SD"
elif rag_explanation == "Source Attribution":
    rag.explanation = "SA"
elif rag_explanation == "Default LLM":
    show_sidebar = False
else:
    rag.explanation = None


rag_LLM = st.selectbox("LLM model:", ["gpt-4o-mini", "mistral-small-latest", "gemini-2.0-flash"])
if rag_LLM == "gpt-4o-mini":
    rag.LLM_name = "gpt-4o-mini"
elif rag_LLM == "mistral-small-latest":
    rag.LLM_name = "mistral-small-latest"
elif rag_LLM == "gemini-2.0-flash":
    rag.LLM_name = "gemini-2.0-flash"

# Only define sidebar content if show_sidebar is True
if show_sidebar:
    sidebar = st.sidebar
    sidebar.title("Documents")
    show_scores = False #sidebar.checkbox("Relevance scores", value=False)
    #sidebar.markdown("---")
else:
    # Fallback if sidebar is hidden
    show_scores = False
    sidebar = None

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
    if rag_explanation == "Default LLM":
        response = {}
        response["answer"] = rag.query_LLM(input_text)
    else:

        response = rag.query(input_text)

        # check if scope prediction exists in response
        if 'scope_prediction' in response:
            if response['scope_prediction'] == 0:
                st.success("Relevant documents found")
            elif response['scope_prediction'] == 1:
                st.warning(f'Would you like to ask: {response['counterfactual']}')
            else:
                st.error("No relevant documents found")

        for i, chunk in enumerate(response['chunks']):
            render_chunk(i, chunk, show_scores)

    st.info(response['answer'])

with st.form("my_form"):
    text = st.text_area("Enter text:", "")
    submitted = st.form_submit_button("Submit")
    if submitted:
        generate_response(text)
