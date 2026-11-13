import streamlit as st

st.set_page_config(page_title="RAG Evaluation Lab", layout="wide")

st.navigation(
    [
        st.Page("chat.py", title="Ask the PDF", icon=":material/chat:"),
        st.Page("dashboard.py", title="Evaluation Results", icon=":material/monitoring:"),
    ]
).run()
