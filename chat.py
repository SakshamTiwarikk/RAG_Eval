import streamlit as st

st.title("Ask the Penguin PDF")
st.caption("RAG chat: local embeddings retrieve chunks, a Groq LLM answers from them.")


@st.cache_resource
def get_ask():
    from RAG1.agent import ask  # indexes the PDF on first import

    return ask


if "history" not in st.session_state:
    st.session_state.history = []

for item in st.session_state.history:
    with st.chat_message(item["role"]):
        st.write(item["text"])
        if item.get("context"):
            with st.expander("Retrieved context"):
                st.write("\n\n---\n\n".join(item["context"]))

if question := st.chat_input("Ask something about penguins"):
    st.session_state.history.append({"role": "user", "text": question})
    with st.chat_message("user"):
        st.write(question)
    with st.chat_message("assistant"):
        with st.spinner("Searching the PDF..."):
            result = get_ask()(question)
        st.write(result["answer"])
        with st.expander("Retrieved context"):
            st.write("\n\n---\n\n".join(result["context"]))
    st.session_state.history.append(
        {"role": "assistant", "text": result["answer"], "context": result["context"]}
    )
