from backend.core import run_llm
import streamlit as st

st.header("Langchain Course: Documentation Helper Bot")

prompt = st.text_input("Prompt", placeholder="Enter your prompt here...")

if "user_prompot_history" not in st.session_state:
    st.session_state.user_prompot_history = []

if "chat_answer_history" not in st.session_state:
    st.session_state.chat_answer_history = []

def create_source_string(sources: set[str]) -> str:
    return "\n".join([f"- {source}" for source in sources])

if prompt:
    with st.spinner("Thinking..."):
        response = run_llm(prompt)
        sources = set([doc.metadata["source"] for doc in response["source_documents"]])
        formatted_response = f"""{response["result"]}\n\nSources:\n{create_source_string(sources)}"""

        st.session_state.user_prompot_history.append(prompt)
        st.session_state.chat_answer_history.append(formatted_response)

if st.session_state.chat_answer_history:
    for chat_answer, user_prompt in zip(st.session_state.chat_answer_history, st.session_state.user_prompot_history):
        st.write(f"User: {user_prompt}")
        st.write(f"Assistant: {chat_answer}")