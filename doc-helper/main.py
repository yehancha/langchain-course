from backend.core import run_llm
import streamlit as st

st.header("Langchain Course: Documentation Helper Bot")

prompt = st.text_input("Prompt", placeholder="Enter your prompt here...")

if "chat_history" not in st.session_state:
    st.session_state.user_prompot_history = []
    st.session_state.chat_answer_history = []
    st.session_state.chat_history = []

def create_source_string(sources: set[str]) -> str:
    return "\n".join([f"- {source}" for source in sources])

if prompt:
    with st.spinner("Thinking..."):
        response = run_llm(prompt, st.session_state.chat_history)
        sources = set([doc.metadata["source"] for doc in response["source_documents"]])
        formatted_response = f"""{response["result"]}\n\nSources:\n{create_source_string(sources)}"""

        st.session_state.user_prompot_history.append(prompt)
        st.session_state.chat_answer_history.append(formatted_response)
        st.session_state.chat_history.append(("human", prompt))
        st.session_state.chat_history.append(("assistant", response["result"]))

if st.session_state.chat_answer_history:
    for chat_answer, user_prompt in zip(st.session_state.chat_answer_history, st.session_state.user_prompot_history):
        st.write(f"User: {user_prompt}")
        st.write(f"Assistant: {chat_answer}")