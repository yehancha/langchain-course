from dotenv import load_dotenv
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_classic.chains.history_aware_retriever import create_history_aware_retriever
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import PromptTemplate
from langchain_pinecone import PineconeVectorStore
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

def run_llm(query: str, chat_history: list[tuple[str, str]] = []) -> str:
    print(chat_history)
    embeddings = OllamaEmbeddings(model="qwen3-embedding:0.6b")
    doc_search = PineconeVectorStore(index_name="langchain-doc-index", embedding=embeddings)
    chat = ChatOllama(model="qwen3:0.6b", verbose=True, temperature=0.0)
    # chat = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.0)

    retrieval_prompt = """
    Answer any use questions based solely on the context below and chat history provided:

    <context>
    {context}
    </context>

    chat_history

    {input}
    """
    retrieval_prompt = PromptTemplate(
        template=retrieval_prompt,
        input_variables=["context", "input", "chat_history"],
    )

    stuff_documents_chain = create_stuff_documents_chain(chat, retrieval_prompt)

    rephrase_prompt = """
    Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.

    Chat History:
    {chat_history}
    Follow Up Input: {input}
    Standalone Question:
    """
    rephrase_prompt = PromptTemplate(
        template=rephrase_prompt,
        input_variables=["input", "chat_history"],
    )

    history_aware_retriever = create_history_aware_retriever(chat, doc_search.as_retriever(), rephrase_prompt)
    
    retrieval_chain = create_retrieval_chain(history_aware_retriever, stuff_documents_chain)
    result = retrieval_chain.invoke(input={"input": query, "chat_history": chat_history})

    return {
        "query": result["input"],
        "result": result["answer"],
        "source_documents": result["context"]
    }

if __name__ == "__main__":
    result = run_llm("What is a LangChain Chain?")
    print(result["result"])