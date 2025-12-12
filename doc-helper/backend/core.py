from dotenv import load_dotenv
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import PromptTemplate
from langchain_pinecone import PineconeVectorStore
from langchain_ollama import ChatOllama, OllamaEmbeddings

load_dotenv()

def run_llm(query: str) -> str:
    embeddings = OllamaEmbeddings(model="qwen3-embedding:0.6b")
    doc_search = PineconeVectorStore(index_name="langchain-doc-index", embedding=embeddings)
    chat = ChatOllama(model="qwen3:0.6b", verbose=True, temperature=0.0)

    retrieval_prompt = """
    Answer any use questions based solely on the context below:

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
    retrieval_chain = create_retrieval_chain(doc_search.as_retriever(), stuff_documents_chain)
    result = retrieval_chain.invoke(input={"input": query})

    return result

if __name__ == "__main__":
    result = run_llm("What is a LangChain Chain?")
    print(result["answer"])