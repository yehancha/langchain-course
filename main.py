import os

from dotenv import load_dotenv
from langchain_classic import hub
from langchain_classic.chains.combine_documents import \
    create_stuff_documents_chain
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def main():
    print("Retrieving...")

    embeddings = OllamaEmbeddings(model="qwen3-embedding:0.6b")
    llm = ChatOllama(model="qwen3:0.6b", temperature=0.0)

    query = "What is Pinecone in machine learning?"

    vectorstore = PineconeVectorStore(
        embedding=embeddings,
        index_name=os.getenv("INDEX_NAME"),
    )

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

    retrieval_chain = (
        {
            "context": vectorstore.as_retriever() | format_docs,
            "input": RunnablePassthrough(),
        }
        | retrieval_prompt
        | llm
    )

    result = retrieval_chain.invoke(query)

    print(result)


if __name__ == "__main__":
    main()
