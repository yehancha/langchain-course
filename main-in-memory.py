from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import CharacterTextSplitter

load_dotenv()


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def main():
    embeddings = OllamaEmbeddings(model="qwen3-embedding:0.6b")

    # print("Embedding documents only once...")

    # loader = PyPDFLoader("2210.03629v3.pdf")
    # documents = loader.load()

    # text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=30, separator="\n")
    # docs = text_splitter.split_documents(documents)

    # vectorstore = FAISS.from_documents(docs, embeddings)
    # vectorstore.save_local("faiss_index")

    # Load the vectorstore
    print("Loading vectorstore...")

    vectorstore = FAISS.load_local(
        "faiss_index", embeddings, allow_dangerous_deserialization=True
    )

    retrieval_prompt = """
    Answer any use questions based solely on the context below. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.

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
    llm = ChatOllama(model="qwen3:0.6b", temperature=0.0)

    retrieval_chain = (
        {
            "context": vectorstore.as_retriever() | format_docs,
            "input": RunnablePassthrough(),
        }
        | retrieval_prompt
        | llm
    )

    result = retrieval_chain.invoke("Give me the gist of ReAct in 3 sentences")
    print(result)


if __name__ == "__main__":
    main()
