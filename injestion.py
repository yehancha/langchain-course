import os

from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_ollama import OllamaEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import CharacterTextSplitter

load_dotenv() 

if __name__ == "__main__":
    print("Injesting...")

    loader = TextLoader("mediumblog1.txt")
    documents = loader.load()

    print("Splitting...")

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)

    print(f"Created {len(texts)} chunks")

    embeddings = OllamaEmbeddings(model="qwen3-embedding:0.6b")

    print("Start injestion...")

    PineconeVectorStore.from_documents(
        texts,
        embeddings,
        index_name=os.getenv("INDEX_NAME"),
    )

    print("Injestion complete")