import asyncio
import os
import ssl
from typing import Any, Dict, List

import certifi
from dotenv import load_dotenv
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_tavily import TavilyCrawl, TavilyExtract, TavilyMap
from logger import (Colors, log_error, log_header, log_info, log_success,
                    log_warning)

load_dotenv()

# Configure SSL context to use certifi certificates
ssl_context = ssl.create_default_context(cafile=certifi.where())
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

embeddings = OllamaEmbeddings(model="qwen3-embedding:0.6b")
vector_store = PineconeVectorStore(
    index_name="langchain-doc-index", embedding=embeddings
)
tavily_extract = TavilyExtract()
tavily_map = TavilyMap(max_depth=5, max_breadth=20, max_pages=1000)
tavily_crawl = TavilyCrawl()


async def index_document(documents: List[Document], batch_size: int = 50):
    log_header("VECTOR STORAGE PHASE")
    log_info(f"VectoreStore: Indexing {len(documents)} documents", Colors.DARKCYAN)

    batches = [
        documents[i : i + batch_size] for i in range(0, len(documents), batch_size)
    ]

    log_info(f"VectoreStore: Split into {len(batches)} batches")

    # Use semaphore to limit concurrent requests to prevent session closure
    # PineconeVectorStore's HTTP session can't handle unlimited concurrent requests
    semaphore = asyncio.Semaphore(1)  # Allow max 3 concurrent requests

    async def add_batch(batch: List[Document], batch_number: int) -> bool:
        async with semaphore:  # Acquire semaphore before making request
            try:
                await vector_store.aadd_documents(batch)
                log_success(
                    f"VectoreStore: Added batch {batch_number} of {len(batches)}"
                )
                return True
            except Exception as e:
                log_error(f"VectoreStore: Error adding batch {batch_number}: {e}")
                return False

    tasks = [add_batch(batch, i + 1) for i, batch in enumerate(batches)]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    success_count = results.count(True)

    if success_count == len(batches):
        log_success(f"VectoreStore: Successfully indexed {success_count} batches")
    else:
        log_warning(
            f"VectoreStore: Indexed {success_count} batches out of {len(batches)}"
        )


async def main():
    log_header("DOCUMENTATION INJESTION PIPELINE")
    log_info("TavilyCrawl: Starting web crawling for documentation", Colors.PURPLE)

    res = tavily_crawl.invoke(
        {
            "url": "https://python.langchain.com/",
            "max_depth": 5,
            "extra_depth": "advanced",
            "instructions": "founder of langchain",
        }
    )
    all_docs = [
        Document(page_content=result["raw_content"], metadata={"source": result["url"]})
        for result in res["results"]
    ]
    log_success(f"TavilyCrawl: Found {len(all_docs)} documents")

    log_header("DOCUMENT CHUNKING")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=200)
    chunks = text_splitter.split_documents(all_docs)
    log_success(
        f"Text Splitter: Split {len(all_docs)} documents into {len(chunks)} chunks"
    )

    await index_document(chunks, batch_size=10)

    log_header("PIPELINE COMPLETED")
    log_success("Documentation ingestion completed successfully")
    log_info("Summary:", Colors.BOLD)
    log_info(f"    URLs mapped: {len(res["results"])}")
    log_info(f"    Documents extracted: {len(all_docs)}")
    log_info(f"    Chunks created: {len(chunks)}")


if __name__ == "__main__":
    asyncio.run(main())
