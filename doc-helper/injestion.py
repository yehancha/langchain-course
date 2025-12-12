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
vector_store = PineconeVectorStore(index_name="langchain-doc-index", embedding=embeddings)
tavily_extract = TavilyExtract()
tavily_map = TavilyMap(max_depth=5, max_breadth=20, max_pages=1000)
tavily_crawl = TavilyCrawl()


async def main():
    log_header("DOCUMENTATION INJESTION PIPELINE")
    log_info("TavilyCrawl: Starting web crawling for documentation", Colors.PURPLE)

    res = tavily_crawl.invoke({
        "url": "https://python.langchain.com/",
        "max_depth": 5,
        "extra_depth": "advanced",
        "instructions": "content on ai agents"
    })
    all_docs = [Document(page_content=result["raw_content"], metadata={ "source": result["url"] }) for result in res["results"]]
    log_success(f"TavilyCrawl: Found {len(all_docs)} documents")


if __name__ == "__main__":
    asyncio.run(main())