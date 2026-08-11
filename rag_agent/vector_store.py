import os
import json

# LangChain
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader


# Ollama Embedding model
from langchain_ollama import OllamaEmbeddings


# Vector Database
from langchain_chroma import Chroma


embeddings = OllamaEmbeddings(
    model="nomic-embed-text")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PERSIST_DIR = os.path.join(PROJECT_ROOT, "data", "vector_db")

FILE_LOCATION = os.path.join(
    PROJECT_ROOT, "data", "Corporate Data Protection and GDPR Compliance Policy.pdf")


# checking local vector database exist
def set_up_vector_db():

    if os.path.exists(PERSIST_DIR) and os.listdir(PERSIST_DIR):
        vector_database = Chroma(
            persist_directory=PERSIST_DIR,
            embedding_function=embeddings
        )
    else:
        file_location = FILE_LOCATION
        loader = PyPDFLoader(file_location)
        pages = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1200,
            chunk_overlap=200,
            separators=["\n\n", "\n", " "]
        )

        chunks = splitter.split_documents(pages)

        vector_database = Chroma.from_documents(documents=chunks,
                                                embedding=embeddings,
                                                persist_directory=PERSIST_DIR)
        print("Vector Database created!")
    return vector_database


def format_data(plain_data):
    formatted_chunks = []

    for doc, _ in plain_data:
        page_label = doc.metadata.get("page", "N/A")
        human_readable_page = page_label + 1
        clean_text = doc.page_content.replace("-\n", "").replace("\n", " ")
        formatted_chunks.append(
            f"[Source: 'Corporate Data Protection and GDPR Compliance Policy', Page: {page_label}]: {clean_text}")

    text_prompt_string = "\n\n".join(formatted_chunks)

    return text_prompt_string


def vector_db_search(user_query):
    vector_database = set_up_vector_db()
    docs_and_scores = vector_database.similarity_search_with_score(
        user_query, k=3)
    vector_db_results = format_data(docs_and_scores)
    return vector_db_results
