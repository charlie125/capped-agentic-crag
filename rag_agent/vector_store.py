import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
import json

from pprint import pprint

embeddings = OllamaEmbeddings(
    model="nomic-embed-text")
PERSIST_DIR = "./vector_db"


# checking local vector database exist
def set_up_vector_db():

    if os.path.exists(PERSIST_DIR) and os.listdir(PERSIST_DIR):
        vector_database = Chroma(
            persist_directory=PERSIST_DIR,
            embedding_function=embeddings
        )
    else:

        file_location = "File path"
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


def formate_data(plaint_data):
    formatted_chunks = []
    for x, (doc, score) in enumerate(plaint_data, 1):
        json_item = {
            "chunk_id": f"chunk{x}",
            "chunk_context": doc.page_content.replace("-\n", "").replace("\n", " "),
            "similarity_score": score
        }
        formatted_chunks.append(json_item)

    json_prompt_string = json.dumps(
        formatted_chunks, ensure_ascii=False, indent=2)
    return json_prompt_string


def vector_db_search(user_query):
    vector_database = set_up_vector_db()
    docs_and_scores = vector_database.similarity_search_with_score(
        user_query, k=3)
    vector_db_results = formate_data(docs_and_scores)
    return vector_db_results
