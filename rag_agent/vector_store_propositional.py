import json
import os
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

embeddings = OllamaEmbeddings(model="nomic-embed-text")
PERSIST_DIR = "./vector_db_propositional"

llm = ChatOllama(
    model="llama3", temperature=0, format="json"
)

PROPOSITION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are an expert task of decomposing complex legal and compliance documents into simple, atomic propositions.
Decompose the following text into clear, self-contained factual propositions.
Rules:
1. Each proposition must be a single, standalone factual statement.
2. Resolve all pronouns (e.g., replace 'it', 'they', 'the Company' with explicit entities like 'The Data Protection Officer' or 'Personal Data').
3. Do not lose any factual details, dates, numbers, or legal conditions.
4. Return ONLY a JSON object with a key "propositions" containing a list of strings.""",
        ),
        ("human", "Text to decompose:\n\n{text}"),
    ]
)

proposition_chain = PROPOSITION_PROMPT | llm | JsonOutputParser()


def set_up_vector_db():
    if os.path.exists(PERSIST_DIR) and os.listdir(PERSIST_DIR):
        vector_database = Chroma(
            persist_directory=PERSIST_DIR, embedding_function=embeddings
        )
    else:
        file_location = "File path"
        loader = PyPDFLoader(file_location)
        pages = loader.load()

        base_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500, chunk_overlap=100
        )
        base_chunks = base_splitter.split_documents(pages)

        proposition_documents = []

        for idx, chunk in enumerate(base_chunks):
            try:
                result = proposition_chain.invoke(
                    {"text": chunk.page_content}
                )
                propositions = result.get("propositions", [])

                for p_idx, prop_text in enumerate(propositions):
                    doc = Document(
                        page_content=prop_text,
                        metadata={
                            "source": chunk.metadata.get("source", ""),
                            "page": chunk.metadata.get("page", 0),
                            "parent_chunk_id": f"base_{idx+1}",
                            "original_context": chunk.page_content[
                                :200
                            ],
                        },
                    )
                    proposition_documents.append(doc)
            except Exception as e:
                continue

        vector_database = Chroma.from_documents(
            documents=proposition_documents,
            embedding=embeddings,
            persist_directory=PERSIST_DIR,
            collection_metadata={"hnsw:space": "cosine"},
        )
        print("Propositional Vector Database created")

    return vector_database


def propositional_vector_search(user_query):
    vector_database = set_up_vector_db()
    docs_and_scores = vector_database.similarity_search_with_score(
        user_query, k=3)
    return docs_and_scores
