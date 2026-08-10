import json
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama
from .vector_store import vector_db_search

llm = ChatOllama(model="llama3.1", temperature=0)

SYSTEM_PROMPT = """You are a strict internal corporate compliance assistant.
    Your core mission is to answer the user's query using ONLY the provided internal database context.
    If the answer cannot be verified purely from the context, respond exactly with: "The relevant answer cannot be found in the internal compliance document database."
    """

HUMAN_TEMPLATE = """
    === INTERNAL CORPORATE DATABASE CONTEXT ===
    {context}
    === END OF CONTEXT ===

    USER QUERY: {query}"""


def run_naive_rag(user_query):
    global llm

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", HUMAN_TEMPLATE),
    ])

    db_result = vector_db_search(user_query)

    chain = prompt | llm | StrOutputParser()

    responses = chain.invoke({
        "query": user_query,
        "context": db_result
    })

    return responses, db_result


def naive_stream(user_query):
    """Generator for Django SSE view."""
    global llm

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", HUMAN_TEMPLATE),
    ])

    db_result = vector_db_search(user_query)
    chain = prompt | llm | StrOutputParser()

    yield {"type": "thinking", "node": "retrieve", "label": "Retrieving relevant documents..."}
    yield {"type": "thinking", "node": "generate", "label": "Generating answer..."}

    for chunk in chain.stream({"query": user_query, "context": db_result}):
        if chunk:
            yield {"type": "token", "text": chunk}


def naive_testing(user_query):
    response, db_result = run_naive_rag(user_query)

    cleaned_contexts = [each["chunk_context"]
                        for each in json.loads(db_result)]

    data = {"retrieved_contexts": cleaned_contexts,
            "response": str(response), }
    return data
