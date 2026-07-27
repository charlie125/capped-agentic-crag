from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama
from vector_store import vector_db_search

SYSTEM_PROMPT = """You are a strict internal corporate compliance assistant.
    Your core mission is to answer the user's query using ONLY the provided internal database context.
    If the answer cannot be verified purely from the context, respond exactly with: "The relevant answer cannot be found in the internal compliance document database."
    """

HUMAN_TEMPLATE = """
    === INTERNAL CORPORATE DATABASE CONTEXT ===
    {context}
    === END OF CONTEXT ===

    USER QUERY: {query}"""


def run_linear_rag(user_query):
    llm = ChatOllama(model="llama3.1", temperature=0)

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


def linear_rag_respones(user_query):
    response = run_linear_rag(user_query)
    return response


def linear_rag_testing(user_query):
    response, db_result = run_linear_rag(user_query)
    data = {"retrieved_contexts": [db_result], "response": str(response), }
    return data
