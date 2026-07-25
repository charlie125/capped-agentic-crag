from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper, llm_factory
from ragas import SingleTurnSample, EvaluationDataset
from ragas.embeddings.base import embedding_factory
from openai import AsyncOpenAI
import pandas as pd
from datasets import load_dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from langchain_ollama import OllamaEmbeddings, ChatOllama
from ragas.run_config import RunConfig
import json
import os
import pprint
from kxy500.rag_agent.capped_agentic_rag import capped_rag_testing
from kxy500.rag_agent.uncapped_agentic_rag import uncapped_rag
from kxy500.rag_agent.linear_rag import linear_rag_testing


script_dir = os.path.dirname(os.path.abspath(__file__))
json_path = os.path.join(script_dir, "ragas_test.json")

sample_space = []

# Create Sample for testing
# user_input: input,
# retrieved_contexts: retrieved contexts from vector db,
# response:  AI response,
# reference: Ground truth / answer

with open(json_path, "r") as files:
    cases = json.load(files)

    for each in cases:
        # This section is to switch for testing three different systems

        # capped_data = capped_rag_testing(user_query=each["user_input"])

        # sample = SingleTurnSample(
        #     user_input=each["user_input"],
        #     retrieved_contexts=capped_data["retrieved_contexts"],
        #     response=capped_data["response"],
        #     reference=each["reference"]
        # )

        # linear_data = linear_rag_testing(user_query=each["user_input"])

        # sample = SingleTurnSample(
        #     user_input=each["user_input"],
        #     retrieved_contexts=linear_data["retrieved_contexts"],
        #     response=linear_data["response"],
        #     reference=each["reference"]
        # )

        # uncapped_data = uncapped_rag_testing(user_query=each["user_input"])

        # sample = SingleTurnSample(
        #     user_input=each["user_input"],
        #     retrieved_contexts=uncapped_data["retrieved_contexts"],
        #     response=uncapped_data["response"],
        #     reference=each["reference"]
        # )

        sample_space.append(sample)

created_dataset = EvaluationDataset(samples=sample_space)
print(f"Length of dataset: {len(sample_space)}")
print(created_dataset)


# LLM and embeddings
lc_chat = ChatOllama(
    model="llama3.1",
    temperature=0,
    base_url="http://localhost:11434"
)

lc_embed = OllamaEmbeddings(
    model="nomic-embed-text",
    base_url="http://localhost:11434"
)

llm = LangchainLLMWrapper(lc_chat)
embedding = LangchainEmbeddingsWrapper(lc_embed)


# Evaluation steps
run_config = RunConfig(timeout=600, max_retries=2, max_workers=1)

result = evaluate(
    dataset=created_dataset,
    metrics=[
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    ],
    llm=llm,
    embeddings=embedding,
    run_config=run_config
)


print("\n--- Evaluation Score ---")
print(result)

df = result.to_pandas()
print("\n--- Statistics ---")
print(df)
