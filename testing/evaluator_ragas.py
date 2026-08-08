import json
import os
import time
import pandas as pd

# ragas metrics
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas import SingleTurnSample, EvaluationDataset
from ragas.embeddings.base import embedding_factory
from ragas.run_config import RunConfig
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)


# Local model and embedding
from langchain_ollama import OllamaEmbeddings, ChatOllama


# import three rag system
from naive_rag import naive_testing
from uncapped_rag import uncapped_testing
from capped_rag import capped_testing


script_dir = os.path.dirname(os.path.abspath(__file__))
json_path = os.path.join(script_dir, "ragas_test.json")


# Create Sample for testing
# user_input: input,
# retrieved_contexts: retrieved contexts from vector db,
# response:  AI response,
# reference: Ground truth / answer

def set_up_sample(condition):

    sample_space = []

    with open(json_path, "r") as files:
        cases = json.load(files)

        counts = 0
        with open(f"{condition}_latency.txt", "a") as f:
            for each in cases:
                start = time.perf_counter()

                if condition == "naive":
                    retrieved = naive_rag_testing(
                        user_query=each["user_input"])
                elif condition == "uncapped":
                    retrieved = uncapped_rag_testing(
                        user_query=each["user_input"])
                elif condition == "capped":
                    retrieved = capped_rag_testing(
                        user_query=each["user_input"])

                end = time.perf_counter()

                total = round(end - start, 2)
                print(total)

                t = {"id": counts, "start": start, "end": end,
                     "total": total, "condition": condition}

                f.write(f"{json.dumps(t)}\n")
                counts += 1

                sample = SingleTurnSample(
                    user_input=each["user_input"],
                    retrieved_contexts=retrieved["retrieved_contexts"],
                    response=retrieved["response"],
                    reference=each["reference"]
                )

                sample_space.append(sample)

    created_dataset = EvaluationDataset(samples=sample_space)
    print(f"Length of dataset: {len(sample_space)}")
    print(created_dataset)
    return created_dataset


def evaluator(created_dataset):
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
