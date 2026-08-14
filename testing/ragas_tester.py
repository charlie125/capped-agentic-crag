import json

# ---------------------------------------------------------
# 1. RAGAS Metrics & LLM Imports
# ---------------------------------------------------------
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas import SingleTurnSample, EvaluationDataset
from ragas.run_config import RunConfig
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from langchain_ollama import OllamaEmbeddings, ChatOllama


# ---------------------------------------------------------
# 2. Stage 2: Load Stage-1 Report (offline, no generation)
# ---------------------------------------------------------
def load_report_as_ragas_dataset(report_path: str) -> EvaluationDataset:
    """
    Loads the resource_report_{condition}.json produced by
    pure_resource_collector.py and builds a RAGAS EvaluationDataset
    directly from the `response` / `retrieved_contexts` already stored
    in it.

    This function never calls naive_testing / uncapped_testing /
    capped_testing and never regenerates any response. This guarantees
    that the answers being scored are exactly the same ones produced
    during the resource-measurement pass, avoiding any mismatch that
    could arise from LLM generation non-determinism across separate runs.
    """
    with open(report_path, "r", encoding="utf-8") as f:
        records = json.load(f)

    sample_space = []
    for record in records:
        sample = SingleTurnSample(
            user_input=record["user_input"],
            retrieved_contexts=record["retrieved_contexts"],
            response=record["response"],
            reference=record["reference"],
        )
        sample_space.append(sample)

    return EvaluationDataset(samples=sample_space)


# ---------------------------------------------------------
# 3. RAGAS Evaluator
# ---------------------------------------------------------
def run_ragas_evaluation(created_dataset: EvaluationDataset, condition: str):
    """Runs the RAGAS metric suite (faithfulness, answer_relevancy,
    context_precision, context_recall) against the given dataset using
    a local llama3.1 instance as judge, and exports the results to CSV."""

    print("\n--- Starting RAGAS Evaluation (offline scoring) ---")
    lc_chat = ChatOllama(model="llama3.1", temperature=0,
                         base_url="http://localhost:11434")
    lc_embed = OllamaEmbeddings(
        model="nomic-embed-text", base_url="http://localhost:11434")
    llm = LangchainLLMWrapper(lc_chat)
    embedding = LangchainEmbeddingsWrapper(lc_embed)

    run_config = RunConfig(timeout=600, max_retries=2, max_workers=1)

    result = evaluate(
        dataset=created_dataset,
        metrics=[faithfulness, answer_relevancy,
                 context_precision, context_recall],
        llm=llm,
        embeddings=embedding,
        run_config=run_config
    )

    df = result.to_pandas()
    csv_filename = f"ragas_results_{condition}.csv"
    df.to_csv(csv_filename, index=False)

    print(f"\n--- RAGAS Statistics for {condition} ---")
    print(result)
    print(f"Results saved to {csv_filename}")

    return result


# ---------------------------------------------------------
# 4. Stage 2 Main Entry
# ---------------------------------------------------------
def run_offline_ragas_scoring(condition: str, report_path: str = None):
    """
    Stage 2 (Offline Scoring):
    Takes the Stage-1 JSON report that already contains `response` and
    `retrieved_contexts`, and feeds it directly to RAGAS for grading.
    No model generation call is made at this stage.
    """

    if report_path is None:
        report_path = f"resource_report_{condition}.json"

    print("=" * 60)
    print(f"STAGE 2 - OFFLINE RAGAS SCORING: {condition.upper()}")
    print(f"Reading pre-generated responses from: {report_path}")
    print("=" * 60)

    evaluation_dataset = load_report_as_ragas_dataset(report_path)
    run_ragas_evaluation(evaluation_dataset, condition)


if __name__ == "__main__":
    target_condition = "capped"
    run_offline_ragas_scoring(target_condition)
