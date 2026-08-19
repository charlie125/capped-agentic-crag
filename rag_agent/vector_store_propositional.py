import os

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

embeddings = OllamaEmbeddings(model="nomic-embed-text")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Kept in step with vector_store.py: both stores live in kxy500/data/.
DATA_ROOT = os.path.join(PROJECT_ROOT, "data")

PERSIST_DIR = os.path.join(DATA_ROOT, "vector_db_propositional")

FILE_LOCATION = os.path.join(
    DATA_ROOT, "Corporate Data Protection and GDPR Compliance Policy.pdf")

DOCUMENT_NAME = "Corporate Data Protection and GDPR Compliance Policy"

llm = ChatOllama(model="llama3.1", temperature=0, format="json")

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
            persist_directory=PERSIST_DIR,
            embedding_function=embeddings
        )
    else:
        loader = PyPDFLoader(FILE_LOCATION)
        pages = loader.load()

        # Base splitter kept identical to vector_store.py so that chunking
        # granularity is the only variable separating this pipeline from
        # the standard one in the ablation study.
        base_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1200,
            chunk_overlap=200,
            separators=["\n\n", "\n", " "]
        )
        base_chunks = base_splitter.split_documents(pages)

        proposition_documents = []

        for idx, chunk in enumerate(base_chunks):
            try:
                result = proposition_chain.invoke({"text": chunk.page_content})
                propositions = result.get("propositions", [])
            except Exception as e:
                print(f"[warn] chunk {idx+1} decomposition failed: {e}")
                continue

            for prop_text in propositions:
                doc = Document(
                    page_content=prop_text,
                    metadata={
                        "source": chunk.metadata.get("source", ""),
                        "page": chunk.metadata.get("page", 0),
                        "parent_chunk_id": f"base_{idx+1}",
                    },
                )
                proposition_documents.append(doc)

            print(
                f"[{idx+1}/{len(base_chunks)}] -> {len(propositions)} propositions")

        # No collection_metadata override: Chroma's default distance metric is
        # used here exactly as in vector_store.py, keeping the similarity
        # measure constant across the two modes of the ablation.
        vector_database = Chroma.from_documents(
            documents=proposition_documents,
            embedding=embeddings,
            persist_directory=PERSIST_DIR,
        )
        print(f"Propositional Vector Database created "
              f"({len(proposition_documents)} propositions from {len(base_chunks)} base chunks)")

    return vector_database


def format_one(doc):
    """Render a single retrieved unit exactly as vector_store.format_data does.

    Factored out so that the budgeted retriever can price each candidate by the
    string that will actually enter the prompt, citation prefix included, rather
    than by its raw page_content.
    """
    page_label = doc.metadata.get("page", "N/A")
    human_readable_page = page_label + 1
    clean_text = doc.page_content.replace("-\n", "").replace("\n", " ")
    return f"[Source: '{DOCUMENT_NAME}', Page: {human_readable_page}]: {clean_text}"


def format_data(plain_data):
    formatted_chunks = [format_one(doc) for doc, _ in plain_data]

    text_prompt_string = "\n\n".join(formatted_chunks)

    return text_prompt_string


# ---------------------------------------------------------------------------
# Retrieval — two modes for the granularity ablation
#
# Mode 1 (fixed depth):   propositional_vector_search(query, k=3)
#   Drop-in replacement for vector_store.vector_db_search. Retrieves the same
#   NUMBER of units as the standard pipeline. Because a proposition is far
#   shorter than a 1200-character chunk, this mode feeds the LLM substantially
#   less context, so it answers the practical question: what happens if the
#   vector store is swapped out and nothing else changes?
#
# Mode 2 (fixed budget):  propositional_vector_search_budgeted(query, ...)
#   Follows the evaluation protocol of Chen et al. (2023) §6, which compares
#   granularities "under the same computation budget" by capping the maximum
#   number of retrieved tokens at l rather than fixing the unit count. Units
#   are accumulated in similarity order until the budget is reached, so both
#   modes feed the LLM a comparable amount of context. This answers the
#   academic question the Dense X Retrieval paper actually poses: at equal
#   budget, does the finer granularity pack more query-relevant information?
# ---------------------------------------------------------------------------

# Average characters per token, used to convert the token budget into the
# character budget that governs accumulation. The realised Prefill token counts
# are recorded independently by ResourceSampler and must be cross-checked
# against the standard mode to confirm the two budgets actually matched.
CHARS_PER_TOKEN = 4

# Token budget for Mode 2, calibrated by calibrate_budget_from_standard(): the
# standard pipeline supplies a mean of 3,225 characters of formatted context at
# k=3 across the 25 evaluation questions, i.e. ~806 tokens. At this budget Mode 2
# admits roughly 21 propositions per query, well inside MAX_CANDIDATES.
RETRIEVAL_TOKEN_BUDGET = 806

# Number of candidates pulled from the store before budget truncation. Must be
# large enough that the budget, not the candidate pool, is the binding limit;
# calibrate_budget_from_standard() reports whether that holds.
MAX_CANDIDATES = 80


def propositional_vector_search(user_query, k=3):
    """Mode 1 — fixed retrieval depth, mirroring the standard pipeline's k."""
    vector_database = set_up_vector_db()
    docs_and_scores = vector_database.similarity_search_with_score(
        user_query, k=k)
    return format_data(docs_and_scores)


def propositional_vector_search_budgeted(user_query,
                                         token_budget=RETRIEVAL_TOKEN_BUDGET,
                                         max_candidates=MAX_CANDIDATES):
    """Mode 2 — take the top-ranked propositions that fit within a token budget.

    Candidates are consumed strictly in similarity order and accumulation stops
    at the first unit that would overrun the budget, mirroring the truncation of
    a ranked list at l tokens in Chen et al. (2023) §6.

    Cost is measured on the *formatted* unit, citation prefix and separator
    included, because that is the string that actually enters the prompt. A
    proposition carries the same ~70-character prefix as a 1200-character chunk,
    so pricing on raw text alone would let this mode consume markedly more Prefill
    than the standard mode and quietly break the equal-budget premise.

    One deliberate deviation from Chen et al.: whole propositions are admitted or
    rejected and a unit is never cut in half, since severing a proposition would
    destroy the atomicity that motivates the representation. The realised budget
    may therefore fall marginally below l.
    """
    vector_database = set_up_vector_db()
    candidates = vector_database.similarity_search_with_score(
        user_query, k=max_candidates)

    char_budget = token_budget * CHARS_PER_TOKEN
    selected = []
    used_chars = 0

    for doc, score in candidates:
        # len(SEPARATOR) accounts for the "\n\n" that format_data will insert
        # before this unit; the first unit carries no separator.
        cost = len(format_one(doc)) + (0 if not selected else 2)
        if used_chars + cost > char_budget:
            break
        selected.append((doc, score))
        used_chars += cost

    # Guarantee at least one unit even if the top proposition alone overruns
    # the budget, so the pipeline never receives an empty context.
    if not selected and candidates:
        selected = [candidates[0]]

    return format_data(selected)


def report_proposition_stats():
    """Print the length distribution of the stored propositions."""
    import statistics

    vector_database = set_up_vector_db()
    stored = vector_database.get()
    texts = stored.get("documents", []) or []

    if not texts:
        print("[warn] store is empty — nothing to report")
        return

    lengths = sorted(len(t) for t in texts)
    mean_chars = sum(lengths) / len(lengths)

    print(f"\nStored propositions: {len(texts)}")
    print(f"  chars  min {lengths[0]}  median {statistics.median(lengths):.0f}  "
          f"mean {mean_chars:.0f}  max {lengths[-1]}")
    print(f"  approx words per proposition: {mean_chars / 5.5:.1f}  "
          f"(Chen et al. report ~10-20 words for Wikipedia propositions)")
    print(f"  approx tokens per proposition, prefix included: "
          f"{(mean_chars + 70) / CHARS_PER_TOKEN:.1f}\n")


def calibrate_budget_from_standard():
    """Measure the standard pipeline's context volume to fix Mode 2's budget.

    Runs vector_store.vector_db_search over the 25 evaluation questions and
    reports the length of the formatted context it hands to the LLM. The mean is
    the value RETRIEVAL_TOKEN_BUDGET should take, so that both modes enter the
    ablation on the equal computation budget that Chen et al. (2023) require.

    Retrieval only — no LLM is invoked, so this is cheap to run.
    """
    import json
    import statistics
    from pathlib import Path

    from .vector_store import vector_db_search

    dataset_path = (Path(PROJECT_ROOT) / "testing" /
                    "resource_ragas_dataset.json")
    with open(dataset_path, "r", encoding="utf-8") as f:
        cases = json.load(f)

    char_lengths = []
    for case in cases:
        context = vector_db_search(case["user_input"])
        char_lengths.append(len(context))

    mean_chars = sum(char_lengths) / len(char_lengths)
    recommended = round(mean_chars / CHARS_PER_TOKEN)

    print(f"\nStandard pipeline, k=3, n={len(char_lengths)} questions")
    print(f"  formatted context chars  min {min(char_lengths)}  "
          f"median {statistics.median(char_lengths):.0f}  "
          f"mean {mean_chars:.0f}  max {max(char_lengths)}")
    print(f"  -> set RETRIEVAL_TOKEN_BUDGET = {recommended}")

    # Confirm the candidate pool is deep enough that the budget, not
    # MAX_CANDIDATES, is what binds Mode 2.
    prop_db = set_up_vector_db()
    prop_texts = prop_db.get().get("documents", []) or []
    if prop_texts:
        mean_unit = sum(len(t) for t in prop_texts) / len(prop_texts) + 72
        print(f"  at that budget Mode 2 admits roughly "
              f"{mean_chars / mean_unit:.0f} propositions per query "
              f"(MAX_CANDIDATES={MAX_CANDIDATES})\n")

    return recommended


if __name__ == "__main__":
    # Run from the kxy500/ directory so the relative imports resolve:
    #   python -m rag_agent.vector_store_propositional          (build + stats)
    #   python -m rag_agent.vector_store_propositional calibrate
    import sys

    set_up_vector_db()
    report_proposition_stats()

    if len(sys.argv) > 1 and sys.argv[1] == "calibrate":
        calibrate_budget_from_standard()
