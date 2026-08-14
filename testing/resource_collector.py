import time
import threading
import statistics
import os
import json
import psutil
from pathlib import Path

# ---------------------------------------------------------
# 1. System Pipelines
# ---------------------------------------------------------
from rag_agent.naive_rag import naive_testing
from rag_agent.uncapped_crag import uncapped_testing
from rag_agent.capped_crag import capped_testing


TARGET_KEYWORDS = ("ollama", "llama-server", "llama_server")


def find_target_processes() -> list:
    """Return psutil.Process objects for all processes matching TARGET_KEYWORDS."""

    matches = []
    for proc in psutil.process_iter(['pid', 'name']):
        name = (proc.info['name'] or "").lower()
        if any(keyword in name for keyword in TARGET_KEYWORDS):
            matches.append(psutil.Process(proc.info['pid']))
    return matches


class ResourceSampler:
    """
    Background sampler that periodically polls CPU% and RSS memory
    for all target processes (ollama / llama-server) and aggregates
    them into summary statistics (avg / peak) once stopped.
    """

    def __init__(self, interval: float = 0.2):
        self.interval = interval
        self._samples = []
        self._stop_event = threading.Event()
        self._thread = None
        self._monitored_procs = {}

    def _run(self):
        while not self._stop_event.is_set():
            total_cpu = 0.0
            total_ram_mb = 0.0
            active_names = []
            current_pids = set()

            for proc in psutil.process_iter(['pid', 'name']):
                name = (proc.info['name'] or "").lower()
                if any(keyword in name for keyword in TARGET_KEYWORDS):
                    current_pids.add(proc.info['pid'])

            # Drop processes that have exited since the last tick
            for old_pid in list(self._monitored_procs.keys()):
                if old_pid not in current_pids:
                    del self._monitored_procs[old_pid]

            for pid in current_pids:
                try:
                    if pid not in self._monitored_procs:
                        # First time seeing this PID: prime cpu_percent()
                        # so the next call returns a meaningful delta
                        # instead of 0 (psutil quirk when re-instantiating
                        # Process objects every tick).
                        new_proc = psutil.Process(pid)
                        new_proc.cpu_percent(interval=None)
                        self._monitored_procs[pid] = new_proc
                        continue
                    proc = self._monitored_procs[pid]
                    total_cpu += proc.cpu_percent(interval=None)
                    total_ram_mb += proc.memory_info().rss / (1024 ** 2)
                    active_names.append(proc.name())
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    self._monitored_procs.pop(pid, None)
                    continue

            self._samples.append({
                "t": time.time(),
                "cpu_pct": total_cpu,
                "ram_mb": total_ram_mb,
                "n_processes": len(active_names),
            })
            time.sleep(self.interval)

    def start(self):
        """Reset sample buffer and start the background sampling thread."""
        self._samples = []
        self._monitored_procs = {}
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> dict:
        """Stop sampling and return aggregated avg/peak CPU & RAM stats."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=2)

        if not self._samples:
            return {
                "avg_cpu_pct": None, "peak_cpu_pct": None,
                "avg_ram_mb": None, "peak_ram_mb": None
            }

        cpu_vals = [s["cpu_pct"] for s in self._samples]
        ram_vals = [s["ram_mb"] for s in self._samples]

        return {
            "avg_cpu_pct": round(statistics.mean(cpu_vals), 2),
            "peak_cpu_pct": round(max(cpu_vals), 2),
            "avg_ram_mb": round(statistics.mean(ram_vals), 2),
            "peak_ram_mb": round(max(ram_vals), 2),
        }


# ---------------------------------------------------------
# 2. Experimental Control Procedures
# ---------------------------------------------------------
def reset_context_and_warmup(condition: str, is_initial=False):
    """
    Executes the Context Reset and Warm-up Procedure defined in Chapter 3.3.

    Stops the Ollama server (clearing the KV cache) and re-warms the
    target pipeline with a fixed dummy query before resuming evaluation.
    """

    print("\n[System] Stopping Ollama to clear KV Cache...")
    os.system("ollama stop llama3.1")
    os.system("ollama stop nomic-embed-text")
    time.sleep(2)

    warm_up_times = 3 if is_initial else 1
    print(f"[System] Executing Warm-up Procedure ({warm_up_times} runs)...")

    dummy_query = "What is the mandatory deadline for reporting a suspected data breach to the DPO?"
    for i in range(warm_up_times):
        if condition == "naive":
            naive_testing(user_query=dummy_query)
        elif condition == "uncapped":
            uncapped_testing(user_query=dummy_query)
        elif condition == "capped":
            capped_testing(user_query=dummy_query)
    print("[System] Warm-up complete. Ready for evaluation.\n")


# ---------------------------------------------------------
# 3. Stage 1: Resource + Response Collection Main Loop
# ---------------------------------------------------------
def run_resource_collection(condition: str):
    """
    Stage 1 (Hardware Collection):
    Runs the target pipeline against every test question and, for each one,
    simultaneously records:
      - wall-clock latency and CPU/RAM usage (via ResourceSampler)
      - the model's actual generated `response` and `retrieved_contexts`

    All fields are packed into a single record per question and written
    to one JSON report. This report is later consumed offline by
    ragas_tester.py (Stage 2) for quality scoring. No RAGAS evaluation
    logic is invoked at this stage.
    """

    current_dir = Path(__file__).resolve().parent
    json_path = current_dir / "resource_ragas_dataset.json"

    with open(json_path, "r", encoding="utf-8") as f:
        cases = json.load(f)

    report_records = []
    sampler = ResourceSampler(interval=0.2)

    # Initial 3-run warm-up
    reset_context_and_warmup(condition, is_initial=True)

    for idx, each in enumerate(cases):
        # Chapter 3 Context Reset: restart and warm up once every 3 questions
        if idx > 0 and idx % 3 == 0:
            reset_context_and_warmup(condition, is_initial=False)

        user_query = each["user_input"]
        ground_truth = each["reference"]
        category = each["category"]
        print(f"[{idx+1}/{len(cases)}] Collecting {condition} | Category: {category}")

        # Start monitoring
        start_time = time.perf_counter()
        sampler.start()

        # Route to the target pipeline architecture
        if condition == "naive":
            retrieved = naive_testing(user_query=user_query)
        elif condition == "uncapped":
            retrieved = uncapped_testing(user_query=user_query)
        elif condition == "capped":
            retrieved = capped_testing(user_query=user_query)
        else:
            raise ValueError(f"Unknown condition: {condition}")

        # Stop monitoring
        stats = sampler.stop()
        elapsed = round(time.perf_counter() - start_time, 2)

        print(
            f"   -> Latency: {elapsed}s | Peak RAM: {stats['peak_ram_mb']} MB")

        # Bundle resource metrics together with the generation artifacts
        # (response / retrieved_contexts) in a single record, guaranteeing
        # both evaluation dimensions originate from the same model call.
        record = {
            "id": idx,
            "category": category,
            "user_input": user_query,
            "reference": ground_truth,
            "response": retrieved["response"],
            "retrieved_contexts": retrieved["retrieved_contexts"],
            "wall_clock_seconds": elapsed,
            "avg_cpu_pct": stats["avg_cpu_pct"],
            "peak_cpu_pct": stats["peak_cpu_pct"],
            "avg_ram_mb": stats["avg_ram_mb"],
            "peak_ram_mb": stats["peak_ram_mb"],
        }
        report_records.append(record)

    # Write a single unified report (consumed by ragas_tester.py)
    report_filename = f"resource_report_{condition}.json"
    with open(report_filename, "w", encoding="utf-8") as f:
        json.dump(report_records, f, indent=4, ensure_ascii=False)
    print(f"\n[Success] Resource + response report saved to {report_filename}")

    return report_filename


if __name__ == "__main__":
    target_condition = "naive"

    print("=" * 60)
    print(f"STAGE 1 - RESOURCE COLLECTION: {target_condition.upper()}")
    print("=" * 60)

    run_resource_collection(target_condition)

    os.system("ollama stop llama3.1")
    os.system("ollama stop nomic-embed-text")
    print("All models stopped. Stage 1 collection gracefully completed.")
