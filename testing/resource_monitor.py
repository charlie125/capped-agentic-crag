import time
import threading
import statistics
import os
import json
import psutil
from langchain_core.messages import HumanMessage
from rag_agent.capped_crag import build_capped_graph
from rag_agent.uncapped_crag import build_uncapped_graph


TARGET_KEYWORDS = ("ollama", "llama-server", "llama_server")


def find_target_processes() -> list:
    """
    Re-scans ALL running processes and returns psutil.Process handles for
    every one whose name matches our target keywords. Called on every
    sample tick (not just once), because llama-server is spawned/killed
    dynamically during the model's lifecycle.
    """
    matches = []
    for proc in psutil.process_iter(['pid', 'name']):
        name = (proc.info['name'] or "").lower()
        if any(keyword in name for keyword in TARGET_KEYWORDS):
            matches.append(psutil.Process(proc.info['pid']))
    return matches


class ResourceSampler:
    """
    Background thread that re-discovers matching processes on every tick
    and sums their CPU%/RAM across all of them. This handles:
      - llama-server not existing yet when start() is called
      - llama-server appearing partway through the run
      - llama-server disappearing (model unloaded) before stop() is called
      - multiple matching processes existing simultaneously
    """

    def __init__(self, interval: float = 0.2):
        self.interval = interval
        self._samples = []
        self._stop_event = threading.Event()
        self._thread = None
        # KEY FIX: persist Process objects across ticks, keyed by PID.
        # psutil.Process.cpu_percent() stores its "last call" state on the
        # object INSTANCE itself, not globally per-PID. Re-creating a new
        # psutil.Process(pid) every scan means cpu_percent() always sees a
        # "fresh" object with no prior call to compare against, and
        # therefore always returns 0.0 -- even though the same OS process
        # is being monitored. Reusing the same instance across ticks lets
        # its internal state accumulate correctly.
        self._monitored_procs = {}  # {pid: psutil.Process}

    def _run(self):
        while not self._stop_event.is_set():
            total_cpu = 0.0
            total_ram_mb = 0.0
            active_names = []

            # Discover current PIDs matching our target keywords
            current_pids = set()
            for proc in psutil.process_iter(['pid', 'name']):
                name = (proc.info['name'] or "").lower()
                if any(keyword in name for keyword in TARGET_KEYWORDS):
                    current_pids.add(proc.info['pid'])

            # Drop tracking for PIDs that no longer exist (process ended)
            for old_pid in list(self._monitored_procs.keys()):
                if old_pid not in current_pids:
                    del self._monitored_procs[old_pid]

            for pid in current_pids:
                try:
                    if pid not in self._monitored_procs:
                        # First time seeing this PID -- create ONE Process
                        # object and keep reusing it. Prime it now; skip
                        # using its number this tick (meaningless on first call).
                        new_proc = psutil.Process(pid)
                        new_proc.cpu_percent(interval=None)
                        self._monitored_procs[pid] = new_proc
                        continue

                    # reuse the SAME instance
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
                "process_names": active_names,
            })

            time.sleep(self.interval)

    def start(self):
        self._samples = []
        self._monitored_procs = {}
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> dict:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=2)

        if not self._samples:
            return {
                "avg_cpu_pct": None, "peak_cpu_pct": None,
                "avg_ram_mb": None, "peak_ram_mb": None,
                "max_n_processes": 0, "n_samples": 0,
            }

        cpu_vals = [s["cpu_pct"] for s in self._samples]
        ram_vals = [s["ram_mb"] for s in self._samples]
        n_proc_vals = [s["n_processes"] for s in self._samples]

        return {
            "avg_cpu_pct": statistics.mean(cpu_vals),
            "peak_cpu_pct": max(cpu_vals),
            "avg_ram_mb": statistics.mean(ram_vals),
            "peak_ram_mb": max(ram_vals),
            "max_n_processes": max(n_proc_vals),
            "n_samples": len(self._samples),
        }


def get_target_processes_info():
    """
    Returns a list of dictionaries containing info about the detected target processes.
    """
    procs = find_target_processes()
    info_list = []

    if not procs:
        print("No matching processes found right now (ollama serve idle, "
              "or llama-server not currently loaded).")
        return info_list

    for p in procs:
        try:
            rss_mb = p.memory_info().rss / 1024**2
            cmdline = " ".join(p.cmdline())
            name = p.name()
            print(
                f"PID {p.pid} | {name} | RSS {rss_mb:.1f} MB | cmdline: {cmdline}")

            info_list.append({
                "pid": p.pid,
                "name": name,
                "rss_mb": round(rss_mb, 2),
                "cmdline": cmdline
            })
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            print(f"PID {p.pid} | (process ended before we could read it)")

    return info_list


def monitor_run(graph, query, sample_interval: float = 0.2) -> tuple:
    """
    Wraps graph execution with resource monitoring.
    Returns (result, stats).
    """
    config = {"recursion_limit": 1000}
    sampler = ResourceSampler(interval=sample_interval)

    sampler.start()
    start_time = time.perf_counter()

    result = graph.invoke(
        {"messages": [HumanMessage(content=query)], "rewrite_counts": 0}, config)

    elapsed = time.perf_counter() - start_time
    stats = sampler.stop()
    stats["wall_clock_seconds"] = elapsed

    # Calculate token usage from messages
    prompt_tokens = 0
    completion_tokens = 0
    total_tokens = 0

    messages = []
    if isinstance(result, dict) and "messages" in result:
        messages = result.get("messages", [])
    elif isinstance(result, tuple) and len(result) > 0:
        if hasattr(result[0], "response_metadata") or hasattr(result[0], "usage_metadata"):
            messages = [result[0]]

    if not isinstance(messages, list):
        messages = [messages]

    for msg in messages:
        if not hasattr(msg, "response_metadata") and not hasattr(msg, "usage_metadata"):
            continue

        p_tok = 0
        c_tok = 0
        t_tok = 0

        # 1. Try new LangChain usage_metadata
        if hasattr(msg, "usage_metadata") and msg.usage_metadata:
            p_tok = msg.usage_metadata.get("input_tokens", 0)
            c_tok = msg.usage_metadata.get("output_tokens", 0)
            t_tok = msg.usage_metadata.get("total_tokens", 0)

        # 2. Try older response_metadata
        if p_tok == 0 and c_tok == 0 and hasattr(msg, "response_metadata") and msg.response_metadata:
            token_usage = msg.response_metadata.get("token_usage", {})
            p_tok = token_usage.get("prompt_tokens", 0)
            c_tok = token_usage.get("completion_tokens", 0)

            # 3. Try Ollama specific keys in response_metadata
            if p_tok == 0 and c_tok == 0:
                p_tok = msg.response_metadata.get("prompt_eval_count", 0)
                c_tok = msg.response_metadata.get("eval_count", 0)

            if t_tok == 0:
                t_tok = token_usage.get("total_tokens", 0)

        if t_tok == 0:
            t_tok = p_tok + c_tok

        prompt_tokens += p_tok
        completion_tokens += c_tok
        total_tokens += t_tok

    stats["prompt_tokens"] = prompt_tokens
    stats["completion_tokens"] = completion_tokens
    stats["total_tokens"] = total_tokens

    if elapsed > 0:
        stats["tokens_per_second"] = round(completion_tokens / elapsed, 2)
    else:
        stats["tokens_per_second"] = 0.0

    return result, stats


def warm_up_llm(graph, n=3):
    warm_times = []

    print(f"Warming up LLM ({n} runs)...")
    for i in range(n):
        result, stats = monitor_run(
            graph,
            "What is the mandatory deadline for reporting a suspected data breach to the DPO?"
        )
        warm_times.append(stats["wall_clock_seconds"])
        print(f"Run {i+1}: {stats['wall_clock_seconds']:.2f}s")
        print(f"Times of warm: {len(warm_times)}")

    print("Warm up complete")


def main(question, graph, test_name):
    # Path to the test file

    print("-" * 80)

    print("Targets BEFORE invoke:")
    targets_before = get_target_processes_info()
    print()

    warm_up_llm(graph)
    print()

    all_run_stats = []

    result, stats = monitor_run(graph, question)

    stats["question"] = question
    all_run_stats.append(stats)

    print(
        f"  -> Finished in {stats['wall_clock_seconds']:.2f}s | TPS: {stats.get('tokens_per_second', 0)}")

    print()
    print("Targets AFTER invoke:")
    targets_after = get_target_processes_info()

    # Calculate some aggregated averages
    n_runs = len(all_run_stats)
    avg_tps = sum(s.get("tokens_per_second", 0)
                  for s in all_run_stats) / n_runs if n_runs > 0 else 0
    avg_time = sum(s.get("wall_clock_seconds", 0)
                   for s in all_run_stats) / n_runs if n_runs > 0 else 0

    final_report = {
        "summary": {
            "total_runs": n_runs,
            "average_tps": round(avg_tps, 2),
            "average_wall_clock_seconds": round(avg_time, 2)
        },
        "diagnostic_targets_before": targets_before,
        "diagnostic_targets_after": targets_after,
        "runs": all_run_stats
    }

    print()
    print("=" * 80)
    print("BATCH RESOURCE STATS SUMMARY")
    print("=" * 80)

    print(json.dumps(final_report["summary"], indent=4))

    # Export to a JSON file
    output_filename = f"resource_stats_{test_name}.json"
    with open(output_filename, "w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=4, ensure_ascii=False)

    print("-" * 80)
    print(
        f"Detailed batch stats have been successfully exported to {output_filename}")


if __name__ == "__main__":
    capped_graph = build_capped_graph(use_memory=False)
    uncapped_graph = build_uncapped_graph(use_memory=False)

    question = "What is the mandatory deadline for reporting a suspected data breach to the DPO?"

    print("=== RUNNING CAPPED TEST ===")
    main(question, capped_graph, test_name="capped")

    # print("=== RUNNING UNCAPPED TEST ===")
    # main(question, uncapped_graph, test_name="uncapped")

    os.system("ollama stop llama3.1")
    os.system("ollama stop nomic-embed-text")
    print("All model has been stopped!")
