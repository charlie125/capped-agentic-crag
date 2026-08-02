"""
Local Resource Monitor for Ollama-backed LangGraph pipelines (v2)
====================================================================
FIX from v1: Ollama on Apple Silicon spawns a SEPARATE child process
called `llama-server` to actually run inference (load model weights,
run GPU compute). The `ollama serve` process itself is just a thin
API gateway/scheduler and barely uses any CPU/RAM.

`llama-server` is DYNAMIC -- it may not exist yet when you start
monitoring (if Ollama hasn't loaded a model), and it may disappear
after inference completes (if Ollama unloads idle models). So instead
of grabbing ONE process handle up front, we re-scan for matching
processes on EVERY sample tick, and sum resource usage across
whatever matches are found at that moment.

Usage:
    from resource_monitor import monitor_run
    from capped_agentic_rag import graph

    result, stats = monitor_run(
        graph, {"messages": "your question", "rewrite_counts": 0})
    print(stats)
"""

import time
import threading
import statistics
import psutil
import os
import sys
from kxy500.rag_agent.capped_agentic_rag import graph as capped_graph


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


# Process names to track. Both are needed:
#   - "ollama"        -> catches `ollama serve` (API gateway, low usage)
#   - "llama-server"   -> catches the dynamically-spawned inference worker
#                         (real GPU/CPU/RAM usage happens here)
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


def monitor_run(graph, invoke_input: dict, sample_interval: float = 0.2) -> tuple:
    """
    Wraps graph.invoke(invoke_input) with resource monitoring.
    Returns (result, stats).
    """

    sampler = ResourceSampler(interval=sample_interval)

    sampler.start()
    start_time = time.perf_counter()

    result = graph.invoke(invoke_input, {"recursion_limit": 1000})

    elapsed = time.perf_counter() - start_time
    stats = sampler.stop()
    stats["wall_clock_seconds"] = elapsed

    return result, stats


# =========================================================================
# Diagnostic helper -- run this FIRST to sanity-check which processes are
# being detected, before trusting any monitor_run() output.
# =========================================================================
def debug_print_targets():
    procs = find_target_processes()
    if not procs:
        print("No matching processes found right now (ollama serve idle, "
              "or llama-server not currently loaded).")
        return
    for p in procs:
        try:
            print(f"PID {p.pid} | {p.name()} | RSS {p.memory_info().rss / 1024**2:.1f} MB "
                  f"| cmdline: {' '.join(p.cmdline())}")
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            print(f"PID {p.pid} | (process ended before we could read it)")


is_warm = False


def warm_up_llm(graph, n=3):
    global is_warm
    warm_times = []

    if not is_warm:
        for i in range(n):
            result, stats = monitor_run(
                graph,
                {"messages": "What is the mandatory deadline for reporting a suspected data breach to the DPO?",
                 "rewrite_counts": 0}
            )

            warm_times.append(stats["wall_clock_seconds"])
            print(f"Run {i+1}: {stats['wall_clock_seconds']:.2f}s")
            print(f"Times of warm: {len(warm_times)}")

        is_warm = True

    print("Warm up complete")


# =========================================================================
# Example usage
# =========================================================================
if __name__ == "__main__":

    warm_up_llm(capped_graph)

    print("Targets BEFORE invoke:")
    debug_print_targets()
    print()

    result, stats = monitor_run(
        capped_graph,
        {"messages": "What compensation or remedy is offered to a data subject whose data was breached?", "rewrite_counts": 0},
    )

    print()
    print("Targets AFTER invoke:")
    debug_print_targets()

    print()
    print("=" * 80)
    print("RESOURCE STATS")
    print("=" * 80)
    for k, v in stats.items():
        print(f"{k}: {v}")
