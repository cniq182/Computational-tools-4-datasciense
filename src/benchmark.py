"""A-priori vs. PCY benchmark — step 4.

Runtime: median of ``reps`` runs with ``time.perf_counter`` (inside the algorithms,
per pass). Memory: one extra run under ``tracemalloc`` (peak Python allocation,
which includes NumPy buffers allocated through Python's allocator), plus the
theoretical size of the main data structures. Loading, plotting and rule
generation are outside every measurement.
"""

from __future__ import annotations

import platform
import tracemalloc

import numpy as np
import pandas as pd

import itemsets as I


def environment() -> dict:
    """Software and hardware the measurements were taken on."""
    return {"python": platform.python_version(), "numpy": np.__version__,
            "pandas": pd.__version__, "machine": platform.machine(),
            "processor": platform.processor(), "system": platform.platform()}


def _run(algorithm: str, T, n_items: int, min_count: int, max_k: int, n_buckets: int):
    if algorithm == "apriori":
        return I.apriori(T, n_items, min_count, max_k)
    return I.pcy(T, n_items, min_count, n_buckets, max_k)


def peak_memory(algorithm, T, n_items, min_count, max_k, n_buckets) -> float:
    """Peak traced allocation (MB) of one run; started and stopped the same way for both."""
    tracemalloc.start()
    tracemalloc.reset_peak()
    _run(algorithm, T, n_items, min_count, max_k, n_buckets)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak / 2**20


def benchmark(T, n_items: int, min_count: int, max_k: int, n_buckets: int,
              reps: int) -> tuple[pd.DataFrame, dict]:
    """Time both algorithms ``reps`` times on identical input; check they agree.

    Returns (one row per algorithm, the PCY candidate statistics).
    Raises if the two algorithms return different itemsets.
    """
    results, rows = {}, []
    for algo in ("apriori", "pcy"):
        runs = [_run(algo, T, n_items, min_count, max_k, n_buckets) for _ in range(reps)]
        results[algo] = runs[-1]
        t = pd.DataFrame([r.timings for r in runs])
        rows.append({"algorithm": algo, "n_transactions": len(T), "min_count": min_count,
                     "max_k": max_k, "reps": reps,
                     **{f"{c}_s": float(t[c].median()) for c in t},
                     "total_s": float(t.sum(axis=1).median()),
                     "total_s_min": float(t.sum(axis=1).min()),
                     "total_s_max": float(t.sum(axis=1).max()),
                     "peak_mem_mb": peak_memory(algo, T, n_items, min_count, max_k, n_buckets)})
    a, p = results["apriori"], results["pcy"]
    if a.itemsets != p.itemsets:
        raise AssertionError("A-priori and PCY disagree: fix before reporting performance")
    cand = {**{k: v for k, v in p.stats.items() if not k.startswith("_")},
            **I.pcy_candidate_stats(p),
            "apriori_candidate_pairs": a.stats["apriori_candidate_pairs"],
            "triangular_bytes": a.stats["triangular_bytes"],
            "itemsets_by_size": a.by_size()}
    return pd.DataFrame(rows), cand


def bucket_sweep(T, n_items: int, min_count: int, bucket_counts) -> pd.DataFrame:
    """Pairs-only PCY for several bucket counts: memory vs. filtering."""
    rows = []
    for nb in bucket_counts:
        r = I.pcy(T, n_items, min_count, nb, max_k=2)
        c = I.pcy_candidate_stats(r)
        rows.append({"n_buckets": nb, "frequent_buckets": r.stats["frequent_buckets"],
                     "share_frequent_buckets": r.stats["frequent_buckets"] / nb,
                     "pcy_candidate_pairs": c["pcy_candidate_pairs"],
                     "candidate_reduction_pct": c["candidate_reduction_pct"],
                     "pcy_pairs_counted": r.stats["pcy_pairs_counted"],
                     "bucket_array_kb": r.stats["bucket_array_bytes"] / 1024,
                     "bitmap_kb": r.stats["bitmap_bytes"] / 1024,
                     "hash_table_kb_course": r.stats["hash_table_bytes_course"] / 1024,
                     "peak_mem_mb": peak_memory("pcy", T, n_items, min_count, 2, nb),
                     "pass1_s": r.timings["pass1"], "pass2_s": r.timings["pass2_pairs"]})
    return pd.DataFrame(rows)
