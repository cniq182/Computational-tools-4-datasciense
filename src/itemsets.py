"""Frequent itemsets — step 4 (A-priori and PCY, written from scratch).

Baskets are the users' liked sets of step 3 (``taste.build_user_sets``): one basket
per user, items = liked restaurants or the (filtered) categories of liked restaurants.
Items are integer indices into ``Baskets.item_ids``; a basket is a sorted array of
distinct indices.

Both algorithms share the same pass 1 (item counts) and the same extension to
itemsets of size >= 3 (A-priori candidate generation + subset pruning, counted
against the transactions). They differ in how pairs are counted:

* **A-priori**: pairs of frequent items counted in a **triangular array**.
* **PCY**: pass 1 also hashes every pair into a bucket array, which becomes a
  **bitmap**; pass 2 counts, in a **hash table**, only pairs of frequent items whose
  bucket is frequent.

Itemsets are sorted tuples of item indices; supports are exact counts.
"""

from __future__ import annotations

import math
import time
from collections import Counter
from dataclasses import dataclass, field
from functools import lru_cache
from itertools import combinations

import numpy as np

# Pair -> bucket hash: the pair is packed into one 64-bit key (i << 32 | j) and
# scrambled with the MurmurHash3 64-bit finaliser, then taken modulo n_buckets.
# Fixed constants, so the hash is identical in every run and process (unlike
# Python's built-in hash() on strings, which is salted per process). A mixer is
# used instead of a linear (A*i + B*j) mod P hash because a linear hash
# degenerates when a constant shares structure with n_buckets.
FMIX_1 = np.uint64(0xFF51AFD7ED558CCD)
FMIX_2 = np.uint64(0xC4CEB9FE1A85EC53)
SHIFT_33 = np.uint64(33)
SHIFT_32 = np.uint64(32)
CHUNK = 500                     # baskets processed per vectorised block
BYTES_PER_COUNTER = 4           # int32 counters (triangular array, buckets)
BYTES_PER_HASH_ENTRY = 12       # course convention: (i, j, count) as three 4-byte ints


# --------------------------------------------------------------------------
# Baskets
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Baskets:
    """Mining transactions of one basket type (one basket per kept user)."""
    version: str
    user_ids: np.ndarray            # user of each basket
    item_ids: np.ndarray            # item label of each item index
    sets: tuple                     # sorted np.ndarray of item indices, one per user
    excluded_users: np.ndarray      # users with fewer than min_items items

    @property
    def n_items(self) -> int:
        return len(self.item_ids)

    def subset(self, users) -> list[np.ndarray]:
        """Baskets of the given users (users without a basket are ignored)."""
        pos = {u: i for i, u in enumerate(self.user_ids)}
        return [self.sets[pos[u]] for u in users if u in pos]


def baskets_from_user_sets(user_sets, all_users, min_items: int = 2) -> Baskets:
    """Turn a ``taste.UserSets`` into mining baskets, dropping baskets with
    fewer than ``min_items`` items (they cannot contain a pair).

    ``all_users`` is the full population, so users with an empty set are
    reported as excluded too.
    """
    sizes = np.array([len(s) for s in user_sets.sets])
    keep = sizes >= min_items
    kept_users = np.asarray(user_sets.user_ids)[keep]
    excluded = np.setdiff1d(np.asarray(all_users), kept_users)
    sets = tuple(np.asarray(s, dtype=np.int64) for s, k in zip(user_sets.sets, keep) if k)
    return Baskets(user_sets.version, kept_users, np.asarray(user_sets.item_ids), sets, excluded)


def min_support_count(min_support: float, n_transactions: int, floor: int = 1) -> int:
    """Absolute threshold: ``max(ceil(min_support * n), floor)``.

    A small epsilon protects against float error (0.1 * 100 = 10.000000000000002).
    """
    if not 0 < min_support <= 1:
        raise ValueError(f"min_support must be in (0, 1], got {min_support}")
    return max(math.ceil(min_support * n_transactions - 1e-9), floor)


# --------------------------------------------------------------------------
# Shared helpers
# --------------------------------------------------------------------------

@lru_cache(maxsize=4096)
def _triu(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Positions (a, b), a < b, of every pair in a basket of length n."""
    return np.triu_indices(n, 1)


def basket_pairs(items: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """All unordered pairs (i, j), i < j, of a sorted basket."""
    a, b = _triu(len(items))
    return items[a], items[b]


def triangular_index(i: np.ndarray, j: np.ndarray, m: int) -> np.ndarray:
    """Position of pair (i, j), 0 <= i < j < m, in a triangular array of size m(m-1)/2."""
    return i * (2 * m - i - 1) // 2 + (j - i - 1)


def triangular_pair(k: np.ndarray, m: int) -> tuple[np.ndarray, np.ndarray]:
    """Inverse of ``triangular_index``."""
    k = np.asarray(k, dtype=np.int64)
    i = (m - 2 - np.floor(np.sqrt(-8 * k + 4 * m * (m - 1) - 7) / 2 - 0.5)).astype(np.int64)
    j = k + i + 1 - m * (m - 1) // 2 + (m - i) * (m - i - 1) // 2
    return i, j


def pair_bucket(i: np.ndarray, j: np.ndarray, n_buckets: int) -> np.ndarray:
    """Deterministic bucket of pair (i, j), with i < j (original item indices).

    uint64 multiplication wraps around modulo 2^64, as the hash intends.
    """
    k = (np.atleast_1d(np.asarray(i, dtype=np.uint64)) << SHIFT_32) | np.atleast_1d(np.asarray(j, dtype=np.uint64))
    k = k ^ (k >> SHIFT_33)
    k = k * FMIX_1
    k = k ^ (k >> SHIFT_33)
    k = k * FMIX_2
    k = k ^ (k >> SHIFT_33)
    return (k % np.uint64(n_buckets)).astype(np.int64)


def bitmap_get(bitmap: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Bit h of a bitmap made by ``np.packbits`` (big-endian bit order)."""
    return (bitmap[h >> 3] >> (7 - (h & 7))) & 1


def _chunks(transactions):
    for s in range(0, len(transactions), CHUNK):
        yield transactions[s:s + CHUNK]


def count_items(transactions, n_items: int) -> np.ndarray:
    """Pass 1: support count of every item."""
    if not transactions:
        return np.zeros(n_items, dtype=np.int64)
    return np.bincount(np.concatenate(transactions), minlength=n_items)


def frequent_item_map(counts: np.ndarray, min_count: int) -> tuple[np.ndarray, np.ndarray]:
    """Frequent items and a map original index -> new index 0..m-1 (or -1)."""
    freq = np.flatnonzero(counts >= min_count)
    remap = np.full(len(counts), -1, dtype=np.int64)
    remap[freq] = np.arange(len(freq))
    return freq, remap


@dataclass
class MiningResult:
    """Output of A-priori or PCY on one transaction collection."""
    algorithm: str
    n_transactions: int
    min_count: int
    itemsets: dict                   # sorted tuple of item indices -> support count
    timings: dict = field(default_factory=dict)   # seconds per pass
    stats: dict = field(default_factory=dict)     # candidate counts, structure sizes

    def by_size(self) -> dict[int, int]:
        return dict(sorted(Counter(len(k) for k in self.itemsets).items()))


# --------------------------------------------------------------------------
# Itemsets of size >= 3 (shared by both algorithms)
# --------------------------------------------------------------------------

def generate_candidates(frequent_prev: list[tuple], k: int) -> list[tuple]:
    """A-priori candidates of size k from the frequent itemsets of size k-1.

    Join two (k-1)-itemsets that share their first k-2 items, then prune the
    candidate if any of its (k-1)-subsets is not frequent.
    """
    prev = sorted(frequent_prev)
    prev_set = set(prev)
    by_prefix: dict[tuple, list] = {}
    for s in prev:
        by_prefix.setdefault(s[:-1], []).append(s[-1])
    out = []
    for prefix, lasts in by_prefix.items():
        for a, b in combinations(lasts, 2):
            cand = prefix + (a, b)
            if all(sub in prev_set for sub in combinations(cand, k - 1)):
                out.append(cand)
    return out


def count_candidates(transactions, candidates: list[tuple], n_items: int,
                     batch: int = 2000) -> np.ndarray:
    """Exact support of each candidate itemset.

    Each item used by a candidate becomes a boolean column over the transactions;
    a transaction contains the candidate iff it has every column set.
    """
    if not candidates:
        return np.zeros(0, dtype=np.int64)
    used = np.unique(np.array(candidates).ravel())
    col = np.full(n_items, -1, dtype=np.int64)
    col[used] = np.arange(len(used))
    M = np.zeros((len(transactions), len(used)), dtype=bool)
    for r, t in enumerate(transactions):
        c = col[t]
        M[r, c[c >= 0]] = True
    C = col[np.array(candidates)]
    out = np.empty(len(candidates), dtype=np.int64)
    for s in range(0, len(C), batch):
        out[s:s + batch] = M[:, C[s:s + batch]].all(axis=2).sum(axis=0)
    return out


def mine_larger(transactions, frequent_pairs: dict, n_items: int, min_count: int,
                max_k: int) -> tuple[dict, dict, dict]:
    """Levels k = 3..max_k. Returns (frequent itemsets, candidates per k, seconds)."""
    found, n_cand, t0 = {}, {}, time.perf_counter()
    prev = list(frequent_pairs)
    for k in range(3, max_k + 1):
        cands = generate_candidates(prev, k)
        n_cand[k] = len(cands)
        if not cands:
            break
        sup = count_candidates(transactions, cands, n_items)
        level = {c: int(s) for c, s in zip(cands, sup) if s >= min_count}
        found.update(level)
        prev = list(level)
        if not level:
            break
    return found, n_cand, time.perf_counter() - t0


def _singletons(freq: np.ndarray, counts: np.ndarray) -> dict:
    return {(int(i),): int(counts[i]) for i in freq}


# --------------------------------------------------------------------------
# A-priori
# --------------------------------------------------------------------------

def apriori_pairs(transactions, freq: np.ndarray, remap: np.ndarray, min_count: int):
    """Pass 2 of A-priori: count pairs of frequent items in a triangular array."""
    m = len(freq)
    tri = np.zeros(m * (m - 1) // 2, dtype=np.int32)
    for chunk in _chunks(transactions):
        idx = []
        for t in chunk:
            f = remap[t]
            f = f[f >= 0]
            if len(f) >= 2:
                a, b = basket_pairs(f)
                idx.append(triangular_index(a, b, m))
        if idx:
            np.add.at(tri, np.concatenate(idx), 1)
    k = np.flatnonzero(tri >= min_count)
    i, j = triangular_pair(k, m)
    pairs = {(int(freq[a]), int(freq[b])): int(tri[x]) for a, b, x in zip(i, j, k)}
    return pairs, len(tri)


def apriori(transactions, n_items: int, min_count: int, max_k: int = 3) -> MiningResult:
    """Frequent itemsets with A-priori (pairs counted in a triangular array)."""
    if min_count < 1:
        raise ValueError("min_count must be >= 1")
    t0 = time.perf_counter()
    counts = count_items(transactions, n_items)
    freq, remap = frequent_item_map(counts, min_count)
    t1 = time.perf_counter()
    pairs, n_tri = apriori_pairs(transactions, freq, remap, min_count) if max_k >= 2 else ({}, 0)
    t2 = time.perf_counter()
    larger, n_cand, t_k = mine_larger(transactions, pairs, n_items, min_count, max_k)
    itemsets = {**_singletons(freq, counts), **pairs, **larger}
    stats = {"frequent_items": len(freq), "apriori_candidate_pairs": n_tri,
             "frequent_pairs": len(pairs), "candidates_k": n_cand,
             "triangular_bytes": n_tri * BYTES_PER_COUNTER}
    timings = {"pass1": t1 - t0, "pass2_pairs": t2 - t1, "larger": t_k}
    return MiningResult("apriori", len(transactions), min_count, itemsets, timings, stats)


# --------------------------------------------------------------------------
# PCY
# --------------------------------------------------------------------------

def pcy_pass1(transactions, n_items: int, n_buckets: int):
    """Item counts and pair-bucket counts in one pass."""
    counts = count_items(transactions, n_items)
    buckets = np.zeros(n_buckets, dtype=np.int32)
    n_hashed = 0
    for chunk in _chunks(transactions):
        hs = []
        for t in chunk:
            if len(t) >= 2:
                a, b = basket_pairs(t)
                hs.append(pair_bucket(a, b, n_buckets))
        if hs:
            h = np.concatenate(hs)
            n_hashed += len(h)
            np.add.at(buckets, h, 1)
    return counts, buckets, n_hashed


def pcy_pass2(transactions, freq, remap, bitmap, n_buckets: int, min_count: int):
    """Count, in a hash table, pairs of frequent items whose bucket is frequent."""
    m = len(freq)
    table: Counter = Counter()
    for chunk in _chunks(transactions):
        keys = []
        for t in chunk:
            f = remap[t]
            keep = f >= 0
            if keep.sum() < 2:
                continue
            a, b = basket_pairs(f[keep])
            ok = bitmap_get(bitmap, pair_bucket(freq[a], freq[b], n_buckets)).astype(bool)
            keys.append(a[ok] * m + b[ok])
        if keys:
            table.update(np.concatenate(keys).tolist())
    pairs = {(int(freq[k // m]), int(freq[k % m])): c for k, c in table.items() if c >= min_count}
    return pairs, len(table)


def pcy(transactions, n_items: int, min_count: int, n_buckets: int, max_k: int = 3) -> MiningResult:
    """Frequent itemsets with PCY (pairs); sizes >= 3 use the shared A-priori extension."""
    if min_count < 1 or n_buckets < 1:
        raise ValueError("min_count and n_buckets must be >= 1")
    t0 = time.perf_counter()
    counts, buckets, n_hashed = pcy_pass1(transactions, n_items, n_buckets)
    frequent_bucket = buckets >= min_count
    bucket_stats = {"pairs_hashed_pass1": n_hashed,
                    "nonempty_buckets": int((buckets > 0).sum()),
                    "frequent_buckets": int(frequent_bucket.sum())}
    bitmap = np.packbits(frequent_bucket)
    del buckets, frequent_bucket                      # only the bitmap survives pass 1
    freq, remap = frequent_item_map(counts, min_count)
    t1 = time.perf_counter()
    pairs, n_counted = (pcy_pass2(transactions, freq, remap, bitmap, n_buckets, min_count)
                        if max_k >= 2 else ({}, 0))
    t2 = time.perf_counter()
    larger, n_cand, t_k = mine_larger(transactions, pairs, n_items, min_count, max_k)
    itemsets = {**_singletons(freq, counts), **pairs, **larger}
    stats = {**bucket_stats, "frequent_items": len(freq), "n_buckets": n_buckets,
             "pcy_pairs_counted": n_counted, "frequent_pairs": len(pairs),
             "candidates_k": n_cand,
             "bucket_array_bytes": n_buckets * BYTES_PER_COUNTER,
             "bitmap_bytes": int(bitmap.nbytes),
             "hash_table_bytes_course": n_counted * BYTES_PER_HASH_ENTRY,
             "_bitmap": bitmap, "_freq": freq}
    timings = {"pass1": t1 - t0, "pass2_pairs": t2 - t1, "larger": t_k}
    return MiningResult("pcy", len(transactions), min_count, itemsets, timings, stats)


def pcy_candidate_stats(result: MiningResult) -> dict:
    """Candidate pairs among all C(m, 2) pairs of frequent items (analysis only,
    outside the timed algorithm): how many pass the bitmap and how many it removes."""
    freq, bitmap, nb = result.stats["_freq"], result.stats["_bitmap"], result.stats["n_buckets"]
    m = len(freq)
    total = m * (m - 1) // 2
    passing = 0
    for a in range(m - 1):
        b = np.arange(a + 1, m)
        passing += int(bitmap_get(bitmap, pair_bucket(freq[a], freq[b], nb)).sum())
    return {"pairs_of_frequent_items": total, "pcy_candidate_pairs": passing,
            "removed_by_bitmap": total - passing,
            "candidate_reduction_pct": 100 * (total - passing) / total if total else 0.0}
