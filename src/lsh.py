"""MinHash + LSH banding, implemented from scratch with NumPy (step 3, owner: Cesi).

Input everywhere: ``sets``, a list of sorted ``np.ndarray`` of item indices
(one per user, items numbered 0 .. n_items-1), as produced by ``taste.build_user_sets``.

Pipeline
--------
1. ``minhash_signatures``   sets -> signature matrix (n_users x n_hashes)
2. ``lsh_candidates``       signatures -> candidate pairs (users sharing >= 1 band bucket)
3. ``verify_pairs``         candidate pairs -> pairs with exact Jaccard >= threshold
4. ``brute_force_pairs``    exact all-pairs Jaccard (ground truth for recall and runtime)

``run_lsh`` chains 1-3 and times each stage; ``lsh_sweep`` runs it for several (b, r).
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import scipy.sparse as sp

PRIME = 2_147_483_647  # 2^31 - 1 (Mersenne prime); a*x + b stays below 2^63, no int64 overflow


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def sets_to_csr(sets: list[np.ndarray], n_items: int) -> sp.csr_matrix:
    """Binary user x item matrix (used for exact Jaccard, not by MinHash itself)."""
    indptr = np.concatenate([[0], np.cumsum([len(s) for s in sets])])
    indices = np.concatenate(sets) if sets else np.array([], dtype=np.int64)
    data = np.ones(len(indices), dtype=np.int32)
    return sp.csr_matrix((data, indices, indptr), shape=(len(sets), n_items))


def jaccard(a: np.ndarray, b: np.ndarray) -> float:
    """Exact Jaccard similarity of two sets of item indices."""
    inter = len(np.intersect1d(a, b, assume_unique=True))
    union = len(a) + len(b) - inter
    return inter / union if union else 0.0


def lsh_threshold(bands: int, rows: int) -> float:
    """Approximate similarity at which a pair has a 50 % chance of becoming a candidate."""
    return (1 / bands) ** (1 / rows)


def s_curve(s: np.ndarray, bands: int, rows: int) -> np.ndarray:
    """Probability that a pair with Jaccard ``s`` becomes a candidate: 1 - (1 - s^r)^b."""
    return 1 - (1 - np.asarray(s) ** rows) ** bands


# --------------------------------------------------------------------------
# 1. MinHash
# --------------------------------------------------------------------------

def minhash_signatures(sets: list[np.ndarray], n_hashes: int = 120, seed: int = 42,
                       n_items: int | None = None) -> np.ndarray:
    """MinHash signature matrix, shape (n_users, n_hashes).

    Hash functions h_k(x) = (a_k * x + b_k) mod p, with a_k, b_k random and p = 2^31 - 1.
    Signature[u, k] = min over the items x of user u of h_k(x).

    Vectorised: hash every item once (n_items x n_hashes table), then take the
    minimum over each user's rows with ``np.minimum.reduceat`` (no Python loop over users).
    """
    if any(len(s) == 0 for s in sets):
        raise ValueError("empty set: drop empty users before MinHash")
    n_items = n_items or int(max(s.max() for s in sets)) + 1
    rng = np.random.default_rng(seed)
    a = rng.integers(1, PRIME, size=n_hashes, dtype=np.int64)
    b = rng.integers(0, PRIME, size=n_hashes, dtype=np.int64)

    x = np.arange(n_items, dtype=np.int64)[:, None]
    item_hash = (a[None, :] * x + b[None, :]) % PRIME           # (n_items, n_hashes)

    flat = np.concatenate(sets)
    starts = np.concatenate([[0], np.cumsum([len(s) for s in sets])[:-1]])
    return np.minimum.reduceat(item_hash[flat], starts, axis=0)  # (n_users, n_hashes)


def minhash_estimate(sig: np.ndarray, u: int, v: int) -> float:
    """Estimated Jaccard: share of hash functions where the two signatures agree."""
    return float(np.mean(sig[u] == sig[v]))


# --------------------------------------------------------------------------
# 2. LSH banding
# --------------------------------------------------------------------------

def lsh_candidates(sig: np.ndarray, bands: int, rows: int,
                   max_bucket: int | None = 1000) -> tuple[np.ndarray, dict]:
    """Candidate pairs: users whose signatures are identical in at least one band.

    The signature is cut into ``bands`` bands of ``rows`` rows (b * r must equal
    the number of hash functions). In each band, users with exactly the same
    ``rows`` values fall in the same bucket.

    Very large buckets (very popular items) would create millions of pairs, so
    buckets with more than ``max_bucket`` users are skipped and reported in ``info``.

    Returns ``(pairs, info)``: ``pairs`` is an (n, 2) int array with u < v, no duplicates.
    """
    n_users, n_hashes = sig.shape
    if bands * rows != n_hashes:
        raise ValueError(f"bands * rows = {bands * rows} != n_hashes = {n_hashes}")

    codes = []                       # pair encoded as u * n_users + v
    bucket_sizes = []
    skipped = []
    for k in range(bands):
        band = sig[:, k * rows:(k + 1) * rows]
        _, bucket = np.unique(band, axis=0, return_inverse=True)   # bucket id per user
        bucket = bucket.ravel()
        order = np.argsort(bucket, kind="stable")
        ids, starts, counts = np.unique(bucket[order], return_index=True, return_counts=True)
        for st, c in zip(starts[counts > 1], counts[counts > 1]):
            if max_bucket is not None and c > max_bucket:
                skipped.append(int(c))
                continue
            members = np.sort(order[st:st + c])
            i, j = np.triu_indices(c, k=1)
            codes.append(members[i].astype(np.int64) * n_users + members[j])
            bucket_sizes.append(int(c))

    codes = np.unique(np.concatenate(codes)) if codes else np.array([], dtype=np.int64)
    pairs = np.column_stack([codes // n_users, codes % n_users])
    info = {
        "n_candidates": len(pairs),
        "n_buckets_used": len(bucket_sizes),
        "max_bucket_seen": max(bucket_sizes + skipped, default=0),
        "n_buckets_skipped": len(skipped),
        "skipped_bucket_sizes": sorted(skipped, reverse=True)[:10],
    }
    return pairs, info


# --------------------------------------------------------------------------
# 3. Verification with the exact Jaccard
# --------------------------------------------------------------------------

def pair_jaccard(X: sp.csr_matrix, pairs: np.ndarray, batch: int = 500_000) -> np.ndarray:
    """Exact Jaccard for many pairs at once, using the sparse user x item matrix."""
    sizes = np.asarray(X.sum(axis=1)).ravel()
    out = np.empty(len(pairs))
    for s in range(0, len(pairs), batch):
        u, v = pairs[s:s + batch, 0], pairs[s:s + batch, 1]
        inter = np.asarray(X[u].multiply(X[v]).sum(axis=1)).ravel()
        out[s:s + batch] = inter / (sizes[u] + sizes[v] - inter)
    return out


def verify_pairs(X: sp.csr_matrix, pairs: np.ndarray, threshold: float) -> pd.DataFrame:
    """Keep the candidate pairs whose exact Jaccard is >= ``threshold``."""
    if len(pairs) == 0:
        return pd.DataFrame({"u": [], "v": [], "jaccard": []})
    j = pair_jaccard(X, pairs)
    keep = j >= threshold
    return pd.DataFrame({"u": pairs[keep, 0], "v": pairs[keep, 1], "jaccard": j[keep]})


# --------------------------------------------------------------------------
# 4. Brute force (ground truth)
# --------------------------------------------------------------------------

def brute_force_pairs(X: sp.csr_matrix, threshold: float, chunk: int = 1000) -> pd.DataFrame:
    """Exact Jaccard of **all** pairs of users, keeping those >= ``threshold``.

    Intersections come from the matrix product X X^T, computed ``chunk`` rows at
    a time so memory stays small. Cost grows with n_users^2.
    """
    n = X.shape[0]
    sizes = np.asarray(X.sum(axis=1)).ravel()
    XT = X.T.tocsc()
    us, vs, js = [], [], []
    for s in range(0, n, chunk):
        inter = (X[s:s + chunk] @ XT).toarray()                     # (chunk, n)
        jac = inter / (sizes[s:s + chunk, None] + sizes[None, :] - inter)
        r, c = np.nonzero(jac >= threshold)
        r = r + s
        keep = r < c                                                # each pair once, no self-pairs
        us.append(r[keep]); vs.append(c[keep]); js.append(jac[r[keep] - s, c[keep]])
    return pd.DataFrame({"u": np.concatenate(us), "v": np.concatenate(vs),
                         "jaccard": np.concatenate(js)})


def random_pair_jaccard(X: sp.csr_matrix, n_pairs: int = 200_000, seed: int = 42) -> np.ndarray:
    """Exact Jaccard of random user pairs: what 'similarity by chance' looks like."""
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    u = rng.integers(0, n, n_pairs)
    v = rng.integers(0, n, n_pairs)
    keep = u != v
    return pair_jaccard(X, np.column_stack([u[keep], v[keep]]))


# --------------------------------------------------------------------------
# Full run + parameter sweep
# --------------------------------------------------------------------------

def run_lsh(sets: list[np.ndarray], n_items: int, bands: int, rows: int, threshold: float,
            seed: int = 42, max_bucket: int | None = 1000, sig: np.ndarray | None = None,
            X: sp.csr_matrix | None = None) -> dict:
    """MinHash -> banding -> verification, with the time of each stage.

    ``sig`` and ``X`` can be passed in to reuse them across a sweep (then the
    MinHash time reported is 0 for that run).
    """
    t0 = time.perf_counter()
    if sig is None:
        sig = minhash_signatures(sets, bands * rows, seed, n_items)
    t1 = time.perf_counter()
    pairs, info = lsh_candidates(sig, bands, rows, max_bucket)
    t2 = time.perf_counter()
    if X is None:
        X = sets_to_csr(sets, n_items)
    edges = verify_pairs(X, pairs, threshold)
    t3 = time.perf_counter()
    return {"bands": bands, "rows": rows, "t_approx": round(lsh_threshold(bands, rows), 3),
            "jaccard_threshold": threshold, **info, "n_verified": len(edges),
            "precision": round(len(edges) / max(len(pairs), 1), 4),
            "time_minhash_s": round(t1 - t0, 2), "time_banding_s": round(t2 - t1, 2),
            "time_verify_s": round(t3 - t2, 2), "time_total_s": round(t3 - t0, 2),
            "edges": edges}


def recall(found: pd.DataFrame, truth: pd.DataFrame) -> float:
    """Share of the true pairs (brute force) that LSH also found."""
    if len(truth) == 0:
        return float("nan")
    f = set(zip(found["u"].tolist(), found["v"].tolist()))
    t = set(zip(truth["u"].tolist(), truth["v"].tolist()))
    return len(f & t) / len(t)


def lsh_sweep(sets, n_items, configs, threshold, truth=None, seed=42, max_bucket=1000):
    """Run LSH for each (bands, rows) in ``configs`` with the same signatures.

    Returns a table (one row per config) and a dict {(b, r): edges}.
    If ``truth`` (brute-force pairs) is given, recall is added.
    """
    n_hashes = configs[0][0] * configs[0][1]
    assert all(b * r == n_hashes for b, r in configs), "all configs need the same b*r"
    t0 = time.perf_counter()
    sig = minhash_signatures(sets, n_hashes, seed, n_items)
    t_sig = time.perf_counter() - t0
    X = sets_to_csr(sets, n_items)
    rows, edges = [], {}
    for b, r in configs:
        res = run_lsh(sets, n_items, b, r, threshold, seed, max_bucket, sig=sig, X=X)
        res["time_minhash_s"] = round(t_sig, 2)
        res["time_total_s"] = round(res["time_total_s"] + t_sig, 2)
        e = res.pop("edges")
        if truth is not None:
            res["recall"] = round(recall(e, truth), 4)
        rows.append(res)
        edges[(b, r)] = e
    return pd.DataFrame(rows), edges


def band_configs(n_hashes: int, max_rows: int = 8, min_rows: int = 2) -> list[tuple[int, int]]:
    """All (bands, rows) with bands * rows = n_hashes and min_rows <= rows <= max_rows."""
    return [(n_hashes // r, r) for r in range(min_rows, max_rows + 1) if n_hashes % r == 0]


def choose_config(table: pd.DataFrame, min_recall: float = 0.90) -> tuple[int, int]:
    """Cheapest (fewest candidates) configuration whose recall is >= ``min_recall``.
    If none reaches it, the one with the highest recall."""
    ok = table[table["recall"] >= min_recall]
    row = ok.loc[ok["n_candidates"].idxmin()] if len(ok) else table.loc[table["recall"].idxmax()]
    return int(row["bands"]), int(row["rows"])


# --------------------------------------------------------------------------
# k-nearest-neighbour graph (used to build the taste graph)
# --------------------------------------------------------------------------
#
# A single global Jaccard threshold leaves most users isolated, because Jaccard
# depends strongly on set size. Instead, every user is linked to its k most
# similar users: LSH proposes the candidates, the exact Jaccard ranks them.

def _top_k(u: np.ndarray, v: np.ndarray, j: np.ndarray, k: int) -> pd.DataFrame:
    """Keep, for every user, the k pairs with the highest Jaccard (pairs counted in both directions).
    Returns directed rows (user, neighbour, jaccard)."""
    a = np.concatenate([u, v]); b = np.concatenate([v, u]); w = np.concatenate([j, j])
    order = np.lexsort((b, -w, a))                     # by user, then Jaccard desc, then neighbour id (ties)
    a, b, w = a[order], b[order], w[order]
    starts = np.r_[0, np.flatnonzero(np.diff(a)) + 1]
    rank = np.arange(len(a)) - np.repeat(starts, np.diff(np.r_[starts, len(a)]))
    keep = rank < k
    return pd.DataFrame({"user": a[keep], "neighbour": b[keep], "jaccard": w[keep]})


def knn_edges(directed: pd.DataFrame) -> pd.DataFrame:
    """Undirected kNN graph: edge u-v if v is among u's k nearest OR u among v's."""
    u = np.minimum(directed["user"], directed["neighbour"]).to_numpy()
    v = np.maximum(directed["user"], directed["neighbour"]).to_numpy()
    e = pd.DataFrame({"u": u, "v": v, "jaccard": directed["jaccard"].to_numpy()})
    return e.drop_duplicates(["u", "v"]).sort_values(["u", "v"]).reset_index(drop=True)


def knn_from_candidates(X: sp.csr_matrix, pairs: np.ndarray, k: int,
                        min_jaccard: float = 0.0) -> pd.DataFrame:
    """Approximate kNN: exact Jaccard of the LSH candidate pairs, then top-k per user.
    Pairs with Jaccard <= ``min_jaccard`` are never neighbours."""
    if len(pairs) == 0:
        return pd.DataFrame({"user": [], "neighbour": [], "jaccard": []})
    j = pair_jaccard(X, pairs)
    ok = j > min_jaccard
    return _top_k(pairs[ok, 0], pairs[ok, 1], j[ok], k)


def brute_force_knn(X: sp.csr_matrix, k: int, min_jaccard: float = 0.0, chunk: int = 1000) -> pd.DataFrame:
    """Exact kNN by comparing every pair (ground truth). Directed rows (user, neighbour, jaccard)."""
    n = X.shape[0]
    sizes = np.asarray(X.sum(axis=1)).ravel()
    XT = X.T.tocsc()
    out = []
    for s in range(0, n, chunk):
        inter = (X[s:s + chunk] @ XT).toarray()
        jac = inter / (sizes[s:s + chunk, None] + sizes[None, :] - inter)
        rows = np.arange(jac.shape[0])
        jac[rows, rows + s] = -1                                       # not your own neighbour
        top = np.argpartition(-jac, k, axis=1)[:, :k]
        w = jac[rows[:, None], top]
        r, c = np.nonzero(w > min_jaccard)
        out.append(pd.DataFrame({"user": r + s, "neighbour": top[r, c], "jaccard": w[r, c]}))
    return pd.concat(out, ignore_index=True)


def knn_recall(found: pd.DataFrame, truth: pd.DataFrame) -> float:
    """Share of the exact neighbour relations (user -> neighbour) that LSH also found.
    A neighbour tied in Jaccard with the true k-th one counts as found."""
    kth = truth.groupby("user")["jaccard"].min()
    f = found.merge(kth.rename("kth"), left_on="user", right_index=True)
    good = f[f["jaccard"] >= f["kth"] - 1e-12].groupby("user").size()
    need = truth.groupby("user").size()
    return float(good.reindex(need.index, fill_value=0).clip(upper=need).sum() / need.sum())


def knn_sweep(sets, n_items, configs, k, truth, min_jaccard=0.0, seed=42, max_bucket=1000):
    """LSH kNN for each (bands, rows): candidates, recall vs exact kNN, runtime."""
    n_hashes = configs[0][0] * configs[0][1]
    assert all(b * r == n_hashes for b, r in configs), "all configs need the same b*r"
    t0 = time.perf_counter()
    sig = minhash_signatures(sets, n_hashes, seed, n_items)
    t_sig = time.perf_counter() - t0
    X = sets_to_csr(sets, n_items)
    rows, found = [], {}
    for b, r in configs:
        t1 = time.perf_counter()
        pairs, info = lsh_candidates(sig, b, r, max_bucket)
        t2 = time.perf_counter()
        d = knn_from_candidates(X, pairs, k, min_jaccard)
        t3 = time.perf_counter()
        users_with = d["user"].nunique()
        rows.append({"bands": b, "rows": r, "t_approx": round(lsh_threshold(b, r), 3), **info,
                     "recall": round(knn_recall(d, truth), 4),
                     "users_without_neighbour": len(sets) - users_with,
                     "time_minhash_s": round(t_sig, 2), "time_banding_s": round(t2 - t1, 2),
                     "time_knn_s": round(t3 - t2, 2), "time_total_s": round(t_sig + t3 - t1, 2)})
        found[(b, r)] = d
    return pd.DataFrame(rows), found
