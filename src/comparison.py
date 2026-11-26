"""Step 5 -- Comparison of friend communities and taste communities (owner: Jose).

Question: are friend communities also taste communities? Steps 2 and 3 produced two
partitions of the *same* 7,947 Philadelphia users. This module measures how much they
agree and whether the agreement is more than chance **once the period of activity is
controlled for** (step 2 found that friend communities line up with the years in which
their members were active, NMI 0.22 vs 0.001 by chance).

Nothing here changes steps 1-4: it only reads their output files.

What is computed
----------------
Partition level (friend partition vs taste partition)
    * NMI (arithmetic normalisation, as scikit-learn), ARI and, for the headline pairs,
      AMI (chance-corrected NMI).
    * For the chosen runs and across the 20 saved runs of each partition (20 x 20 grid),
      for both taste versions (category = main, restaurant = secondary).
    * A *noise ceiling*: how much two runs of the **same** method agree with each other
      (both partitions are only moderately stable, so they cannot agree with each other
      more than they agree with themselves).
    * Period baselines: NMI/ARI of each partition against the period bins.
    * Permutation test: taste labels are shuffled **within period bins** (friend partition
      fixed). This keeps the link between taste and period but destroys the link between
      taste and friends, so it asks: do friends agree more than period alone explains?
      The plain shuffle (ignoring period) is also reported to show how much the control
      matters.
    * Sensitivity: isolated users included, tiny communities merged / dropped.

Pair level (friend pairs vs random pairs)
    * Share of friend pairs in the same taste community, and mean Jaccard similarity of
      friend pairs. Two nulls: random user pairs (the version in the report) and
      period-controlled (every user's taste data is shuffled among users of the same
      period, friend edges stay fixed).

Labels are arbitrary: only the grouping matters, never compare label values.

Run everything from the command line (from the repository root)::

    python src/comparison.py

or step by step in ``notebooks/05_comparison.ipynb``.
"""

from __future__ import annotations

import json
import sys
import warnings
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Constants (values come from steps 1-3; see STEP2_FRIEND_COMMUNITIES.md and notebook 03)
# --------------------------------------------------------------------------

CITY_PREFIX = "philadelphia_pa"
VERSIONS = ("category", "business")          # taste versions; "category" is the main one
PERIOD_ORDER = ["<=2010", "2011-13", "2014-16", ">=2017"]   # same bins as notebooks 02 and 03
MIN_COMMUNITY = 10                           # "tiny" community = fewer users than this
TOL = 1e-12
VNAME = {"category": "category taste", "business": "restaurant taste"}

# What the report / hand-off notes say; used to catch a regenerated dataset that differs.
EXPECTED_REPORT = {
    "n_users": 7947,
    "friend_communities": 51,                              # chosen run, notebook 02
    # chosen runs, notebook 03 (incl. isolated singletons). Restaurant: 17 = 11 + 6 isolated, as regenerated
    # by notebook 03 with the pinned requirements.txt (earlier unpinned runs gave 18 and 19).
    "taste_communities": {"category": 17, "business": 17},
    "users_with_set": 7941,                                # notebook 03, section 5.1
    "n_items": {"category": 196, "business": 5414},        # notebook 03, section 5.1
}

# Parameters of notebook 03 (also read from data/results/taste_summary.json when present).
TASTE_PARAM_DEFAULTS = {
    "positive": "is_positive",
    "generic_max_frac": 0.80,
    "max_cats_per_business": 20,
    "min_businesses_per_category": 5,
}


def _log(on: bool, msg: str) -> None:
    if on:
        print(msg, flush=True)


# --------------------------------------------------------------------------
# Loading and aligning the inputs
# --------------------------------------------------------------------------

def find_project_root(start: Path | None = None) -> Path:
    """Repository root: the first parent folder that contains ``data/processed``."""
    p = Path(start or Path.cwd()).resolve()
    for candidate in (p, *p.parents):
        if (candidate / "data" / "processed").exists():
            return candidate
    raise FileNotFoundError("data/processed not found: run notebook 01 first.")


@dataclass
class Inputs:
    """Everything step 5 needs, aligned on the same users (``users`` is sorted)."""
    users: np.ndarray                      # user ids, position = user index
    friend_main: np.ndarray                # (n,) chosen friend partition (int codes)
    friend_seeds: np.ndarray               # (n, n_friend_seeds)
    taste_main: dict                       # version -> (n,) chosen taste partition
    taste_seeds: dict                      # version -> (n, n_taste_seeds)
    isolated: dict                         # version -> (n,) bool: user has no similar pair
    period: np.ndarray                     # (n,) int code of the activity-period bin
    period_labels: list
    edge_u: np.ndarray                     # friend edges as user indices, u < v
    edge_v: np.ndarray

    @property
    def n(self) -> int:
        return len(self.users)


def _seed_columns(df: pd.DataFrame) -> list:
    cols = [c for c in df.columns if str(c).startswith("seed_")]
    return sorted(cols, key=lambda c: int(str(c).split("_")[1]))


def factorize(x) -> np.ndarray:
    """Relabel to 0..k-1 (labels are arbitrary; only the grouping matters)."""
    return pd.factorize(np.asarray(x))[0].astype(np.int64)


def remap_edges(edge_u: np.ndarray, edge_v: np.ndarray, sel: np.ndarray, n_total: int):
    """Keep the edges whose two ends are in ``sel`` and renumber them 0..len(sel)-1."""
    new = np.full(n_total, -1, dtype=np.int64)
    new[sel] = np.arange(len(sel))
    u, v = new[edge_u], new[edge_v]
    keep = (u >= 0) & (v >= 0)
    return u[keep], v[keep]


def align_inputs(friend_main: pd.Series, friend_seeds: pd.DataFrame, taste: pd.DataFrame,
                 taste_seeds: dict, edges: pd.DataFrame) -> Inputs:
    """Put the friend and taste files on the same users (intersection, with a warning if
    anything is lost) and turn them into arrays.

    ``friend_main``: Series ``friend_community`` indexed by ``user_id``.
    ``friend_seeds``: DataFrame ``seed_0..seed_k`` indexed by ``user_id``.
    ``taste``: ``taste_communities.parquet`` indexed by ``user_id``
               (``louvain_*``, ``isolated_*``, ``period``).
    ``taste_seeds``: {version: DataFrame ``seed_0..seed_k`` indexed by ``user_id``}.
    ``edges``: friend edges with columns ``u``, ``v`` (user ids).
    """
    frames = {"friend_main": friend_main, "friend_seeds": friend_seeds, "taste": taste,
              **{f"taste_seeds_{v}": d for v, d in taste_seeds.items()}}
    for name, obj in frames.items():
        if not obj.index.is_unique:
            raise ValueError(f"{name}: duplicated user_id")
    users = friend_main.index
    for obj in frames.values():
        users = users.intersection(obj.index)
    users = users.sort_values()
    lost = {name: int(len(obj.index) - len(users)) for name, obj in frames.items() if len(obj.index) != len(users)}
    if lost:
        warnings.warn(f"the files do not cover exactly the same users; restricted to the "
                      f"{len(users):,} common users (dropped per file: {lost})")

    fseed_cols = _seed_columns(friend_seeds)
    tmain = {v: taste.loc[users, f"louvain_{v}"].to_numpy() for v in VERSIONS}
    per = taste.loc[users, "period"].astype(str)
    order = PERIOD_ORDER + sorted(set(per) - set(PERIOD_ORDER))     # unknown labels get their own bin
    pcode = per.map({p: i for i, p in enumerate(order)}).to_numpy(dtype=np.int64)

    pos = pd.Series(np.arange(len(users)), index=users)
    eu = pos.reindex(edges["u"]).to_numpy()
    ev = pos.reindex(edges["v"]).to_numpy()
    ok = ~(np.isnan(eu) | np.isnan(ev))
    eu, ev = eu[ok].astype(np.int64), ev[ok].astype(np.int64)
    lo, hi = np.minimum(eu, ev), np.maximum(eu, ev)
    key = np.unique(lo * len(users) + hi)
    lo, hi = key // len(users), key % len(users)
    keep = lo != hi
    lo, hi = lo[keep], hi[keep]

    return Inputs(
        users=users.to_numpy(),
        friend_main=friend_main.loc[users].to_numpy(),
        friend_seeds=friend_seeds.loc[users, fseed_cols].to_numpy(),
        taste_main=tmain,
        taste_seeds={v: taste_seeds[v].loc[users, _seed_columns(taste_seeds[v])].to_numpy() for v in VERSIONS},
        isolated={v: taste.loc[users, f"isolated_{v}"].to_numpy(dtype=bool) for v in VERSIONS},
        period=pcode,
        period_labels=order,
        edge_u=lo.astype(np.int64),
        edge_v=hi.astype(np.int64),
    )


def load_inputs(root: Path, prefix: str = CITY_PREFIX) -> Inputs:
    """Read the files written by notebooks 01, 02 and 03 and align them."""
    proc, res = root / "data" / "processed", root / "data" / "results"
    paths = {
        "friend communities (notebook 02)": proc / f"{prefix}_friend_communities.parquet",
        "friend communities, 20 seeds (notebook 02)": proc / f"{prefix}_friend_communities_all_seeds.parquet",
        "friend edges (notebook 01)": proc / f"{prefix}_friend_edges.parquet",
        "taste communities (notebook 03)": res / "taste_communities.parquet",
        **{f"taste communities, 20 seeds, {v} (notebook 03)": res / f"taste_communities_all_seeds_{v}.parquet"
           for v in VERSIONS},
    }
    missing = [f"  - {label}: {p}" for label, p in paths.items() if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing input files. Run notebooks 01, 02 and 03 first (data/ is not in git):\n" + "\n".join(missing))
    friend = pd.read_parquet(paths["friend communities (notebook 02)"]).set_index("user_id")["friend_community"]
    fseeds = pd.read_parquet(paths["friend communities, 20 seeds (notebook 02)"]).set_index("user_id")
    taste = pd.read_parquet(paths["taste communities (notebook 03)"]).set_index("user_id")
    tseeds = {v: pd.read_parquet(res / f"taste_communities_all_seeds_{v}.parquet").set_index("user_id")
              for v in VERSIONS}
    edges = pd.read_parquet(paths["friend edges (notebook 01)"])
    return align_inputs(friend, fseeds, taste, tseeds, edges)


# --------------------------------------------------------------------------
# Agreement measures from a contingency table (fast enough for 10^6 evaluations)
# --------------------------------------------------------------------------

def contingency(f: np.ndarray, t: np.ndarray, nf: int | None = None, nt: int | None = None) -> np.ndarray:
    """Contingency table of two label vectors with codes 0..nf-1 / 0..nt-1."""
    nf = int(f.max()) + 1 if nf is None else nf
    nt = int(t.max()) + 1 if nt is None else nt
    return np.bincount(f * nt + t, minlength=nf * nt).reshape(nf, nt)


def _entropy(counts: np.ndarray, n: float) -> float:
    c = counts[counts > 0]
    p = c / n
    return float(-(p * np.log(p)).sum())


def table_stats(tab: np.ndarray) -> tuple:
    """(NMI, ARI) of a contingency table.

    NMI = MI / mean(H_f, H_t) (arithmetic normalisation, the scikit-learn default);
    ARI = (index - expected) / (max index - expected). Zero rows/columns are allowed.
    """
    tab = np.asarray(tab, dtype=np.float64)
    n = tab.sum()
    if n < 2:
        return 1.0, 1.0
    a, b = tab.sum(axis=1), tab.sum(axis=0)
    ii, jj = np.nonzero(tab)
    nij = tab[ii, jj]
    mi = float((nij / n * (np.log(nij) + np.log(n) - np.log(a[ii]) - np.log(b[jj]))).sum())
    ha, hb = _entropy(a, n), _entropy(b, n)
    nmi = 1.0 if (ha == 0.0 and hb == 0.0) else max(mi, 0.0) / (0.5 * (ha + hb))
    sum_ij = float((nij * (nij - 1) / 2).sum())
    sum_a, sum_b = float((a * (a - 1) / 2).sum()), float((b * (b - 1) / 2).sum())
    expected = sum_a * sum_b / (n * (n - 1) / 2)
    denom = 0.5 * (sum_a + sum_b) - expected
    ari = 1.0 if denom == 0 else (sum_ij - expected) / denom
    return min(nmi, 1.0), ari


def pair_stats(f: np.ndarray, t: np.ndarray, nf: int | None = None, nt: int | None = None) -> tuple:
    """(NMI, ARI) between two label vectors."""
    return table_stats(contingency(f, t, nf, nt))


def group_indices(strata: np.ndarray) -> list:
    return [np.flatnonzero(strata == s) for s in np.unique(strata)]


def within_stats(f: np.ndarray, t: np.ndarray, groups: list, nf: int, nt: int) -> tuple:
    """NMI and ARI computed inside each stratum (period bin), averaged with weights = size.

    Compares friends and tastes *among users who were active in the same period*.
    """
    n = sum(len(g) for g in groups)
    nm = ar = 0.0
    for g in groups:
        s_nmi, s_ari = table_stats(np.bincount(f[g] * nt + t[g], minlength=nf * nt).reshape(nf, nt))
        nm += len(g) * s_nmi
        ar += len(g) * s_ari
    return nm / n, ar / n


def verify_against_sklearn(f: np.ndarray, t: np.ndarray, atol: float = 1e-9) -> dict:
    """Check our NMI/ARI against scikit-learn (raises if they differ)."""
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    ours = pair_stats(factorize(f), factorize(t))
    ref = (normalized_mutual_info_score(f, t), adjusted_rand_score(f, t))
    if not np.allclose(ours, ref, atol=atol):
        raise AssertionError(f"NMI/ARI differ from scikit-learn: ours {ours}, sklearn {ref}")
    return {"nmi": float(ours[0]), "ari": float(ours[1]), "sklearn_nmi": float(ref[0]), "sklearn_ari": float(ref[1])}


# --------------------------------------------------------------------------
# Permutations
# --------------------------------------------------------------------------

class Shuffler:
    """Random permutation of the users that only moves users inside their own stratum.

    ``Shuffler(period)`` = shuffle within period bins; ``Shuffler(zeros)`` = plain shuffle.
    Calling it returns ``idx``; ``labels[idx]`` is the shuffled label vector.
    """

    def __init__(self, strata: np.ndarray, seed: int = 42):
        self.n = len(strata)
        self.groups = group_indices(strata)
        self.rng = np.random.default_rng(seed)

    def __call__(self) -> np.ndarray:
        idx = np.arange(self.n)
        for g in self.groups:
            idx[g] = self.rng.permutation(g)
        return idx


def p_upper(obs: float, null: np.ndarray) -> float:
    """One-sided empirical p-value P(null >= obs), with the +1 correction (never 0)."""
    return float((1 + np.sum(null >= obs - TOL)) / (1 + len(null)))


def null_summary(obs: float, null: np.ndarray) -> dict:
    sd = float(np.std(null, ddof=1)) if len(null) > 1 else float("nan")
    mean = float(np.mean(null))
    return {"observed": float(obs), "null_mean": mean, "null_sd": sd, "excess": float(obs - mean),
            "z": float((obs - mean) / sd) if sd and sd > 0 else float("nan"), "p": p_upper(obs, null)}


# --------------------------------------------------------------------------
# A view of the data for one taste version
# --------------------------------------------------------------------------

@dataclass
class View:
    version: str
    sel: np.ndarray                 # indices (into Inputs) of the users compared
    f_main: np.ndarray
    f_seeds: list
    t_main: np.ndarray
    t_seeds: list
    period: np.ndarray
    eu: np.ndarray
    ev: np.ndarray
    n_isolated_excluded: int


def make_view(inp: Inputs, version: str, exclude_isolated: bool = True) -> View:
    """Users compared for one taste version.

    Users with no similar pair in the taste graph (6-8 users) are singleton "communities";
    by default they are excluded, because a singleton says nothing about taste similarity.
    """
    mask = ~inp.isolated[version] if exclude_isolated else np.ones(inp.n, dtype=bool)
    sel = np.flatnonzero(mask)
    eu, ev = remap_edges(inp.edge_u, inp.edge_v, sel, inp.n)
    return View(
        version=version, sel=sel,
        f_main=factorize(inp.friend_main[sel]),
        f_seeds=[factorize(inp.friend_seeds[sel, k]) for k in range(inp.friend_seeds.shape[1])],
        t_main=factorize(inp.taste_main[version][sel]),
        t_seeds=[factorize(inp.taste_seeds[version][sel, k]) for k in range(inp.taste_seeds[version].shape[1])],
        period=factorize(inp.period[sel]),
        eu=eu, ev=ev,
        n_isolated_excluded=int(inp.n - len(sel)),
    )


# --------------------------------------------------------------------------
# Partition level
# --------------------------------------------------------------------------

def compare_pair(f: np.ndarray, t: np.ndarray, strata: np.ndarray, n_perm: int = 10_000,
                 seed: int = 42, with_ami: bool = True) -> tuple:
    """Friend partition ``f`` vs taste partition ``t``, with permutation tests.

    Observed: NMI, ARI (AMI optional) and the within-period NMI/ARI.
    Null "within period": taste labels shuffled among users of the same period bin.
    Null "plain": taste labels shuffled among all users (ignores the confounder).
    Returns (summary dict, dict of null arrays).
    """
    nf, nt = int(f.max()) + 1, int(t.max()) + 1
    groups = group_indices(strata)
    nmi, ari = pair_stats(f, t, nf, nt)
    w_nmi, w_ari = within_stats(f, t, groups, nf, nt)

    sh_period, sh_plain = Shuffler(strata, seed), Shuffler(np.zeros(len(f), dtype=np.int64), seed + 1)
    null_p = np.empty((n_perm, 4))      # NMI, ARI, within NMI, within ARI
    null_n = np.empty((n_perm, 2))      # NMI, ARI
    for k in range(n_perm):
        tp = t[sh_period()]
        null_p[k] = (*pair_stats(f, tp, nf, nt), *within_stats(f, tp, groups, nf, nt))
        null_n[k] = pair_stats(f, t[sh_plain()], nf, nt)

    out = {
        "n_users": int(len(f)), "n_friend_communities": nf, "n_taste_communities": nt,
        "n_permutations": int(n_perm),
        "observed": {"nmi": nmi, "ari": ari, "within_period_nmi": w_nmi, "within_period_ari": w_ari},
        "within_period_shuffle": {"nmi": null_summary(nmi, null_p[:, 0]), "ari": null_summary(ari, null_p[:, 1]),
                                  "within_period_nmi": null_summary(w_nmi, null_p[:, 2]),
                                  "within_period_ari": null_summary(w_ari, null_p[:, 3])},
        "plain_shuffle": {"nmi": null_summary(nmi, null_n[:, 0]), "ari": null_summary(ari, null_n[:, 1])},
    }
    if with_ami:
        from sklearn.metrics import adjusted_mutual_info_score
        out["observed"]["ami"] = float(adjusted_mutual_info_score(f, t))
    return out, {"within_period": null_p, "plain": null_n}


def ami_over_friend_seeds(view: View) -> dict:
    """AMI (chance-corrected NMI) of each of the 20 friend runs against the chosen taste partition."""
    from sklearn.metrics import adjusted_mutual_info_score
    vals = np.array([adjusted_mutual_info_score(f, view.t_main) for f in view.f_seeds])
    return {"mean": float(vals.mean()), "sd": float(vals.std(ddof=1)), "min": float(vals.min()),
            "max": float(vals.max())}


def seed_grid(f_list: list, t_list: list, strata: np.ndarray, n_perm: int = 1000, seed: int = 42,
              progress: bool = True) -> pd.DataFrame:
    """NMI and ARI of every friend run vs every taste run, each with a within-period permutation null.

    The same permutations are used for every cell (common random numbers); only sums are
    stored, so memory does not grow with ``n_perm``.
    """
    nF, nT = len(f_list), len(t_list)
    nfs = [int(f.max()) + 1 for f in f_list]
    nts = [int(t.max()) + 1 for t in t_list]
    obs = np.array([[pair_stats(f_list[i], t_list[j], nfs[i], nts[j]) for j in range(nT)] for i in range(nF)])
    ge = np.zeros((nF, nT, 2))
    s1, s2 = np.zeros((nF, nT, 2)), np.zeros((nF, nT, 2))
    sh = Shuffler(strata, seed)
    for k in range(n_perm):
        idx = sh()
        for j in range(nT):
            tp = t_list[j][idx]
            for i in range(nF):
                v = np.array(pair_stats(f_list[i], tp, nfs[i], nts[j]))
                ge[i, j] += v >= obs[i, j] - TOL
                s1[i, j] += v
                s2[i, j] += v * v
        if progress and (k + 1) % 250 == 0:
            _log(True, f"    grid permutations {k + 1}/{n_perm}")
    mean = s1 / n_perm
    sd = np.sqrt(np.maximum(s2 / n_perm - mean ** 2, 0) * n_perm / max(n_perm - 1, 1))
    p = (1 + ge) / (1 + n_perm)
    rows = []
    for i in range(nF):
        for j in range(nT):
            rows.append({"friend_seed": i, "taste_seed": j,
                         "nmi": obs[i, j, 0], "ari": obs[i, j, 1],
                         "null_nmi_mean": mean[i, j, 0], "null_nmi_sd": sd[i, j, 0], "p_nmi": p[i, j, 0],
                         "null_ari_mean": mean[i, j, 1], "null_ari_sd": sd[i, j, 1], "p_ari": p[i, j, 1]})
    return pd.DataFrame(rows)


def _spread(x) -> dict:
    x = np.asarray(x, dtype=float)
    return {"mean": float(x.mean()), "sd": float(x.std(ddof=1)) if len(x) > 1 else float("nan"),
            "min": float(x.min()), "max": float(x.max())}


def summarise_grid(grid: pd.DataFrame) -> dict:
    """Mean / sd / range over the grid cells, and how often the result beats the null."""
    out = {"n_cells": int(len(grid))}
    for m in ("nmi", "ari"):
        out[m] = _spread(grid[m])
        out[f"null_{m}_mean"] = _spread(grid[f"null_{m}_mean"])
        out[f"excess_{m}"] = _spread(grid[m] - grid[f"null_{m}_mean"])
        out[f"share_cells_p_below_0.05_{m}"] = float((grid[f"p_{m}"] < 0.05).mean())
        out[f"max_p_{m}"] = float(grid[f"p_{m}"].max())
    return out


def seed_ceiling(parts: list) -> dict:
    """Agreement between runs of the same method: the most two partitions of one kind can agree."""
    stats = np.array([pair_stats(a, b) for a, b in combinations(parts, 2)])
    return {"n_pairs": int(len(stats)), "nmi": _spread(stats[:, 0]), "ari": _spread(stats[:, 1])}


def period_baselines(view: View, n_perm: int = 1000, seed: int = 42) -> dict:
    """NMI/ARI of each partition against the period bins, with a plain-shuffle null.

    If NMI(taste, period) is as large as NMI(taste, friends), "friends share tastes" cannot be
    told apart from "friends were active in the same years".
    """
    per = view.period
    np_ = int(per.max()) + 1
    sh = Shuffler(np.zeros(len(per), dtype=np.int64), seed)
    out = {}
    for name, main, seeds in (("friend", view.f_main, view.f_seeds), ("taste", view.t_main, view.t_seeds)):
        nm = int(main.max()) + 1
        nmi, ari = pair_stats(main, per, nm, np_)
        null = np.array([pair_stats(main, per[sh()], nm, np_) for _ in range(n_perm)])
        over_seeds = np.array([pair_stats(s, per) for s in seeds])
        out[name] = {"chosen": {"nmi": null_summary(nmi, null[:, 0]), "ari": null_summary(ari, null[:, 1])},
                     "over_seeds": {"nmi": _spread(over_seeds[:, 0]), "ari": _spread(over_seeds[:, 1])}}
    return out


def merge_small(codes: np.ndarray, min_size: int = MIN_COMMUNITY) -> np.ndarray:
    """Merge every community with fewer than ``min_size`` users into one 'other' group."""
    counts = np.bincount(codes)
    small = counts < min_size
    out = codes.copy()
    if small.any():
        out[small[codes]] = codes.max() + 1
    return factorize(out)


def _apply_variant(f, t, strata, mode: str, min_size: int):
    if mode == "merge":
        return merge_small(f, min_size), merge_small(t, min_size), strata
    if mode == "drop":
        keep = (np.bincount(f)[f] >= min_size) & (np.bincount(t)[t] >= min_size)
        return factorize(f[keep]), factorize(t[keep]), factorize(strata[keep])
    return f, t, strata


def sensitivity_variants(inp: Inputs, version: str, n_perm: int = 1000, seed: int = 42,
                         min_size: int = MIN_COMMUNITY) -> list:
    """How much the headline NMI/ARI depends on the choices made for isolated users and tiny communities."""
    specs = [
        ("main: users isolated in the taste graph excluded", True, "none"),
        ("isolated users included (each is a singleton community)", False, "none"),
        (f"tiny communities (< {min_size} users) merged into one group", True, "merge"),
        (f"users in tiny communities (< {min_size}) dropped", True, "drop"),
    ]
    rows = []
    for label, excl, mode in specs:
        view = make_view(inp, version, excl)
        f, t, strata = _apply_variant(view.f_main, view.t_main, view.period, mode, min_size)
        nf, nt = int(f.max()) + 1, int(t.max()) + 1
        nmi, ari = pair_stats(f, t, nf, nt)
        sh = Shuffler(strata, seed)
        null = np.array([pair_stats(f, t[sh()], nf, nt) for _ in range(n_perm)])
        seeds_obs = np.array([pair_stats(*_apply_variant(fs, view.t_main, view.period, mode, min_size)[:2])
                              for fs in view.f_seeds])
        rows.append({"variant": label, "n_users": int(len(f)), "n_friend_communities": nf,
                     "n_taste_communities": nt,
                     "nmi": null_summary(nmi, null[:, 0]), "ari": null_summary(ari, null[:, 1]),
                     "mean_nmi_over_20_friend_runs": float(seeds_obs[:, 0].mean()),
                     "mean_ari_over_20_friend_runs": float(seeds_obs[:, 1].mean())})
    return rows


# --------------------------------------------------------------------------
# Pair level
# --------------------------------------------------------------------------

def _random_pairs(rng: np.random.Generator, n: int, m: int) -> tuple:
    """``m`` random user pairs (u != v), uniformly from n users."""
    u = rng.integers(0, n, size=m)
    v = (u + 1 + rng.integers(0, n - 1, size=m)) % n
    return u, v


def edge_share_test(view: View, n_perm: int = 1000, seed: int = 42, progress: bool = True) -> dict:
    """Share of friend pairs that sit in the same taste community, for the chosen taste partition
    and all 20 taste runs.

    Null "random pairs": as many random user pairs as there are friend pairs (the report's version).
    Null "within period": taste labels shuffled among users of the same period, friend edges fixed.
    """
    eu, ev, n, m = view.eu, view.ev, len(view.sel), len(view.eu)
    parts = [view.t_main] + view.t_seeds
    obs = np.array([(t[eu] == t[ev]).mean() for t in parts])
    sh = Shuffler(view.period, seed)
    rng = np.random.default_rng(seed + 1)
    null_p = np.empty((n_perm, len(parts)))
    null_r = np.empty((n_perm, len(parts)))
    for k in range(n_perm):
        idx = sh()
        a, b = idx[eu], idx[ev]
        u, v = _random_pairs(rng, n, m)
        for j, t in enumerate(parts):
            null_p[k, j] = (t[a] == t[b]).mean()
            null_r[k, j] = (t[u] == t[v]).mean()
        if progress and (k + 1) % 250 == 0:
            _log(True, f"    edge-share permutations {k + 1}/{n_perm}")
    seeds = slice(1, None)
    return {
        "n_friend_pairs": int(m), "n_permutations": int(n_perm),
        "chosen_taste_partition": {"within_period": null_summary(obs[0], null_p[:, 0]),
                                   "random_pairs": null_summary(obs[0], null_r[:, 0])},
        "over_taste_runs": {
            "observed": _spread(obs[seeds]),
            "excess_within_period": _spread(obs[seeds] - null_p[:, seeds].mean(axis=0)),
            "excess_random_pairs": _spread(obs[seeds] - null_r[:, seeds].mean(axis=0)),
            "share_runs_p_below_0.05_within_period": float(np.mean(
                [p_upper(obs[j], null_p[:, j]) < 0.05 for j in range(1, len(parts))])),
        },
    }


def pair_jaccard(X, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Exact Jaccard similarity of the item sets of user pairs (u[i], v[i]); X = binary CSR (users x items)."""
    sizes = np.asarray(X.sum(axis=1)).ravel()
    inter = np.asarray(X[u].multiply(X[v]).sum(axis=1)).ravel()
    union = sizes[u] + sizes[v] - inter
    out = np.zeros(len(u), dtype=np.float64)
    np.divide(inter, union, out=out, where=union > 0)
    return out


def jaccard_pair_test(X, eu: np.ndarray, ev: np.ndarray, strata: np.ndarray, n_perm: int = 1000,
                      seed: int = 42, progress: bool = True) -> tuple:
    """Do friends have more similar liked-sets than other pairs of users?

    Statistics: mean Jaccard over the friend pairs, and the share of friend pairs with at least
    one liked item in common. ``X`` must only contain users with a non-empty set.
    Null "random pairs": as many random user pairs as friend pairs (the report's version).
    Null "within period": every user's set is swapped with a random user of the same period bin and
    the friend edges stay fixed, so each friend pair is compared with pairs active in the same periods.
    """
    n, m = X.shape[0], len(eu)
    j_obs = pair_jaccard(X, eu, ev)
    obs = (float(j_obs.mean()), float((j_obs > 0).mean()))
    sh = Shuffler(strata, seed)
    rng = np.random.default_rng(seed + 1)
    null_p = np.empty((n_perm, 2))
    null_r = np.empty((n_perm, 2))
    for k in range(n_perm):
        idx = sh()
        jp = pair_jaccard(X, idx[eu], idx[ev])
        u, v = _random_pairs(rng, n, m)
        jr = pair_jaccard(X, u, v)
        null_p[k] = (jp.mean(), (jp > 0).mean())
        null_r[k] = (jr.mean(), (jr > 0).mean())
        if progress and (k + 1) % 250 == 0:
            _log(True, f"    Jaccard permutations {k + 1}/{n_perm}")
    out = {"n_users": int(n), "n_friend_pairs": int(m), "n_permutations": int(n_perm)}
    for name, i in (("mean_jaccard", 0), ("share_pairs_with_common_item", 1)):
        out[name] = {"random_pairs": null_summary(obs[i], null_r[:, i]),
                     "within_period": null_summary(obs[i], null_p[:, i])}
        out[name]["ratio_to_random_pairs"] = float(obs[i] / np.mean(null_r[:, i]))
        out[name]["ratio_to_within_period"] = float(obs[i] / np.mean(null_p[:, i]))
    return out, {"within_period": null_p, "random_pairs": null_r, "observed": np.array(obs)}


# --------------------------------------------------------------------------
# User sets for the Jaccard test (rebuilt with the parameters of notebook 03)
# --------------------------------------------------------------------------

def usersets_to_matrix(us, users: np.ndarray):
    """Binary CSR matrix (len(users) x items) from a ``taste.UserSets``; users without a set get an empty row."""
    import scipy.sparse as sp
    pos = pd.Index(users).get_indexer(us.user_ids)
    if (pos < 0).any():
        raise ValueError("some users of the taste sets are not in the aligned user list")
    lens = np.fromiter((len(s) for s in us.sets), dtype=np.int64, count=len(us.sets))
    rows = np.repeat(pos, lens)
    cols = np.concatenate(us.sets) if len(us.sets) else np.empty(0, dtype=np.int64)
    return sp.csr_matrix((np.ones(len(cols), dtype=np.int32), (rows, cols)),
                         shape=(len(users), len(us.item_ids)))


def taste_params(root: Path) -> dict:
    """Parameters used by notebook 03 (read from taste_summary.json if it exists)."""
    params = dict(TASTE_PARAM_DEFAULTS)
    f = root / "data" / "results" / "taste_summary.json"
    if f.exists():
        saved = json.loads(f.read_text()).get("params", {})
        params.update({k: saved[k] for k in params if k in saved})
    return params


def build_taste_matrices(root: Path, users: np.ndarray, strict: bool = True) -> tuple:
    """Rebuild the liked-restaurant and liked-category sets (steps 1 and 3 files only).

    Careful: ``taste.build_user_sets`` has *different defaults* than notebook 03 (generic
    threshold 0.30 vs 0.80, max categories 10 vs 20), so the notebook's values are passed explicitly.
    Returns ({version: CSR matrix}, info dict).
    """
    try:
        import taste
    except ImportError:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import taste
    params = taste_params(root)
    reviews, businesses, users_df = taste.load_inputs(root)
    mats, info = {}, {"params": params}
    for v in VERSIONS:
        us = taste.build_user_sets(reviews, businesses, users_df, v, params["positive"],
                                   generic_max_frac=params["generic_max_frac"],
                                   max_cats_per_business=params["max_cats_per_business"],
                                   min_businesses_per_category=params["min_businesses_per_category"])
        mats[v] = usersets_to_matrix(us, users)
        info[v] = {"users_with_set": len(us.sets), "n_items": len(us.item_ids),
                   "mean_set_size": float(us.sizes.mean())}
        problems = []
        if len(us.sets) != EXPECTED_REPORT["users_with_set"]:
            problems.append(f"users with a set: {len(us.sets)} vs {EXPECTED_REPORT['users_with_set']}")
        if len(us.item_ids) != EXPECTED_REPORT["n_items"][v]:
            problems.append(f"items: {len(us.item_ids)} vs {EXPECTED_REPORT['n_items'][v]}")
        info[v]["problems"] = problems
        if problems:
            _raise_or_warn(f"{v} sets differ from notebook 03: " + "; ".join(problems), strict)
    return mats, info


def jaccard_view(inp: Inputs, X) -> tuple:
    """Restrict to the users with a non-empty set (Jaccard is undefined for two empty sets)."""
    sizes = np.asarray(X.sum(axis=1)).ravel()
    sel = np.flatnonzero(sizes > 0)
    eu, ev = remap_edges(inp.edge_u, inp.edge_v, sel, inp.n)
    return X[sel], eu, ev, factorize(inp.period[sel])


# --------------------------------------------------------------------------
# Sanity checks and orchestration
# --------------------------------------------------------------------------

def _raise_or_warn(msg: str, strict: bool) -> None:
    if strict:
        raise AssertionError(msg + "  (set strict=False to continue anyway)")
    warnings.warn(msg)


def check_against_report(inp: Inputs, strict: bool = True) -> dict:
    """The regenerated partitions must match what the report and the hand-off notes say.

    Each team member regenerates the data locally; if a library version changes the Louvain runs,
    the numbers here would silently disagree with the report. Community counts include the
    singleton communities of isolated users, as in notebook 03.
    """
    found = {"n_users": inp.n,
             "friend_communities": int(len(np.unique(inp.friend_main))),
             "taste_communities": {v: int(len(np.unique(inp.taste_main[v]))) for v in VERSIONS},
             "n_isolated_users": {v: int(inp.isolated[v].sum()) for v in VERSIONS},
             "n_friend_pairs": int(len(inp.edge_u)),
             "n_friend_runs": int(inp.friend_seeds.shape[1]),
             "n_taste_runs": {v: int(inp.taste_seeds[v].shape[1]) for v in VERSIONS}}
    problems = []
    if found["n_users"] != EXPECTED_REPORT["n_users"]:
        problems.append(f"users: {found['n_users']} vs {EXPECTED_REPORT['n_users']}")
    if found["friend_communities"] != EXPECTED_REPORT["friend_communities"]:
        problems.append(f"friend communities: {found['friend_communities']} vs {EXPECTED_REPORT['friend_communities']}")
    for v in VERSIONS:
        if found["taste_communities"][v] != EXPECTED_REPORT["taste_communities"][v]:
            problems.append(f"{v} taste communities: {found['taste_communities'][v]} vs "
                            f"{EXPECTED_REPORT['taste_communities'][v]}")
    if problems:
        _raise_or_warn("regenerated data differ from the report: " + "; ".join(problems), strict)
    return {"found": found, "expected": EXPECTED_REPORT, "problems": problems, "ok": not problems}


def analyse_version(inp: Inputs, version: str, n_perm_main: int = 10_000, n_perm_grid: int = 1000,
                    n_perm_other: int = 1000, seed: int = 42, progress: bool = True) -> tuple:
    """All partition-level results for one taste version. Returns (results dict, tables/nulls dict)."""
    view = make_view(inp, version)
    _log(progress, f"[{version}] {len(view.sel):,} users ({view.n_isolated_excluded} isolated excluded), "
                   f"{len(view.eu):,} friend pairs")
    verify_against_sklearn(view.f_main, view.t_main)

    _log(progress, f"[{version}] chosen partitions, {n_perm_main:,} permutations")
    main, nulls = compare_pair(view.f_main, view.t_main, view.period, n_perm_main, seed)
    ami_seeds = ami_over_friend_seeds(view)

    _log(progress, f"[{version}] {len(view.f_seeds)} x {len(view.t_seeds)} grid, {n_perm_grid:,} permutations")
    grid = seed_grid(view.f_seeds, view.t_seeds, view.period, n_perm_grid, seed + 2, progress)

    _log(progress, f"[{version}] ceiling, period baselines, sensitivity, friend-pair share")
    ceiling = {"friend_runs": seed_ceiling(view.f_seeds), "taste_runs": seed_ceiling(view.t_seeds)}
    period = period_baselines(view, n_perm_other, seed + 3)
    variants = sensitivity_variants(inp, version, n_perm_other, seed + 4)
    share = edge_share_test(view, n_perm_other, seed + 5, progress)

    results = {"version": version, "n_users_compared": int(len(view.sel)),
               "n_isolated_excluded": view.n_isolated_excluded, "main": main,
               "ami_over_friend_runs": ami_seeds, "grid": summarise_grid(grid), "ceiling": ceiling,
               "period_baselines": period, "sensitivity": variants, "friend_pair_share": share}
    return results, {"grid": grid, "main_nulls": nulls}


def analyse_pairs(inp: Inputs, matrices: dict, n_perm: int = 1000, seed: int = 42,
                  progress: bool = True) -> tuple:
    """Friend-pair Jaccard test for each taste version. Returns (results, nulls)."""
    res, nulls = {}, {}
    for v in VERSIONS:
        X, eu, ev, strata = jaccard_view(inp, matrices[v])
        _log(progress, f"[{v}] Jaccard test: {X.shape[0]:,} users, {len(eu):,} friend pairs, {n_perm:,} permutations")
        res[v], nulls[v] = jaccard_pair_test(X, eu, ev, strata, n_perm, seed, progress)
    return res, nulls


def taste_vs_taste(inp: Inputs) -> dict:
    """Agreement between the category-based and the restaurant-based taste partitions (chosen runs)."""
    sel = np.flatnonzero(~(inp.isolated["category"] | inp.isolated["business"]))
    a, b = factorize(inp.taste_main["category"][sel]), factorize(inp.taste_main["business"][sel])
    nmi, ari = pair_stats(a, b)
    return {"n_users": int(len(sel)), "nmi": nmi, "ari": ari}


def build_table_v(results: dict) -> pd.DataFrame:
    """Compact table for the report (Table V): NMI and ARI, mean [min, max] over the 20 x 20 runs."""
    def cell(s):
        return f"{s['mean']:.3f} [{s['min']:.3f}, {s['max']:.3f}]"
    rows = []
    for v in VERSIONS:
        r = results["versions"][v]
        g, c, per = r["grid"], r["ceiling"], r["period_baselines"]
        m = r["main"]
        label = VNAME[v]
        rows += [
            (label, "chosen runs", m["observed"]["nmi"], m["observed"]["ari"]),
            (label, "20x20 runs, mean [min, max]", cell(g["nmi"]), cell(g["ari"])),
            (label, "chance, shuffled within period", cell(g["null_nmi_mean"]), cell(g["null_ari_mean"])),
            (label, "friend vs period (20 runs)", cell(per["friend"]["over_seeds"]["nmi"]),
             cell(per["friend"]["over_seeds"]["ari"])),
            (label, "taste vs period (20 runs)", cell(per["taste"]["over_seeds"]["nmi"]),
             cell(per["taste"]["over_seeds"]["ari"])),
            (label, "ceiling: friend vs friend runs", cell(c["friend_runs"]["nmi"]), cell(c["friend_runs"]["ari"])),
            (label, "ceiling: taste vs taste runs", cell(c["taste_runs"]["nmi"]), cell(c["taste_runs"]["ari"])),
        ]
    df = pd.DataFrame(rows, columns=["taste version", "comparison", "NMI", "ARI"])
    for col in ("NMI", "ARI"):
        df[col] = df[col].map(lambda x: f"{x:.3f}" if isinstance(x, float) else x)
    return df


def main_table(results: dict) -> pd.DataFrame:
    """Chosen partitions: observed agreement vs the two null models (one row per measure)."""
    rows = []
    for v in VERSIONS:
        m = results["versions"][v]["main"]
        obs, w, pl = m["observed"], m["within_period_shuffle"], m["plain_shuffle"]
        for key, name in (("nmi", "NMI"), ("ari", "ARI")):
            rows.append({"taste version": VNAME[v], "measure": name, "observed": obs[key],
                         "chance, shuffled within period": w[key]["null_mean"], "excess": w[key]["excess"],
                         "z": w[key]["z"], "p (within period)": w[key]["p"],
                         "chance, plain shuffle": pl[key]["null_mean"], "p (plain shuffle)": pl[key]["p"]})
        for key, name in (("within_period_nmi", "NMI inside period bins"), ("within_period_ari", "ARI inside period bins")):
            rows.append({"taste version": VNAME[v], "measure": name, "observed": obs[key],
                         "chance, shuffled within period": w[key]["null_mean"], "excess": w[key]["excess"],
                         "z": w[key]["z"], "p (within period)": w[key]["p"],
                         "chance, plain shuffle": np.nan, "p (plain shuffle)": np.nan})
        if "ami" in obs:
            rows.append({"taste version": VNAME[v], "measure": "AMI", "observed": obs["ami"]})
    return pd.DataFrame(rows)


def grid_table(results: dict) -> pd.DataFrame:
    """The 20 x 20 runs: mean [min, max] and how often the result beats the within-period null."""
    rows = []
    for v in VERSIONS:
        g = results["versions"][v]["grid"]
        for m in ("nmi", "ari"):
            rows.append({"taste version": VNAME[v], "measure": m.upper(),
                         "observed mean": g[m]["mean"], "sd": g[m]["sd"], "min": g[m]["min"], "max": g[m]["max"],
                         "chance mean": g[f"null_{m}_mean"]["mean"], "mean excess": g[f"excess_{m}"]["mean"],
                         "share of cells with p<0.05": g[f"share_cells_p_below_0.05_{m}"],
                         "largest p": g[f"max_p_{m}"]})
    return pd.DataFrame(rows)


def sensitivity_table(results: dict) -> pd.DataFrame:
    rows = []
    for v in VERSIONS:
        for r in results["versions"][v]["sensitivity"]:
            rows.append({"taste version": VNAME[v], "variant": r["variant"], "users": r["n_users"],
                         "friend comm.": r["n_friend_communities"], "taste comm.": r["n_taste_communities"],
                         "NMI": r["nmi"]["observed"], "NMI chance": r["nmi"]["null_mean"], "NMI p": r["nmi"]["p"],
                         "ARI": r["ari"]["observed"], "ARI chance": r["ari"]["null_mean"], "ARI p": r["ari"]["p"],
                         "mean NMI, 20 friend runs": r["mean_nmi_over_20_friend_runs"]})
    return pd.DataFrame(rows)


def pair_table(results: dict) -> pd.DataFrame:
    """Friend pairs vs other pairs: same-taste-community share and Jaccard similarity of the liked sets."""
    rows = []
    for v in VERSIONS:
        s = results["versions"][v]["friend_pair_share"]["chosen_taste_partition"]
        rows.append({"taste version": VNAME[v], "statistic": "share of friend pairs in the same taste community",
                     "friend pairs": s["within_period"]["observed"],
                     "random pairs": s["random_pairs"]["null_mean"], "p (random pairs)": s["random_pairs"]["p"],
                     "same-period pairs": s["within_period"]["null_mean"], "p (within period)": s["within_period"]["p"]})
        if "jaccard" in results:
            for key, name in (("mean_jaccard", "mean Jaccard of the liked sets"),
                              ("share_pairs_with_common_item", "share of friend pairs with a liked item in common")):
                j = results["jaccard"][v][key]
                rows.append({"taste version": VNAME[v], "statistic": name,
                             "friend pairs": j["within_period"]["observed"],
                             "random pairs": j["random_pairs"]["null_mean"], "p (random pairs)": j["random_pairs"]["p"],
                             "same-period pairs": j["within_period"]["null_mean"], "p (within period)": j["within_period"]["p"]})
    return pd.DataFrame(rows)


def to_jsonable(x):
    """Convert numpy / pandas objects so that json can write them."""
    if isinstance(x, dict):
        return {str(k): to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [to_jsonable(v) for v in x]
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    return x


def save_results(root: Path, results: dict, tables: dict, pair_nulls: dict | None = None) -> list:
    """Write everything to data/results/ (not in git): one JSON, CSV tables, null distributions."""
    out = root / "data" / "results"
    out.mkdir(parents=True, exist_ok=True)
    written = []
    path = out / "comparison_summary.json"
    path.write_text(json.dumps(to_jsonable(results), indent=2))
    written.append(path)
    for v, t in tables.items():
        p = out / f"comparison_grid_{v}.csv"
        t["grid"].to_csv(p, index=False)
        written.append(p)
        nul = t["main_nulls"]
        p = out / f"comparison_main_nulls_{v}.csv"
        pd.DataFrame({"nmi_within_period": nul["within_period"][:, 0], "ari_within_period": nul["within_period"][:, 1],
                      "within_nmi_within_period": nul["within_period"][:, 2],
                      "within_ari_within_period": nul["within_period"][:, 3],
                      "nmi_plain": pd.Series(nul["plain"][:, 0]), "ari_plain": pd.Series(nul["plain"][:, 1])}
                     ).to_csv(p, index=False)
        written.append(p)
    if pair_nulls:
        for v, n in pair_nulls.items():
            p = out / f"comparison_jaccard_nulls_{v}.csv"
            pd.DataFrame({"mean_jaccard_within_period": n["within_period"][:, 0],
                          "mean_jaccard_random_pairs": n["random_pairs"][:, 0],
                          "share_common_item_within_period": n["within_period"][:, 1],
                          "share_common_item_random_pairs": n["random_pairs"][:, 1]}).to_csv(p, index=False)
            written.append(p)
    p = out / "comparison_table_v.csv"
    build_table_v(results).to_csv(p, index=False)
    written.append(p)
    return written


def run_all(root: Path | None = None, n_perm_main: int = 10_000, n_perm_grid: int = 1000,
            n_perm_other: int = 1000, n_perm_pairs: int = 1000, seed: int = 42, strict: bool = True,
            with_jaccard: bool = True, progress: bool = True) -> tuple:
    """Full step 5. Returns (results, tables, pair_nulls)."""
    root = find_project_root(root)
    inp = load_inputs(root)
    results = {"parameters": {"n_perm_main": n_perm_main, "n_perm_grid": n_perm_grid,
                              "n_perm_other": n_perm_other, "n_perm_pairs": n_perm_pairs, "seed": seed,
                              "min_community": MIN_COMMUNITY, "period_bins": inp.period_labels},
               "sanity": check_against_report(inp, strict), "versions": {}}
    tables = {}
    for v in VERSIONS:
        results["versions"][v], tables[v] = analyse_version(inp, v, n_perm_main, n_perm_grid, n_perm_other,
                                                            seed, progress)
    results["taste_vs_taste"] = taste_vs_taste(inp)
    pair_nulls = None
    if with_jaccard:
        mats, info = build_taste_matrices(root, inp.users, strict)
        results["sets"] = info
        results["jaccard"], pair_nulls = analyse_pairs(inp, mats, n_perm_pairs, seed + 6, progress)
    results["table_v"] = build_table_v(results).to_dict(orient="records")
    written = save_results(root, results, tables, pair_nulls)
    _log(progress, "written:\n  " + "\n  ".join(str(p.relative_to(root)) for p in written))
    return results, tables, pair_nulls


# --------------------------------------------------------------------------
# Figures (project palette: blue = category taste, orange = restaurant taste, grey = baselines)
# --------------------------------------------------------------------------

BLUE, ORANGE, INK, MUTED, GREY = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#9a9993"
VCOLOR = {"category": BLUE, "business": ORANGE}


def _style(plt) -> None:
    plt.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 150, "savefig.bbox": "tight",
        "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": MUTED,
        "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
        "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.6, "axes.axisbelow": True,
        "axes.titlesize": 10.5, "axes.titleweight": "bold", "axes.titlelocation": "left",
    })


def plot_main_vs_null(results: dict, tables: dict, path: Path):
    """Chosen partitions: observed NMI / ARI against the two null distributions."""
    import matplotlib.pyplot as plt
    _style(plt)
    fig, axes = plt.subplots(2, 2, figsize=(9, 5.6))
    for r, v in enumerate(VERSIONS):
        res = results["versions"][v]["main"]
        nul = tables[v]["main_nulls"]
        for c, (m, ci) in enumerate((("nmi", 0), ("ari", 1))):
            ax = axes[r, c]
            obs = res["observed"][m]
            lo = min(nul["plain"][:, ci].min(), nul["within_period"][:, ci].min())
            hi = max(obs, nul["within_period"][:, ci].max()) * 1.08
            bins = np.linspace(lo, hi, 50)
            ax.hist(nul["plain"][:, ci], bins=bins, color=GREY, alpha=0.8, label="shuffled among all users")
            ax.hist(nul["within_period"][:, ci], bins=bins, color=VCOLOR[v], alpha=0.75,
                    label="shuffled within period")
            ax.axvline(obs, color=INK, lw=2, label="observed")
            p = res["within_period_shuffle"][m]["p"]
            ax.set_title(f"{VNAME[v]}: {m.upper()} = {obs:.3f}\np = {p:.4g} (within period)")
            ax.set_xlabel(m.upper())
            ax.locator_params(axis="x", nbins=6)     # fewer ticks: neighbouring panels' labels collided
            ax.set_yticks([])
            if r == 0 and c == 0:
                ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.suptitle("Friend vs taste partition: observed agreement and chance (chosen runs)", x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path)
    return fig


def plot_agreement_levels(results: dict, path: Path):
    """Observed friend-taste agreement next to chance, period and the run-to-run ceiling (20 runs each)."""
    import matplotlib.pyplot as plt
    _style(plt)
    labels = ["friend vs taste (observed)", "friend vs taste, expected by chance within period",
              "friend vs period", "taste vs period",
              "ceiling: friend run vs friend run", "ceiling: taste run vs taste run"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    for ax, m in zip(axes, ("nmi", "ari")):
        for k, v in enumerate(VERSIONS):
            r = results["versions"][v]
            stats = [r["grid"][m], r["grid"][f"null_{m}_mean"], r["period_baselines"]["friend"]["over_seeds"][m],
                     r["period_baselines"]["taste"]["over_seeds"][m], r["ceiling"]["friend_runs"][m],
                     r["ceiling"]["taste_runs"][m]]
            y = np.arange(len(labels)) + (-0.13 if k == 0 else 0.13)
            ax.hlines(y, [s["min"] for s in stats], [s["max"] for s in stats], color=VCOLOR[v], lw=2)
            ax.plot([s["mean"] for s in stats], y, "o", color=VCOLOR[v], ms=7, mec="white", mew=1.2,
                    label=VNAME[v])
        ax.set_yticks(np.arange(len(labels)))
        ax.set_yticklabels(labels)
        ax.set_xlabel(f"{m.upper()} (dot = mean, line = min-max over runs)")
        ax.set_title(m.upper())
        ax.grid(axis="y", visible=False)
    axes[0].invert_yaxis()          # shared y axis: invert once
    axes[0].legend(frameon=False, fontsize=8, loc="upper right")
    fig.suptitle("How much do friend and taste communities agree, compared with chance, period and run-to-run noise?",
                 x=0.01, ha="left", fontsize=10.5)
    fig.tight_layout()
    fig.savefig(path)
    return fig


def plot_pair_tests(results: dict, pair_nulls: dict, path: Path, column: bool = False):
    """Mean Jaccard similarity of friend pairs vs random pairs and vs period-matched pairs.

    ``column=True`` stacks the two panels for one column of a two-column paper (report version).
    """
    import matplotlib.pyplot as plt
    _style(plt)
    if column:
        fig, axes = plt.subplots(2, 1, figsize=(3.5, 4.4))
    else:
        fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6))
    for ax, v in zip(axes, VERSIONS):
        n = pair_nulls[v]
        obs = results["jaccard"][v]["mean_jaccard"]["random_pairs"]["observed"]
        lo = min(n["random_pairs"][:, 0].min(), n["within_period"][:, 0].min(), obs)
        hi = max(n["random_pairs"][:, 0].max(), n["within_period"][:, 0].max(), obs)
        pad = 0.04 * (hi - lo) if hi > lo else 0.01
        bins = np.linspace(lo - pad, hi + pad, 60)
        ax.hist(n["random_pairs"][:, 0], bins=bins, color=GREY, alpha=0.8, label="random user pairs")
        ax.hist(n["within_period"][:, 0], bins=bins, color=VCOLOR[v], alpha=0.75, label="same-period shuffle")
        ax.axvline(obs, color=INK, lw=2, label="friend pairs")
        p = results["jaccard"][v]["mean_jaccard"]["within_period"]["p"]
        ax.set_yticks([])
        if column:
            ax.set_title(f"{VNAME[v].capitalize()}: friend pairs {obs:.3f}", fontsize=9)
            ax.set_xlabel("mean Jaccard similarity of the liked sets", fontsize=8)
            ax.tick_params(labelsize=7)
            ax.locator_params(axis="x", nbins=5)
            ax.legend(frameon=False, fontsize=7, loc="upper center")
        else:
            ax.set_title(f"{VNAME[v]}: mean Jaccard {obs:.3f}\np = {p:.4g} (within period)")
            ax.set_xlabel("mean Jaccard similarity of the liked sets")
            ax.legend(frameon=False, fontsize=8, loc="upper left")
    if not column:
        fig.suptitle("Do friends like the same things more than other users?", x=0.01, ha="left", fontsize=10.5)
    fig.tight_layout()
    fig.savefig(path)
    return fig


# --------------------------------------------------------------------------
# Command line
# --------------------------------------------------------------------------

def main(argv=None) -> None:
    import argparse
    ap = argparse.ArgumentParser(description="Step 5: compare friend and taste partitions.")
    ap.add_argument("--root", type=Path, default=None, help="repository root (default: auto-detect)")
    ap.add_argument("--perm-main", type=int, default=10_000)
    ap.add_argument("--perm-grid", type=int, default=1000)
    ap.add_argument("--perm-other", type=int, default=1000)
    ap.add_argument("--perm-pairs", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-strict", action="store_true", help="only warn if the regenerated data differ from the report")
    ap.add_argument("--no-jaccard", action="store_true", help="skip the friend-pair Jaccard test")
    a = ap.parse_args(argv)
    root = find_project_root(a.root)
    results, tables, pair_nulls = run_all(root, a.perm_main, a.perm_grid, a.perm_other, a.perm_pairs, a.seed,
                                          strict=not a.no_strict, with_jaccard=not a.no_jaccard)
    import matplotlib
    matplotlib.use("Agg")           # command line: write the PNGs without opening windows
    fig_dir = root / "figures"
    fig_dir.mkdir(exist_ok=True)
    plot_main_vs_null(results, tables, fig_dir / "05_main_vs_null.png")
    plot_agreement_levels(results, fig_dir / "05_agreement_levels.png")
    if pair_nulls:
        plot_pair_tests(results, pair_nulls, fig_dir / "05_pair_tests.png")
        plot_pair_tests(results, pair_nulls, fig_dir / "05_pair_tests_report.png", column=True)
    print(build_table_v(results).to_string(index=False))


if __name__ == "__main__":
    main()
