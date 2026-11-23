"""Association rules per community and their comparison — step 4.

* ``rules_from_itemsets``: rules X -> Y from frequent itemsets (exact counts).
* ``SupportIndex``: exact support of any itemset in a fixed transaction set
  (used for population-level confidence/lift and the same-business tag share).
* ``mine_partition``: mine every eligible community of a partition.
* ``overlap`` / ``null_model``: how distinctive and how shared the rule sets are,
  against random partitions with the same community sizes.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
import pandas as pd

import itemsets as I

RULE_COLS = ["antecedent", "consequent", "support_count", "support", "confidence", "lift"]


# --------------------------------------------------------------------------
# Rules
# --------------------------------------------------------------------------

def rules_from_itemsets(itemsets: dict, n: int, min_confidence: float = 0.0,
                        min_lift: float = 0.0) -> tuple[pd.DataFrame, int]:
    """All rules X -> Y with X, Y non-empty and disjoint, X ∪ Y a frequent itemset.

    support = s(X∪Y)/n, confidence = s(X∪Y)/s(X), lift = confidence / (s(Y)/n).
    Every subset of a frequent itemset is frequent, so s(X), s(Y) are in ``itemsets``.
    Returns (filtered rules sorted deterministically, number of rules before filtering).
    """
    rows = []
    for s in sorted(k for k in itemsets if len(k) >= 2):
        c = itemsets[s]
        for r in range(1, len(s)):
            for X in combinations(s, r):
                Y = tuple(x for x in s if x not in X)
                conf = c / itemsets[X]
                rows.append((X, Y, c, c / n, conf, conf / (itemsets[Y] / n)))
    df = pd.DataFrame(rows, columns=RULE_COLS)
    n_before = len(df)
    df = df[(df.confidence >= min_confidence) & (df.lift >= min_lift)]
    df = df.sort_values(["lift", "confidence", "antecedent", "consequent"],
                        ascending=[False, False, True, True], kind="mergesort")
    return df.reset_index(drop=True), n_before


class SupportIndex:
    """Exact support of itemsets over fixed rows (users or businesses), cached.

    ``rows`` is a list of sorted item-index arrays; the index is a dense boolean
    row x item matrix, so the support of an itemset is the number of rows that
    have all its columns set.
    """

    def __init__(self, rows, n_items: int):
        self.n = len(rows)
        self.M = np.zeros((self.n, n_items), dtype=bool)
        for r, items in enumerate(rows):
            self.M[r, items] = True
        self.cache: dict[tuple, int] = {}

    def support(self, sets) -> np.ndarray:
        missing = sorted({s for s in sets if s not in self.cache}, key=len)
        by_len: dict[int, list] = {}
        for s in missing:
            by_len.setdefault(len(s), []).append(s)
        for group in by_len.values():
            idx = np.array(group)
            for a in range(0, len(idx), 2000):
                cnt = self.M[:, idx[a:a + 2000]].all(axis=2).sum(axis=0)
                self.cache.update(zip(group[a:a + 2000], cnt.tolist()))
        return np.array([self.cache[s] for s in sets], dtype=np.int64)


def add_population_metrics(rules: pd.DataFrame, pop: SupportIndex) -> pd.DataFrame:
    """Confidence and lift of the same rule on the whole population, and
    ``lift_ratio`` = lift in the community / lift in the population."""
    if rules.empty:
        return rules.assign(confidence_pop=[], lift_pop=[], lift_ratio=[])
    xy = [tuple(sorted(a + c)) for a, c in zip(rules.antecedent, rules.consequent)]
    s_xy, s_x, s_y = (pop.support(xy), pop.support(list(rules.antecedent)),
                      pop.support(list(rules.consequent)))
    conf = np.divide(s_xy, s_x, out=np.zeros(len(s_x)), where=s_x > 0)
    lift = np.divide(conf * pop.n, s_y, out=np.zeros(len(s_y)), where=s_y > 0)
    ratio = np.divide(rules.lift.to_numpy(), lift, out=np.full(len(lift), np.inf), where=lift > 0)
    return rules.assign(confidence_pop=conf, lift_pop=lift, lift_ratio=ratio)


def add_same_business_share(rules: pd.DataFrame, biz: SupportIndex) -> pd.DataFrame:
    """Share of businesses tagged with all of X that are also tagged with all of Y.

    High values mean the rule mostly restates how Yelp tags single restaurants
    (e.g. Sushi Bars -> Japanese) rather than two tastes of the same user.
    """
    if rules.empty:
        return rules.assign(same_business_share=[])
    xy = [tuple(sorted(a + c)) for a, c in zip(rules.antecedent, rules.consequent)]
    s_xy, s_x = biz.support(xy), biz.support(list(rules.antecedent))
    share = np.divide(s_xy, s_x, out=np.zeros(len(s_x)), where=s_x > 0)
    return rules.assign(same_business_share=share)


# --------------------------------------------------------------------------
# Mining a partition
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class MiningParams:
    min_support: float
    min_confidence: float
    min_lift: float
    max_k: int
    min_count_floor: int
    min_community_size: int
    lift_ratio_min: float


def mine_partition(baskets: I.Baskets, labels: pd.Series, p: MiningParams,
                   pop: SupportIndex, biz: SupportIndex | None = None
                   ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Mine each community (labels: user_id -> community) on its members' baskets.

    Communities with fewer than ``p.min_community_size`` mining transactions are
    reported as excluded, not silently skipped. Returns (per-community table, rules).
    """
    summary, all_rules = [], []
    for comm, members in labels.groupby(labels).groups.items():
        T = baskets.subset(members)
        row = {"community_id": comm, "n_assigned": len(members), "n_transactions": len(T)}
        if len(T) < p.min_community_size:
            summary.append({**row, "eligible": False, "reason": "too small"})
            continue
        mc = I.min_support_count(p.min_support, len(T), p.min_count_floor)
        res = I.apriori(T, baskets.n_items, mc, p.max_k)
        rules, n_before = rules_from_itemsets(res.itemsets, len(T), p.min_confidence, p.min_lift)
        rules = add_population_metrics(rules, pop)
        if biz is not None:
            rules = add_same_business_share(rules, biz)
        rules = rules.assign(community_id=comm,
                             specific=rules.lift_ratio >= p.lift_ratio_min)
        items_here = np.unique(np.concatenate(T)) if T else []
        sizes = res.by_size()
        summary.append({**row, "eligible": True, "reason": "", "n_items": len(items_here),
                        "min_support": p.min_support, "min_count": mc,
                        **{f"frequent_k{k}": sizes.get(k, 0) for k in range(1, p.max_k + 1)},
                        "rules_before_filter": n_before, "rules": len(rules),
                        "rules_specific": int(rules.specific.sum())})
        all_rules.append(rules)
    all_rules = [r for r in all_rules if not r.empty]
    rules = pd.concat(all_rules, ignore_index=True) if all_rules else pd.DataFrame()
    return pd.DataFrame(summary), rules


# --------------------------------------------------------------------------
# Comparing rule sets
# --------------------------------------------------------------------------

def rule_keys(rules: pd.DataFrame) -> pd.Series:
    """Hashable, directed identity of a rule: (antecedent, consequent)."""
    return pd.Series(list(zip(rules.antecedent, rules.consequent)), index=rules.index)


def rule_sets(rules: pd.DataFrame, specific_only: bool = False,
              exclude_tag: float | None = None) -> dict:
    """{community: set of rule keys}, optionally only specific / non-tag rules."""
    if rules.empty:
        return {}
    r = rules
    if specific_only:
        r = r[r.specific]
    if exclude_tag is not None and "same_business_share" in r:
        r = r[r.same_business_share < exclude_tag]
    keys = rule_keys(r)
    return {c: set(keys[g]) for c, g in r.groupby("community_id").groups.items()}


def jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if (a or b) else 0.0


def overlap(sets: dict) -> dict:
    """Distinctiveness of the communities of one partition.

    * ``mean_pairwise_jaccard``: average Jaccard between the rule sets of two
      communities (low = communities have different rules);
    * ``share_unique``: share of distinct rules found in exactly one community.
    """
    if not sets:
        return {"n_rules_distinct": 0, "mean_pairwise_jaccard": np.nan, "share_unique": np.nan}
    pj = [jaccard(sets[a], sets[b]) for a, b in combinations(sorted(sets), 2)]
    counts: dict = {}
    for s in sets.values():
        for k in s:
            counts[k] = counts.get(k, 0) + 1
    uniq = sum(1 for v in counts.values() if v == 1)
    return {"n_rules_distinct": len(counts),
            "mean_pairwise_jaccard": float(np.mean(pj)) if pj else np.nan,
            "share_unique": uniq / len(counts) if counts else np.nan}


def union(sets: dict) -> set:
    out: set = set()
    for s in sets.values():
        out |= s
    return out


def shuffled_labels(labels: pd.Series, seed: int) -> pd.Series:
    """Random partition with exactly the same community sizes."""
    rng = np.random.default_rng(seed)
    return pd.Series(rng.permutation(labels.to_numpy()), index=labels.index)


def pair_rules(rules: pd.DataFrame) -> pd.DataFrame:
    """Rules with one item on each side. Comparisons use these: rules from triples
    grow combinatorially in communities of heavy reviewers and would dominate
    every overlap measure."""
    if rules.empty:
        return rules
    return rules[(rules.antecedent.str.len() == 1) & (rules.consequent.str.len() == 1)]


def partition_measures(rules: pd.DataFrame, exclude_tag: float | None = None,
                       reference: dict | None = None) -> dict:
    """Comparison measures of one partition, on its pair rules.

    * ``median_lift_ratio`` / ``share_lift_ratio_lt1``: does the partition absorb the
      population's associations (lift inside communities below population lift)?
    * ``rules_specific``: rules with lift_ratio >= the cut-off;
    * overlap measures (see ``overlap``) on all and on specific rules;
    * ``jaccard_vs_<name>``: union of specific rules vs each reference rule set.
    """
    r = pair_rules(rules)
    if exclude_tag is not None and "same_business_share" in r:
        r = r[r.same_business_share < exclude_tag]
    row = {"pair_rules": len(r),
           "median_lift_ratio": float(r.lift_ratio.median()) if len(r) else np.nan,
           "share_lift_ratio_lt1": float((r.lift_ratio < 1).mean()) if len(r) else np.nan,
           "rules_specific": int(r.specific.sum()) if len(r) else 0}
    for tag, kw in (("all", {}), ("specific", {"specific_only": True})):
        row.update({f"{tag}_{k}": v for k, v in overlap(rule_sets(r, **kw)).items()})
    for name, ref in (reference or {}).items():
        row[f"jaccard_vs_{name}"] = jaccard(union(rule_sets(r, specific_only=True)), ref)
    return row


def null_model(baskets, labels, p, pop, n_shuffles: int, seed: int, biz=None,
               reference: dict | None = None, exclude_tag: float | None = None) -> pd.DataFrame:
    """``partition_measures`` on ``n_shuffles`` size-preserving random partitions."""
    rows = []
    for s in range(n_shuffles):
        _, r = mine_partition(baskets, shuffled_labels(labels, seed + s), p, pop, biz)
        rows.append({"shuffle": s, **partition_measures(r, exclude_tag, reference)})
    return pd.DataFrame(rows)
