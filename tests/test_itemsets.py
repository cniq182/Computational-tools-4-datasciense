"""Tests for src/itemsets.py: A-priori and PCY against a brute-force reference."""

import sys
from collections import Counter
from itertools import combinations
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import itemsets as I  # noqa: E402


def brute_force(transactions, min_count, max_k):
    """Every itemset up to max_k with its exact support (reference answer)."""
    c = Counter()
    for t in transactions:
        items = sorted(set(int(x) for x in t))
        for k in range(1, max_k + 1):
            c.update(combinations(items, k))
    return {s: n for s, n in c.items() if n >= min_count}


def tx(*baskets):
    return [np.array(sorted(b), dtype=np.int64) for b in baskets]


TOY = tx({0, 1, 2}, {0, 1}, {0, 2}, {1, 2, 3}, {0, 1, 2, 3}, {3})


# --- helpers -------------------------------------------------------------

def test_min_support_count_rounds_up_and_applies_floor():
    assert I.min_support_count(0.1, 100) == 10          # no float error
    assert I.min_support_count(0.1, 101) == 11
    assert I.min_support_count(0.01, 100, floor=5) == 5
    with pytest.raises(ValueError):
        I.min_support_count(0.0, 10)


@pytest.mark.parametrize("m", [2, 3, 7, 50])
def test_triangular_index_is_a_bijection_and_invertible(m):
    i, j = np.triu_indices(m, 1)
    k = I.triangular_index(i, j, m)
    assert np.array_equal(k, np.arange(m * (m - 1) // 2))
    ii, jj = I.triangular_pair(k, m)
    assert np.array_equal(ii, i) and np.array_equal(jj, j)


def test_pair_bucket_is_deterministic_and_in_range():
    i, j = np.array([0, 5, 100]), np.array([1, 7, 4000])
    h1, h2 = I.pair_bucket(i, j, 97), I.pair_bucket(i, j, 97)
    assert np.array_equal(h1, h2) and h1.min() >= 0 and h1.max() < 97


@pytest.mark.parametrize("n_buckets", [1_009, 100_003, 1_000_003])
def test_pair_bucket_spreads_distinct_pairs(n_buckets):
    """Regression: a linear hash whose constant equalled n_buckets put 19,900 pairs
    into ~200 buckets. A good hash leaves few collisions when buckets >> pairs."""
    i, j = np.triu_indices(200, 1)
    used = len(np.unique(I.pair_bucket(i, j, n_buckets)))
    expected = n_buckets * (1 - (1 - 1 / n_buckets) ** len(i))   # balls into bins
    assert used >= 0.95 * expected


def test_frequent_pairs_sit_in_frequent_buckets():
    rng = np.random.default_rng(1)
    T = [np.unique(rng.integers(0, 60, 15)) for _ in range(500)]
    r = I.pcy(T, 60, 40, n_buckets=1_000_003, max_k=2)
    assert r.stats["frequent_buckets"] >= r.stats["frequent_pairs"] > 0
    pairs = np.array([k for k in r.itemsets if len(k) == 2])
    bits = I.bitmap_get(r.stats["_bitmap"], I.pair_bucket(pairs[:, 0], pairs[:, 1], 1_000_003))
    assert bits.all()                                   # no false negatives


def test_bitmap_roundtrip():
    bits = np.array([1, 0, 0, 1, 1, 0, 1, 0, 1, 1], dtype=bool)
    packed = np.packbits(bits)
    assert np.array_equal(I.bitmap_get(packed, np.arange(len(bits))).astype(bool), bits)


def test_generate_candidates_prunes_infrequent_subsets():
    pairs = [(0, 1), (0, 2), (1, 2), (0, 3)]               # (1, 3), (2, 3) missing
    assert I.generate_candidates(pairs, 3) == [(0, 1, 2)]


# --- algorithms ------------------------------------------------------------

@pytest.mark.parametrize("algo", ["apriori", "pcy"])
@pytest.mark.parametrize("min_count", [1, 2, 3])
def test_matches_brute_force_on_toy(algo, min_count):
    expected = brute_force(TOY, min_count, 3)
    r = (I.apriori(TOY, 4, min_count, 3) if algo == "apriori"
         else I.pcy(TOY, 4, min_count, n_buckets=11, max_k=3))
    assert r.itemsets == expected


@pytest.mark.parametrize("n_buckets", [1, 3, 1009])
def test_pcy_equals_apriori_on_random_data_any_bucket_count(n_buckets):
    """n_buckets=1 forces every pair into one bucket (all collide): no false negatives."""
    rng = np.random.default_rng(0)
    T = [np.unique(rng.integers(0, 30, rng.integers(0, 12))) for _ in range(300)]
    a = I.apriori(T, 30, 8, 3)
    p = I.pcy(T, 30, 8, n_buckets, 3)
    assert a.itemsets == p.itemsets == brute_force(T, 8, 3)


def test_support_exactly_at_threshold_is_frequent():
    T = tx({0, 1}, {0, 1}, {0})
    r = I.apriori(T, 2, 2, 2)
    assert r.itemsets[(0, 1)] == 2 and r.itemsets[(0,)] == 3


def test_empty_and_singleton_baskets_do_not_break_counting():
    T = tx(set(), {0}, {0, 1}, set(), {0, 1})
    for r in (I.apriori(T, 2, 2, 3), I.pcy(T, 2, 2, 5, 3)):
        assert r.itemsets == {(0,): 3, (1,): 2, (0, 1): 2}


def test_no_frequent_singletons_gives_nothing():
    T = tx({0, 1}, {2, 3})
    assert I.apriori(T, 4, 2).itemsets == {}
    assert I.pcy(T, 4, 2, 7).itemsets == {}


def test_no_transactions():
    assert I.apriori([], 3, 1).itemsets == {}
    assert I.pcy([], 3, 1, 5).itemsets == {}


def test_duplicate_items_are_collapsed_by_baskets_from_user_sets():
    class US:  # minimal stand-in for taste.UserSets
        version, user_ids, item_ids = "x", np.array(["a", "b", "c"]), np.array(["p", "q"])
        sets = [np.array([0, 1]), np.array([0]), np.array([0, 1])]
    b = I.baskets_from_user_sets(US, ["a", "b", "c", "d"], min_items=2)
    assert list(b.user_ids) == ["a", "c"]
    assert sorted(b.excluded_users) == ["b", "d"]


def test_pcy_stats_and_candidate_reduction():
    T = tx({0, 1}, {0, 1}, {0, 2}, {1, 2}, {0, 1, 2})
    r = I.pcy(T, 3, 3, n_buckets=101)
    s = I.pcy_candidate_stats(r)
    assert s["pairs_of_frequent_items"] == 3
    assert s["pcy_candidate_pairs"] + s["removed_by_bitmap"] == 3
    assert r.stats["pairs_hashed_pass1"] == 1 + 1 + 1 + 1 + 3
    assert r.itemsets.get((0, 1)) == 3


def test_invalid_parameters_raise():
    with pytest.raises(ValueError):
        I.apriori(TOY, 4, 0)
    with pytest.raises(ValueError):
        I.pcy(TOY, 4, 1, 0)
