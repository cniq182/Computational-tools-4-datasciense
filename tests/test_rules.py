"""Tests for src/rules.py: rule metrics, population metrics, tag share, comparison."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import itemsets as I  # noqa: E402
import rules as R  # noqa: E402


def tx(*baskets):
    return [np.array(sorted(b), dtype=np.int64) for b in baskets]


def by_rule(df):
    """{(antecedent, consequent): row as dict}."""
    return {(a, c): row for a, c, row in zip(df.antecedent, df.consequent, df.to_dict("records"))}


# 10 transactions: s(0)=6, s(1)=5, s(0,1)=4, s(2)=3
T = tx({0, 1}, {0, 1}, {0, 1}, {0, 1}, {0}, {0}, {1, 2}, {2}, {2}, set())


def test_rule_metrics_on_known_example():
    res = I.apriori(T, 3, min_count=2, max_k=2)
    rules, n_before = R.rules_from_itemsets(res.itemsets, len(T))
    r = by_rule(rules)
    assert n_before == 2                                    # 0->1 and 1->0
    assert r[(0,), (1,)]["support"] == pytest.approx(0.4)
    assert r[(0,), (1,)]["confidence"] == pytest.approx(4 / 6)
    assert r[(0,), (1,)]["lift"] == pytest.approx((4 / 6) / 0.5)
    assert r[(1,), (0,)]["confidence"] == pytest.approx(4 / 5)


def test_rules_from_triple_are_disjoint_nonempty_and_unique():
    its = {(0,): 5, (1,): 5, (2,): 5, (0, 1): 4, (0, 2): 4, (1, 2): 4, (0, 1, 2): 3}
    rules, n_before = R.rules_from_itemsets(its, 10)
    assert n_before == 3 * 2 + 6                            # 3 pairs x 2 + 1 triple x 6
    keys = list(zip(rules.antecedent, rules.consequent))
    assert len(keys) == len(set(keys))
    assert all(a and c and not set(a) & set(c) for a, c in keys)


def test_confidence_filter_and_count_before_filter():
    res = I.apriori(T, 3, 2, 2)
    rules, n_before = R.rules_from_itemsets(res.itemsets, len(T), min_confidence=0.7)
    assert n_before == 2 and len(rules) == 1 and rules.antecedent[0] == (1,)


def test_support_index_and_lift_ratio():
    pop = R.SupportIndex(T, 3)
    assert list(pop.support([(0,), (0, 1), (1, 2), (2,)])) == [6, 4, 1, 3]
    # community = first 6 transactions: s(0)=6, s(1)=4, s(0,1)=4 -> lift = (4/6)/(4/6) = 1
    res = I.apriori(T[:6], 3, 2, 2)
    rules, _ = R.rules_from_itemsets(res.itemsets, 6)
    row = by_rule(R.add_population_metrics(rules, pop))[(1,), (0,)]
    assert row["lift"] == pytest.approx(1.0)
    assert row["lift_pop"] == pytest.approx((4 / 5) / 0.6)
    assert row["lift_ratio"] == pytest.approx(1.0 / ((4 / 5) / 0.6))


def test_same_business_share():
    # businesses tagged with categories: 0=Sushi, 1=Japanese, 2=Pizza
    biz = R.SupportIndex(tx({0, 1}, {0, 1}, {0}, {1}, {2}), 3)
    rules = pd.DataFrame({"antecedent": [(0,), (1,), (0,)], "consequent": [(1,), (0,), (2,)]})
    share = R.add_same_business_share(rules, biz).same_business_share.tolist()
    assert share == pytest.approx([2 / 3, 2 / 3, 0.0])


def test_overlap_jaccard_and_unique():
    sets = {0: {"a", "b"}, 1: {"b", "c"}, 2: {"d"}}
    o = R.overlap(sets)
    assert o["n_rules_distinct"] == 4
    assert o["share_unique"] == pytest.approx(3 / 4)
    assert o["mean_pairwise_jaccard"] == pytest.approx((1 / 3 + 0 + 0) / 3)
    assert R.jaccard(set(), set()) == 0.0


def test_shuffled_labels_preserve_sizes_and_are_seeded():
    labels = pd.Series([0] * 5 + [1] * 3 + [2] * 2, index=list("abcdefghij"))
    a, b = R.shuffled_labels(labels, 1), R.shuffled_labels(labels, 1)
    assert a.equals(b)
    assert a.value_counts().sort_index().tolist() == [5, 3, 2]


def test_mine_partition_reports_small_communities():
    class US:
        version, item_ids = "x", np.array(["p", "q", "r"])
        user_ids = np.array([f"u{i}" for i in range(len(T))])
        sets = T
    B = I.baskets_from_user_sets(US, US.user_ids, min_items=1)
    labels = pd.Series([0] * 6 + [1] * 4, index=US.user_ids)
    p = R.MiningParams(min_support=0.3, min_confidence=0.0, min_lift=0.0, max_k=2,
                       min_count_floor=1, min_community_size=5, lift_ratio_min=1.2)
    summary, rules = R.mine_partition(B, labels, p, R.SupportIndex(list(B.sets), 3))
    s = summary.set_index("community_id")
    assert bool(s.loc[0, "eligible"]) and not bool(s.loc[1, "eligible"])
    assert s.loc[1, "reason"] == "too small"
    assert set(rules.community_id) == {0}


def test_pair_rules_and_partition_measures():
    rules = pd.DataFrame({
        "antecedent": [(0,), (0,), (0, 1), (2,)], "consequent": [(1,), (2,), (2,), (0,)],
        "lift_ratio": [0.5, 2.0, 3.0, 1.0], "specific": [False, True, True, False],
        "community_id": [0, 0, 0, 1]})
    assert len(R.pair_rules(rules)) == 3
    m = R.partition_measures(rules, reference={"ref": {((0,), (2,))}})
    assert m["pair_rules"] == 3 and m["rules_specific"] == 1
    assert m["median_lift_ratio"] == pytest.approx(1.0)
    assert m["share_lift_ratio_lt1"] == pytest.approx(1 / 3)
    assert m["jaccard_vs_ref"] == pytest.approx(1.0)
