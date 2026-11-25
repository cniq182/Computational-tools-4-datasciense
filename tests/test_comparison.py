"""Tests for src/comparison.py (step 5) on synthetic data.

The synthetic worlds have a known answer:
* "independent": friend partition, taste partition and period are unrelated;
* "confounded": friends and taste are both tied to the period, but not to each other;
* "signal": taste follows the friend partition *beyond* the period.
A correct permutation test must reject only in the "signal" world when it shuffles within period.
"""

import sys
import warnings
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

matplotlib.use("Agg")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import comparison as C  # noqa: E402
import taste  # noqa: E402


# --------------------------------------------------------------------------
# synthetic worlds
# --------------------------------------------------------------------------

def make_world(kind: str, n: int = 600, seed: int = 0, n_runs: int = 3):
    rng = np.random.default_rng(seed)
    period = rng.integers(0, 4, n)
    if kind == "independent":
        friend = rng.integers(0, 8, n)
        taste_lab = rng.integers(0, 4, n)
    else:
        # friend community: 2 sub-communities per period, 70 % determined by the period
        friend = np.where(rng.random(n) < 0.7, 2 * period + rng.integers(0, 2, n), rng.integers(0, 8, n))
        if kind == "confounded":
            taste_lab = np.where(rng.random(n) < 0.6, period, rng.integers(0, 4, n))
        else:                                    # signal: follows the friend sub-community, beyond the period
            taste_lab = np.where(rng.random(n) < 0.7, friend % 2 + 2 * (period % 2), rng.integers(0, 4, n))
    # friend "runs": noisy copies of the chosen partition
    def noisy(lab, k):
        r = np.random.default_rng(seed * 100 + k)
        return np.where(r.random(n) < 0.15, r.integers(0, lab.max() + 1, n), lab)
    return {"period": period, "friend": friend, "taste": taste_lab,
            "friend_runs": [noisy(friend, k) for k in range(n_runs)],
            "taste_runs": [noisy(taste_lab, k + 50) for k in range(n_runs)]}


def make_edges(group: np.ndarray, m: int, seed: int = 1):
    """m random edges between users of the same ``group``."""
    rng = np.random.default_rng(seed)
    n = len(group)
    by = {g: np.flatnonzero(group == g) for g in np.unique(group)}
    u = rng.integers(0, n, m)
    v = np.array([rng.choice(by[group[a]]) for a in u])
    keep = u != v
    return u[keep], v[keep]


def make_sets(group: np.ndarray, seed: int = 2, n_items: int = 80):
    """Sparse binary matrix: 5 items from the pool of the user's group + 3 random items."""
    rng = np.random.default_rng(seed)
    rows, cols = [], []
    for i, g in enumerate(group):
        items = set(10 * (g % 6) + rng.choice(10, 5, replace=False)) | set(rng.choice(n_items, 3, replace=False))
        rows += [i] * len(items)
        cols += sorted(items)
    return sp.csr_matrix((np.ones(len(rows), dtype=np.int32), (rows, cols)), shape=(len(group), n_items))


def to_inputs(w, edges=None, n_isolated: int = 3) -> C.Inputs:
    """Build the Inputs object through the same path as the real files (align_inputs)."""
    n = len(w["period"])
    users = pd.Index([f"u{i:04d}" for i in range(n)], name="user_id")
    rng = np.random.default_rng(5)
    iso = {v: np.isin(np.arange(n), rng.choice(n, n_isolated, replace=False)) for v in C.VERSIONS}
    labels = np.array(C.PERIOD_ORDER)[w["period"]]
    taste_df = pd.DataFrame({"louvain_category": w["taste"], "louvain_business": w["taste"],
                             "isolated_category": iso["category"], "isolated_business": iso["business"],
                             "period": labels}, index=users)
    fseeds = pd.DataFrame({f"seed_{k}": r for k, r in enumerate(w["friend_runs"])}, index=users)
    tseeds = {v: pd.DataFrame({f"seed_{k}": r for k, r in enumerate(w["taste_runs"])}, index=users) for v in C.VERSIONS}
    if edges is None:
        edges = make_edges(w["period"], 1500)
    e = pd.DataFrame({"u": users[edges[0]], "v": users[edges[1]]})
    return C.align_inputs(pd.Series(w["friend"], index=users, name="friend_community"), fseeds, taste_df, tseeds, e)


# --------------------------------------------------------------------------
# measures
# --------------------------------------------------------------------------

def test_nmi_ari_match_sklearn_on_random_partitions():
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    rng = np.random.default_rng(0)
    for _ in range(30):
        n = int(rng.integers(20, 400))
        f, t = rng.integers(0, int(rng.integers(1, 12)), n), rng.integers(0, int(rng.integers(1, 12)), n)
        # make them dependent half of the time
        if rng.random() < 0.5:
            t = np.where(rng.random(n) < 0.6, f, t)
        ours = C.pair_stats(C.factorize(f), C.factorize(t))
        ref = (normalized_mutual_info_score(f, t), adjusted_rand_score(f, t))
        assert np.allclose(ours, ref, atol=1e-10), (ours, ref)


def test_degenerate_partitions_match_sklearn():
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    one = np.zeros(50, dtype=int)
    single = np.arange(50)
    mixed = np.arange(50) % 5
    for a, b in [(one, one), (single, single), (one, mixed), (mixed, one), (single, mixed), (mixed, mixed)]:
        ours = C.pair_stats(C.factorize(a), C.factorize(b))
        ref = (normalized_mutual_info_score(a, b), adjusted_rand_score(a, b))
        assert np.allclose(ours, ref, atol=1e-10), (a[:3], b[:3], ours, ref)


def test_identical_partitions_give_one_and_label_names_do_not_matter():
    f = np.random.default_rng(1).integers(0, 6, 300)
    relabelled = (f * 7 + 3) % 11 + 100 * (f % 2)       # injective relabelling is not guaranteed -> use a map
    mapping = {c: 1000 + i for i, c in enumerate(np.unique(f))}
    relabelled = np.array([mapping[c] for c in f])
    nmi, ari = C.pair_stats(C.factorize(f), C.factorize(relabelled))
    assert nmi == pytest.approx(1.0) and ari == pytest.approx(1.0)
    assert C.verify_against_sklearn(f, relabelled)["nmi"] == pytest.approx(1.0)


def test_tables_with_zero_rows_and_columns():
    tab = np.array([[5, 0, 0], [0, 0, 0], [1, 4, 0]])
    ref = C.table_stats(tab[[0, 2]][:, [0, 1]])
    assert np.allclose(C.table_stats(tab), ref)


# --------------------------------------------------------------------------
# shuffling and p-values
# --------------------------------------------------------------------------

def test_shuffler_keeps_users_inside_their_stratum():
    strata = np.random.default_rng(0).integers(0, 4, 400)
    sh = C.Shuffler(strata, seed=3)
    for _ in range(5):
        idx = sh()
        assert sorted(idx) == list(range(400))               # a permutation
        assert (strata[idx] == strata).all()                 # nobody left its stratum
    assert (sh() != np.arange(400)).any()                    # and it does move users


def test_p_value_never_zero_and_counts_ties():
    null = np.array([0.1, 0.2, 0.3])
    assert C.p_upper(1.0, null) == pytest.approx(1 / 4)
    assert C.p_upper(0.2, null) == pytest.approx(3 / 4)       # ties count as >=


# --------------------------------------------------------------------------
# permutation tests: the control for the confounder is what matters
# --------------------------------------------------------------------------

def _p_values(kind, reps=25, n_perm=99):
    out = []
    for r in range(reps):
        w = make_world(kind, n=500, seed=r)
        f, t, per = C.factorize(w["friend"]), C.factorize(w["taste"]), C.factorize(w["period"])
        res, _ = C.compare_pair(f, t, per, n_perm=n_perm, seed=r, with_ami=False)
        out.append((res["within_period_shuffle"]["nmi"]["p"], res["plain_shuffle"]["nmi"]["p"],
                    res["within_period_shuffle"]["ari"]["p"]))
    return np.array(out)


def test_independent_world_is_not_rejected():
    p = _p_values("independent")
    assert (p[:, 0] < 0.05).mean() <= 0.2
    assert 0.3 < p[:, 0].mean() < 0.7                         # roughly uniform p-values


def test_confounded_world_naive_test_rejects_but_period_control_does_not():
    p = _p_values("confounded")
    assert (p[:, 1] <= 0.011).all()                           # plain shuffle: "friends share tastes" (wrongly)
    assert (p[:, 0] < 0.05).mean() <= 0.2                     # shuffle within period: no evidence
    assert (p[:, 2] < 0.05).mean() <= 0.2


def test_signal_world_is_detected_with_period_control():
    p = _p_values("signal", reps=8)
    assert (p[:, 0] <= 0.011).all()
    assert (p[:, 2] <= 0.011).all()


def test_compare_pair_summary_is_consistent():
    w = make_world("signal", n=400, seed=3)
    f, t, per = C.factorize(w["friend"]), C.factorize(w["taste"]), C.factorize(w["period"])
    res, nulls = C.compare_pair(f, t, per, n_perm=50, seed=1)
    assert nulls["within_period"].shape == (50, 4) and nulls["plain"].shape == (50, 2)
    assert res["observed"]["nmi"] == pytest.approx(C.pair_stats(f, t)[0])
    s = res["within_period_shuffle"]["nmi"]
    assert s["excess"] == pytest.approx(s["observed"] - s["null_mean"])
    assert 0 < res["observed"]["ami"] <= 1
    assert res["n_users"] == 400


def test_within_period_stats_are_weighted_means():
    w = make_world("signal", n=400, seed=4)
    f, t, per = C.factorize(w["friend"]), C.factorize(w["taste"]), C.factorize(w["period"])
    nf, nt = f.max() + 1, t.max() + 1
    nm, ar = C.within_stats(f, t, C.group_indices(per), nf, nt)
    exp = [(len(g), *C.pair_stats(C.factorize(f[g]), C.factorize(t[g]))) for g in C.group_indices(per)]
    assert nm == pytest.approx(sum(n * a for n, a, _ in exp) / 400)
    assert ar == pytest.approx(sum(n * b for n, _, b in exp) / 400)


# --------------------------------------------------------------------------
# grid
# --------------------------------------------------------------------------

def test_seed_grid_matches_direct_computation():
    w = make_world("signal", n=300, seed=5, n_runs=3)
    f = [C.factorize(x) for x in w["friend_runs"]]
    t = [C.factorize(x) for x in w["taste_runs"]]
    per = C.factorize(w["period"])
    g = C.seed_grid(f, t, per, n_perm=1, seed=9, progress=False)
    assert len(g) == 9
    idx = C.Shuffler(per, 9)()                                   # same permutation as inside seed_grid
    for _, r in g.iterrows():
        i, j = int(r.friend_seed), int(r.taste_seed)
        assert (r.nmi, r.ari) == pytest.approx(C.pair_stats(f[i], t[j]))
        assert (r.null_nmi_mean, r.null_ari_mean) == pytest.approx(C.pair_stats(f[i], t[j][idx]))
        assert 0 < r.p_nmi <= 1


def test_seed_grid_summary_and_ceiling():
    w = make_world("signal", n=300, seed=6, n_runs=4)
    f = [C.factorize(x) for x in w["friend_runs"]]
    t = [C.factorize(x) for x in w["taste_runs"]]
    g = C.seed_grid(f, t, C.factorize(w["period"]), n_perm=40, seed=1, progress=False)
    s = C.summarise_grid(g)
    assert s["n_cells"] == 16 and s["nmi"]["min"] <= s["nmi"]["mean"] <= s["nmi"]["max"]
    assert s["share_cells_p_below_0.05_nmi"] > 0.9               # strong planted signal
    c = C.seed_ceiling(f)
    assert c["n_pairs"] == 6 and 0 < c["nmi"]["mean"] < 1


# --------------------------------------------------------------------------
# alignment, views, variants
# --------------------------------------------------------------------------

def test_align_inputs_restricts_to_common_users_and_keeps_valid_edges():
    w = make_world("independent", n=100, seed=0)
    n = 100
    users = pd.Index([f"u{i:04d}" for i in range(n)], name="user_id")
    taste_df = pd.DataFrame({"louvain_category": w["taste"], "louvain_business": w["taste"],
                             "isolated_category": False, "isolated_business": False,
                             "period": np.array(C.PERIOD_ORDER)[w["period"]]}, index=users)
    fseeds = pd.DataFrame({"seed_10": w["friend_runs"][0], "seed_2": w["friend_runs"][1]}, index=users)
    tseeds = {v: pd.DataFrame({"seed_0": w["taste_runs"][0]}, index=users) for v in C.VERSIONS}
    friend = pd.Series(w["friend"], index=users)
    edges = pd.DataFrame({"u": ["u0003", "u0001", "u0001", "u0005", "ghost", "u0002"],
                          "v": ["u0001", "u0003", "u0002", "u0005", "u0001", "u0004"]})
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        inp = C.align_inputs(friend.iloc[:90], fseeds, taste_df, tseeds, edges)       # friend file misses 10 users
    assert inp.n == 90 and any("same users" in str(c.message) for c in caught)
    # duplicates (3-1 and 1-3), self loop (5-5) and the unknown user are removed; u<v
    assert sorted(zip(inp.edge_u.tolist(), inp.edge_v.tolist())) == [(1, 2), (1, 3), (2, 4)]
    # seed columns sorted numerically (seed_2 before seed_10)
    assert (inp.friend_seeds[:, 0] == w["friend_runs"][1][:90]).all()
    assert inp.period_labels[:4] == C.PERIOD_ORDER


def test_make_view_excludes_isolated_users_and_remaps_edges():
    inp = to_inputs(make_world("independent", n=300, seed=1), n_isolated=5)
    v = C.make_view(inp, "category")
    assert len(v.sel) == 295 and v.n_isolated_excluded == 5
    assert not inp.isolated["category"][v.sel].any()
    assert v.eu.max() < 295 and (v.eu < v.ev).all()
    keep = ~inp.isolated["category"]
    kept_edges = int((keep[inp.edge_u] & keep[inp.edge_v]).sum())
    assert len(v.eu) == kept_edges
    assert len(C.make_view(inp, "category", exclude_isolated=False).sel) == 300


def test_merge_small_and_variants():
    codes = np.array([0] * 30 + [1] * 25 + [2] * 3 + [3] * 2)
    merged = C.merge_small(codes, 10)
    assert len(np.unique(merged)) == 3 and merged[-1] == merged[-5] != merged[-6]
    inp = to_inputs(make_world("signal", n=400, seed=2), n_isolated=4)
    rows = C.sensitivity_variants(inp, "category", n_perm=20, seed=1, min_size=10)
    assert len(rows) == 4
    assert rows[0]["n_users"] == 396 and rows[1]["n_users"] == 400
    assert rows[3]["n_users"] <= rows[0]["n_users"]
    assert all(0 < r["nmi"]["p"] <= 1 for r in rows)


def test_period_baselines_see_the_confounder():
    inp = to_inputs(make_world("confounded", n=600, seed=3), n_isolated=3)
    b = C.period_baselines(C.make_view(inp, "category"), n_perm=40, seed=1)
    assert b["friend"]["chosen"]["nmi"]["observed"] > 5 * b["friend"]["chosen"]["nmi"]["null_mean"]
    assert b["taste"]["chosen"]["nmi"]["p"] <= 0.03


# --------------------------------------------------------------------------
# pair level
# --------------------------------------------------------------------------

def test_pair_jaccard_matches_python_sets():
    X = make_sets(np.random.default_rng(0).integers(0, 6, 120), seed=3)
    u, v = np.random.default_rng(1).integers(0, 120, 200), np.random.default_rng(2).integers(0, 120, 200)
    sets = [set(X.indices[X.indptr[i]:X.indptr[i + 1]]) for i in range(120)]
    ref = [len(sets[a] & sets[b]) / len(sets[a] | sets[b]) for a, b in zip(u, v)]
    assert np.allclose(C.pair_jaccard(X, u, v), ref)


def test_pair_jaccard_empty_sets_give_zero():
    X = sp.csr_matrix(np.array([[1, 1, 0], [0, 0, 0], [0, 0, 0], [1, 0, 0]], dtype=np.int32))
    assert list(C.pair_jaccard(X, np.array([1, 0, 0]), np.array([2, 1, 3]))) == [0.0, 0.0, 0.5]


def test_jaccard_test_rejects_when_friends_share_a_pool_beyond_period():
    n = 400
    rng = np.random.default_rng(0)
    period = rng.integers(0, 4, n)
    group = rng.integers(0, 6, n)                              # unrelated to period
    X = make_sets(group)
    eu, ev = make_edges(group, 800)
    res, _ = C.jaccard_pair_test(X, eu, ev, C.factorize(period), n_perm=60, seed=1, progress=False)
    mj = res["mean_jaccard"]
    assert mj["within_period"]["p"] <= 0.02 and mj["random_pairs"]["p"] <= 0.02
    assert mj["ratio_to_within_period"] > 1.1


def test_jaccard_test_period_control_removes_a_period_only_effect():
    n = 600
    rng = np.random.default_rng(1)
    period = rng.integers(0, 4, n)
    X = make_sets(period)                                      # sets depend on the period only
    eu, ev = make_edges(period, 1200)                          # friends are active in the same period
    res, _ = C.jaccard_pair_test(X, eu, ev, C.factorize(period), n_perm=80, seed=2, progress=False)
    mj = res["mean_jaccard"]
    assert mj["random_pairs"]["p"] <= 0.02                     # naive: "friends share tastes"
    assert mj["within_period"]["p"] > 0.05                     # controlled: no extra similarity


def test_edge_share_test_on_planted_world():
    inp = to_inputs(make_world("signal", n=500, seed=4), edges=None, n_isolated=2)
    # friends: same friend community
    eu, ev = make_edges(inp.friend_main, 900)
    inp.edge_u, inp.edge_v = np.minimum(eu, ev), np.maximum(eu, ev)
    r = C.edge_share_test(C.make_view(inp, "category"), n_perm=60, seed=1, progress=False)
    assert r["chosen_taste_partition"]["within_period"]["observed"] > r["chosen_taste_partition"]["within_period"]["null_mean"]
    assert r["chosen_taste_partition"]["within_period"]["p"] <= 0.02
    assert r["over_taste_runs"]["excess_within_period"]["mean"] > 0


def test_usersets_to_matrix_with_taste_usersets():
    users = np.array(["a", "b", "c", "d"])
    us = taste.UserSets(version="category", user_ids=np.array(["a", "c", "d"]),
                        item_ids=np.array(["x", "y", "z"]),
                        sets=[np.array([0, 2]), np.array([1]), np.array([0, 1, 2])],
                        empty_users=np.array(["b"]))
    X = C.usersets_to_matrix(us, users)
    assert X.shape == (4, 3)
    assert X.toarray().tolist() == [[1, 0, 1], [0, 0, 0], [0, 1, 0], [1, 1, 1]]


# --------------------------------------------------------------------------
# sanity check, orchestration, saving, figures
# --------------------------------------------------------------------------

def test_check_against_report_strict_and_lenient():
    inp = to_inputs(make_world("independent", n=200, seed=0))
    with pytest.raises(AssertionError, match="differ from the report"):
        C.check_against_report(inp, strict=True)
    with pytest.warns(UserWarning, match="differ from the report"):
        out = C.check_against_report(inp, strict=False)
    assert not out["ok"] and out["found"]["n_users"] == 200


def test_end_to_end_small_run_saves_files_and_figures(tmp_path):
    w = make_world("signal", n=400, seed=7, n_runs=3)
    inp = to_inputs(w, n_isolated=3)
    results = {"parameters": {}, "sanity": C.check_against_report(inp, strict=False), "versions": {}}
    tables = {}
    for v in C.VERSIONS:
        results["versions"][v], tables[v] = C.analyse_version(inp, v, n_perm_main=40, n_perm_grid=20,
                                                              n_perm_other=20, seed=1, progress=False)
    results["taste_vs_taste"] = C.taste_vs_taste(inp)
    mats = {v: make_sets(w["friend"]) for v in C.VERSIONS}
    results["jaccard"], pair_nulls = C.analyse_pairs(inp, mats, n_perm=20, seed=1, progress=False)
    assert results["versions"]["category"]["n_users_compared"] == 397
    assert results["taste_vs_taste"]["nmi"] == pytest.approx(1.0)       # same synthetic partition in both versions
    assert len(tables["category"]["grid"]) == 9

    for tab in (C.main_table(results), C.grid_table(results), C.sensitivity_table(results), C.pair_table(results)):
        assert len(tab) > 0 and tab.iloc[:, 0].isin(["category taste", "restaurant taste"]).all()
    assert len(C.main_table(results)) == 2 * 5 and len(C.sensitivity_table(results)) == 8
    assert len(C.pair_table(results)) == 2 * 3
    table_v = C.build_table_v(results)
    assert set(table_v.columns) == {"taste version", "comparison", "NMI", "ARI"} and len(table_v) == 14
    written = C.save_results(tmp_path, results, tables, pair_nulls)
    assert all(p.exists() for p in written)
    import json
    back = json.loads((tmp_path / "data" / "results" / "comparison_summary.json").read_text())
    assert back["versions"]["category"]["main"]["observed"]["nmi"] == pytest.approx(
        results["versions"]["category"]["main"]["observed"]["nmi"])

    for name, fn in (("a.png", lambda p: C.plot_main_vs_null(results, tables, p)),
                     ("b.png", lambda p: C.plot_agreement_levels(results, p)),
                     ("c.png", lambda p: C.plot_pair_tests(results, pair_nulls, p))):
        p = tmp_path / name
        fig = fn(p)
        assert p.exists() and p.stat().st_size > 5_000
        import matplotlib.pyplot as plt
        plt.close(fig)


def test_to_jsonable_handles_numpy():
    import json
    obj = {"a": np.float64(1.5), "b": np.arange(3), "c": [np.int64(2)], 3: (np.float32(0.5),)}
    assert json.loads(json.dumps(C.to_jsonable(obj))) == {"a": 1.5, "b": [0, 1, 2], "c": [2], "3": [0.5]}
