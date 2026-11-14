"""Taste communities — step 3 (owner: Cesi).

Step 5.1: build, for every user, the set of items they like, in two versions:

* ``"business"``: set of liked restaurants (``business_id``);
* ``"category"``: set of categories of the liked restaurants, without the
  overly generic ones (``Food``, ``Nightlife``, ...).

"Liked" means ``is_positive`` (``stars >= 3.5``), the definition produced by
step 1, so that every step of the project uses the same notion of "liked".

All inputs come from step 1 (``data/processed/philadelphia_pa_*.parquet``);
nothing is recomputed from the raw Yelp files.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Paths and loading
# --------------------------------------------------------------------------

CITY_PREFIX = "philadelphia_pa"


def find_project_root(start: Path | None = None) -> Path:
    """Return the repository root: the first parent folder containing ``data/processed``."""
    p = Path(start or Path.cwd()).resolve()
    for candidate in (p, *p.parents):
        if (candidate / "data" / "processed").exists():
            return candidate
    raise FileNotFoundError("data/processed not found: run notebook 01 first.")


def load_inputs(root: Path, prefix: str = CITY_PREFIX):
    """Load the cleaned reviews, businesses and users written by step 1."""
    d = root / "data" / "processed"
    reviews = pd.read_parquet(
        d / f"{prefix}_reviews.parquet",
        columns=["user_id", "business_id", "stars", "stars_norm", "is_positive", "date"],
    )
    businesses = pd.read_parquet(d / f"{prefix}_businesses.parquet",
                                 columns=["business_id", "name", "categories_taste"])
    users = pd.read_parquet(d / f"{prefix}_users.parquet", columns=["user_id"])
    return reviews, businesses, users


# --------------------------------------------------------------------------
# User sets
# --------------------------------------------------------------------------

@dataclass
class UserSets:
    """Users' item sets, with ids mapped to contiguous integers.

    ``sets[i]`` is a sorted ``np.ndarray`` of item indices for user ``user_ids[i]``.
    Users with an empty set are **not** in ``sets``; they are listed in
    ``empty_users`` so that later steps can mark them as isolated.
    """
    version: str
    user_ids: np.ndarray            # original Yelp user ids, position = user index
    item_ids: np.ndarray            # original item ids (business_id or category), position = item index
    sets: list[np.ndarray]
    empty_users: np.ndarray         # users of users.parquet with no liked item after filtering
    dropped_items: dict[str, float] = field(default_factory=dict)  # generic categories -> share of users

    @property
    def sizes(self) -> np.ndarray:
        return np.array([len(s) for s in self.sets])

    def summary(self) -> dict:
        s = self.sizes
        return {
            "version": self.version,
            "users_kept": len(self.sets),
            "users_empty": len(self.empty_users),
            "n_items": len(self.item_ids),
            "set_size_mean": round(float(s.mean()), 2),
            "set_size_median": float(np.median(s)),
            "set_size_min": int(s.min()),
            "set_size_max": int(s.max()),
            "generic_dropped": list(self.dropped_items),
        }


def liked_pairs(reviews: pd.DataFrame, positive_col: str = "is_positive") -> pd.DataFrame:
    """(user_id, business_id) pairs the user liked. ``positive_col`` can be changed
    for the robustness check (e.g. a column ``stars_norm > 0``)."""
    return reviews.loc[reviews[positive_col], ["user_id", "business_id"]].drop_duplicates()


def generic_categories(user_cats: pd.DataFrame, n_users: int, max_frac: float) -> dict[str, float]:
    """Categories liked by more than ``max_frac`` of the users, with their share.

    A category almost everybody likes ("Food", "Nightlife") makes everybody look
    similar and carries no information about taste, so it is removed.
    """
    share = user_cats.groupby("category")["user_id"].nunique() / n_users
    gen = share[share > max_frac].sort_values(ascending=False)
    return {str(k): round(float(v), 3) for k, v in gen.items()}


def _to_sets(pairs: pd.DataFrame, item_col: str, all_users: pd.Series, version: str,
             dropped: dict[str, float]) -> UserSets:
    """Map ids to contiguous integers and group items per user."""
    items = np.sort(pairs[item_col].unique())
    item_index = pd.Index(items)
    pairs = (pairs.assign(item=item_index.get_indexer(pairs[item_col]).astype(np.int64))
                  [["user_id", "item"]].drop_duplicates()
                  .sort_values(["user_id", "item"]))
    user_ids, starts = np.unique(pairs["user_id"].to_numpy(), return_index=True)
    sets = np.split(pairs["item"].to_numpy(), starts[1:])
    empty = np.setdiff1d(all_users.to_numpy(), user_ids)
    return UserSets(version=version, user_ids=user_ids, item_ids=items,
                    sets=sets, empty_users=empty, dropped_items=dropped)


def clean_business_categories(businesses: pd.DataFrame, max_cats_per_business: int | None = 10,
                               min_businesses_per_category: int = 5):
    """Category of each restaurant, for the category version, without two artefacts:

    * **multi-purpose venues** (markets, food halls) tagged with more than
      ``max_cats_per_business`` categories: liking one of them would add dozens of
      "tastes" at once, so they are left out of the category version (they stay in
      the restaurant version);
    * **rare categories** present in fewer than ``min_businesses_per_category``
      restaurants: they describe one place, not a kind of food.

    Returns (long table business_id-category, info dict).
    """
    cats = businesses[["business_id", "categories_taste"]].copy()
    n_cats = cats["categories_taste"].apply(len)
    venues = businesses.loc[n_cats > max_cats_per_business] if max_cats_per_business else businesses.iloc[:0]
    if max_cats_per_business:
        cats = cats[n_cats <= max_cats_per_business]
    long = cats.explode("categories_taste").rename(columns={"categories_taste": "category"}).dropna()
    per_cat = long.groupby("category")["business_id"].nunique()
    rare = per_cat[per_cat < min_businesses_per_category]
    long = long[~long["category"].isin(rare.index)]
    info = {"venues_dropped": {str(r.get("name", r["business_id"])): int(len(r["categories_taste"]))
                               for _, r in venues.iterrows()},
            "rare_categories_dropped": len(rare),
            "rare_categories_examples": sorted(rare.index.astype(str))[:30]}
    return long, info


def build_user_sets(reviews: pd.DataFrame, businesses: pd.DataFrame, users: pd.DataFrame,
                    version: str, positive_col: str = "is_positive",
                    generic_max_frac: float = 0.30, max_cats_per_business: int | None = 10,
                    min_businesses_per_category: int = 5) -> UserSets:
    """Build the user sets for ``version`` in {"business", "category"}.

    Only users of ``users.parquet`` are considered (the same 7,947 users as the
    friend partition). The category filters (venues with too many categories,
    rare categories, generic categories) are used only by the category version.
    """
    all_users = users["user_id"].drop_duplicates()
    pairs = liked_pairs(reviews, positive_col)
    pairs = pairs[pairs["user_id"].isin(all_users)]

    if version == "business":
        return _to_sets(pairs, "business_id", all_users, version, dropped={})

    if version == "category":
        cats, _ = clean_business_categories(businesses, max_cats_per_business, min_businesses_per_category)
        user_cats = pairs.merge(cats, on="business_id")[["user_id", "category"]].drop_duplicates()
        dropped = generic_categories(user_cats, len(all_users), generic_max_frac)
        user_cats = user_cats[~user_cats["category"].isin(dropped)]
        return _to_sets(user_cats, "category", all_users, version, dropped)

    raise ValueError(f"unknown version {version!r}: use 'business' or 'category'")


def generic_sweep(reviews, businesses, users, fracs=(0.2, 0.3, 0.4, 0.5),
                  positive_col: str = "is_positive", max_cats_per_business: int | None = 10,
                  min_businesses_per_category: int = 5) -> pd.DataFrame:
    """How the generic-category threshold changes the category sets."""
    rows = []
    for f in fracs:
        s = build_user_sets(reviews, businesses, users, "category", positive_col, f,
                            max_cats_per_business, min_businesses_per_category)
        rows.append({"max_frac": f, "n_dropped": len(s.dropped_items),
                     "dropped": ", ".join(s.dropped_items),
                     "users_empty": len(s.empty_users),
                     "set_size_mean": round(float(s.sizes.mean()), 2)})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Period of activity (confounder found in step 2), saved for step 5
# --------------------------------------------------------------------------

PERIOD_BINS = [-np.inf, 2010, 2013, 2016, np.inf]
PERIOD_LABELS = ["<=2010", "2011-13", "2014-16", ">=2017"]


def user_activity_period(reviews: pd.DataFrame) -> pd.DataFrame:
    """Median review year per user and the same period bins as notebook 02."""
    year = reviews.groupby("user_id")["date"].apply(lambda d: d.dt.year.median())
    period = pd.cut(year, PERIOD_BINS, labels=PERIOD_LABELS)
    return pd.DataFrame({"median_year": year, "period": period})


def item_popularity(s: UserSets, top: int = 15) -> pd.DataFrame:
    """Most frequent items (share of users), useful to inspect the sets."""
    c = Counter(int(i) for st in s.sets for i in st)
    rows = [(s.item_ids[i], n, n / len(s.sets)) for i, n in c.most_common(top)]
    return pd.DataFrame(rows, columns=["item", "n_users", "share_users"])


# --------------------------------------------------------------------------
# 5.3 Taste-similarity graph and communities
# --------------------------------------------------------------------------

def build_taste_graph(all_user_ids, edges: pd.DataFrame):
    """Undirected graph: one node per user (all users, also those without similar
    pairs, which stay isolated), one edge per similar pair, weight = Jaccard."""
    import networkx as nx
    G = nx.Graph()
    G.add_nodes_from(all_user_ids)
    G.add_weighted_edges_from(edges[["u", "v", "jaccard"]].itertuples(index=False, name=None))
    return G


def relabel_by_size(part: dict) -> dict:
    """Renumber communities by size: 0 = largest (ties broken by smallest member id)."""
    groups: dict = {}
    for u, c in part.items():
        groups.setdefault(c, []).append(u)
    order = sorted(groups.values(), key=lambda m: (-len(m), min(m)))
    return {u: new for new, members in enumerate(order) for u in members}


def louvain_seeds(G, n_seeds: int = 20, gamma: float = 1.0, weight="weight"):
    """Run Mattia's Louvain with ``n_seeds`` seeds, fix disconnected communities,
    and return a table (one row per seed) plus the partitions {seed: {user: community}}."""
    import louvain as L
    rows, parts = [], {}
    for s in range(n_seeds):
        raw, _ = L.louvain(G, gamma=gamma, seed=s, weight=weight)
        fixed, n_split = L.split_disconnected(G, raw)
        part = relabel_by_size(fixed)
        sizes = pd.Series(part).value_counts()
        rows.append({"seed": s, "modularity": L.modularity(G, part, gamma, weight),
                     "n_communities": len(sizes),
                     "n_disconnected_fixed": n_split,
                     "n_communities_ge_50": int((sizes >= 50).sum()),
                     "share_users_in_5_largest": float(sizes.iloc[:5].sum() / len(part))})
        parts[s] = part
    return pd.DataFrame(rows), parts


def pairwise_stability(parts: dict, nodes) -> pd.Series:
    """ARI and NMI between every pair of runs (how stable the partition is across seeds)."""
    from itertools import combinations
    from sklearn.metrics import adjusted_rand_score as ari, normalized_mutual_info_score as nmi
    labels = {s: [p[u] for u in nodes] for s, p in parts.items()}
    a = [ari(labels[i], labels[j]) for i, j in combinations(labels, 2)]
    n = [nmi(labels[i], labels[j]) for i, j in combinations(labels, 2)]
    return pd.Series({"ARI_mean": np.mean(a), "ARI_min": np.min(a),
                      "NMI_mean": np.mean(n), "NMI_min": np.min(n)})


def community_profile(cat_sets: UserSets, part: dict, min_size: int = 50,
                      top: int = 5, min_share: float = 0.25) -> pd.DataFrame:
    """Describe each community by its most characteristic categories.

    For each community with >= ``min_size`` members: the categories liked by at
    least ``min_share`` of its members, ranked by **lift** = share in the community /
    share among all users (lift 2 = twice as common as in the whole population).
    """
    users = pd.Series(range(len(cat_sets.user_ids)), index=cat_sets.user_ids)
    n_all = len(users)
    overall = np.zeros(len(cat_sets.item_ids))
    for s in cat_sets.sets:
        overall[s] += 1
    overall /= n_all
    comm = pd.Series(part).reindex(cat_sets.user_ids)
    rows = []
    for c, members in comm.groupby(comm).groups.items():
        if len(members) < min_size:
            continue
        cnt = np.zeros(len(cat_sets.item_ids))
        for u in members:
            cnt[cat_sets.sets[users[u]]] += 1
        share = cnt / len(members)
        lift = np.where(overall > 0, share / np.where(overall > 0, overall, 1), 0)
        ok = np.where(share >= min_share)[0]
        best = ok[np.argsort(-lift[ok])][:top]
        rows.append({"community": int(c), "size": len(members),
                     "top categories (share in community, lift)": "; ".join(
                         f"{cat_sets.item_ids[i]} ({share[i]:.0%}, ×{lift[i]:.1f})" for i in best)})
    return pd.DataFrame(rows).sort_values("community").reset_index(drop=True)
