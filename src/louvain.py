"""Louvain community detection, implemented from scratch.

Written by Mattia for step 2 (notebook 02, section 2b) and moved here so that
step 3 (taste communities) uses exactly the same algorithm.

Only change: an optional ``weight`` argument. With ``weight=None`` (default)
every edge counts 1 and the results are identical to Mattia's original code;
with ``weight="weight"`` the edge attribute is used (taste graph: weight = Jaccard).
"""

import random
from collections import Counter, defaultdict

import networkx as nx


def _w(d, weight):
    return d.get(weight, 1.0) if weight else 1.0


def modularity(G, labels, gamma=1.0, weight=None):
    # Modularity of a partition {node: community}, straight from the definition (independent of Louvain).
    m2 = 2 * G.size(weight=weight)
    inside, total = Counter(), Counter()
    for u, v, d in G.edges(data=True):
        if labels[u] == labels[v]:
            inside[labels[u]] += 2 * _w(d, weight)
    for u, deg in G.degree(weight=weight):
        total[labels[u]] += deg
    return sum(inside[c] / m2 - gamma * (total[c] / m2) ** 2 for c in total)


def to_adjacency(G, index, weight=None):
    adj = [dict() for _ in range(len(index))]
    for u, v, d in G.edges(data=True):
        if u != v:
            w = _w(d, weight)
            adj[index[u]][index[v]] = w
            adj[index[v]][index[u]] = w
    return adj


def local_moving(adj, gamma, rng):
    # Phase 1: move single nodes to the neighbouring community with the best gain until no node moves.
    n = len(adj)
    k = [sum(a.values()) for a in adj]          # degree of each (super-)node, self-loop included
    m2 = sum(k)                                  # 2m
    comm = list(range(n))                        # every node starts alone
    tot = k[:]                                   # sum of degrees of each community
    order = list(range(n))
    moved_any, improved = False, True
    while improved:
        improved = False
        rng.shuffle(order)
        for i in order:
            ci = comm[i]
            w = defaultdict(float)               # weight from i to each neighbouring community
            for j, wij in adj[i].items():
                if j != i:
                    w[comm[j]] += wij
            tot[ci] -= k[i]                      # take i out of its community
            best, best_gain = ci, w.get(ci, 0.0) - gamma * tot[ci] * k[i] / m2
            for c, wc in w.items():
                gain = wc - gamma * tot[c] * k[i] / m2
                if gain > best_gain + 1e-12:     # strict improvement, so ties keep the node where it is
                    best, best_gain = c, gain
            tot[best] += k[i]
            if best != ci:
                comm[i] = best
                improved = moved_any = True
    return comm, moved_any


def aggregate(adj, comm):
    # Phase 2: one super-node per community; parallel links are summed, internal links become a self-loop.
    ids = {c: x for x, c in enumerate(sorted(set(comm)))}
    c2 = [ids[c] for c in comm]
    new = [defaultdict(float) for _ in ids]
    for i, a in enumerate(adj):
        for j, w in a.items():
            new[c2[i]][c2[j]] += w
    return [dict(d) for d in new], c2


def louvain(G, gamma=1.0, seed=0, weight=None):
    # Returns (partition {node: community}, list of partitions after each aggregation level).
    rng = random.Random(seed)
    nodes = list(G.nodes())
    index = {u: i for i, u in enumerate(nodes)}
    adj = to_adjacency(G, index, weight)
    membership = list(range(len(nodes)))         # original node -> current super-node
    levels = []
    while True:
        comm, moved = local_moving(adj, gamma, rng)
        if not moved:
            break
        adj, c2 = aggregate(adj, comm)
        membership = [c2[m] for m in membership]
        levels.append({u: membership[index[u]] for u in nodes})
    if not levels:                               # nothing moved: every node is its own community
        levels.append({u: index[u] for u in nodes})
    return levels[-1], levels


def n_communities(part):
    return len(set(part.values()))


def split_disconnected(G, part):
    # Replace every internally disconnected community by its connected components.
    # Returns (new partition, number of communities that had to be split). Never lowers Q.
    groups = defaultdict(list)
    for u, c in part.items():
        groups[c].append(u)
    new, n_split, next_id = {}, 0, 0
    for c in sorted(groups):
        comps = list(nx.connected_components(G.subgraph(groups[c])))
        n_split += len(comps) > 1
        for comp in sorted(comps, key=lambda s: (-len(s), min(s))):   # deterministic order
            for u in comp:
                new[u] = next_id
            next_id += 1
    return new, n_split


def louvain_connected(G, gamma=1.0, seed=0, weight=None):
    part, _ = louvain(G, gamma=gamma, seed=seed, weight=weight)
    return split_disconnected(G, part)[0]
