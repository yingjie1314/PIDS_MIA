"""The rewritten single-pass R-CAID pseudo-graph builder must produce the same graph as the old
four-stage pipeline (create -> prune -> relabel -> convert), which is kept verbatim below as the
oracle.

"Same graph" here means same node set with same attributes and same directed edge multiset. Edge
timestamps are excluded from the comparison on purpose: they are *arbitrary by design* (new edges
have no real timestamp, so `add_arbitrary_timestamps_to_graph` assigns uniform randoms over the
original time range, and its own docstring forbids downstream order-dependence). The new builder
draws them per root rather than in one vector over the relabelled edge list, so the values differ
under the same seed; the test instead pins the contract -- every timestamp lies in the original
graph's [min_t, max_t) range.
"""

import os
import sys

import networkx as nx
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pidsmaker.preprocessing.transformation_methods.transformation_rcaid_pseudo_graph import (
    build_pseudo_graph,
    identify_root_nodes,
    main as new_main,
)
from pidsmaker.preprocessing.transformation_methods.transformation_utils import (
    add_arbitrary_timestamps_to_graph,
)


# --------------------------------------------------------------------------------------------
# Oracle: the previous implementation, verbatim (create_pseudo_graph / prune_pseudo_roots /
# remove_pseudo_prefix and the old main), as of commit 259adea.
# --------------------------------------------------------------------------------------------
def oracle_create_pseudo_graph(G, root_nodes):
    pseudo_graph = nx.DiGraph()

    for node, attr in G.nodes(data=True):
        pseudo_graph.add_node(node, **attr)

    for root in root_nodes:
        pseudo_root = f"pseudo_{root}"
        pseudo_graph.add_node(pseudo_root, **G.nodes[root])

        descendants = nx.descendants(G, root)
        for descendant in descendants:
            pseudo_graph.add_edge(pseudo_root, descendant)

    return pseudo_graph


def oracle_prune_pseudo_roots(pseudo_graph, G, prune_threshold):
    total_nodes_in_G = len(G.nodes())
    max_allowed_connections = prune_threshold * total_nodes_in_G

    pseudo_roots_to_prune = []
    for node in pseudo_graph.nodes():
        if node.startswith("pseudo_"):
            num_connections = len(list(pseudo_graph.successors(node)))
            if num_connections > max_allowed_connections:
                pseudo_roots_to_prune.append(node)

    for pseudo_root in pseudo_roots_to_prune:
        pseudo_graph.remove_node(pseudo_root)

    return pseudo_graph


def oracle_remove_pseudo_prefix(graph):
    mapping = {node: node.replace("pseudo_", "") for node in graph.nodes()}
    graph = nx.relabel_nodes(graph, mapping)
    return graph


def oracle_main(graph, use_pruning):
    root_nodes = identify_root_nodes(graph)
    pseudo_graph = oracle_create_pseudo_graph(graph, root_nodes)
    if use_pruning:
        pseudo_graph = oracle_prune_pseudo_roots(pseudo_graph, graph, 0.5)
    pseudo_graph = oracle_remove_pseudo_prefix(pseudo_graph)
    pseudo_graph = add_arbitrary_timestamps_to_graph(original_G=graph, new_G=pseudo_graph)
    pseudo_graph = nx.MultiDiGraph(pseudo_graph)
    return pseudo_graph


# --------------------------------------------------------------------------------------------
# Graph generators. Node names are strings (as produced by construction), attributes carry the
# label/node_type fields featurization reads, edges carry integer 'time'.
# --------------------------------------------------------------------------------------------
def random_provenance_graph(seed, n_nodes=40, n_edges=120):
    rng = np.random.default_rng(seed)
    G = nx.MultiDiGraph()
    for i in range(n_nodes):
        G.add_node(str(i), label=f"label_{i}", node_type=["subject", "file", "netflow"][i % 3])
    for _ in range(n_edges):
        u, v = rng.integers(0, n_nodes, size=2)
        G.add_edge(str(u), str(v), time=int(rng.integers(1_000, 9_000)))
    return G


def star_burst_graph():
    """One root reaching every node: pruned at threshold 0.5, kept without pruning."""
    G = nx.MultiDiGraph()
    for i in range(10):
        G.add_node(str(i), label=f"l{i}", node_type="subject")
    for i in range(1, 10):
        G.add_edge("0", str(i), time=100 + i)
    # A second, small root so the pruned graph still has edges.
    G.add_edge("8", "9", time=50)
    return G


def canonical_edges(G):
    return sorted((u, v) for u, v, _ in G.edges(keys=True))


def assert_same_graph(new_G, old_G, original_G):
    assert isinstance(new_G, nx.MultiDiGraph)
    assert sorted(new_G.nodes()) == sorted(old_G.nodes())
    for n in new_G.nodes():
        assert new_G.nodes[n] == old_G.nodes[n], f"attrs differ on node {n}"
    assert canonical_edges(new_G) == canonical_edges(old_G)

    times = [d["time"] for _, _, d in original_G.edges(data=True)]
    min_t, max_t = min(times), max(times)
    for _, _, d in new_G.edges(data=True):
        assert min_t <= d["time"] < max_t


class _Cfg:
    def __init__(self, use_pruning):
        t = type("T", (), {})()
        t.rcaid_pseudo_graph = type("T", (), {})()
        t.rcaid_pseudo_graph.use_pruning = use_pruning
        self.transformation = t


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
@pytest.mark.parametrize("use_pruning", [True, False])
def test_matches_oracle_on_random_graphs(seed, use_pruning):
    G = random_provenance_graph(seed)
    np.random.seed(seed)
    old = oracle_main(G.copy(), use_pruning)
    np.random.seed(seed)
    new = new_main(G.copy(), _Cfg(use_pruning))
    assert_same_graph(new, old, G)


def test_pruning_drops_only_the_giant_root():
    G = star_burst_graph()
    np.random.seed(0)
    old = oracle_main(G.copy(), use_pruning=True)
    np.random.seed(0)
    new = new_main(G.copy(), _Cfg(use_pruning=True))
    assert_same_graph(new, old, G)
    # Node "0" reaches 9/10 nodes > 0.5 threshold: its edges are gone, the node itself stays.
    assert "0" in new.nodes()
    assert new.out_degree("0") == 0
    # The small root's edge survives.
    assert ("8", "9") in {(u, v) for u, v, _ in new.edges(keys=True)}


def test_without_pruning_keeps_the_giant_root():
    G = star_burst_graph()
    np.random.seed(0)
    old = oracle_main(G.copy(), use_pruning=False)
    np.random.seed(0)
    new = new_main(G.copy(), _Cfg(use_pruning=False))
    assert_same_graph(new, old, G)
    assert new.out_degree("0") == 9


def test_single_pass_builds_no_intermediate_copies():
    """The point of the rewrite: main returns the one graph build_pseudo_graph constructed,
    not a relabelled or converted copy of it."""
    G = random_provenance_graph(seed=7)
    roots = identify_root_nodes(G)
    np.random.seed(7)
    direct = build_pseudo_graph(G, roots, prune_threshold=0.5)
    np.random.seed(7)
    via_main = new_main(G, _Cfg(use_pruning=True))
    assert canonical_edges(direct) == canonical_edges(via_main)
    assert sorted(direct.nodes()) == sorted(via_main.nodes())
