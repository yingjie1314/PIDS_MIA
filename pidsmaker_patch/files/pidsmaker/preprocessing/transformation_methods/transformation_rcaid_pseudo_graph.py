import networkx as nx
import numpy as np


def identify_root_nodes(G):
    root_nodes = set()

    for node in G.nodes():
        out_edges = list(G.out_edges(node, data=True))
        in_edges = list(G.in_edges(node, data=True))

        # out edge
        earliest_out_time = min(edge[2]["time"] for edge in out_edges) if out_edges else None
        # in
        earliest_in_time = min(edge[2]["time"] for edge in in_edges) if in_edges else None

        # if root
        if earliest_out_time is not None and (
            earliest_in_time is None or earliest_out_time < earliest_in_time
        ):
            root_nodes.add(node)

    return root_nodes


def build_pseudo_graph(G, root_nodes, prune_threshold=None):
    """Build the R-CAID pseudo-graph in a single pass.

    The result is the same graph the previous four-stage pipeline produced
    (create -> prune -> relabel -> convert): all original nodes with their attributes, and one
    root->descendant edge per (root, descendant) pair, each carrying an arbitrary timestamp drawn
    uniformly from the original graph's time range. Roots reaching more than
    `prune_threshold * |V|` descendants contribute no edges (the old code removed their pseudo-node,
    which deleted exactly those edges and nothing else, since original edges were never copied in).

    It is built in one pass because the staged version held up to two full copies of the result
    alive at once (`relabel_nodes` and the `nx.MultiDiGraph(...)` conversion each build a new graph
    before the old one is released). On dense THEIA windows the pseudo-graph runs to 10^8 edges and
    a single copy is tens of GB of NetworkX dicts, so the second transient copy was the difference
    between finishing and rc=-9 -- THEIA_E3_SHADOW's transformation was OOM-killed twice running
    alone with 239 GB free.

    Timestamps are arbitrary by design (see `add_arbitrary_timestamps_to_graph`): new edges have no
    real timestamp, so random ones in the original range are assigned and downstream models must not
    rely on edge order. The draw here happens per root instead of in one vector over the final edge
    list, so the *values* differ from the old implementation under the same seed -- the
    distribution, dtype and range are identical, which is the only contract.
    """
    timestamps = [edge[2]["time"] for edge in G.edges(data=True)]
    min_t, max_t = min(timestamps), max(timestamps)

    pseudo_graph = nx.MultiDiGraph()
    for node, attr in G.nodes(data=True):
        pseudo_graph.add_node(node, **attr)

    max_allowed = (
        prune_threshold * G.number_of_nodes() if prune_threshold is not None else float("inf")
    )

    for root in root_nodes:
        descendants = nx.descendants(G, root)
        if len(descendants) > max_allowed:
            continue
        rand_t = np.random.randint(min_t, max_t, size=len(descendants))
        for t, descendant in zip(rand_t, descendants):
            pseudo_graph.add_edge(root, descendant, time=t)

    return pseudo_graph


def main(graph: nx.Graph, cfg) -> nx.Graph:
    root_nodes = identify_root_nodes(graph)
    use_pruning = cfg.transformation.rcaid_pseudo_graph.use_pruning
    prune_threshold = 0.5 if use_pruning else None
    return build_pseudo_graph(graph, root_nodes, prune_threshold=prune_threshold)
