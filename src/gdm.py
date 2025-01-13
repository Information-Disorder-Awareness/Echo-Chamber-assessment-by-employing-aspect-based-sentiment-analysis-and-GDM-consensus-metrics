import os
import scipy
import networkx as nx
import numpy as np
from tqdm import tqdm
from itertools import combinations


def compute_consensus(m1, adj=None):

    # init similarity vector
    sv = np.zeros(m1.shape[1], dtype=np.float64)
    total = 0
    skipped = 0

    # iter on combination of user pairs
    user_pairs = combinations(list(range(0, len(m1))), 2)

    couples = len(m1)*(len(m1)-1)/2

    for u1, u2 in tqdm(user_pairs, total=couples):
        if adj is not None and adj[u1, u2] == 0:
            skipped += 1
            continue

        try:
            v1 = m1[u1]
            v2 = m1[u2]
            v1 = (v1 + 1) / 2
            v2 = (v2 + 1) / 2
            diff = np.abs(v1) - np.abs(v2)

            # sv = sv + np.power(diff, 2)
            sv = sv + abs(diff)
        except KeyError as e:
            skipped += 1

        total += 1

    print(f"Skipped {skipped} user pairs")

    # cs = np.sqrt(sv) / np.sqrt(total)
    cs = sv / total

    # cc = np.linalg.norm(cs, ord=2) / np.sqrt(len(cs))
    cc = np.linalg.norm(cs, ord=1)/len(cs)

    return cs, cc


def compute_metrics(df, communities, graph=None):

    results = {}
    adj = None

    if graph:
        # G: nx.DiGraph = nx.read_gml(os.path.join(
        # path, f'Graph/Final_DiGraph_{name.capitalize()}.gml'))
        adj: scipy.sparse.csr_matrix = nx.adjacency_matrix(
            graph.subgraph(communities[0]))

    m1 = df.filter(items=communities[0], axis=0).to_numpy()
    results["community0"] = compute_consensus(m1, adj)

    if graph:
        adj: scipy.sparse.csr_matrix = nx.adjacency_matrix(
            graph.subgraph(communities[1]))

    m2 = df.filter(items=communities[1], axis=0).to_numpy()
    results["community1"] = compute_consensus(m2, adj)

    m1_avg = np.mean(m1, axis=0)
    m2_avg = np.mean(m2, axis=0)

    gpm = np.vstack((m1_avg, m2_avg))
    results["inter_community"] = compute_consensus(gpm, adj)

    return results
