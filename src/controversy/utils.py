import logging
import os

import pandas as pd

from lib.echochamber.preprocessing.ops_on_raw_data import check_directory_absence
import ast
from lib.echochamber.preprocessing.utilities import add_edge, manage_and_save
from tqdm import tqdm
import networkx as nx

import pickle
from lib.echochamber.preprocessing.ops_sentiment_vader import add_sent_weight, \
    add_sent_weight_without_scaling

from lib.echochamber.community.com_utilities import community_detection, note_difference

from lib.echochamber.controversy_detection.GMCK import start_GMCK
from lib.echochamber.controversy_detection.change_side_controversy import change_side_controversy
from lib.echochamber.controversy_detection.random_walks import random_walks, random_walks_centrality

from lib.echochamber.preprocessing.ops_build_graph import add_consensus_weight


def build_vaccination_graph(path):
    df = pd.read_csv(path + '/processed/' + 'final_vax_data.csv', lineterminator='\n')

    G_dg = nx.DiGraph()
    G_g = nx.Graph()

    for _, row in tqdm(df.iterrows(), desc="Rows processed"):
        mentions = ast.literal_eval(row[3])
        if 'self' in mentions:
            G_dg = add_edge(G_dg, row[2], row[7], row[6], row[5], row[4], row[1], row[1])
        else:
            for mention in mentions:
                G_dg = add_edge(G_dg, row[2], row[7], row[6], row[5], row[4], row[1], mention)
                G_g = add_edge(G_g, None, None, None, None, None, row[1], mention)

    G_dg.name = 'Starter vax Direct Graph'
    G_g.name = 'Starter vax Graph'

    graphs = [G_dg, G_g]
    manage_and_save(graphs, path)


def ops_on_vac(path):
    os.chdir(os.path.join(path, 'raw'))

    df = pd.read_csv('./twitter_dataset.csv', usecols=['date', 'username', 'replies_count', 'retweets_count',
                                                       'likes_count', 'hashtags', 'mentions', 'tweet'])
    os.chdir(os.path.join(path, 'processed'))
    df['mentions'].replace('[]', "['self']", inplace=True)
    df['hashtags'].replace('[]', "['none']", inplace=True)
    df.to_csv('final_vax_data.csv', encoding='utf-8', index=False)

    print('VACCINATION DATA FINAL SHAPE')
    print(df.shape)

    os.chdir(path)


def ops_on_rot(path):
    os.chdir(os.path.join(path, 'raw'))
    df = pd.read_csv('./twitter_dataset.csv',
                     usecols=['date', 'username', 'replies_count', 'retweets_count', 'likes_count', 'hashtags',
                              'mentions', 'tweet'])
    os.chdir(os.path.join(path, 'processed'))
    df['mentions'].replace('[]', "['self']", inplace=True)
    df['mentions'].fillna("['self']", inplace=True)

    df.to_csv('final_rot_data.csv', index=False)

    print(df.shape)
    os.chdir(path)


def build_graph(path, name):
    if check_directory_absence('Graph', path):
        os.mkdir('Graph')
        os.chdir(os.path.join(path, 'processed'))
        if name == "vax":
            build_vaccination_graph(path)
        elif name == "rot":
            build_rot_graph(path)
    os.chdir(path)


def build_rot_graph(path):
    df = pd.read_csv(path + '/processed/' + 'final_rot_data.csv', lineterminator='\n')

    G_dg = nx.DiGraph()
    G_g = nx.Graph()
    print(df.columns)
    for _, row in tqdm(df.iterrows(), desc="Rows processed"):
        if row[3] == "mentions":
            continue
        mentions = ast.literal_eval(row[3])
        if 'self' in mentions:
            G_dg = add_edge(G_dg, row[2], "", 0, 0, 0, row[1], row[1])
        else:
            for mention in mentions:
                G_dg = add_edge(G_dg, row[2], "", 0, 0, 0, row[1], mention)
                G_g = add_edge(G_g, None, None, None, None, None, row[1], mention)

    G_dg.name = 'Starter rot Direct Graph'
    G_g.name = 'Starter rot Graph'

    graphs = [G_dg, G_g]
    manage_and_save(graphs, path)


def sentiment(starting_path, name, no_scaling=True):
    ####
    stop_words = []
    with open(f"{starting_path}/raw/stopwords_vader.txt", "rb") as fp:
        stop_words = pickle.load(fp)
    ###
    path = os.path.join(starting_path, 'Graph')
    os.chdir(os.path.join(path))

    CompGraph = nx.read_gml(f'Final_Graph_{name.capitalize()}.gml')
    print(nx.info(CompGraph))
    print()
    DiGraph = nx.read_gml(f'Final_DiGraph_{name.capitalize()}.gml')
    print(nx.info(DiGraph))
    print()
    if no_scaling:
        add_sent_weight_without_scaling(DiGraph, CompGraph, stop_words, name.capitalize())
    else:
        add_sent_weight(DiGraph, CompGraph, stop_words, name.capitalize())
    os.chdir(starting_path)


def graph_community_detection(starting_path, name):
    path = os.path.join(starting_path, 'Graph')
    os.chdir(path)
    print("weight")
    info_no_sent_metis, info_no_sent_fluid = community_detection(name.capitalize(), 1, 'weight')
    print("sentiment")
    info_sent_metis, info_sent_fluid = community_detection(name.capitalize(), 1, 'sentiment')
    #######
    print("consensus")
    info_cons_metis, info_cons_fluid = community_detection(name.capitalize(), 1, 'consensus')

    note_difference(info_no_sent_metis, info_sent_metis, 'Metis', 'sentiment')
    note_difference(info_no_sent_metis, info_cons_metis, 'Metis', 'consensus')
    os.chdir(starting_path)


def graph_controversy_detection(starting_path, name):
    path = os.path.join(starting_path, 'Graph')
    os.chdir(path)

    graph = nx.read_gml(f'Final_Graph_{name.capitalize()}.gml')
    logging.basicConfig(filename='community_log.log', level=logging.INFO, format='%(message)s')
    logging.info(10 * "*")

    print("SHORTEST PATH")
    print("WARNING: HARDCODED SHORTEST PATH")
    # shortest_path = average_shortest_path_length(graph)
    shortest_path = 3.236347712221066
    print(f'Average shortest path: {shortest_path}')

    random_walks(graph, 0.6, shortest_path * 2, opt=0)
    random_walks(graph, 0.6, shortest_path * 2, opt=1)
    random_walks(graph, 0.6, shortest_path * 2, opt=3)
    random_walks_centrality(graph, 1, opt=0)
    random_walks_centrality(graph, 1, opt=1)
    random_walks_centrality(graph, 1, opt=3)
    print()

    change_side_controversy(graph, 0.6, shortest_path * 2, opt=0)
    change_side_controversy(graph, 0.6, shortest_path * 2, opt=1)
    # test metric calculation with consensus
    change_side_controversy(graph, 0.6, shortest_path * 2, opt=3)
    print()

    start_GMCK(graph, 'weightComm')
    start_GMCK(graph, 'sentimentComm')
    start_GMCK(graph, 'consensusComm')
    print()

    os.chdir(starting_path)


def add_consensus(starting_path, name):
    path = os.path.join(starting_path, 'Graph')
    os.chdir(os.path.join(path))

    CompGraph = nx.read_gml(f'Final_Graph_{name.capitalize()}.gml')
    if not 'weightWithConsensus' in list(CompGraph.edges(data=True))[0][2]:
        print("Adding cons")
        print(nx.info(CompGraph))
        print()
        DiGraph = nx.read_gml(f'Final_DiGraph_{name.capitalize()}.gml')
        print(nx.info(DiGraph))
        print()
        add_consensus_weight(DiGraph, CompGraph, name)

    os.chdir(starting_path)
    pass


def run(path, name):
    if name == "rot":
        ops_on_rot(path)
    elif name == "vax":
        ops_on_vac(path)
    os.chdir(path)
    build_graph(path, name)
    sentiment(path, name, False)
    add_consensus(path, name)


import scipy
from modules.compute_metrics import _numpy_compute, compute_consensus


def compute_metrics_vader(path, name, communities, topology):
    """Computes consensus metric over calculated communities.
    """
    sentiment(path, name, True)
    extract_sentiment(path, name)
    df = pd.read_csv(os.path.join(path, f"processed/vader_sentiment_{name}.csv"), index_col="username")[["sentiment"]]

    results = {"users": len(df.index)}
    adj = None
    inv = None
    print(results["users"])

    if topology:
        G: nx.DiGraph = nx.read_gml(os.path.join(path, f'Graph/Final_DiGraph_{name.capitalize()}.gml'))
        adj: scipy.sparse.csr_matrix = nx.adjacency_matrix(G.subgraph(communities[0]))
    m1 = df.filter(items=communities[0], axis=0).to_numpy()

    sv, n = _numpy_compute(m1=m1, adj=adj, communities=communities)
    results["community0"] = compute_consensus(sv, n)

    if topology:
        adj: scipy.sparse.csr_matrix = nx.adjacency_matrix(G.subgraph(communities[1]))

    m2 = df.filter(items=communities[1], axis=0).to_numpy()

    sv, n = _numpy_compute(m1=m2, adj=adj, communities=communities)
    results["community1"] = compute_consensus(sv, n)

    if topology:
        adj = nx.adjacency_matrix(G)
        nodes = list(G.nodes())
        inv = {nodes[i]: i for i in range(len(nodes))}

    sv, n = _numpy_compute(m1=m1, m2=m2, adj=adj, communities=communities, nodes=inv)
    print(sv)
    results["inter_community"] = compute_consensus(sv, n)

    return results


def extract_sentiment(path, name):
    G = nx.read_gml(os.path.join(path, f'Graph/Final_DiGraph_{name.capitalize()}.gml'))

    data = []
    for node in G.nodes():
        row = {"username": node,
               "sentiment": G.nodes[node]["sentiment"]}
        data.append(row)

    df = pd.DataFrame(data, columns=["username", "sentiment"])
    df.to_csv(os.path.join(path, f"processed/vader_sentiment_{name}.csv"))
    print("Extracted sentiment")
