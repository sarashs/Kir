import numpy as np
import networkx as nx
import scipy.sparse as scisparse
import itertools


def get_paths_for_nodes(g, node_list, sparse=False):
    # make sure there are no duplicates in node_list
    assert len(node_list) == np.unique(node_list).size

    num_nodes = len(node_list)
    all_paths = []
    if sparse:
        path_mat = scisparse.csr_matrix((num_nodes, num_nodes))
    else:
        path_mat = np.zeros((num_nodes, num_nodes))
    for i, j in itertools.permutations(range(num_nodes), 2):
        source = node_list[i]
        target = node_list[j]

        # raises NodeNotFound if source is not in G, and NetworkXNoPath target
        # is not reachable from source (including if target is not in G)
        try:
            length = nx.dijkstra_path_length(g, source, target)
            all_paths.append((source, target))
            path_mat[i, j] = length
            # display(f'({source}, {target}): {length}')
        except Exception:
            pass
    return path_mat, all_paths


def get_paths_for_nodes_bfs(g, node_list, sparse=False):
    # make sure there are no duplicates in node_list
    assert len(node_list) == np.unique(node_list).size
    num_nodes = len(node_list)
    if sparse:
        path_mat = scisparse.csr_matrix((num_nodes, num_nodes))
    else:
        path_mat = np.zeros((num_nodes, num_nodes))

    for i in range(num_nodes):
        source = node_list[i]
        pred_map = dict(nx.bfs_predecessors(g, source))
        for j in range(num_nodes):
            # skip self loops
            if i == j:
                continue
            target = node_list[j]
            # if target is unreachable, leave path lenght as zero
            if target not in pred_map:
                continue
            # traverse back from each target towards the source and count the
            # number of steps
            while target != source:
                target = pred_map[target]
                path_mat[i, j] += 1
    return path_mat
