from pathlib import Path
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from qpr.vtr_utils import parse_rr_graph_xml
# from qpr.routing import NetList, Route

fpath = Path('sharedgraphstuff/rrgraph.xml')
graph, node_dict = parse_rr_graph_xml(fpath)

# find the nodes with high degrees and print them
node_list1 = np.array(graph.nodes)
deg_list1 = np.array([graph.degree(i) for i in graph.nodes])
inds1 = deg_list1 > 9
pairs1 = list(zip(node_list1[inds1], deg_list1[inds1]))
pairs1 = sorted(pairs1, key=lambda x: x[1])
print(pairs1)

print(np.unique([node_dict[i]['type'] for i in node_dict]))

# plot histogram of the node degrees
min_deg, max_deg = np.min(deg_list1), np.max(deg_list1)
hist_bin_edges1 = np.arange(min_deg, max_deg + 2) - 0.5
print(min_deg, max_deg)
# plt.hist(deg_list1, bins=hist_bin_edges1)
# plt.show()

node_subset = list(node_list1[inds1])
print(f"neighbors: {list(graph.to_undirected(as_view=True)['240'])}")
node_subset += list(graph.to_undirected(as_view=True)['240'])

# plot the graphs
fig, ax = plt.subplots(nrows=1, ncols=2)
nx.draw_networkx(graph, ax=ax[0])
nx.draw_networkx(graph.subgraph(node_subset), ax=ax[1])
plt.show()
