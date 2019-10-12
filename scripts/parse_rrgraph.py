import xml.etree.ElementTree as ET
from pathlib import Path
# from qpr.routing import NetList, Route
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

fpath = Path('sharedgraphstuff/rrgraph.xml')
with open(fpath, 'r') as f:
    tree = ET.parse(f)
root = tree.getroot()

# <rr_graph>
#     [..]
#     <rr_edges>
#         <edge src_node="548" sink_node="477" switch_id="1"/>
# this finds the first rr_edge element
rr_edges = root.find('rr_edges')
edge_list = []
for edge in rr_edges:
    edge_list.append((edge.attrib['src_node'], edge.attrib['sink_node']))

graph = nx.DiGraph(edge_list)

node_list1 = np.array(graph.nodes)
deg_list1 = np.array([graph.degree(i) for i in graph.nodes])
inds1 = deg_list1 > 9
pairs1 = list(zip(node_list1[inds1], deg_list1[inds1]))
pairs1 = sorted(pairs1, key=lambda x: x[1])
print(pairs1)


# <rr_graph>
#     [..]
#     <rr_nodes>
#         <node id="0" type="SINK" capacity="1">
#             <loc xlow="0" ylow="1" xhigh="0" yhigh="1" ptc="0"/>
#             <timing R="0" C="0"/>
#         </node>
rr_nodes = root.find('rr_nodes')
node_list2, node_cap_list2 = [], []
pos2 = dict()
for node in rr_nodes:
    node_list2.append(node.attrib['id'])
    node_cap_list2.append(int(node.attrib['capacity']))
    if node.attrib['id'] in node_list1[inds1]:
        assert node.attrib['id'] not in pos2
        # assume there is exactly one loc element
        loc = node.find('loc')
        x0, x1, y0, y1 = [
            int(loc.attrib[i]) for i in ['xlow', 'xhigh', 'ylow', 'yhigh']
        ]
        pos2[node.attrib['id']] = ((x0 + x1) / 2, (y0 + y1) / 2)

min_cap, max_cap = np.min(node_cap_list2), np.max(node_cap_list2)
hist_bin_edges2 = np.arange(min_cap, max_cap + 2) - 0.5
print(min_cap, max_cap, len(node_list2))
plt.hist(node_cap_list2, bins=hist_bin_edges2)
plt.show()

min_deg, max_deg = np.min(deg_list1), np.max(deg_list1)
hist_bin_edges1 = np.arange(min_deg, max_deg + 2) - 0.5
print(min_deg, max_deg)
plt.hist(deg_list1, bins=hist_bin_edges1)
plt.show()

nx.draw_networkx(graph)
plt.show()
