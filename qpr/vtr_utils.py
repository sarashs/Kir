import xml.etree.ElementTree as ET
import networkx as nx


def parse_rr_graph_xml(fpath):
    with open(fpath, 'r') as f:
        tree = ET.parse(f)
    root = tree.getroot()

    # parse the edge elements
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

    # parse the node elements
    # <rr_graph>
    #     [..]
    #     <rr_nodes>
    #         <node id="0" type="SINK" capacity="1">
    #             <loc xlow="0" ylow="1" xhigh="0" yhigh="1" ptc="0"/>
    #             <timing R="0" C="0"/>
    #         </node>
    rr_nodes = dict()
    for node in root.find('rr_nodes'):
        nid = node.attrib['id']
        rr_nodes[nid] = dict(cap=int(node.attrib['capacity']))
        rr_nodes[nid]['type'] = node.attrib['type']
        # assume there is exactly one loc element
        loc = node.find('loc')
        for i in ['xlow', 'xhigh', 'ylow', 'yhigh', 'ptc']:
            rr_nodes[nid][i] = int(loc.attrib[i])
        rr_nodes[nid]['posx'] = (
            rr_nodes[nid]['xlow'] + rr_nodes[nid]['xhigh']
        ) / 2
        rr_nodes[nid]['posy'] = (
            rr_nodes[nid]['ylow'] + rr_nodes[nid]['yhigh']
        ) / 2

    return graph, rr_nodes
