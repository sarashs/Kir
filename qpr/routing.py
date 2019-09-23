import numpy as np
import re
import networkx as nx
import dimod
from .graph_utils import get_paths_for_nodes
from .quantum_utils import get_all_min_energy
from itertools import combinations_with_replacement as cwr


class NetList(object):
    def __init__(self, net_list):
        '''
            net_list: a sequence of (source, targets) pairs where source is
                      a single node and targets is a sequence of nodes
                      e.g. [[1, [2, 3]], [4, [6]], [7, [8, 9, 10]]
        '''
        # TODO verify that no source node appears in the set of target nodes
        # and vice versa

        self.net_list = net_list
        net_edges = [(s, t) for s, targets in self.net_list for t in targets]
        self.graph = nx.DiGraph(net_edges)

        self.all_source_nodes = set(s for s, _ in self.net_list)
        self.all_target_nodes = self.graph.nodes - self.all_source_nodes

        # each CLB gets one source node which connects to all of its input pin
        # nodes and one target node that connects to all of its output nodes
        # (i.e. separated source and target nodes). Make sure the netlist
        # honors this.
        assert np.all([
            i not in self.all_source_nodes for i in self.all_target_nodes
        ])
        assert np.all([
            i not in self.all_target_nodes for i in self.all_source_nodes
        ])

    def map(self, map_):
        ''' map_: a dict for mapping node names, returns a new Netlist obj '''
        net_list = [
            [map_[source], [map_[target] for target in target_list]]
            for source, target_list in self.net_list
        ]
        return NetList(net_list)


class Route(object):
    def __init__(self, arch_graph, netlist, edge_list, node_map, pos=None):
        '''
            arch_graph: nx.DiGraph
            netlist: Netlist
            edge_list: a set/sequence of edges from arch_graph used for
                       routing, forming route_subgraph
            node_map: a dict, keys are netlist graph node names and values are
                      corresponding arch_graph/route_subgraph nodes
                      (essentially the placement).
            pos: a dict, keys are arch_graph nodes and values are coordinates,
                 used in draw() only.
        '''
        assert type(edge_list) is set
        assert type(netlist) is NetList
        assert type(node_map) is dict

        self.arch_graph = arch_graph
        self.netlist = netlist
        self.edge_list = edge_list
        self.node_map = node_map
        self.pos = pos

        self.placed_netlist = self.netlist.map(self.node_map)
        self.all_source_nodes = self.placed_netlist.all_source_nodes
        self.all_target_nodes = self.placed_netlist.all_target_nodes
        # non-terminal nodes in arch_graph
        self.all_nt_nodes = (
            self.arch_graph.nodes - self.all_source_nodes -
            self.all_target_nodes
        )

        self.route_subgraph = self.arch_graph.edge_subgraph(self.edge_list)

        # draw properties: node size, source color, target color, route width,
        #                  route color, default edge style,
        self.dp = dict(ns=900, sc='y', tc='c', rw=8, rc='r', ds='dotted')

    def draw_arch(self, ax=None):
        nx.draw_networkx(
            self.arch_graph, pos=self.pos, node_size=self.dp['ns'], ax=ax,
        )

    def draw(self, draw_netlist=True, draw_subgraph=True, axes=None):
        if axes is None:
            axes = [None] * 3

        # edges not used in the routing graph
        inactive_edges = self.arch_graph.edges - self.edge_list

        # Draw routing subgraph
        nx.draw_networkx_edges(
            self.arch_graph, edgelist=self.edge_list, pos=self.pos,
            width=self.dp['rw'], node_size=self.dp['ns'],
            edge_color=self.dp['rc'], ax=axes[0],
        )
        # draw architecture graph
        # TODO: it doesn't seem to honor the edge style, not for a directed
        #       graphs anyway
        # node_size is required to make sure the arrow heads appear in the
        # right place
        nx.draw_networkx_edges(
            self.arch_graph, edgelist=inactive_edges, pos=self.pos,
            node_size=self.dp['ns'], ax=axes[0],
        )

        # draw source nodes
        custom_labs = {
            self.node_map[i]: f'{self.node_map[i]}_{i}'
            for i in self.netlist.all_source_nodes
        }
        # suppress the edges since we draw them separately
        nx.draw_networkx(
            self.arch_graph, nodelist=self.all_source_nodes, pos=self.pos,
            node_color=self.dp['sc'], edgelist=[], labels=custom_labs,
            node_size=self.dp['ns'], ax=axes[0],
        )

        # draw target nodes
        custom_labs = {
            self.node_map[i]: f'{self.node_map[i]}_{i}'
            for i in self.netlist.all_target_nodes
        }
        # suppress the edges since we draw them separately
        nx.draw_networkx(
            self.arch_graph, nodelist=self.all_target_nodes, pos=self.pos,
            node_color=self.dp['tc'], edgelist=[], labels=custom_labs,
            node_size=self.dp['ns'], ax=axes[0],
        )
        # draw non-terminal nodes
        # specifying nodelist determines which nodes get a circle, but it
        # displays *all* nodes labels. To suppress that, labels=custom_labs is
        # required.
        custom_labs = {i: i for i in self.all_nt_nodes}
        nx.draw_networkx(
            self.arch_graph, nodelist=self.all_nt_nodes, pos=self.pos,
            edgelist=[], labels=custom_labs, node_size=self.dp['ns'],
            ax=axes[0],
        )

        if draw_netlist:
            nx.draw_networkx(self.netlist.graph, ax=axes[1])
        if draw_subgraph:
            nx.draw_networkx(self.route_subgraph, pos=self.pos, ax=axes[2])

    def validate(self):
        # the routing should satisfy the netlist i.e. what's connected in
        # netlist should be connected in route_subgraph and what's not
        # connected in netlist shouldn't be connected in route_subgraph
        # returns a sparse matrix
        netlist_A = nx.adjacency_matrix(self.netlist.graph)
        node_list = [self.node_map[i] for i in self.netlist.graph.nodes]

        route_paths, _ = get_paths_for_nodes(self.route_subgraph, node_list)

        # route_paths contains the path lenght but netlist_A contains 0 or 1
        if not np.all((route_paths > 0) == (netlist_A > 0)):
            return False

        # if there are dedicated source and target nodes, the node constraints
        # 1-4 along with the connectivity constraint gurantees no loops,
        # I [SP] think..
        """
        # there should be no cycle:
        try:
            # does DFS and returns the first cycle
            nx.find_cycle(self.route_subgraph.to_undirected(as_view=True))
            return False
        except nx.NetworkXNoCycle:
            pass

        # there should be no dangling edges, i.e. only terminal nodes can have
        # degree one
        non_terminal = (
            self.route_subgraph.nodes - self.all_source_nodes -
            self.all_target_nodes
        )
        # .degree() ignores the edge directions in DiGraphs but that's okay
        non_term_degrees = np.array([
            deg for n, deg in self.route_subgraph.degree(non_terminal)
        ])
        # np.any() returns false for an empty array, np.all() returns true for
        # an empty array
        if np.any(non_term_degrees == 1):
            return False
        """

        route_A = nx.adjacency_matrix(self.route_subgraph)

        # source nodes, create the approproate array for slicing route_A
        nodes = np.array([
            i in self.all_source_nodes for i in self.route_subgraph.nodes
        ])
        # 1-no edge should end at a source node
        if route_A[:, nodes].sum() > 0:
            return False
        # target nodes
        nodes = np.array([
            i in self.all_target_nodes for i in self.route_subgraph.nodes
        ])
        # 2-no edge should start from a target node
        if route_A[nodes, :].sum() > 0:
            return False
        # non-terminal nodes
        nodes = np.array([
            i not in self.all_source_nodes and i not in self.all_target_nodes
            for i in self.route_subgraph.nodes
        ])
        # 3-non-terminal nodes, i.e. wires, shouldn't have more than one input
        if np.any(route_A[:, nodes].sum(axis=0) > 1):
            return False
        # nodes that have at least one input
        # .sum() produces a dense output, it's a 1xn matrix, has to be
        # squeezed to be used for slicing route_A, matrix can't be squeezed
        # so convert to array
        nodes_with_input = np.array(route_A.sum(axis=0) > 0).squeeze()
        # 4-non-terminal nodes that have at least one input
        nonterm_with_input = np.array([
            with_input and nonterm
            for with_input, nonterm in zip(nodes_with_input, nodes)
        ])
        # 4-must have an output otherwise they are dangling
        if np.any(route_A[nonterm_with_input, :].sum(axis=1) == 0):
            return False

        return True

    def score(self):
        if not self.validate():
            return np.inf
        return len(self.edge_list)


def get_qubo(
    arch_graph, netlist, w_obj=0, w_target=1, w_nonterm=1, w_source=1, w_and=6
):
    '''
        w_obj: objective (formerly w1)
        w_target: constraint on target nodes (formerly w2)
        w_nonterm: constraint on non-terminal nodes (formerly w3)
        w_source: constraint on the source nodes (formerly w4)
        w_and: yiyj and weightm
    '''
    assert len(netlist.net_list) == 1
    assert len(netlist.net_list[0][1]) == 1

    net_end = [netlist.net_list[0][1][0]]
    net_start = [netlist.net_list[0][0]]
    n = len(net_end)

    Q = {}
    # move away from y1, y2, etc as var names and do y(1, 2), y(3, 'i1'), etc
    # to make the edge-variable correspondence clear. Use w(1, 2)(3, 'i1')
    # format for auxiliary/'and' variables.
    for edge1, edge2 in cwr(arch_graph.edges(), 2):
        Q[(f'y{edge1}', f'y{edge2}')] = 0

    starting_node = []
    end_nodes = {}
    other_nodes = {}
    for item_1 in net_end:
        end_nodes[item_1] = []
    for item_1 in arch_graph.nodes:
        if item_1 not in net_end and item_1 not in net_start:
            other_nodes[item_1] = []

    # variables are defined for each edge, iterate over numerical edge labels
    # and store what edge is connected to which type of node.
    for edge in arch_graph.edges:
        Q[(f'y{edge}', f'y{edge}')] = 1 * w_obj
        # Starting node
        if edge[0] in net_start or edge[1] in net_start:
            starting_node.append(edge)
        # end node
        if edge[0] in net_end:
            end_nodes[edge[0]].append(edge)
        if edge[1] in net_end:
            end_nodes[edge[1]].append(edge)
        # other nodes
        # dangling non-terminal nodes are irrelevant
        if edge[0] in other_nodes:  # and arch_graph.degree(edge[0]) > 1:
            other_nodes[edge[0]].append(edge)
        if edge[1] in other_nodes:  # and arch_graph.degree(edge[1]) > 1:
            other_nodes[edge[1]].append(edge)

    # each item is a list of (numerical) edge labels ending in the node
    for node, edge_list in end_nodes.items():
        # for each pair of edges
        for i, j in cwr(edge_list, 2):
            if i == j:
                # ######Removing end node edges from the objective
                # Q[(f'y{i}', f'y{j}')] += -1 * w_obj
                # #############################
                Q[(f'y{i}', f'y{j}')] += -1 * w_target
            else:
                Q[(f'y{i}', f'y{j}')] += 2 * w_target

    # iterate over numerical edge labels
    for node, edge_list in other_nodes.items():
        for i, j in cwr(edge_list, 2):
            if i == j:
                Q[(f'y{i}', f'y{j}')] += w_nonterm
            else:
                Q[(f'y{i}', f'y{j}')] += -2 * w_nonterm  # 2
        for i, j, k in cwr(edge_list, 3):
            if i != j and j != k and i != k:
                if (f'w{i}{j}', f'y{k}') not in Q:
                    Q[(f'w{i}{j}', f'y{k}')] = 0
                Q[(f'w{i}{j}', f'y{k}')] += 6 * w_nonterm  # 2
                Q[(f'y{i}', f'y{j}')] += 1 * w_and  # 2
                if (f'w{i}{j}', f'y{i}') not in Q:
                    Q[(f'w{i}{j}', f'y{i}')] = 0
                Q[(f'w{i}{j}', f'y{i}')] += -2 * w_and
                if (f'w{i}{j}', f'y{j}') not in Q:
                    Q[(f'w{i}{j}', f'y{j}')] = 0
                Q[(f'w{i}{j}', f'y{j}')] += -2 * w_and
                if (f'w{i}{j}', f'w{i}{j}') not in Q:
                    Q[(f'w{i}{j}', f'w{i}{j}')] = 0
                Q[(f'w{i}{j}', f'w{i}{j}')] += 3 * w_and

    # starting_node is a list of numerical labels and not a dict
    for i, j in cwr(starting_node, 2):
        if i == j:
            # ######Removing starting node edges from the objective
            # Q[(f'y{i}', f'y{j}')] += -1*w_obj
            # #############################
            Q[(f'y{i}', f'y{j}')] += -(2 * n - 1) * w_source
        else:
            Q[(f'y{i}', f'y{j}')] += 2 * w_source

    return Q


def edge_list_from_qubo_answer_dict(ans):
    # eval converts the string "(1, 2)" or "(3, 'i1')" to tuples
    return [eval(i[1:]) for i in ans if ans[i] == 1 and i[0] == 'y']


class DimodExact(object):
    solver = None

    @classmethod
    def solve(cls, Q):
        if cls.solver is None:
            cls.solver = dimod.ExactSolver()
        response = cls.solver.sample_qubo(Q)
        min_energy_sols, _ = get_all_min_energy(response)
        return [edge_list_from_qubo_answer_dict(i) for i in min_energy_sols]


# TODO these verify_xx and qubo_violation funtions can be combined and made
# more efficient..
def verify_and(ans_dict):
    and_vars = [i for i in ans_dict if i[0] == 'w']
    var_pairs = [re.match(r'w(\(.*, .*\))(\(.*, .*\))', i) for i in and_vars]
    var_pairs = [
        (f'y{i.group(1)}', f'y{i.group(2)}') for i in var_pairs
        if i is not None
    ]
    broken = [
        (w12, v1, v2, f'{ans_dict[v1]}x{ans_dict[v2]}<->{ans_dict[w12]}')
        for (v1, v2), w12 in zip(var_pairs, and_vars)
        if ans_dict[v1] * ans_dict[v2] != ans_dict[w12]
    ]
    if len(and_vars) == 0:
        return 0, None
    return len(broken) / len(and_vars), broken


def verify_term(ans_dict, node_list):
    ''' node_list: a set of terminal (source or target) nodes
    '''
    active_dict = {k: v for k, v in ans_dict.items() if v == 1}
    edges = edge_list_from_qubo_answer_dict(active_dict)
    counts = np.zeros(len(node_list))
    for i, n in enumerate(node_list):
        for e in edges:
            if n in e:
                counts[i] += 1
    return (counts != 1).sum() / counts.size, zip(node_list, counts)


def verify_nonterm(ans_dict, node_list):
    active_dict = {k: v for k, v in ans_dict.items() if v == 1}
    edges = edge_list_from_qubo_answer_dict(active_dict)
    counts = np.zeros(len(node_list))
    for i, n in enumerate(node_list):
        for e in edges:
            if n in e:
                counts[i] += 1
    return (
        ((counts != 0) & (counts != 2)).sum() / counts.size,
        list(zip(node_list, counts))
    )


def qubo_violation(ans_dict, source_list, target_list, nonterm_list):
    active_dict = {k: v for k, v in ans_dict.items() if v == 1}
    active_edges = edge_list_from_qubo_answer_dict(active_dict)
    num_edges = len([i for i in ans_dict if i[0] == 'y'])
    assert num_edges > 0

    obj_score = len(active_edges) / num_edges
    and_score, _ = verify_and(ans_dict)
    source_score, _ = verify_term(ans_dict, source_list)
    target_score, _ = verify_term(ans_dict, target_list)
    nt_score, _ = verify_nonterm(ans_dict, nonterm_list)

    return {
        'obj': obj_score, 'and': and_score, 's': source_score,
        't': target_score, 'nt': nt_score
    }
