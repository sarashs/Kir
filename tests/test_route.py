import networkx as nx
from qpr.routing import Route, NetList


def get_3x3_arch_graph():
    edges = [
        (6, 7), (7, 8), (6, 3), (7, 4), (8, 5),
        (3, 4), (4, 5), (3, 0), (4, 1), (5, 2),
        (0, 1), (1, 2),
    ]
    rev_edges = [(j, i) for i, j in edges]
    edges = edges + rev_edges
    edges = edges + [
        (0, 'i0'), ('o0', 0), (1, 'i1'), ('o1', 1),
        (2, 'i2'), ('o2', 2), (3, 'i3'), ('o3', 3),
        (5, 'i5'), ('o5', 5), (6, 'i6'), ('o6', 6),
        (7, 'i7'), ('o7', 7), (8, 'i8'), ('o8', 8),
    ]

    return nx.DiGraph(edges)


def test_dangling_node():
    super_graph = get_3x3_arch_graph()

    # (1, 4) is a dangling node
    route = Route(
        super_graph,
        NetList([('s1', ['t1']), ]),
        set([('o0', 0), (0, 1), (1, 2), (2, 'i2'), (1, 4)]),  # edge_list
        dict(s1='o0', t1='i2'),  # node_map
        None
    )
    assert route.validate() is False


def test_loop():
    super_graph = get_3x3_arch_graph()

    # there is a loop in the wires
    route = Route(
        super_graph,
        NetList([('s1', ['t1']), ]),
        set([
            ('o0', 0), (0, 3), (3, 4), (3, 6), (6, 7), (4, 7), (7, 'i7')
        ]),  # edge_list
        dict(s1='o0', t1='i7'),  # node_map
        None
    )
    assert route.validate() is False


def test_loop2():
    super_graph = get_3x3_arch_graph()

    # there is a loop in the wires
    route = Route(
        super_graph,
        NetList([('s1', ['t1']), ]),
        set([
            ('o0', 0), (0, 3), (3, 4), (3, 6), (6, 7), (7, 4), (7, 'i7')
        ]),  # edge_list
        dict(s1='o0', t1='i7'),  # node_map
        None
    )
    assert route.validate() is False
