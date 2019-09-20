import networkx as nx
from qpr.routing import Route, NetList, get_qubo, DimodExact


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


def test_two_node_qubo():
    two_node_graph = nx.Graph([(0, 1)])
    netlist = NetList([(0, [1])])
    Q = get_qubo(
        two_node_graph, netlist,
        w_obj=1, w_target=1, w_nonterm=1, w_source=1, w_and=1
    )

    assert list(Q.keys()) == [('y(0, 1)', 'y(0, 1)')]
    assert Q[('y(0, 1)', 'y(0, 1)')] == -1


def test_qubo_weights():
    two_node_graph = nx.Graph([(0, 1)])
    netlist = NetList([(0, [1])])

    ws = dict(w_obj=10, w_target=100, w_nonterm=1e3, w_source=1e4, w_and=1e5)
    Q = get_qubo(two_node_graph, netlist, **ws)

    Q_exp = {
        # source: -y1, target: -y1, nonterm: 0, obj: +y1
        ('y(0, 1)', 'y(0, 1)'):
            -1 * ws['w_source'] - 1 * ws['w_target'] + 1 * ws['w_obj'] +
            0 * ws['w_nonterm'] + 0 * ws['w_and'],
    }

    assert Q == Q_exp


def test_three_node_line_qubo():
    three_node_graph = nx.Graph([(0, 1), (1, 2)])
    netlist = NetList([(0, [2])])

    ws = dict(w_obj=10, w_target=100, w_nonterm=1e3, w_source=1e4, w_and=1e5)
    Q = get_qubo(three_node_graph, netlist, **ws)

    Q_exp = {
        ('y(0, 1)', 'y(0, 1)'):
            -1 * ws['w_source'] + 1 * ws['w_obj'] + 1 * ws['w_nonterm'],
        ('y(0, 1)', 'y(1, 2)'): -2 * ws['w_nonterm'],
        ('y(1, 2)', 'y(1, 2)'):
            -1 * ws['w_target'] + 1 * ws['w_obj'] + 1 * ws['w_nonterm'],
    }

    assert Q == Q_exp

    min_energy_sols = DimodExact.solve(Q)
    assert min_energy_sols == [[(0, 1), (1, 2)]]


def test_six_node_qubo():
    # o   o   o
    # |   |   |
    # S - o - T
    #
    arch_graph = nx.Graph([(0, 1), (1, 2), (0, 3), (1, 4), (2, 5)])
    netlist = NetList([(0, [2])])

    ws = dict(w_obj=10, w_target=100, w_nonterm=1e3, w_source=1e4, w_and=1e5)
    Q = get_qubo(arch_graph, netlist, **ws)

    Q_exp = {
        ('y(0, 1)', 'y(0, 1)'):
            -1 * ws['w_source'] + 1 * ws['w_obj'] + 1 * ws['w_nonterm'],
        ('y(1, 2)', 'y(1, 2)'):
            -1 * ws['w_target'] + 1 * ws['w_obj'] + 1 * ws['w_nonterm'],
        ('y(0, 3)', 'y(0, 3)'):
            1 * ws['w_obj'] - 1 * ws['w_source'] + 1 * ws['w_nonterm'],
        ('y(1, 4)', 'y(1, 4)'):
            1 * ws['w_nonterm'] + 1 * ws['w_obj'] + 1 * ws['w_nonterm'],
        ('y(2, 5)', 'y(2, 5)'):
            -1 * ws['w_target'] + 1 * ws['w_obj'] + 1 * ws['w_nonterm'],
        ('w(0, 1)(1, 2)', 'w(0, 1)(1, 2)'): 3 * ws['w_and'],

        ('y(0, 1)', 'y(1, 2)'): -2 * ws['w_nonterm'] + 1 * ws['w_and'],
        ('y(1, 2)', 'y(2, 5)'): 2 * ws['w_target'],
        ('y(0, 1)', 'y(1, 4)'): -2 * ws['w_nonterm'],
        ('y(1, 2)', 'y(1, 4)'): -2 * ws['w_nonterm'],
        ('y(0, 1)', 'y(0, 3)'): 2 * ws['w_source'],

        ('w(0, 1)(1, 2)', 'y(1, 4)'): 6 * ws['w_nonterm'],
        ('w(0, 1)(1, 2)', 'y(0, 1)'): -2 * ws['w_and'],
        ('w(0, 1)(1, 2)', 'y(1, 2)'): -2 * ws['w_and'],

        ('y(0, 3)', 'y(1, 4)'): 0,
        ('y(1, 4)', 'y(2, 5)'): 0,
        ('y(0, 3)', 'y(2, 5)'): 0,
        ('y(0, 1)', 'y(2, 5)'): 0,
        ('y(0, 3)', 'y(1, 2)'): 0,
    }

    assert Q == Q_exp

    min_energy_sols = DimodExact.solve(Q)
    assert min_energy_sols == [[(0, 1), (1, 2)]]
