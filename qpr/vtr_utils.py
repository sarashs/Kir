import xml.etree.ElementTree as ET
import networkx as nx
import numpy as np


# https://docs.verilogtorouting.org/en/latest/vpr/file_formats/
def parse_rr_graph_xml(fpath):
    with open(fpath, 'r') as f:
        tree = ET.parse(f)
    root = tree.getroot()

    # <block_types>
    #     <block_type id="1" name="io" width="1" height="1">
    #         <pin_class type="INPUT">
    #             <pin ptc="2">io[0].clock[0]</pin>
    # e.g. id=0 <-> clb, id=1 <-> io, etc
    bt_map = dict()
    clock_nodes = dict()
    rr_block_types = root.find('block_types')
    for btype_elem in rr_block_types:
        bt_map[btype_elem.attrib['id']] = btype_elem.attrib['name']

        clock_nodes[btype_elem.attrib['name']] = dict(
            IPIN=[], OPIN=[], SINK=[], SOURCE=[]
        )
        for pin_class_idx, pin_class_elem in enumerate(btype_elem):
            is_clock = False
            for pin_elem in pin_class_elem:
                # TODO verify vvvvvvv:
                # clocks pins may be grouped together, take each pins ptc
                # to deal with I/OPINS, take the pin_class index within
                # the block_type element as the ptc for SOURCE/SINK nodes
                # assigned to the I/OPINS
                if 'clock' in pin_elem.text or 'clk' in pin_elem.text:
                    if pin_class_elem.attrib['type'] == 'INPUT':
                        _key = 'IPIN'
                    else:
                        _key = 'OPIN'
                    # clock_nodes[clb/io][ipin/opin].append(ptc)
                    clock_nodes[btype_elem.attrib['name']][_key].append(
                            int(pin_elem.attrib['ptc'])
                        )
                    # TODO: seeing one clock pin suggests a clock class
                    is_clock = True
            if is_clock:
                if pin_class_elem.attrib['type'] == 'INPUT':
                    _key = 'SINK'
                else:
                    _key = 'SOURCE'
                # TODO verify that there is one sink/source per class
                # clock_nodes[clb/io][sink/source].append(inferred ptc)
                clock_nodes[btype_elem.attrib['name']][_key].append(
                    pin_class_idx
                )
    # <grid>
    #     <grid_loc x="0" y="0" block_type_id="0" width_offset="0"
    #     height_offset="0"/>
    rr_grid = root.find('grid')
    grid_locs = dict()
    for elem in rr_grid:
        x, y, type_id = [elem.attrib[i] for i in ('x', 'y', 'block_type_id')]
        grid_locs[(int(x), int(y))] = bt_map[type_id]

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
        try:
            rr_nodes[nid]['side'] = loc.attrib['side']
        except KeyError:
            pass
        rr_nodes[nid]['posx'] = (
            rr_nodes[nid]['xlow'] + rr_nodes[nid]['xhigh']
        ) / 2
        rr_nodes[nid]['posy'] = (
            rr_nodes[nid]['ylow'] + rr_nodes[nid]['yhigh']
        ) / 2

        # TODO this only works if blocks fit on a single tile..
        btype = grid_locs[(
            rr_nodes[nid]['xlow'], rr_nodes[nid]['ylow']
        )]
        rr_nodes[nid]['block_type'] = btype
        # if it's not a wire segment and its ptc matches
        rr_nodes[nid]['is_clock'] = (
            rr_nodes[nid]['type'] not in ('CHANX', 'CHANY') and
            rr_nodes[nid]['ptc'] in clock_nodes[btype][rr_nodes[nid]['type']]
        )

    return graph, rr_nodes, grid_locs, clock_nodes


def pos_rr_graph(
    node_dict, grid_dict,
    side_deltas=dict(
       RIGHT=np.array([1, 1, 0, -1]), BOTTOM=np.array([1, -1, -1, 0]),
        LEFT=np.array([-1, -1, 0, 1]), TOP=np.array([-1, 1, 1, 0])
    ),
    fudge_factor=dict(SINK=.1, SOURCE=0, IPIN=.1, OPIN=0),
    chan_bbox=dict(CHANX=(1, .1, 0, 1), CHANY=(.1, 1, 1, 0)),
    bbb_half_edge=.2,
    rnbb_half_edge=.15
):
    '''
        node_dict: dict of node ids and node data parsed from rr_graph xml
            produced by parse_rr_graph_xml()
        grid_dict: dict of block grid locs (pairs of ints) and the block type
            produced by parse_rr_graph_xml()
        side_deltas:
            dx, dy: from the center of the tile to get to the start of the bbox
                edge, and laying out the pins in order clockwise around the
                bounding box of the (sub?)block
            step_dx, step_dy: which way to move along the edge to place the
                pins
        fudge_factor: fudge node classes to avoid overlap
            IOPIN: create concentric squares
            SINK/SOURCE: linear set of nodes just offset vertically
        chan_bbox:
            w, h: channel bounding box size
            orig_offset_x, orig_offset_y: origin of the coordinate system for
                channel location is offset relative the coordinate system for
                blocks
        bbb_half_edge: the edges of the bounding box of a block is this much
            away from its center
        rnbb_half_edge: put the route node (SINK/SOURCE) bouding boxs in side
            of a smaller bounding box inside of the block bounding box
    '''
    pos = dict()
    nodes_left = set(node_dict.keys())

    for posx, posy in grid_dict:
        if grid_dict[(posx, posy)] == 'EMPTY':
            continue
        grid_nodes = [
            i for i in node_dict
            if node_dict[i]['posx'] == posx and node_dict[i]['posy'] == posy
        ]

        # place wire segment nodes
        for chan in 'CHANX', 'CHANY':
            chan_nodes = [
                (i, node_dict[i]) for i in grid_nodes
                if node_dict[i]['type'] == chan
            ]
            if not chan_nodes:
                continue
            chan_nodes = sorted(chan_nodes, key=lambda x: x[1]['ptc'])

            w, h, orig_offset_x, orig_offset_y = chan_bbox[chan]
            chan_step_x, chan_step_y = w / len(chan_nodes), h / len(chan_nodes)

            for s_ind, (nid, _) in enumerate(chan_nodes):
                assert nid not in pos
                nodes_left.remove(nid)

                # enable short edge offset (0/1), enable along x for vertical
                # boxes
                en_seo_x, en_seo_y = w / 2, 0
                if chan == 'CHANX':
                    # enable along y for horizontal boxes
                    en_seo_x, en_seo_y = 0, h / 2

                x = (
                    orig_offset_x + posx - en_seo_x +
                    chan_step_x * (s_ind + .5)
                )
                y = (
                    orig_offset_y + posy - en_seo_y +
                    chan_step_y * (s_ind + .5)
                )
                pos[nid] = (x, y)

        # place I/OPIN nodes
        block_pins = [
            i for i in grid_nodes if node_dict[i]['type'] in ('IPIN', 'OPIN')
        ]
        cx, cy = posx + .5, posy + .5
        # do one side of the block at a time
        for side in 'TOP', 'RIGHT', 'BOTTOM', 'LEFT':
            side_pins = [i for i in block_pins if node_dict[i]['side'] == side]
            for ntype in 'IPIN', 'OPIN':
                pins = [
                    (j, node_dict[j]) for j in side_pins
                    if node_dict[j]['type'] == ntype
                ]
                if not pins:
                    continue
                pins = sorted(pins, key=lambda x: x[1]['ptc'])
                assert pins

                for p_idx, (nid, ndict) in enumerate(pins):
                    assert nid not in pos
                    nodes_left.remove(nid)
                    fudge = fudge_factor[ndict['type']]
                    fudge_half_edge = bbb_half_edge + fudge
                    side_step = 2 * fudge_half_edge / len(pins)
                    dx, dy, step_dx, step_dy = side_deltas[side]
                    # center to bbox edge, then p_idx steps, then offset pin
                    # loc by half a step, then fudge so that IPINs and OPINs
                    # don't overlap
                    x = (
                        cx + dx * fudge_half_edge +
                        step_dx * side_step * (p_idx + .5)
                    )
                    y = (
                        cy + dy * fudge_half_edge +
                        step_dy * side_step * (p_idx + .5)
                    )
                    pos[nid] = (x, y)

        for ntype in 'SOURCE', 'SINK':
            route_nodes = [
                (i, node_dict[i]) for i in grid_nodes
                if node_dict[i]['type'] == ntype
            ]
            route_nodes = sorted(route_nodes, key=lambda x: x[1]['ptc'])
            step = 2 * rnbb_half_edge / len(route_nodes)
            assert route_nodes
            for p_idx, (nid, ndict) in enumerate(route_nodes):
                assert nid not in pos
                nodes_left.remove(nid)
                fudge = fudge_factor[ndict['type']]
                pos[nid] = (
                    cx + fudge, cy - rnbb_half_edge + step * (p_idx + .5)
                )
    assert not nodes_left

    return pos


def draw_rr_graph(
    graph, node_dict, grid_dict, fig, ax,
    node_cols=dict(
        SOURCE='r', SINK='tab:blue', CHANX='y', CHANY='y', IPIN='g', OPIN='m'
    ),
    **kwargs
):
    '''
        The kwargs are passed on to pos_rr_graph()
    '''
    pos = pos_rr_graph(node_dict, grid_dict, **kwargs)

    draw_subgraph = graph.subgraph(list(pos.keys()))
    for i in node_cols:
        draw_nodes = [j for j in node_dict if node_dict[j]['type'] == i]
        nx.draw_networkx(
            draw_subgraph, nodelist=draw_nodes, edgelist=[], pos=pos,
            node_color=node_cols[i],
        )
    draw_nodes = [i for i in node_dict if node_dict[i]['is_clock']]
    nx.draw_networkx(
        draw_subgraph, nodelist=draw_nodes, edgelist=[], pos=pos,
        node_color='tab:brown',
    )

    nx.draw_networkx_edges(draw_subgraph, pos=pos)
    # nx.draw_networkx(draw_subgraph, pos=pos, ax=ax)
    ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
    ax.grid(True)
    fig.tight_layout()
