def create_qubo(G,weight_objective=1,weight_end=1,weight_start=1,weight_others=1,weight_and=6):
    Q={}
    # these are 'y0', 'y1', etc
    for var1, var2 in cwr(G.np2qv.values(), 2):
        Q[(var1, var2)]=0
    
    Starting_node=[]
    end_nodes={}
    other_nodes={}
    for item_1 in net_end:
        end_nodes[item_1]=[]
    for item_1 in G.nodes:
        if item_1 not in net_end and item_1 not in net_start:
            other_nodes[item_1]=[]
    
    # iterate over numerical edge labels
    for edge_pair, num_lab in G.np2l.items():
        # objective
        w1=weight_objective
        Q[(f'y{num_lab}',f'y{num_lab}')]=1*w1
        # According to Arash, VPR defines nets as having one start and multiple ends
        #Starting node
        if edge_pair[0] in net_start or edge_pair[1] in net_start:
            Starting_node.append(num_lab)
        #end node
        if edge_pair[0] in net_end:
            end_nodes[edge_pair[0]].append(num_lab)
        if edge_pair[1] in net_end:
            end_nodes[edge_pair[1]].append(num_lab)
        #other nodes
        if edge_pair[0] in other_nodes:
            other_nodes[edge_pair[0]].append(num_lab)
        if edge_pair[1] in other_nodes:
            other_nodes[edge_pair[1]].append(num_lab)   
    
    #constraint on end nodes
    w2=weight_end
    # each item is a list of (numerical) edge labels ending in the node
    for node, num_lab_list in end_nodes.items():
        # for each pair of edges
        for i,j in cwr(num_lab_list,2):
            if i==j:
                #######Removing end node edges from the objective
                Q[(f'y{i}',f'y{j}')]+=-1*w1 
                ##############################
                Q[(f'y{i}',f'y{j}')]+=-1*w2
            else:
                Q[(f'y{i}',f'y{j}')]+=2*w2
    
    #constraint on other nodes
    w3=weight_others
    # yiyj and weightm
    w_and=weight_and
    
    # iterate over numerical edge labels
    for node, num_lab_list in other_nodes.items():
        for i,j in cwr(num_lab_list,2):
            if i==j:
                Q[(f'y{i}',f'y{j}')]+=w3
            else:
                Q[(f'y{i}',f'y{j}')]+=-2*w3 #2
        for i,j,k in cwr(num_lab_list,3):
            if i !=j and j!=k and i !=k:
                if (f'w{i}{j}',f'y{k}') not in Q:
                    Q[(f'w{i}{j}',f'y{k}')]=0
                Q[(f'w{i}{j}',f'y{k}')]+=6*w3 #2
                Q[(f'y{i}',f'y{j}')]+=1*w_and #2
                if (f'w{i}{j}',f'y{i}') not in Q:
                    Q[(f'w{i}{j}',f'y{i}')]=0
                Q[(f'w{i}{j}',f'y{i}')]+=-2*w_and
                if (f'w{i}{j}',f'y{j}') not in Q:
                    Q[(f'w{i}{j}',f'y{j}')]=0
                Q[(f'w{i}{j}',f'y{j}')]+=-2*w_and
                if (f'w{i}{j}',f'w{i}{j}') not in Q:
                    Q[(f'w{i}{j}',f'w{i}{j}')]=0
                Q[(f'w{i}{j}',f'w{i}{j}')]+=3*w_and            
    #constraint on the starting node
    w4=weight_start
    # Starting_node is a list of numerical labels and not a dict
    for i,j in cwr(Starting_node,2):
        if i==j:
            #######Removing starting node edges from the objective
            Q[(f'y{i}',f'y{j}')]+=-1*w1 
            ##############################
            Q[(f'y{i}',f'y{j}')]+=-(2*n-1)*w4
        else:
            Q[(f'y{i}',f'y{j}')]+=2*w4
    return Q
def longest_chain_in_embed(e):
    return np.max([len(i) for i in e.values()])

def find_embedding_minorminer(Q, A, num_tries=100):
    best_embedding = None
    best_chain_len = np.inf

    for i in range(num_tries):
        e = minorminer.find_embedding(Q, A)
        if e: #to guarantee an embedding is produced
            chain_len = longest_chain_in_embed(e)
            if chain_len < best_chain_len:
                best_embedding = e
                best_chain_len = chain_len

    return best_embedding, best_chain_len   

# flip the key-value in a dict
def reverse_map(_dict):
    # make sure the values are unique, i.e. nothing gets clobbered
    assert len(_dict) == len(set(_dict.values()))
    return {val: key for key, val in _dict.items()}


class RectGridGraph(nx.Graph):
    def __init__(self, nrow, ncol):
        super().__init__(nx.grid_2d_graph(nrow, ncol))
        
        self.pos = {n: n for n in self.nodes()}
        self.np2l = {edge: idx for idx, edge in enumerate(self.edges)}  # node pair to numerical labels
        self.np2qv = {edge: f'y{idx}' for idx, edge in enumerate(self.edges)}  # node pair to qubo var name
        self.l2np = reverse_map(self.np2l)
        self.qv2np = reverse_map(self.np2qv)

    def draw(self, ax=None, edge_labs=True):
        nx.draw_networkx_nodes(self, self.pos, node_size=100, node_color='r', ax=ax)
        nx.draw_networkx_edges(self, self.pos, ax=ax)
        if edge_labs:
            nx.draw_networkx_edge_labels(self, self.pos, self.np2l, font_size=20, ax=ax)

    def qubo_answer2node_pairs(self, ans):
        # answer is a dict of {qubo var name: 0/1} e.g. {'y0': 0, 'y1': 1, etc}
        # it can have auxiliary variables not found in self.np2qv or self.qv2np
        return [self.qv2np[var_name] for var_name in self.qv2np if ans[var_name] == 1]
    
    def highlight_edge_list(self, edge_list, ax=None):
        nx.draw_networkx_edges(
            self, self.pos, edgelist=edge_list, width=8, edge_color='r', ax=ax
        )
