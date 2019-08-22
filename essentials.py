#import the necessary shit
import matplotlib.pyplot as plt
import numpy as np
import random
from copy import deepcopy
from itertools import combinations_with_replacement as cwr

import dwave_networkx as dnx
import networkx as nx
import minorminer
from dwave.system.samplers import DWaveSampler
from dwave.system.composites import FixedEmbeddingComposite
import hybrid
import dimod

####

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
#####

def create_qubo(G, net_start, net_end, params={'weight_objective': 1, 'weight_end': 1, 'weight_start': 1, 'weight_others': 1,'weight_and': 6}):
    Q = {}
    n = len(net_end)
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
        w1=params['weight_objective']
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
    w2=params['weight_end']
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
    w3=params['weight_others']
    # yiyj and weightm
    w_and=params['weight_and']
    
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
    w4=params['weight_start']
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

#####

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

#####

def optimize_qannealer(sampler, Q, params={'chain_strength': 7, 'annealing_time': 99, 'num_reads': 10000}):
    response = sampler.sample_qubo(
        Q, chain_strength=params['chain_strength'], annealing_time=params['annealing_time'], auto_scale=True, num_reads=params['num_reads']
    )
    return response

#####

def make_ax_grid(n, ax_h=4, ax_w=6, ncols=4):
    nrows = int(np.ceil(n / ncols))
    fig_h = nrows * ax_h
    fig_w = ncols * ax_w
    return plt.subplots(nrows=nrows, ncols=ncols, figsize=(fig_w, fig_h))

#####

def get_all_min_energy(sample_set):
    min_energy = np.min(sample_set.record.energy)
    # use .record since it is slicing friendly, this returns a 2d-array-like recarray
    records = sample_set.record[sample_set.record.energy == min_energy]
    # make dicts out of each answer using the original var names (i.e. sample_set.variables)
    return [dict(zip(sample_set.variables, i.sample)) for i in records], min_energy

#####

def plot_all_exact_solutions(min_energy_sols):
    fig, axes = make_ax_grid(len(min_energy_sols))
    display(len(min_energy_sols))
    
    for ax, answer_dict in zip(axes.flat, min_energy_sols):
        G.draw(edge_labs=False, ax=ax)  # edge_labs=False)
    
        edge_set = G.qubo_answer2node_pairs(answer_dict)
        G.highlight_edge_list(edge_set, ax=ax)
    fig.tight_layout()

#####

def check_against_exact(ans,exact_min_energy_sols):
    #ans is the answer from QPU or hybrid solver. 
    #exact_min_energy_sols is the set of all possible solutions from the exact solver
    return (ans in exact_min_energy_sols)

#####

def is_this_an_answer(ans, G, net_start, net_end): #q_response.samples()[0]
    edge_set = G.qubo_answer2node_pairs(ans)
    edge_end = net_start[0]
    out = 0
    for j in range(len(net_end)):
        while edge_set:
            for i, item in enumerate(edge_set):
                if edge_end == item[0]:
                    edge_end = item[1]
                    edge_set.pop(i)
                    break
                else:
                    edge_set=[]
            if edge_end == net_end[j]:
                out += 1
                break
    return out

#####

list_of_qubo_params = ['weight_objective', 'weight_end', 'weight_start', 'weight_others', 'weight_and']
list_of_anneal_params = ['num_reads', 'annealing_time', 'chain_strength']
class SA(object):
    """ Simulated Annealing optimizer.
    Parameters
    --------------
    T: float
        starting temperature.
    T_min: float
        final temperature.
    alpha: float
        temperature scaling factor.
    max_iter: int
        maximum number of iterations per temperature.
    params : dictionary 
        format: params= {'param_#':[init_value, min, max, integer_flag],...}
    
    Attributes
    --------------
    cost_ : float
        Cost calculated by the cost function, needs to be minimized.
    sol_  : dictionary
        current solution.
        format: sol_= {'param_#': value}
    costs : list
        List of costs over time
    sols : list (of dictionary)
        historical data on the best solution
        format: sols= [{'param_#': value}]
    """
    def __init__(self, graph_size, params={'weight_objective': [1, 0, 2, 0], 'weight_end': [1, 0, 2, 0],
                                           'weight_start': [1, 0, 2, 0] ,'weight_others': [1, 0, 2, 0],
                                           'weight_and': [6, 4, 15, 0],
                                           'chain_strength': [7, 4, 15, 0], 'annealing_time': [99, 10, 10000, 1]
                                          },
                 T=1, T_min=0.00001, alpha=0.9, max_iter=50
                ):
        self.graph_size = graph_size
        self.params = params
        self.T = T 
        self.T_min = T_min
        self.alpha = alpha
        self.max_iter = max_iter
        self.cost_= 0
        self.sol_= {key: val[0] for key, val in params.items()}
        self.sols = []
        self.energies = {}
    def param_generator(self):
        """Generates the next solution.
        Returns
        -------
        self : object
        """
        for j in self.sol_.keys():
            if self.params[j][1]==self.params[j][2]: # in order to fix a certain parameter
                sol_ = self.params[j][1]
            else:
                 if self.params[j][3] == 0: # if the parameter isn't an integer
                     while True:
                         sol_ = self.params[j][0] + (0.5-random.random()) # new parameter between -0.5,0.5
                         if sol_ > self.params[j][1] and sol_ < self.params[j][2]: #see if the new parameter is within range
                             break
                 else:
                     list_of_integers = list(range(self.params[j][1], self.params[j][2]))
                     list_of_integers.append(self.params[j][2])
                     while True:
                         sol_ += random.choice([1, -1])
                         if sol_ in list_of_integers:
                             break
            self.sol_[j]=sol_
    def cost_function(self, G, embedding):
        error = 0
        net_start = [(0,0)]
        net_end = [(0,0)]
        Q_params = {}
        anneal_params = {}
        for i in self.sol_.keys():
            if i in list_of_qubo_params:
                Q_params[i] = self.sol_[i]
            elif i in list_of_anneal_params:
                anneal_params[i] = self.sol_[i]
        anneal_params['num_reads'] = np.floor(1000000/anneal_params['annealing_time'])
        if anneal_params['num_reads'] > 10000:
            anneal_params['num_reads'] = 10000
        for i in range(self.graph_size): #we try self.graph_size number of sets of start and end nodes
            while net_start == net_end:
                net_start = random.choice(list(G.nodes))
                net_end = random.choice(list(G.nodes))
            Q=create_qubo(G, [net_start], [net_end], Q_params)
            fixed_sampler = FixedEmbeddingComposite(
            DWaveSampler(solver={'lower_noise': True, 'qpu': True}), embedding
            )
            q_response = optimize_qannealer(fixed_sampler, Q, anneal_params)
            error += is_this_an_answer(q_response.samples()[0], G, net_start, net_end)#a function to compare the best_q_answer vs the correct answer
            self.cost_ = error 
    def accept_prob(self,c_old,c_new):
        """Computes the acceptance probability.
        Returns
        -------
        self : object
        """
        if -(c_new-c_old)/self.T > 0:
            ap = 1.1 # to deal with
        else:
            ap = np.exp(-(c_new-c_old)/self.T)
        return ap
    def anneal(self):
	###########
        G = RectGridGraph(self.graph_size, self.graph_size) #create the graph only once
        net_start = [(0,0)]
        net_end = [(0,0)]
        Q=create_qubo(G, net_start, net_end) #in order to produce the embedding we will run this once
        dwave_sampler = DWaveSampler(solver={'lower_noise': True, 'qpu': True})
        A = dwave_sampler.edgelist
        embedding, _ = find_embedding_minorminer(Q, A) #create the embedding only once
	###########
        self.cost_function(G, embedding)
        best_sol = self.sol_
        cost_old = self.cost_
        self.costs = [cost_old]
        self.sols = [best_sol]
        while self.T > self.T_min:
            ##
            print(self.T)
            ##
            i = 1
            while i <= self.max_iter:
                self.param_generator()
                self.cost_function(G, embedding)
                cost_new = self.cost_
                ap = self.accept_prob(cost_old, cost_new)
                if ap > random.random():
                    best_sol = self.sol_
                    cost_old = cost_new
                else:
                    self.cost_ = cost_old
                    self.sol_ = best_sol
                i += 1
            self.costs.append(self.cost_)
            self.sols.append(self.sol_)
            self.T = self.T*self.alpha
