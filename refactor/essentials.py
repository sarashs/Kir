#import the necessary shit
import matplotlib.pyplot as plt
import numpy as np
import random
from copy import deepcopy
from itertools import combinations_with_replacement as cwr
import pickle

import dwave_networkx as dnx
import networkx as nx
from dwave.system.samplers import DWaveSampler
from dwave.system.composites import FixedEmbeddingComposite
import hybrid
import dimod

#for garbace collection
import gc

from qpr.quantum_utils import find_embedding_minorminer, get_all_min_energy
from qpr.notebook_utils import make_ax_grid

####
def save_data(data_dict,name):
    pickle_out = open(name+".pickle","wb")
    pickle.dump(data_dict, pickle_out)
    pickle_out.close()

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
                #Q[(f'y{i}',f'y{j}')]+=-1*w1 
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
                Q[(f'y{i}',f'y{j}')]+=6*w_and #2
                if (f'w{i}{j}',f'y{i}') not in Q:
                    Q[(f'w{i}{j}',f'y{i}')]=0
                Q[(f'w{i}{j}',f'y{i}')]+=-12*w_and
                if (f'w{i}{j}',f'y{j}') not in Q:
                    Q[(f'w{i}{j}',f'y{j}')]=0
                Q[(f'w{i}{j}',f'y{j}')]+=-12*w_and
                if (f'w{i}{j}',f'w{i}{j}') not in Q:
                    Q[(f'w{i}{j}',f'w{i}{j}')]=0
                Q[(f'w{i}{j}',f'w{i}{j}')]+=18*w_and

                
    #constraint on the starting node
    w4=params['weight_start']
    # Starting_node is a list of numerical labels and not a dict
    for i,j in cwr(Starting_node,2):
        if i==j:
            #######Removing starting node edges from the objective
            #Q[(f'y{i}',f'y{j}')]+=-1*w1 
            ##############################
            Q[(f'y{i}',f'y{j}')]+=-(2*n-1)*w4
        else:
            Q[(f'y{i}',f'y{j}')]+=2*w4
    return Q

#####


#####

def optimize_qannealer(sampler, Q, params={'chain_strength': 7, 'annealing_time': 99, 'num_reads': 10000, 'anneal_schedule': None}):
    if params['anneal_schedule'] is None:
        response = sampler.sample_qubo(
                   Q, chain_strength=params['chain_strength'], annealing_time=params['annealing_time'], auto_scale=True, num_reads=params['num_reads']
        )
    else:
        response = sampler.sample_qubo(
                   Q, chain_strength=params['chain_strength'], anneal_schedule=params['anneal_schedule'], auto_scale=True, num_reads=params['num_reads']
        )
    return response

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
    path=[]
    out = 0
    no_chain_break_flag = True
    for j in range(len(net_end)):
        while no_chain_break_flag:
            no_chain_break_flag = False
            for item in edge_set:
                if edge_end == item[0]:
                    no_chain_break_flag = True
                    edge_end = item[1]
                    path.append(item)
                    break
            if (edge_end == net_end[j]) and (len(path) == len(edge_set)):
                out += 1
                break
            elif edge_end == net_end[j]:
                out += 0.5
                break
    return out

#####

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
        format: params= {'param_#':[init_value, min, max, integer_flag, parameter_change_weight],...}
        format params for anneal schedule: [init_schedule(list), min_max_list, parameter_change_weight_list=[S_weight, time_weight]]
    
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
    list_of_qubo_params = [
        'weight_objective', 'weight_end', 'weight_start', 'weight_others',
        'weight_and'
    ]
    list_of_anneal_params = ['num_reads', 'annealing_time', 'chain_strength', 'anneal_schedule'] 

    def __init__(self, graph_size, params={'weight_objective': [1, 0, 2, 0, 0.1], 'weight_end': [1, 0, 2, 0, 0.1],
                                           'weight_start': [1, 0, 2, 0, 0.1] ,'weight_others': [1, 0, 2, 0, 0.1],
                                           'weight_and': [6, 4, 15, 0, 0.1],
                                           'chain_strength': [7, 4, 15, 0, 0.1], 'annealing_time': [99, 10, 10000, 1, 10], 
                                           'anneal_schedule': None
                                          },
                 T=1, T_min=0.00001, alpha=0.9, max_iter=50, embedding_pickle=None
                ):
        assert ((params['anneal_schedule'] is None) or (params['annealing_time'] is None)),"anneal schedule or time? pick one!"
        self.graph_size = graph_size
        self.params = params
        self.T = T 
        self.T_min = T_min
        self.alpha = alpha
        self.max_iter = max_iter
        self.cost_= 0
        self.sol_= {}
        if embedding_pickle is None:
            self.embedding = None
        else:
            with open(embedding_pickle, 'rb') as f:
                pickle_data = pickle.load(f)
            self.embedding, _, _, _ = pickle_data[:4]
        for key, val in params.items():
            if val is None:
                self.sol_[key] = None
            else:
                self.sol_[key] = val[0]
        self.sols = None
        self.costs = None
        self.energies = {}
        self.sampler = None
    def param_generator(self):
        """Generates the next solution.
        Returns
        -------
        self : object
        """
        for j in self.sol_.keys():
            sol_ = deepcopy(self.sol_[j])
            if sol_ is None:
                pass
            else:
                if j in ['anneal_schedule']: #[[0.0, 0.0],[time_1, b], [time_3, 1.0]], [[[0, 0], [0, 0]],[[0.0, 100],[0.0,0.8]], [[80, 999],[0.8,1.0]], [[1, 1],[1000, 1100]], [0.1, 1]]
                    #params[j][0] current params list, params[j][1] min_max list, params[j][2] param change weight list
                    schedule_length = len(self.params[j][0])
                    for i in range(1, schedule_length):
                        #For the S part
                        if self.params[j][1][i][1][0] == self.params[j][1][i][1][1]:
                            sol_[i][1] = self.params[j][1][i][1][0]
                        else:
                            while True:
                                sol_[i][1] = self.sol_[j][i][1]
                                sol_[i][1] += (0.5-random.random()) * self.params[j][2][1] # new parameter between -0.5,0.5
                                if sol_[i][1] >= self.params[j][1][i][1][0] and sol_[i][1] <= self.params[j][1][i][1][1] and sol_[i][1] >= sol_[i-1][1]: #see if the new parameter is within range
                                    break                        
                      
                            #For the time part
                        if self.params[j][1][i][0][0] == self.params[j][1][i][0][1]:
                            sol_[i][0] = self.params[j][1][i][0][0]
                        else:
                            list_of_integers = list(range(self.params[j][1][i][0][0], self.params[j][1][i][0][1]))
                            list_of_integers.append(self.params[j][1][i][0][1])
                            while True:
                                sol_[i][0] = self.sol_[j][i][0]
                                sol_[i][0] += random.choice([1, -1])*self.params[j][2][0]
                                if sol_[i][0] in list_of_integers and sol_[i][0] > sol_[i-1][0] :
                                    break
                    self.sol_[j] = deepcopy(sol_)              
                else:
                    if self.params[j][1]==self.params[j][2]: # in order to fix a certain parameter
                        sol_ = self.params[j][1]
                    else:
                        if self.params[j][3] == 0: # if the parameter isn't an integer
                            while True:
                                sol_ = self.sol_[j]
                                sol_ += (0.5-random.random())*self.params[j][4] # new parameter between -0.5,0.5
                                if sol_ > self.params[j][1] and sol_ < self.params[j][2]: #see if the new parameter is within range
                                    break
                        else:
                            list_of_integers = list(range(self.params[j][1], self.params[j][2]))
                            list_of_integers.append(self.params[j][2])
                            while True:
                                sol_ = self.sol_[j]
                                sol_ += random.choice([1, -1])*self.params[j][4]
                                if sol_ in list_of_integers:
                                    break
                    self.sol_[j] = sol_
            
    def global_sampler(self):
        fixed_sampler = FixedEmbeddingComposite(
            DWaveSampler(solver={'lower_noise': True, 'qpu': True}), self.embedding
            )
        return fixed_sampler
    def cost_function(self, G, fixed_sampler):
        error = 0
        net_start = [(0,0)]
        net_end = [(0,0)]
        Q_params = {}
        anneal_params = {}
        for i in self.sol_.keys():
            if i in self.list_of_qubo_params:
                Q_params[i] = self.sol_[i]
            elif i in self.list_of_anneal_params:
                anneal_params[i] = self.sol_[i]
        if anneal_params['annealing_time'] is None:
            anneal_params['num_reads'] = int(np.floor(999000/anneal_params['anneal_schedule'][-1][0]))
        else:
            anneal_params['num_reads'] = int(np.floor(999000/anneal_params['annealing_time']))
            #999000 due to Dwave's own bug
        if anneal_params['num_reads'] > 10000:
            anneal_params['num_reads'] = 10000
        #print([anneal_params['annealing_time'], anneal_params['num_reads']])
        for i in range(int(np.ceil(self.graph_size**2/2))): #we trying number of sets of start and end nodes
            net_start = random.choice(list(G.nodes))
            net_end = random.choice(list(G.nodes))
            while net_start == net_end:
                net_start = random.choice(list(G.nodes))
                net_end = random.choice(list(G.nodes))
            Q=create_qubo(G, [net_start], [net_end], Q_params)
            q_response = optimize_qannealer(fixed_sampler, Q, anneal_params)
            error -= is_this_an_answer(q_response.samples()[0], G, net_start, net_end)#a function to compare the best_q_answer vs the correct answer
        #print(error)
        self.cost_ = error
        ## memory improvement
        #garbages = gc.collect()
       
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
    
    def reset(self):
        """Restarts the annealing and discards the current solutions/costs collected"""
        self.costs = None    
    
    def anneal(self):
	###########Perform this only the first time object is instanciated
        G = RectGridGraph(self.graph_size, self.graph_size) #create the graph only once
        net_start = [(0,0)]
        net_end = [(0,0)]
        Q=create_qubo(G, net_start, net_end) #in order to produce the embedding we will run this once
        dwave_sampler = DWaveSampler(solver={'lower_noise': True, 'qpu': True})
        A = dwave_sampler.edgelist
        if self.embedding is None:
            self.embedding, _ = find_embedding_minorminer(Q, A) #create the embedding only once
        #define global sampler here
        fixed_sampler = self.global_sampler()
        if self.costs is None:
            self.cost_function(G, fixed_sampler)
            best_sol = deepcopy(self.sol_)
            cost_old = self.cost_
            self.costs = [cost_old]
            self.sols = [best_sol]
        else:
            cost_old = deepcopy(self.costs[-1])
            best_sol = deepcopy(self.sols[-1])
	###########
        while self.T > self.T_min:
            ## memory improvement
            #garbages = gc.collect()
            ##seems no longer tobe necessary
            print(self.T)
            ##
            i = 1
            while i <= self.max_iter:
                ## memory improvement
                #garbages = gc.collect()
                ######
                self.param_generator()
                self.cost_function(G, fixed_sampler)
                cost_new = self.cost_
                ap = self.accept_prob(cost_old, cost_new)
                if ap > random.random():
                    best_sol = deepcopy(self.sol_)
                    cost_old = cost_new
                    #print(best_sol)
                else:
                    self.cost_ = cost_old
                    self.sol_ = deepcopy(best_sol)
                i += 1
            self.costs.append(cost_old)
            self.sols.append(best_sol)
            self.T = self.T*self.alpha
