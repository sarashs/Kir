from essentials import *

a=SA(3, params={'weight_objective': [1, 0, 2, 0], 'weight_end': [1, 0, 2, 0],
                 'weight_start': [1, 0, 2, 0] ,'weight_others': [1, 0, 2, 0],
                 'weight_and': [6, 4, 15, 0],
                 'chain_strength': [7, 3, 12, 0], 'annealing_time': [99, 99, 99, 1]},
                 T=1, T_min=0.00001, alpha=0.8, max_iter=5)

a.anneal()
print(a.costs)
