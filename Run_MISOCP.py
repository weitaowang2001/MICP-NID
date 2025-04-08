'''
k_d refers to k and density. k is used for sparse and complete graphs (in R code) Note that |E|/(v)=s   2s=k
d is used for dense (in R code) d= |E|/(v*(v-1)). 
'''

                 ############################################################################
                 ################### LIBRARIES ############################################
                 ############################################################################                 

import gurobipy
import math
import OPT_LNConic_L0            # Layered Network for all graph types without valid inequalities
#change here
#import OPT_LN_L0
#import test
#import test1
#import test3

import timeit
import os
#import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import networkx
from multiprocessing import Pool
import multiprocessing
import os
#import OPT_M_L1
#import M
import cvxpy as cp
       ############################################################################
       ############################################################################
       ############################################################################



    ############################################################################
    ############################################################################
    ############################################################################
    
Input =[]
#for Opt_model in [OPT_LN_L1.Optimization, OPT_TO_L1.Optimization, OPT_CP_L1.Optimization, OPT_Linear_L1.Optimization]:

# m = [6,8,9,15,14,16,18,20,27,27]
m = [6,8,9,15,14,16,18,20,27,27,56,70]
datasets = ['1dsep','2asia','3bowling','4insuranceSmall','5rain','6cloud','7funnel','8galaxy','9insurance',
            '10factors', '11hfinder', '12hepar']
# datasets = ['1dsep']
# dt= 5#change here
# Dataset = "5rain"#change here

results = []
for i in range(10, 11):
    dt = i+1
    Dataset = datasets[i]
    for lamda in [12*np.log(m[i])]:  # 5 log(m) it used to be 6.214*2 in the old paper
        for k in range(1, 11):
            for estG in ["corest"]:  # true, corest
                result_i = OPT_LNConic_L0.Optimization([Dataset, estG, k, lamda])
                print(result_i)
                results.append(result_i)
results_df = pd.DataFrame(results, columns=['network', 'p', 'n', 'k', 'RGAP', 'Time', 'd_cpdag', 'SHDs', 'TPR', 'FPR'])
results_df.to_csv('./experiment results/comparison to other benchmarks/MISOCP_est_12log(m)n_11_30.csv', index=False, header=True)

"""
for m in [10,20,30,40]:
	for n in [100,1000]:
		for lamda in [0.1,1]:
			for graph_type in ['Sparse','Complete']: 
				for Dataset in ['Dataset1']:
					for k in [2]:
						Input.append(m)
						Input.append(n)
						Input.append(lamda)
						Input.append(graph_type)
						Input.append(Dataset)
						Input.append(k)
						for Opt_model in [OPT_LN_L1.Optimization, OPT_TO_L1.Optimization, OPT_CP_L1.Optimization, OPT_Linear_L1.Optimization]:
							Opt_model(Input)
						
		"""	 


