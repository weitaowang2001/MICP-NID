import time

import gurobipy as gp
from gurobipy import GRB
import timeit
import networkx as nx
import random
import numpy as np
import pandas as pd
import os

import causaldag as cd
import scipy
import cvxpy as cp

from functions import *
from causaldag import rand, partial_correlation_suffstat, partial_correlation_test, MemoizedCI_Tester, gsp


def read_data(name, kk):
    """
    Read the real world datasets with dataset name and type of errors

    parameters:

        name: name of the dataset

        kk: kk-th dataset
    """
    file_path = './Data/RealWorldDatasetsTXu-NonGaussian/'
    file_path += name + '/'
    file_name = glob.glob(file_path + f"/*n_500_iter_{kk}.csv")
    data = pd.read_csv(file_name[0], header=None)

    graph_name = [i for i in os.listdir(
        file_path) if os.path.isfile(os.path.join(file_path, i)) and 'Sparse_Original_edges' in i][0]
    True_B = pd.read_table(file_path + graph_name, header=None, sep=',')

    moral_graph_name = [i for i in os.listdir(
        file_path) if os.path.isfile(os.path.join(file_path, i)) and 'Sparse_Moral_edges' in i][0]
    True_moral = pd.read_table(file_path + moral_graph_name, header=None, sep=',')

    mgest_name = glob.glob(file_path + f"/*glasso_iter_{kk}.txt")
    mgest = pd.read_table(mgest_name[0], header=None, sep=',')
    return data, True_B, True_moral, mgest


if __name__ == '__main__':
    results = []
    # '1dsep', '2asia', '3bowling', '4insuranceSmall', '5rain', '6cloud', '7funnel', '8galaxy', '9insurance', '10factors', '11hfinder', '12hepar'
    for dataset in ['1dsep', '2asia', '3bowling', '4insuranceSmall', '5rain', '6cloud', '7funnel', '8galaxy', '9insurance', '10factors', '11hfinder', '12hepar']:
        for kk in range(1,11):
            data, True_B, moral, mgest = read_data(dataset, kk)
            nnodes, p = data.shape
            nodes = set(range(nnodes))
            suffstat = partial_correlation_suffstat(data)
            ci_tester = MemoizedCI_Tester(partial_correlation_test, suffstat, alpha=1e-3)

            possible_edges_true = tuple(zip(moral.values[:,0]-1, moral.values[:,1]-1))
            fixed_gaps_true = set()
            for i in range(nnodes):
                for j in range(nnodes):
                    if (i, j) not in possible_edges_true:
                        fixed_gaps_true.add((i, j))
            indices = np.where(mgest != 0)
            possible_edges_est = tuple(zip(indices[0], indices[1]))
            fixed_gaps_est = set()
            for i in range(nnodes):
                for j in range(nnodes):
                    if (i, j) not in possible_edges_est:
                        fixed_gaps_est.add((i, j))
            start_i = time.time()
            est_dag = gsp(nodes, ci_tester, fixed_gaps=fixed_gaps_true)
            end_i = time.time()
            # RGAP, SHD_cpdag, SHDs, TPR, FPR, run_time
            True_B_mat = ind2mat(True_B.values, p)
            true_dag = cd.DAG.from_amat(np.array(True_B_mat))
            true_cpdag = true_dag.cpdag().to_amat()
            B_arcs = est_dag.to_amat()[0]
            estimated_dag = cd.DAG.from_amat(np.array(B_arcs))
            estimated_cpdag = estimated_dag.cpdag().to_amat()
            SHD_cpdag = np.sum(np.abs(estimated_cpdag[0] - true_cpdag[0]))
            skeleton_estimated, skeleton_true = skeleton(B_arcs), skeleton(True_B_mat)
            TPR, FPR = performance(skeleton_estimated, skeleton_true)
            results_i = [dataset, kk, SHD_cpdag, TPR, FPR, end_i-start_i]
            print(results_i)
            results.append(results_i)
        print(pd.DataFrame(results))
        df = pd.DataFrame(results, columns=['network', 'k', 'd_cpdag', 'TPR', 'FPR', 'Time'])
        df.to_csv('GSP_results_non-Gaussian.csv', index=False)