'''
Date: 2023/6/19
Author: Tong Xu
Mixed integer convex programming with perspective strengthening + outer approximation.
'''

# import packages
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


def read_data(name, kk):
    """
    Read the real world datasets with dataset name and type of errors

    parameters:

        name: name of the dataset

        kk: kk-th dataset
    """
    file_path = './Data/RealWorldDatasetsTXu-NonGaussian_30/'
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


def optimization(input):
    dataset = input[0]
    est = input[1]
    kk = input[2]

    # l = 0.08
    data, True_B, moral, mgest = read_data(dataset, kk)
    n, p = data.shape
    l = 12 * np.log(p) / n  # sparsity penalty parameter
    # l = np.log(n) / n/2
    True_B_mat = ind2mat(True_B.values, p)
    E = [(i, j) for i in range(p) for j in range(p) if i != j]  # off diagonal edge sets
    list_edges = []
    if est == 'true':
        for edge in moral.values:
            list_edges.append((edge[0]-1, edge[1]-1))
            list_edges.append((edge[1]-1, edge[0]-1))
    else:
        for edge in mat2ind(mgest, p):
            list_edges.append((edge[0], edge[1]))
            # list_edges.append((edge[1], edge[0]))

    ## Moral Graph
    G_moral = nx.Graph()
    for i in range(p):
        G_moral.add_node(i)
    G_moral.add_edges_from(list_edges)

    non_edges = list(set(E) - set(list_edges))
    # Sigma_hat = data.values.T @ data.values / n
    Sigma_hat = np.cov(data.values.T)
    ############################## Find Delta and Mu ########################################
    #########################################################################################

    # Find the smallest possible \mu such that Sigma_hat + \mu I be PD and stable.
    min_eig = np.min(scipy.linalg.eigh(Sigma_hat, eigvals_only=True))
    if min_eig < 0:
        pmu = np.abs(min_eig)  # due to numerical instability. This is the minimum value for \mu.
    else:
        pmu = 0

    # Find delta using SDP
    Lam = cp.Variable(p)
    # The operator >> denotes matrix inequality.

    constraints = [Sigma_hat + pmu * np.identity(p) - cp.diag(Lam) >> 0] + [Lam[i] >= 0 for i in range(p)]

    prob = cp.Problem(cp.Maximize(cp.sum(Lam)), constraints)
    prob.solve(solver=cp.CVXOPT)

    # Print results
    Delta = Lam.value
    Delta[Delta < 0] = 0  # Due to possible numerical instability

    ##########################################################################################
    ################################# PARAMETERS OF THE MODEL ################################
    ##########################################################################################

    m = gp.Model()
    # Create variables
    # Continuous variables

    S_var = {}
    for j, k in list_edges:
        S_var[j, k] = m.addVar(vtype=GRB.CONTINUOUS, name="s_%s_%s" % (j, k))
    for i in range(p):
        S_var[i, i] = m.addVar(vtype=GRB.CONTINUOUS, name="s_%s_%s" % (i, i))
    Gamma = {}
    for i in range(p):
        for j in range(p):
            if i == j:
                Gamma[i, j] = m.addVar(lb=1e-5, vtype=GRB.CONTINUOUS, name="Gamma%s%s" % (i, j))
            else:
                Gamma[i, j] = m.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="Gamma%s%s" % (i, j))


    # Gamma = m.addMVar((p, p), ub=M, vtype=GRB.CONTINUOUS, name="Gamma")
    psi = m.addMVar((p, 1), lb=1, ub=p, vtype=GRB.CONTINUOUS, name='psi')

    # Integer variables
    # g = m.addMVar((p, p), vtype=GRB.BINARY, name='g')
    g = {}
    for i in range(p):
        for j in range(p):
            g[i, j] = m.addVar(vtype=GRB.BINARY, name="g%s%s" % (i, j))
    # z = m.addMVar((p, p), vtype=GRB.BINARY, name='z')

    # Variables for outer approximation
    T = {}
    for i in range(p):
        T[i] = m.addVar(lb=-10, ub=100, vtype=GRB.CONTINUOUS, name="T%s" % i)
        # This gives Gamma[i,i] a range about [0.0001, 100]

    m._T = T
    m._Gamma = Gamma
    m._g = g

    # define the callback function
    def logarithmic_callback(model, where):
        if where == gp.GRB.Callback.MIPSOL:
            # Get the value of Gamma
            Gamma_val = model.cbGetSolution(model._Gamma)
            for i in range(p):
                model.cbLazy(model._T[i] >= -2 * np.log(Gamma_val[i, i]) - 2 / Gamma_val[i, i] * (
                        model._Gamma[i, i] - Gamma_val[i, i]))

    Q = Sigma_hat - np.diag(Delta) + pmu * np.identity(p)
    min_eig = np.min(scipy.linalg.eigh(Q, eigvals_only=True))
    if min_eig <= 0:
        epsilon = np.abs(min_eig) + 0.0000001  # due to numerical instability. We epsilon = min_eig + 0.0000001
    else:
        epsilon = 0

    D = np.diag(Delta) - pmu * np.identity(p) - epsilon * np.identity(p)
    Q = Sigma_hat - D
    q = np.linalg.cholesky(Q)
    q = q.T


    # Set objective

    # log_term = gp.LinExpr()
    # for i in range(p):
    #     log_term += -2*t[i]
    log_term = 0
    for i in range(p):
        log_term += T[i]

    trace = gp.QuadExpr()
    for k in range(p):
        for i in range(p):
            for j in range(p):
                trace += Gamma[j, k] * Gamma[j, i] * Q[i, k]

    perspective_bigM = gp.LinExpr()
    perspective = gp.LinExpr()
    for i in range(p):
        for j in range(p):
            perspective_bigM += Gamma[j, i]*Gamma[j, i]*D[i, i]
    # for j in range(p):
    #     perspective_bigM += D[j, j]*gp.quicksum(Gamma[k, j]*Gamma[k, j] for k in G_moral.neighbors(j))
    #     perspective_bigM += D[j, j]*Gamma[j, j]*Gamma[j, j]
    for j in range(p):
        perspective += D[j, j]*gp.quicksum(S_var[k, j] for k in G_moral.neighbors(j))
        perspective += D[j, j]*S_var[j, j]

    penalty = gp.LinExpr()
    for i, j in E:
        penalty += l*g[i, j]


    m.setObjective(log_term + trace + perspective_bigM + penalty, GRB.MINIMIZE)

    # solve the problem without constraints to get big_M
    m.Params.lazyConstraints = 1
    m.Params.OutputFlag = 0
    m.optimize(logarithmic_callback)
    Big_M_obj = m.ObjVal
    m.update()

    big_M = 0
    for j, k in list_edges:
        big_M = max(big_M, abs(Gamma[j, k].x))

    M = 2*big_M
    # M = 20

    m.setObjective(log_term + trace + perspective + penalty, GRB.MINIMIZE)

    # log
    # t = m.addMVar((p, 1), vtype='c', lb=-gp.GRB.INFINITY, name='t')

    m.addConstrs(Gamma[i, i] <= M for i in range(p))
    m.addConstrs(Gamma[j, k] <= M*g[j, k] for j, k in list_edges)
    m.addConstrs(Gamma[j, k] >= -M*g[j, k] for j, k in list_edges)
    m.addConstrs(1-p+p*g[j, k] <= psi[k] - psi[j] for j, k in list_edges)


    # # If we use OA, we do not need this general constraint.
    # for i in range(p):
    #     m.addGenConstrLog(Gamma[i, i], t[i])
    # m.Params.FuncPieces = -1

    # if err == 'equal':
    # m.addConstrs(Gamma[j, j] == 1 for j in range(p))  ############################### Just for test!!!!!!!!!!!!

    m.addConstrs(Gamma[j, k] == 0 for j, k in non_edges)  # Use moral structure
    m.addConstrs(g[j, k] == 0 for j, k in non_edges)  # Use moral structure
    m.update()

    # Conic constraints
    for k, j in list_edges:
        m.addConstr(S_var[k, j]*g[k, j] >= Gamma[k, j]*Gamma[k, j])
        m.addConstr(S_var[k, j] <= M*M*g[k, j])
    for i in range(p):
        m.addConstr(S_var[i, i] >= Gamma[i, i]*Gamma[i, i])
        m.addConstr(S_var[i, i] <= M * M)




    # Solve
    m.Params.TimeLimit = 50*p
    m.Params.lazyConstraints = 1
    m.Params.OutputFlag = 1
    start = timeit.default_timer()
    m.optimize(logarithmic_callback)
    end = timeit.default_timer()

    # Extract solutions
    Gamma_ij = [var.X for var in m.getVars() if "Gamma" in var.VarName]
    t_ij = np.array([var.X for var in m.getVars() if "t" in var.VarName])
    g_ij = np.array([var.X for var in m.getVars() if "g" in var.VarName])
    # z_ij = np.array([var.X for var in m.getVars() if "z" in var.VarName])
    Gamma_opt = np.reshape(Gamma_ij, (p, p))
    g_opt = np.reshape(g_ij, (p, p))
    # z_opt = np.reshape(z_ij, (p, p))
    # t_opt = np.reshape(t_ij, (p, 1))

    D_half = np.diag(np.diag(Gamma_opt))
    B = np.eye(p) - np.linalg.inv(D_half)@Gamma_opt.T
    # np.savetxt('.\MISOCP_results\MISOCP_%s_%s_%s.csv' % (dataset, est, np.round(l, 3)), B, fmt="%.18f", delimiter=",")
    B_arcs = [[0 if np.abs(B[i][j]) <= 1e-6 or i == j else 1 for j in range(p)] for i in range(p)]

    # create a dag for computing shd for cpdag
    true_dag = cd.DAG.from_amat(np.array(True_B_mat))
    true_cpdag = true_dag.cpdag().to_amat()
    estimated_dag = cd.DAG.from_amat(np.array(B_arcs))
    estimated_cpdag = estimated_dag.cpdag().to_amat()
    SHD_cpdag = np.sum(np.abs(estimated_cpdag[0] - true_cpdag[0]))

    # SHD = compute_SHD(B_arcs, True_B_mat)
    skeleton_estimated, skeleton_true = skeleton(B_arcs), skeleton(True_B_mat)
    SHDs = compute_SHD(skeleton_estimated, skeleton_true, True)
    TPR, FPR = performance(skeleton_estimated, skeleton_true)
    run_time = end - start
    RGAP = m.MIPGAP
    return RGAP, SHD_cpdag, SHDs, TPR, FPR, run_time


if __name__ == '__main__':
    # print(optimization(['9insurance', 'corest']))
    # print(optimization(['10factors', 'true']))
    results = []
    # '1dsep', '2asia', '3bowling', '4insuranceSmall', '5rain', '6cloud', '7funnel', '8galaxy', '9insurance', '10factors', '11hfinder', '12hepar'
    for dataset in ['12hepar']:
        for kk in range(21,24):
            results_i = optimization([dataset, 'true', kk])
            print([dataset, kk] + list(results_i))
            results.append([dataset, kk] + list(results_i))
        print(pd.DataFrame(results))
        df = pd.DataFrame(results, columns=['network', 'k', 'RGAP', 'd_cpdag', 'SHD', 'TPR', 'FPR', 'Time'])
        df.to_csv('./experiment results/comparison to other benchmarks with non-Gaussian errors/1-30/MICP_NID_results_non-Gaussian_1_30.csv', index=False)
        # print(optimization(['4insuranceSmall', 'corest']))

