'''
Date: 2023/8/8
Author: Tong Xu
Copied from MICP_NID_early_stopping.py. Modified to choose the lambda of synthetic datasets.

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
import matplotlib.pyplot as plt

from functions import *


def read_early_stopping(m, n, k):
    file_path = 'E:/Northwestern/Research/independent study 1/dag/gurobi/MIPDAGextentions/SyntheticDataNID/'
    file_name = "data_m_{}_n_{}_iter_{}.csv".format(m, n, k)
    data = pd.read_csv(file_path + file_name, header=None)
    True_B = pd.read_table(file_path + "DAG_{}.txt".format(m), delimiter=" ", header=None)
    moral = pd.read_table(file_path + "Moral_DAG_{}.txt".format(m), delimiter=" ", header=None)
    return data, True_B, moral


def optimization(input):
    mm = input[0]
    nn = input[1]
    kk = input[2]
    tau = input[3]
    data, True_B, moral = read_early_stopping(mm, nn, kk)
    n, p = data.shape
    # l = np.log(n)/n  # sparsity penalty parameter
    # l = np.log(p)/n*2
    l = input[4]
    True_B_mat = ind2mat(True_B.values, p)
    E = [(i, j) for i in range(p) for j in range(p) if i != j]  # off diagonal edge sets
    list_edges = []
    for edge in moral.values:
        list_edges.append((edge[0]-1, edge[1]-1))
        list_edges.append((edge[1]-1, edge[0]-1))

    ## Moral Graph
    G_moral = nx.Graph()
    for i in range(p):
        G_moral.add_node(i)
    G_moral.add_edges_from(list_edges)

    non_edges = list(set(E) - set(list_edges))
    Sigma_hat = data.values.T @ data.values / n

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
        for j in range(p):
            for i in range(p):
                trace += Gamma[i, k]*Gamma[i, j]*Q[j, k]

    perspective_bigM = gp.LinExpr()
    perspective = gp.LinExpr()
    for j in range(p):
        for i in range(p):
            perspective_bigM += Gamma[i, j]*Gamma[i, j]*D[j, j]
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
    m.Params.TimeLimit = 20
    m.Params.lazyConstraints = 1
    m.Params.OutputFlag = 1
    if tau == 'early':
        m.Params.MIPGapAbs = p*p/n
    elif tau == '0':
        m.Params.MIPGapAbs = 0
    else:
        print("Wrong setting.")
    start = timeit.default_timer()
    m.optimize(logarithmic_callback)
    end = timeit.default_timer()

    # Extract solutions
    Gamma_ij = [var.X for var in m.getVars() if "Gamma" in var.VarName]
    t_ij = np.array([var.X for var in m.getVars() if "t" in var.VarName])
    g_ij = np.array([var.X for var in m.getVars() if "g" in var.VarName])
    # z_ij = np.array([var.X for var in m.getVars() if "z" in var.VarName])
    Gamma_opt = np.reshape(Gamma_ij, (p, p)).T
    Gamma_opt[np.abs(Gamma_opt) <= 1e-6] = 0
    g_opt = np.reshape(g_ij, (p, p))
    # z_opt = np.reshape(z_ij, (p, p))
    # t_opt = np.reshape(t_ij, (p, 1))

    D_half = np.diag(np.diag(Gamma_opt))
    B = np.eye(p) - np.linalg.inv(D_half)@Gamma_opt
    file_path = 'E:/Northwestern/Research/independent study 1/dag/gurobi/MIPDAGextentions/SyntheticDataNID/'
    # np.savetxt(file_path+'MISOCP_%s_%s_%s_%s.csv' % (mm, nn, kk, np.round(l,3)), B, fmt="%.18f", delimiter=",")
    B_arcs = [[0 if np.abs(B[i][j]) <= 1e-6 or i == j else 1 for j in range(p)] for i in range(p)]
    # for i in range(p):
    #     for j in range(p):
    #         if B_arcs[i][j] != 0 and B_arcs[j][i] != 0:
    #             if np.abs(B[i][j]) > np.abs(B[j][i]):
    #                 B_arcs[j][i] = 0
    #             else:
    #                 B_arcs[i][j] = 0



    # # create a dag for computing shd for cpdag
    # true_dag = cd.DAG.from_amat(np.array(True_B_mat))
    # true_cpdag = true_dag.cpdag().to_amat()
    # estimated_dag = cd.DAG.from_amat(np.array(B_arcs))
    # estimated_cpdag = estimated_dag.cpdag().to_amat()
    # SHD_cpdag = np.sum(np.abs(estimated_cpdag[0] - true_cpdag[0]))
    # # SHD = compute_SHD(B_arcs, True_B_mat)
    # skeleton_estimated, skeleton_true = skeleton(B_arcs), skeleton(True_B_mat)
    # SHDs = compute_SHD(skeleton_estimated, skeleton_true, True)
    # TPR, FPR = performance(skeleton_estimated, skeleton_true)
    # run_time = end - start
    # RGAP = m.MIPGAP
    # GAP = m.ObjVal - m.objBound
    # return GAP, RGAP, SHD_cpdag, SHDs, TPR, FPR, run_time

    Ln = log_term + trace + perspective
    k = np.sum(g_opt + p)
    bic = k * np.log(n) + Ln.getValue() * n
    return bic


if __name__ == '__main__':
    # print(optimization([20, 100, 1, "0"]))
    m = [10]
    n = 100
    k = 1

    bics = []
    smallest_bic = np.inf
    best_ll = np.inf

    lambdas = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15])*np.log(10)/n
    # lambdas = [0.018, 0.105]
    for ll in lambdas:
        bic_ll = optimization([10, n, k, "0", ll])
        bics.append(bic_ll)
        if bic_ll < smallest_bic:
            smallest_bic = bic_ll
            best_ll = ll
    print(best_ll, best_ll/(np.log(10)/n))
    f = plt.figure()
    plt.plot(np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]), bics, color="green", marker='s', markerfacecolor='none')
    plt.xlabel("$\lambda = i * \\log(m)/n$")
    plt.ylabel("BIC")
    plt.show()

