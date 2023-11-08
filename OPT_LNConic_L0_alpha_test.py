'''
MISOCP + LN formulation 

This is an MIP conic formulation model for learning Bayesian Networks

Written and maintained by Hasan Manzour
Artcile: Conic Programming for Learning Directed Acyclic Graphs from Continuous Data

Input 
- sample number n 
- number of nodes m  
- true graph 
- raw data of realization of random variables obtained from true graph
- raw data is generated in R
- structure of true graph 
- structure of moral graph
- Four types of data are considered in various levels
     - Three types of graph 1) Sparse 2) Complete 4) High dimentional 
     - Lambda is set as \ln(n) 
     = DIfferent level of \mu is used
     - Different level of size of nodes are considered 
     - Different level of sample size are considered 

Output:
- Given the matrix data as well as the structure of Moral graph, we predict the structure of the network 
- Hence, output is a predicted DAG 
- Note that optimal DAG is the DAG from which we generated the data (i.e., true DAG)
- We record all information including (IP solution and LP solution)
'''

'''
Important notes
There is a very important subtle point. We need g variables. This is L0 norm 
Big-M is solved in a heuristic way and is equvalent for all models for fair comparison 
The formulation itself does not require big-M. Yet, we add to tight the variable S
X^TX is always PD. Yet the diag(d) could be small. Thus, we add \mu in the sensitivity analysis. 
Otherwise, \mu is set to be 0. 
We add epsilonI to X^TX -D to make sure it is PD to get around numerical instability. 
We do not have this issue in our instances though
'''

# ----------------------------------- Libraries and Modules -----------------------------------
from gurobipy import *

#import gurobipy
import timeit
import os
import networkx as nx
from multiprocessing import Pool
import multiprocessing
import pathlib2
from pathlib2 import Path
import scipy

import math
from random import *
import numpy as np
import pandas as pd
from itertools import product
import time

import cvxpy as cp
import causaldag as cd
from numpy import linalg as LA

from functions import *


#-------------------------------------- Optimization code ------------------------------- 

def Optimization(Input):   # M: number of node N:number of samples 
    
    # M=Input[0]
    # N=Input[1]
    # LAMBDA=Input[2]
    # graph_type=Input[3]
    # Dataset=Input[4]
    # density=Input[5]
    # pmu=Input[6]             # positive mu
    # est = Input[7]##HHWU

    est = "true"
    pmu = 0
    M, N, alpha, kk = Input
    LAMBDA = 4*np.log(M)

    ##########################################################################################################
    ###################################################### INPUT #############################################
    ##########################################################################################################

# ----------------------- Constructing Original True DAG ----------------------------
    
    ## Read the directory of True (Original) DAG 
    # datasets_directory = "E:/Northwestern/Research/independent study 1/dag/gurobi/MIPDAGextentions/RealWorldDatasetsHHWu/NonEqVarData"
    # dataset_directory = datasets_directory + '/' + Dataset
    # filename= '%s_Original_edges_%s_%s_%s.txt' %(graph_type, M, N, density)
    # name = "%s/%s" % (dataset_directory, filename)
    # Original_edges = np.loadtxt(name, delimiter=",")    # Read the True DAG
    data, True_B, moral = read_alpha(M, N, alpha, kk)
    x = data.values
    Original_edges = True_B.values
    
    ## True (Original) Graph plot 
    List_Original_edges = []
    Pos = {}
    for i in range(len(Original_edges)):
        List_Original_edges.append((int(Original_edges[i][0]) - 1, int(Original_edges[i][1]) - 1))
    
    G_Original = nx.DiGraph()
    for i in range(M):
        G_Original.add_node(i)
    G_Original.add_edges_from(List_Original_edges)
    #plt.figure(1)
    #plt.subplot(222)
    #plt.title("True DAG")   # Plot the true DAG
    #Pos = nx.circular_layout(G_Original)
    #nx.draw(G_Original, pos=Pos, with_labels=True)

    # create a dag for computing shd for cpdag
    true_dag = cd.DAG.from_nx(G_Original)
    true_cpdag = true_dag.cpdag().to_amat()
    
    ## Read input raw data
    # filename = 'data_%s_n_500_iter_1.csv' % Dataset
    # data_directory = "%s/%s" % (dataset_directory, filename)
    # x = pd.read_csv(data_directory, header=None).values
    
    ## Normalize data
    #mean= np.mean(x, axis=0)
    #std = np.std(x, axis=0)
    #x = (x - mean)/std      
    
    ## Read moral graph structure
    if est == 'true':
        # filename = '%s_Moral_edges_%s_%s_%s.txt' %(graph_type, M, N, density)
        # name = "%s/%s" % (dataset_directory, filename)
        # M_edges = np.loadtxt(name, delimiter=",")
        M_edges = moral.values

        List_edges = []
        for i in range(len(M_edges)):
            List_edges.append((int(M_edges[i][0]) - 1, int(M_edges[i][1]) - 1))
            List_edges.append((int(M_edges[i][1]) - 1, int(M_edges[i][0]) - 1))
    else:
        filename = "mgest_PearsonCorEst.txt"
        # name = "%s/%s" % (dataset_directory, filename)
        # M_edges = np.loadtxt(name, delimiter=",")
        # List_edges = list(zip(*np.where(M_edges == 1)))



    ## Moral Graph
    G_moral = nx.Graph()
    ##HHWU
    for i in range(M):
        G_moral.add_node(i)
    ##HHWU
    G_moral.add_edges_from(List_edges)

    
    #plt.subplot(224)
    #plt.title("Moral DAG")
    #nx.draw(G_moral, pos=Pos, with_labels=True)
    
    ############################## Find Delta and Mu ########################################
    #########################################################################################
    #########################################################################################
    
    # Find the smallest possible \mu such that X^TX + \mu I be PD and stable.   
    if pmu <= 0.0001:
        Decom = np.matmul(x.T,x) 
        min_eig= np.min(scipy.linalg.eigh(Decom, eigvals_only=True))
        if min_eig<0:
            pmu = np.abs(min_eig)  # due to numerical instability. This is the minimum value for \mu.      
        else:
            pmu=0
        
    # Find delta using SDP  
    Lam = cp.Variable(M)
    # The operator >> denotes matrix inequality.
    
    constraints = [np.matmul(x.T,x) + pmu*np.identity(M) -cp.diag(Lam)  >> 0] + [Lam[i] >= 0 for i in range(M)]

    prob = cp.Problem(cp.Maximize(cp.sum(Lam)),constraints)
    prob.solve(solver=cp.CVXOPT)

    # Print results  
    Delta=Lam.value
    Delta[Delta < 0] = 0  # Due to possible numerical instability 
    
    ##########################################################################################
    ################################# PARAMETERS OF THE MODEL ################################
    ##########################################################################################
    
    LN_Solution={}           # Solution dic for Topological order
    
    # ------------------- Parameters of the model ------------------
    # m: Number of nodes/variables/features
    # n: Number of samples
    [n, v] = np.shape(x)
    
    Error=0
    if v != M or n!=N:
        Error= "there is an error"
       
    # Penalty coefficient 
    Lamda = LAMBDA
    
    # Index of all nodes
    J = range(v)
    # Index of all samples
    I = range(n)
     
    ##########################################################################################
    ################################# PARAMETERS OF THE MODEL ################################
    ##########################################################################################

    #-------------------------------------- Optimization code ------------------------------- 
    name = "LN_model %s" % Lamda
    m = Model(name)

    #Variables for conic model 
    S_var = {}
    for j, k in List_edges:
        S_var[j, k] = m.addVar(vtype=GRB.CONTINUOUS, name="s_%s_%s" % (j, k))
        
    # Variables 
    Beta = {}
    for j, k in List_edges:
        Beta[j, k] = m.addVar(lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="beta_%s_%s" % (j, k))
    
    Z = {}
    g={}
    for j, k in List_edges:
        g[j, k] = m.addVar(vtype=GRB.BINARY, name="g_%s_%s" % (j, k))
        if j <k: 
            Z[j, k] = m.addVar(vtype=GRB.BINARY, name="z_%s_%s" % (j, k))        
    gen={}
    for u in J:
        gen[u] = m.addVar(ub= v-1, vtype=GRB.CONTINUOUS, name="gen_%s" % (u))

    m.update()

    #Objective function
            
    # qq^T = Q = X^TX - delta + \muI
    Q = np.matmul(x.T, x) - np.diag(Delta) + pmu*np.identity(M)
    min_eig= np.min(scipy.linalg.eigh(Q, eigvals_only=True))
    if min_eig<=0:
        epsilon= np.abs(min_eig) + 0.0000001     # due to numerical instability. We epsilon = min_eig + 0.0000001 
    else: 
        epsilon = 0
            
    Q = np.matmul(x.T,x) - np.diag(Delta) + pmu*np.identity(M) + epsilon*np.identity(M)    
    q=np.linalg.cholesky(Q)  
    q=q.T
        
    # All objectives 
    Obj1 = LinExpr()
    Obj1 = np.trace(np.matmul(x.T,x)) 
    
    Coeff= {}
    for j in J:
        for k in J:
            Coeff[j,k] = np.matmul(x.T,x)[j,k] #sum(x[l,j]*x[l,k] for l in J)
            #Coeff[j,k] = sum(x[l,j]*x[l,k] for l in I)
    
    Obj2 = LinExpr()
    for j,k in List_edges: 
        Obj2+= (Beta[k,j]+Beta[j,k])*Coeff[j,k] 

    Obj3 = LinExpr()
    for j in J:
        Obj3+= (Delta[j])*quicksum(S_var[j,k] for k in G_moral.neighbors(j)) 
    
    Obj3_prime = LinExpr()
    for j in J:
        Obj3_prime+= (Delta[j])*quicksum(Beta[j,k]*Beta[j,k] for k in G_moral.neighbors(j)) 
        
    Obj4 = LinExpr()
    for i in J:     # i is not sample here. Note q is m by m 
        for j in J:
            Obj4+= (quicksum(Beta[l,j]*q[i,l] for l in G_moral.neighbors(j)))*(quicksum(Beta[l,j]*q[i,l] for l in G_moral.neighbors(j)))
    
    Penalty = LinExpr()
    for j, k in List_edges:
        Penalty += Lamda * (g[j, k])
                  
    Obj = LinExpr()
    Obj_big_M = Obj1-Obj2+ Obj3_prime+ Obj4
    Obj = Obj1-Obj2+ Obj3 + Obj4 + Penalty
    
    m.setObjective(Obj_big_M, GRB.MINIMIZE)
    m.update()
    
    # ---------  To find an upper bound for M: solve the problem with no constraint ---------
    m.optimize()
    Big_M_obj= m.ObjVal
    m.update()
        
    big_M = 0
    for j, k in List_edges:
        big_M = max(big_M, abs(Beta[j, k].x))
        
    big_M=2*big_M
        
    # --------- Real objective ------------
    m.setObjective(Obj, GRB.MINIMIZE)
    m.update()    

    #------- Constraints -----------
    
    #####Conic Constraints 
        
    CNum0 = {}
    for k, j in List_edges:    
        CNum0[k, j] = m.addQConstr(Beta[k,j]*Beta[k,j]<=S_var[k,j]*g[k,j], name="q_%s_%s" % (k, j))
    
    CNum1 = {}
    for k, j in List_edges:    
        CNum1[k, j] = m.addQConstr(S_var[k,j]<= big_M*big_M*g[k,j], name="q_%s_%s" % (k, j))

    CNum6 = {}
    for j, k in List_edges:
        lhs_1 = Beta[j, k] - big_M * g[j, k]
        CNum6[j, k] = m.addConstr(lhs_1 <= 0)     
        lhs_2 = Beta[j, k] + big_M * g[j, k]
        CNum6[j, k] = m.addConstr(lhs_2 >= 0)  
    
    CNum2 = {}
    for j, k in List_edges:
        CNum2[j,k] = None ##debug KeyError: (0, 0)
        if j <k: 
            lhs1 = g[j, k]
            CNum2[j, k] = m.addConstr(lhs1 <= Z[j,k])    
        else:
            lhs1 = g[j, k]
            CNum2[j, k] = m.addConstr(lhs1 <= 1- Z[k,j])    
                 
    # Topological ordering with continious variables. PLease pay attention that you have to write it on both sides 
    # because we reduce the number of variables 
    CNum3 = {}
    for j, k in List_edges:
        if j < k:
            lhs_3 = (v)* Z[j,k] + gen[j] - gen[k] 
            CNum3[j, k] = m.addConstr(lhs_3 <= v-1)
            
    #CNum3_prime = {}
    for j, k in List_edges:
        if j < k:
            lhs_3_prime = (v)* Z[j,k] + gen[j] - gen[k] 
            CNum3[j, k] = m.addConstr(lhs_3_prime >= 1)        
        
    m.update()   
        
    m.write("out.lp")
    # Gurobi Parameters (default)
    m.params.TimeLimit = 50*v
    #m.params.TimeLimit = 1800  
    m.params.MIPGap = 0.01
    #m.params.MIPGap = math.log(v)/(100*math.log(2.718))    
    #m.params.Threads =4
    m.params.OutputFlag = 1
    #m.params.VarBranch=1
    #m.Params.Aggregate = 0
    #m.Params.Presolve = 0  
    #m.Params.MIQCP=0
    #m.Params.NodeLimit=1000000
    
    #Optimizing
    
    start = timeit.default_timer()
    m.optimize()
    End = timeit.default_timer()
    ###########################################################################################
    ################################################# IP results  #############################
    ###########################################################################################    
    name_g = 'LNConicResults_L0_%s.txt' %(est)  
    f_g = open(name_g,'a')
    
     
    # Predicted DAG graph 
    Predicted_DAG = []
    for j, k in List_edges:
    #    print (j,k, Beta[j, k].x*Beta[j, k].x)
    #    print (j,k, S_var[j, k].x)
    #    print (j,k, g[j, k].x)
    #    print("******")
           
        if abs(Beta[j, k].x) >= 0.00001: ## and j!=k
            Predicted_DAG.append((j, k))
            print(j,",",k)
            f_g.write(str(j+1)+","+str(k+1)+"\n")
    B = np.zeros((v, v))
    for j, k in List_edges:
        if abs(Beta[j, k].x) >= 1e-5:
            B[j, k] = Beta[j, k].x
    ## Only for test np.trace((np.eye(v) - B).T @ (np.eye(v) - B) @ (x.T @ x) / 500) - 2 * np.linalg.det(np.eye(v) - B)

    predicted_dag = nx.DiGraph()
    predicted_dag.add_nodes_from(range(v))
    predicted_dag.add_edges_from(Predicted_DAG)
    estimated_cpdag = cd.DAG.from_nx(predicted_dag).cpdag().to_amat()
    SHD_cpdag = np.sum(np.abs(estimated_cpdag[0] - true_cpdag[0]))
    
    True_graph=list(G_Original.edges())[:]
    Predicted_graph=Predicted_DAG[:]
    print("True_graph---------------")
    print(True_graph)
    print("Predicted_graph---------------")
    print(Predicted_graph)
    FN = len(set(True_graph).difference(Predicted_graph))#an edge that should be there that is not there
    FP = len(set(Predicted_graph).difference(True_graph))#an edge that should not be there is there
    f_g.write("FN:"+str(FN)+"\n")
    f_g.write("FP:"+str(FP)+"\n")
    f_g.close()        
    ################################ Graph measures ########################
    
    learned_skeleton=Predicted_DAG[:]
    for item in Predicted_DAG:
        learned_skeleton.append((item[1],item[0]))
    
    True_skeleton=list(G_Original.edges())[:]
    for item in list(G_Original.edges()):
        True_skeleton.append((item[1],item[0]))        
    
    #Jacard similarity of Predicted DAG and True DAG
    Jacard_similarity_IP = float(len(set(Predicted_DAG).intersection(G_Original.edges()))) / (
    len(set(Predicted_DAG)) + len(set(G_Original.edges())) - len(set(Predicted_DAG).intersection(G_Original.edges())))
    
    shp1 = len(set(learned_skeleton).difference(True_skeleton))/2
    shp2 = len(set((True_skeleton)).difference(learned_skeleton))/2
    Reversed =  [(y, x) for x, y in Predicted_DAG]
    shp3 = len(set(list(G_Original.edges())).intersection(Reversed))    
    
    SHD= shp1+shp2+shp3
    SHDs = shp1 + shp2
    
    # False Positive (FP) is an edge that is not in the undirected skeleton of the true graph                                   
    FP = len(set(Predicted_DAG).difference(True_skeleton))
    # TP the corect estimation of direction 
    TP = len(set(Predicted_DAG).intersection(True_skeleton))
    # Length learned DAG
    len_P = len(Predicted_DAG)
    # Length True DAG
    len_T = len(G_Original.edges())
    # Length Reversed edges
    len_R = len(set(list(G_Original.edges())).intersection(Reversed))   
    # len_F False (F) is the set of non-edges in the ground truth graph
    len_F = v*(v-1)/2 - len_T
    
    FDR = float((len_R+FP))/(len_P+0.0000001)
    TRP = float(TP)/len_T
    FPR = float(len_R+FP)/(len_F+0.0005)

    ###########################################################
    
    #Predicted DAG plot 
    #G = nx.DiGraph()
    #G.add_nodes_from(range(v))
    #G.add_edges_from(Predicted_DAG)
    #plt.subplot(221)
    #plt.title("Predicted DAG_TO")
    #nx.draw(G, pos=Pos, with_labels=True)
    #plt.show()    
    
    IP_Obj = m.ObjVal
    # Save all important info regarding IP solution    
    sl_TO=[v, n, alpha, kk, float(format(Lamda, '.2f')), float(format(pmu, '.3f')),
           len(List_Original_edges),
           len(M_edges), m.NumConstrs, m.NumVars, m.NumBinVars, int(m.NodeCount), 
           float(format(m.ObjBound, '.2f')), float(format(m.ObjVal, '.2f')), 
           float(format(End - start, '.1f')), float(format(m.MIPGap, '.3f')), 
           float(format(Jacard_similarity_IP, '.2f')), len_T, len(Predicted_DAG), 
           float(format(FDR, '.2f')), float(format(TRP, '.2f')),float(format(FPR, '.2f')),
           SHD, SHDs, SHD_cpdag]
    
    s2_TO=[v, n, alpha, kk, float(format(Lamda, '.2f')), float(format(pmu, '.3f')),
           len(List_Original_edges),
           len(M_edges), m.NumConstrs, m.NumVars, m.NumBinVars, "NodeCount:" , int(m.NodeCount), 
           float(format(m.ObjBound, '.2f')), "Obj:" ,float(format(m.ObjVal, '.2f')), "Time:",
           float(format(End - start, '.1f')), "MIPGap:", float(format(m.MIPGap, '.3f')), 
           float(format(Jacard_similarity_IP, '.2f')), len_T, len(Predicted_DAG), 
           float(format(FDR, '.2f')), float(format(TRP, '.3f')), float(format(FPR, '.3f')), "SHD:",
           SHD, "SHDs:", SHDs, "SHD_cpdag:", SHD_cpdag]

    alpha_results = [v, n, alpha, kk, Lamda, End - start, m.MIPGap, SHD_cpdag, SHDs, TRP, FPR]

    ##########################################################################################
    ################################## LP Model & Result #####################################
    ##########################################################################################  

    for j, k in List_edges:
        g[j, k].vtype = GRB.CONTINUOUS        
        if j < k: 
            Z[j, k].vtype = GRB.CONTINUOUS
    
    #m.params.Threads =4
    m.params.OutputFlag=1
    
    start = timeit.default_timer()
    m.optimize()
    End = timeit.default_timer()

    CR_Obj = m.ObjVal

    #GAP = ((IP_Obj - CR_Obj) * 100 / CR_Obj)
    
    # Save all important info regarding LP solution    
    sl_TO.append(float(format(m.ObjVal, '.2f')))
    sl_TO.append(float(format(End - start, '.1f')))
    sl_TO.append(float(format(big_M, '.2f')))
    sl_TO.append(float(format(Big_M_obj, '.2f')))
        
    #sl_TO.append(float(format(GAP, '.2f')))
    LN_Solution[Lamda] = sl_TO
    
    m.reset()    #RESENT THE MODEL

    name = "LNConicResults_L0.txt"
    f = open(name,'a')
    f.write('\n' + str(LN_Solution[Lamda]))

    LN_Solution[Lamda] = s2_TO
    f.write('\n' + str(LN_Solution[Lamda]))

    f.close()
    return alpha_results
    
    
if __name__ == "__main__":
    # Input = []
    # for Opt_model in [OPT_LN_L1.Optimization, OPT_TO_L1.Optimization, OPT_CP_L1.Optimization, OPT_Linear_L1.Optimization]:

    # m = [6,8,9,15,14,16,18,20,27,27]
    # m = [6, 8, 9, 15, 14, 16, 18, 20, 27, 27, 56, 70]
    # datasets = ['1dsep', '2asia', '3bowling', '4insuranceSmall', '5rain', '6cloud', '7funnel', '8galaxy', '9insurance',
    #             '10factors', '11hfinder', '12hepar']
    # dt= 5#change here
    # Dataset = "5rain"#change here

    results = []
    for mm in [10,15,20]:
        for nn in [100]:
            for Alpha in [1,2,4]:
                for kk in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
                    Input = [mm, nn, Alpha, kk]
                    print(Input)
                    results.append(Optimization(Input))
    names = ["m", "n", "alpha", "k", "lambda", "Time", "RGAP", "d_cpdag", "SHDs", "TPR", "FPR"]
    results_df = pd.DataFrame(results, columns=names)

    results_df.to_csv("./alpha_results/MISOCP_alpha_results_4log(m).csv", index=False)

    # for i in [1]:
    #     dt = i + 1
    #     Dataset = datasets[i]
    #     for graph_type in ['Sparse']:
    #         for n in [500]:
    #             for lamda in [6.214*2]: #6.214 * 2
    #                 for k in [1]:
    #                     for estG in ["true"]:
    #                         Input = []
    #                         Input.append(m[dt - 1])
    #                         Input.append(n)
    #                         Input.append(lamda)  # lamda = ln m
    #                         Input.append(graph_type)
    #                         Input.append(Dataset)
    #                         Input.append(dt)
    #                         pmu = 0
    #                         Input.append(pmu)
    #                         Input.append(estG)
    #                         Optimization(Input)
