import numpy as np
import pandas as pd
import os
import pcalg
import networkx as nx
import glob


def read_alpha(m, n, alpha, k):
    """
    Read data for the test on changing level of variance difference.
    :param m:
    :param n:
    :param alpha:
    :param k: number of iteration
    :return:
    """
    file_path = 'E:/Northwestern/Research/independent study 1/dag/gurobi/Data/SyntheticDataNID/'
    file_name = "alpha/data_m_{}_n_{}_alpha_{}_iter_{}.csv".format(m, n, alpha, k)
    data = pd.read_csv(file_path + file_name, header=None)
    True_B = pd.read_table(file_path + "DAG_{}.txt".format(m), delimiter=" ", header=None)
    moral = pd.read_table(file_path + "Moral_DAG_{}.txt".format(m), delimiter=" ", header=None)
    return data, True_B, moral


def read_data(name, kk):
    """
    Read the real world datasets with dataset name and type of errors

    parameters:

        name: name of the dataset

        kk: kk-th dataset
    """
    file_path = 'E:/Northwestern/Research/independent study 1/dag/gurobi/Data/RealWorldDatasetsTXu/'
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


def ind2mat(edges, p):
    matrix = [[1 if (i, j) in set(list(map(tuple, edges))) else 0 for j in range(1, p + 1)] for i in range(1, p + 1)]
    return matrix


def tresh_cov(sigma):
    theta = np.linalg.inv(sigma)
    theta[np.abs(theta) < 0.3] = 0
    return theta


def mat2ind(mat, p):
    edges = [(i, j) for i in range(p) for j in range(p) if mat[i][j] == 1]
    return edges


def performance(A, Theta):
    """
    parameters:

    A: The estimated matrix.
    Theta: The ground truth matrix.

    """
    A, Theta = np.array(A), np.array(Theta)
    support_Theta = Theta != 0
    support_A = A != 0
    P = np.count_nonzero(support_Theta)
    p = Theta.shape[0]
    N = p*(p-1) - P
    TP = np.count_nonzero(np.multiply(support_Theta, support_A))
    TPR = TP / P
    FP = np.count_nonzero(support_A) - TP
    FPR = FP / N
    return TPR, FPR


def find_datasets(file_path):
    lists = os.listdir(file_path)
    lists = [file for file in lists if not file.startswith('.')]
    lists = sorted(lists, key=lambda s: int(''.join(filter(str.isdigit, s))))
    return lists


def collect_results(results, datasets):
    """
    Collect results from MIP_DAG_LN_NID()
    :param results: list of results
    :param datasets: list of dataset names
    :return:
    """
    results_eq = pd.DataFrame(results['equal'], columns=['RGAP', 'd_cpdag', 'SHDs', 'TPR', 'FPR', 'Time'])
    results_eq['network'] = datasets
    results_eq = results_eq.set_index('network')
    results_ineq = pd.DataFrame(results['unequal'], columns=['RGAP', 'd_cpdag', 'SHDs', 'TPR', 'FPR', 'Time'])
    results_ineq['network'] = datasets
    results_ineq = results_ineq.set_index('network')
    return results_eq, results_ineq


def orders(lst):
    return [int(''.join(filter(str.isdigit, s))) for s in lst]


def skeleton(dag):
    """
    Given a list of arcs in the dag, return the undirected skeleton.
    This is for the computation of SHDs
    :param dag: list or arcs with 0 or 1 entries
    :return: skeleton np.array
    """
    skeleton_array = np.array(dag) + np.array(dag).T
    return skeleton_array


def compute_SHD(learned_DAG, True_DAG, SHDs=False):
    """
    Compute the stuctural Hamming distrance, which counts the number of arc differences (
    additions, deletions, or reversals)

    :param learned_DAG: list of arcs, represented as adjacency matrix
    :param True_DAG: list of arcs
    :return: shd: integer, non-negative
    """
    if type(learned_DAG) == tuple:
        learned_DAG = learned_DAG[0]
    if type(True_DAG) == tuple:
        True_DAG = True_DAG[0]
    learned_arcs = mat2ind(learned_DAG, len(learned_DAG))
    true_arcs = mat2ind(True_DAG, len(True_DAG))
    learned_skeleton = learned_arcs.copy()
    for item in learned_arcs:
        learned_skeleton.append((item[1], item[0]))
    True_skeleton = true_arcs.copy()
    for item in true_arcs:
        True_skeleton.append((item[1], item[0]))

    shd1 = len(set(learned_skeleton).difference(True_skeleton)) / 2
    shd2 = len(set((True_skeleton)).difference(learned_skeleton)) / 2
    Reversed = [(y, x) for x, y in learned_arcs]
    shd3 = len(set(true_arcs).intersection(Reversed))

    shd = shd1 + shd2 + shd3
    if SHDs:
        return shd1 + shd2
    return shd


def read_B(method, dataset, est, l):
    """
    Read estimated Beta matrices and compute performance.

    method: MISOCP or MICP.
    dataset: name of the dataset
    est: true or corest. true means using true moral graph; corest means using estimated moral graph.
    :return:
    """
    file_path = 'E:/Northwestern/Research/independent study 1/dag/gurobi/MIP_DAG/'
    # B = pd.read_csv(file_path + "%s_results/%s_%s_%s.csv" % (method, method, dataset, est),
    #                 header=None)
    B = pd.read_csv(file_path + "%s_results/%s_%s_%s_%s.csv" % (method, method, dataset, est, np.round(l, 3)), header=None)
    data, True_B, moral, mgest = read_data(dataset, "unequal")
    n, p = data.shape
    True_B_mat = ind2mat(True_B.values, p)
    B_arcs = [[0 if B[j][i] == 0 or i == j else 1 for j in range(p)] for i in range(p)]
    # B_arcs = np.sign(np.abs(B))
    SHD = compute_SHD(B_arcs, True_B_mat)
    skeleton_estimated, skeleton_true = skeleton(B_arcs), skeleton(True_B_mat)
    SHDs = compute_SHD(skeleton_estimated, skeleton_true, True)
    TPR, FPR = performance(skeleton_estimated, skeleton_true)
    return SHD, SHDs, TPR, FPR


if __name__ == '__main__':
    print(read_B("MICP", "3bowling", "true", 0.1))
