import causaldag as cd
import scipy
import cvxpy as cp
import time

from functions import *
from causaldag import rand, partial_correlation_suffstat, partial_correlation_test, MemoizedCI_Tester, gsp



if __name__ == '__main__':
    results = []
    est = 'true'
    for mm in [10, 15, 20]:
        for nn in [100]:
            for Alpha in [1,2,4]:
                # for kk in [10]:
                for kk in range(1, 31):
                    Input = [mm, nn, Alpha, kk]
                    data, True_B, moral, mgest = read_alpha(mm, nn, Alpha, kk)
                    nnodes, p = data.shape
                    nodes = set(range(nnodes))
                    suffstat = partial_correlation_suffstat(data)
                    ci_tester = MemoizedCI_Tester(partial_correlation_test, suffstat, alpha=1e-3)
                    if est == 'true':
                        possible_edges_true = tuple(zip(moral.values[:, 0] - 1, moral.values[:, 1] - 1))
                    else:
                        possible_edges_true = mat2ind(mgest, p)
                    fixed_gaps_true = set()
                    for i in range(nnodes):
                        for j in range(nnodes):
                            if (i, j) not in possible_edges_true:
                                fixed_gaps_true.add((i, j))

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
                    results_i = [mm, kk, Alpha, SHD_cpdag, TPR, FPR, end_i - start_i]
                    print(results_i)
                    results.append(results_i)
    names = ['m', 'k', 'alpha', 'd_cpdag', 'TPR', 'FPR', 'Time']
    results_df = pd.DataFrame(results, columns=names)
    print(results_df)
    results_df.to_csv("./experiment results/variance difference level/1-30/GSP_alpha_results_est_moral_1_30.csv", index=False)
